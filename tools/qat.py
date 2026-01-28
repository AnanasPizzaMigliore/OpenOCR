import os
import sys
import yaml
import time
import numpy as np
from tqdm import tqdm
import torch
import random
import copy
import pdb
# --- Path Setup ---
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# --- Project Imports ---
from tools.data import build_dataloader
from tools.utils.ckpt import load_ckpt, save_ckpt
from tools.utils.logging import get_logger
from tools.utils.utility import AverageMeter

# --- EXECUTORCH & QAT IMPORTS ---

from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge
HAS_EXECUTORCH = True
from torchao.quantization.pt2e import move_exported_model_to_eval, move_exported_model_to_train


# ------------------------------------------------------------------------------
# TRAINER CLASS
# ------------------------------------------------------------------------------
class Trainer(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg.cfg
        self.mode = mode.lower()
        
        # 1. Basic Setup (No Distributed)
        self.task = self.cfg['Global'].get('model_type', 'rec')
        
        # Simple Device Selection
        if self.cfg['Global']['device'] == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
            
        self.cfg['Global']['distributed'] = False # Force False

        self.output_dir = self.cfg['Global'].get('output_dir', 'output_executorch_qat')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = get_logger('openrec', log_file=os.path.join(self.output_dir, 'train.log'))
        self.set_random_seed(self.cfg['Global'].get('seed', 48))

        # 2. Build Dataloaders
        if 'train' in self.mode:
            self.train_dataloader = build_dataloader(self.cfg, 'Train', self.logger, task=self.task)
        
        # 3. Model Init
        if self.task == 'rec':
            self._init_rec_model()
        elif self.task == 'det':
            self._init_det_model()

        # 4. Load Pretrained Float Model
        path = self.cfg['Global'].get('pretrained_model')
        if path and os.path.exists(path):
            self.logger.info(f"Loading pretrained float model: {path}")
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.logger.warning("⚠️ No pretrained model found. QAT requires a converged model to work well.")

        self.model.to(self.device)

    def run_executorch_qat(self):
        """
        ExecuTorch PT2E QAT Workflow (Single Device)
        """
        if not HAS_EXECUTORCH:
            raise ImportError("Cannot run QAT: ExecuTorch or TorchAO is missing.")

        self.logger.info("--- Step 1: Capturing Model Graph (export_for_training) ---")
        
        # Fetch one sample batch for tracing
        sample_batch = next(iter(self.train_dataloader))
        sample_images = sample_batch[0].to(self.device)
        example_args = (sample_images,)

        self.model.eval()
        
        try:
            # export_for_training replaces torch.jit.trace but keeps gradients alive
            exported_model = export(self.model, example_args).module()
        except Exception as e:
            self.logger.error("Failed to export model for training. Ensure model is traceable.")
            raise e

        self.logger.info("--- Step 2: Preparing QAT (XNNPACK Quantizer) ---")
        
        # Configure XNNPACK Quantizer
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
        
        # Inject Fake Quantize Nodes
        qat_model = prepare_qat_pt2e(exported_model.to('cpu'), quantizer)
        qat_model = qat_model.to(self.device)

        self.model.to(self.device)
        
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-5)

        self.logger.info("--- Step 3: QAT Fine-Tuning Loop ---")
        
        epochs = 1
        code_path = os.path.join(self.output_dir, "qat_model_code.py")
        structure_path = os.path.join(self.output_dir, "qat_model_parameters.txt")
        
        for epoch in range(epochs):
            move_exported_model_to_train(qat_model)
            pbar = tqdm(total=len(self.train_dataloader), desc=f'QAT Ep {epoch+1}/{epochs}', position=0)

            for idx, batch in enumerate(self.train_dataloader):
                #batch_tensor = [t.to(self.device) for t in batch]
                #batch_numpy = [t.numpy() for t in batch]
                images = batch[0].to(self.device)
                # Note: 'targets' are used for loss, not passed to the captured graph forward()
                
                optimizer.zero_grad()
                
                # Forward (Graph Execution)
                preds = qat_model(images)
                preds_torch = self.model(images)
                #print(f"QAT keys: {preds.keys()}")
                #print(f"Torch keys: {preds_torch.keys()}")
                q_map = preds['maps']        # Extract tensor
                f_map = preds_torch['maps']  # Extract tensor
                print(f"  QAT Model:   {q_map.shape}")
                print(f"  Float Model: {f_map.shape}")
                print(f"q_map sample: {q_map.flatten()[0:5]}")

                pdb.set_trace()
                
                # Calculate Loss
                batch = [t.to(self.device) for t in batch]
                loss_dict = self.loss_class(preds_torch, batch)
                #loss_dict = self.loss_class(preds, batch)
                loss = loss_dict['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss.update(loss.item(), images.size(0))
                pbar.set_postfix(loss=total_loss.avg)
                pbar.update(1)
            
            pbar.close()
            
            # Save Checkpoint
            torch.save(qat_model.state_dict(), os.path.join(self.output_dir, f'qat_ep{epoch}.pth'))

        self.logger.info("--- Step 4: Convert and Export .pte ---")
        
        # Move to CPU for final conversion
        qat_model = qat_model.cpu()
        quantized_model = convert_pt2e(qat_model)
        
        # Export .pte
        try:
            final_inputs = (sample_images.cpu(),)
            edge_prog = to_edge(export(quantized_model, final_inputs))
            et_prog = edge_prog.to_executorch()
            
            pte_path = os.path.join(self.output_dir, "model_qat.pte")
            with open(pte_path, "wb") as f:
                f.write(et_prog.buffer)
                
            self.logger.info(f"✅ ExecuTorch QAT model saved: {pte_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export .pte: {e}")

    # --- Helper Methods ---
    def _init_rec_model(self):
        from openrec.losses import build_loss
        from openrec.modeling import build_model
        from openrec.postprocess import build_post_process
        self.post_process_class = build_post_process(self.cfg['PostProcess'], self.cfg['Global'])
        char_num = self.post_process_class.get_character_num()
        self.cfg['Architecture']['Decoder']['out_channels'] = char_num
        self.model = build_model(self.cfg['Architecture'])
        self.loss_class = build_loss(self.cfg['Loss'])

    def _init_det_model(self):
        from opendet.losses import build_loss
        from opendet.modeling import build_model
        from opendet.postprocess import build_post_process
        self.post_process_class = build_post_process(self.cfg['PostProcess'], self.cfg['Global'])
        self.model = build_model(self.cfg['Architecture'])
        self.loss_class = build_loss(self.cfg['Loss'])

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        random.seed(seed)
        np.random.seed(seed)

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
class ConfigWrapper:
    def __init__(self, config_dict):
        self.cfg = config_dict
    def __getitem__(self, item):
        return self.cfg[item]

if __name__ == '__main__':
    config_path = "/scratch/penghao/OpenOCR/configs/det/dbnet/timm_real_repvit.yml" 
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if 'Global' not in config_dict: config_dict['Global'] = {}
    
    # Force single gpu in config
    config_dict['Global']['distributed'] = False

    trainer = Trainer(ConfigWrapper(config_dict), mode='train')
    
    print("Starting Single-Device ExecuTorch QAT...")
    trainer.run_executorch_qat()