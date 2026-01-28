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
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
HAS_EXECUTORCH = True



# ------------------------------------------------------------------------------
# TRAINER CLASS
# ------------------------------------------------------------------------------
class Trainer(object):
    def __init__(self, cfg, mode='eval'):
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

        self.output_dir = self.cfg['Global'].get('output_dir', 'output_executorch_ptq')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = get_logger('openrec', log_file=os.path.join(self.output_dir, 'train.log'))
        self.set_random_seed(self.cfg['Global'].get('seed', 48))

        # 2. Build Dataloaders
        if 'train' in self.mode:
            self.train_dataloader = build_dataloader(self.cfg, 'Train', self.logger, task=self.task)
        if 'eval' in self.mode:
            self.eval_dataloader = build_dataloader(self.cfg, 'Eval', self.logger, task=self.task)
        
        # 3. Model Init
        if self.task == 'rec':
            self._init_rec_model()
        elif self.task == 'det':
            self._init_det_model()

        # 4. Load Pretrained Float Model
        path = self.cfg['Global'].get('pretrained_model')
        if path and os.path.exists(path):
            self.logger.info(f"Loading pretrained float model: {path}")
            if self.cfg['Architecture']['algorithm'] == 'SVTRv2_mobile':
                from tools.infer_rec import OpenRecognizer
                self.model = OpenRecognizer(self.cfg).model

            elif self.cfg['Architecture']['algorithm'] == 'DB_mobile':
                from tools.infer_det import OpenDetector
                self.model = OpenDetector(self.cfg).model

        else:
            self.logger.warning("⚠️ No pretrained model found. QAT requires a converged model to work well.")

        self.model.to(self.device)

    def run_executorch_ptq(self):
        """
        ExecuTorch PT2E QAT Workflow (Single Device)
        """
        if not HAS_EXECUTORCH:
            raise ImportError("Cannot run QAT: ExecuTorch or TorchAO is missing.")

        self.logger.info("--- Step 1: Capturing Model Graph (export_for_training) ---")
        
        # Fetch one sample batch for tracing
        sample_batch = next(iter(self.eval_dataloader))
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
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        
        # Inject Fake Quantize Nodes
        ptq_model = prepare_pt2e(exported_model.to('cpu'), quantizer)
        ptq_model = ptq_model.to(self.device)

        self.model.to(self.device)

        self.logger.info("--- Step 3: QAT Fine-Tuning Loop ---")
        


        for idx, batch in enumerate(self.eval_dataloader):
            #batch_tensor = [t.to(self.device) for t in batch]
            #batch_numpy = [t.numpy() for t in batch]
            images = batch[0].to(self.device)
            # Note: 'targets' are used for loss, not passed to the captured graph forward()
            
            # Forward (Graph Execution)
            preds = ptq_model(images)
            preds_torch = self.model(images)
            if idx >= 20:
                break

        self.logger.info("--- Step 4: Convert and Export .pte ---")
        
        # Move to CPU for final conversion
        ptq_model = ptq_model.cpu()
        quantized_model = convert_pt2e(ptq_model)
        
        # Export .pte
        try:
            final_inputs = (sample_images.cpu(),)

            et_program = to_edge_transform_and_lower( # (6)
            torch.export.export(quantized_model, final_inputs),
            partitioner=[XnnpackPartitioner()],
            ).to_executorch()

            
            pte_path = os.path.join(self.output_dir, "export_rec/repvit_cpu_int8.pte")
            with open(pte_path, "wb") as f:
                f.write(et_program.buffer)
                
            self.logger.info(f"✅ ExecuTorch QAT model saved: {pte_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export .pte: {e}")


        from executorch.runtime import Runtime
        runtime = Runtime.get()
        method = runtime.load_program(pte_path).load_method("forward")
        et_out = method.execute(final_inputs)
        self.model.to("cpu")
        torch_out = self.model(sample_images.cpu())    
        from torchao.quantization.utils import compute_error
        sqnr = compute_error(et_out[0], torch_out)
        print(f"SQNR between TorchScript and Executorch outputs: {sqnr} dB")

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
    config_path = "/scratch/penghao/OpenOCR/configs/rec/svtrv2/repsvtr_ch.yml" 
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if 'Global' not in config_dict: config_dict['Global'] = {}
    
    # Force single gpu in config
    config_dict['Global']['distributed'] = False

    trainer = Trainer(ConfigWrapper(config_dict), mode='eval')
    
    print("Starting Single-Device ExecuTorch QAT...")
    trainer.run_executorch_ptq()