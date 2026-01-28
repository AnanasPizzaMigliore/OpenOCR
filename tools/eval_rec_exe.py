import os
import sys
import argparse
import yaml
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed
import random

# --- Path Setup ---
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# --- Project Imports ---
from ptflops import get_model_complexity_info
from tools.data import build_dataloader
from tools.utils.ckpt import load_ckpt, save_ckpt
from tools.utils.logging import get_logger
from tools.utils.stats import TrainingStats
from tools.utils.utility import AverageMeter

# --- EXECUTORCH IMPORTS ---
try:
    from executorch.runtime import Runtime, Program, Method
    HAS_EXECUTORCH = True
except ImportError:
    HAS_EXECUTORCH = False
    print("Warning: ExecuTorch not installed.")

# ------------------------------------------------------------------------------
# 1. EXECUTORCH WRAPPER
# ------------------------------------------------------------------------------
class ExecutorchModel:
    def __init__(self, pte_path):
        if not HAS_EXECUTORCH:
            raise ImportError("ExecuTorch is not installed.")
        
        print(f"Loading ExecuTorch model from: {pte_path}")
        from executorch.runtime import Runtime
        # Load the program and the method
        self.program = Runtime.get().load_program(pte_path)
        self.forward_method = self.program.load_method("forward")
        print("✅ ExecuTorch model loaded successfully.")

    def __call__(self, x):
        # 1. Prepare Input
        if x.device.type != 'cpu':
            x = x.cpu()
        
        # Ensure float32 if the model expects it (most Quantized models still take Float input and quantize internally)
        # If your model expects Int8 input, you must quantize x here manually.
        input_tensor = x.contiguous()

        # 2. Execute
        try:
            # execute() takes a list of inputs
            outputs = self.forward_method.execute([input_tensor])
        except Exception as e:
            print(f"\n❌ Execution Failed. Input Shape: {input_tensor.shape}")
            raise e
        
        # 3. Return Raw Output (Let Trainer decide how to wrap it)
        # Most OCR models return a single tensor (Logits or Probability Map)
        return outputs[0]

# ------------------------------------------------------------------------------
# 2. TRAINER CLASS
# ------------------------------------------------------------------------------
class Trainer(object):
    def __init__(self, cfg, mode='eval'):
        self.cfg = cfg.cfg
        
        # Automatically determine task from config if not explicitly passed
        self.task = self.cfg['Global'].get('model_type', 'rec') # defaults to rec if missing
        
        self.local_rank = (int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0)
        self.set_device(self.cfg['Global']['device'])
        
        self.mode = mode.lower()
        
        # --- CONFIGURATION CHECK ---
        self.use_executorch = self.cfg['Global'].get('use_executorch', False)
        self.executorch_model = None
        
        if torch.cuda.device_count() > 1 and 'train' in self.mode:
            torch.distributed.init_process_group(backend='nccl')
            torch.cuda.set_device(self.device)
            self.cfg['Global']['distributed'] = True
        else:
            self.cfg['Global']['distributed'] = False
            self.local_rank = 0

        self.cfg['Global']['output_dir'] = self.cfg['Global'].get('output_dir', 'output')
        os.makedirs(self.cfg['Global']['output_dir'], exist_ok=True)

        self.logger = get_logger(
            'openrec' if self.task == 'rec' else 'opendet',
            None # No log file for pure eval usually, or set path
        )

        # Standard Random Seed setup
        self.set_random_seed(self.cfg['Global'].get('seed', 48))

        # Build Eval Dataloader
        self.valid_dataloader = None
        if self.cfg['Eval']:
            self.valid_dataloader = build_dataloader(self.cfg, 'Eval', self.logger, task=self.task)
            self.logger.info(f'Valid dataloader has {len(self.valid_dataloader)} iters')

        # Initialize Model Components (PostProcess, Metrics, Loss)
        if self.task == 'rec':
            self._init_rec_model()
        elif self.task == 'det':
            self._init_det_model()
        else:
            raise NotImplementedError(f"Task {self.task} not supported")

        # --- MODEL LOADING LOGIC ---
        if self.use_executorch:
            pte_path = self.cfg['Global'].get('executorch_model_path')
            if not pte_path:
                raise ValueError("use_executorch is True but 'executorch_model_path' is missing in YAML.")
            
            if not os.path.exists(pte_path):
                raise FileNotFoundError(f"ExecuTorch model not found at: {pte_path}")
            
            self.executorch_model = ExecutorchModel(pte_path)
            self.logger.info(f"Using ExecuTorch backend: {pte_path}")
        else:
            # Fallback to standard PyTorch if use_executorch is False
            self.model = self.model.to(self.device)
            if self.cfg['Global']['distributed']:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, [self.local_rank], find_unused_parameters=False)
            
            # Load Pretrained
            path = self.cfg['Global'].get('pretrained_model')
            if path and os.path.exists(path):
                self.load_params(torch.load(path, map_location=self.device))

    def _init_rec_model(self):
        from openrec.losses import build_loss
        from openrec.metrics import build_metric
        from openrec.modeling import build_model
        from openrec.postprocess import build_post_process
        self.post_process_class = build_post_process(self.cfg['PostProcess'], self.cfg['Global'])
        char_num = self.post_process_class.get_character_num()
        self.cfg['Architecture']['Decoder']['out_channels'] = char_num
        self.model = build_model(self.cfg['Architecture'])
        self.loss_class = build_loss(self.cfg['Loss'])
        self.eval_class = build_metric(self.cfg['Metric'])

    def _init_det_model(self):
        from opendet.losses import build_loss
        from opendet.metrics import build_metric
        from opendet.modeling import build_model
        from opendet.postprocess import build_post_process
        self.post_process_class = build_post_process(self.cfg['PostProcess'], self.cfg['Global'])
        self.model = build_model(self.cfg['Architecture'])
        self.loss_class = build_loss(self.cfg['Loss'])
        self.eval_class = build_metric(self.cfg['Metric'])

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        random.seed(seed)
        np.random.seed(seed)

    def set_device(self, device):
        if device == 'gpu' and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.local_rank}')
        else:
            device = torch.device('cpu')
        self.device = device
    
    def load_params(self, params):
        self.model.load_state_dict(params)

    def eval(self):
        if not self.use_executorch:
            self.model.eval()
            
        with torch.no_grad():
            total_frame = 0.0
            total_time = 0.0
            pbar = tqdm(total=len(self.valid_dataloader), desc='Eval:', position=0, leave=True)
            
            for idx, batch in enumerate(self.valid_dataloader):
                # 1. Prepare Inputs
                if self.use_executorch:
                    batch_tensor = [batch[0].cpu()] # ExecuTorch inputs usually on CPU
                else:
                    batch_tensor = [t.to(self.device) for t in batch]
                
                batch_numpy = [t.numpy() for t in batch]
                
                # 2. Inference
                start = time.time()
                
                if self.use_executorch:
                    # PTE model usually expects pure tensor input, no 'data' args
                    preds = self.executorch_model(batch_tensor[0])
                    if idx == 0:
                        print(f"\n[DEBUG] Raw ExecuTorch Output Shape: {preds.shape}")
                        print(f"[DEBUG] Expected PostProcess Input: [Batch, Seq_Len, Num_Classes]")
                        # Check values briefly
                        print(f"[DEBUG] First few values: {preds.flatten()[:5]}")
                    #print(f"DEBUG: Output keys: {preds.keys()}") # <--- Add this temporarily
                    if self.task == 'det':
                        # Detection post-processing usually expects a dict with 'maps'
                        preds = {'maps': preds}
                    else:
                        # Recognition (REC) post-processing usually expects the raw Tensor (softmax/logits)
                        preds = preds
                        # Check if we need softmax (Post-processors usually handle logits, but verify if your export included softmax)
                        # preds = torch.softmax(preds, dim=2)
                else:
                    preds = self.model(batch_tensor[0], data=batch_tensor[1:])

                total_time += time.time() - start
                
                # 3. Post-Process
                # 'preds' from ExecuTorch needs to be compatible with post_process
                post_result = self.post_process_class(preds, batch_numpy)
                self.eval_class(post_result, batch_numpy)

                pbar.update(1)
                total_frame += len(batch[0])
            
            metric = self.eval_class.get_metric()

        pbar.close()
        metric['fps'] = total_frame / total_time
        return metric

# ------------------------------------------------------------------------------
# 3. MAIN EXECUTION BLOCK (AUTO-CONFIG)
# ------------------------------------------------------------------------------
class ConfigWrapper:
    def __init__(self, config_dict):
        self.cfg = config_dict
    def print_cfg(self, logger_func):
        pass
    def save(self, path, cfg):
        pass
    def __getitem__(self, item):
        return self.cfg[item]

if __name__ == '__main__':
    # 1. Determine Config Path
    #config_path = "/scratch/penghao/OpenOCR/configs/rec/svtrv2/repsvtr_ch.yml" # Default
    config_path = "/scratch/penghao/OpenOCR/configs/rec/svtrv2/svtrv2_ch.yml" # Default
    #config_path = "/scratch/penghao/OpenOCR/configs/det/dbnet/timm_real_mobilenetv4.yml" # Default
    if len(sys.argv) > 1:
        config_path = sys.argv[1] # Use argument if provided
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at '{config_path}'")
        print("Usage: python run_eval.py [path/to/config.yml]")
        sys.exit(1)

    print(f"Loading configuration from: {config_path}")

    # 2. Load YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 3. Validation / Overrides (Optional)
    # Ensure Global exists
    if 'Global' not in config_dict:
        config_dict['Global'] = {}

    # 4. Initialize Trainer
    # Note: We don't pass task/model/device explicitly here; 
    # The Trainer class now extracts them from the config_dict['Global']
    cfg_wrapper = ConfigWrapper(config_dict)
    
    try:
        trainer = Trainer(cfg_wrapper, mode='eval')
        
        # 5. Run Eval
        print(f"Starting Evaluation (Task: {trainer.task})...")
        metrics = trainer.eval()
        
        print("\n" + "="*40)
        print("FINAL METRICS:")
        print(metrics)
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()