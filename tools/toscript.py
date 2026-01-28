import os
import sys
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from tools.utility import ArgsParser
from tools.utils.logging import get_logger
import logging
# ✅ CORRECT (New Path)
from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType

# Standard Imports
#from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
#from executorch.backends.vulkan.test.utils import run_and_check_output

def to_torchscript(model, dummy_input, save_path='model.pte', method='trace'):
    """
    Convert a PyTorch model to ExecuTorch (Vulkan) and save it.
    """
    model = model.cpu().eval()
    example_inputs = (dummy_input,)

    # 1. Run PyTorch Baseline (Standard NCHW Layout)
    with torch.no_grad():
        torch_out = model(dummy_input)

    # 2. Export to Edge Graph
    print("⏳ Exporting to Edge IR...")
    exported_program = torch.export.export(model, example_inputs)

    # --- DEBUG SNIPPET START ---
    print("\n🔍 DEBUGGING GRAPH ARGUMENTS 🔍")
    # Adjust variable name 'edge_program' to matches yours (e.g., 'prog', 'exported_program')

    compile_options = {
        "force_fp16": True,
    }

    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    exec_prog = edge_program.to_executorch()

    atol = 1e-3
    rtol = 1e-3

    with open(save_path, "wb") as f:
        exec_prog.write_to_file(f)
    print(f"✅ ExecuTorch model saved to {save_path}")

    # --- VERIFICATION ---
    print("\n🔍 Running Verification...")
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    method = runtime.load_program(save_path).load_method("forward")

    # 6. Fix Memory Layout for Verification (THE CRITICAL FIX)
    # Vulkan GPU requires Channels Last (NHWC). Standard PyTorch uses NCHW.
    #dummy_input_vulkan = dummy_input.to(memory_format=torch.channels_last)

    # Execute using the NHWC input
    et_out = method.execute(dummy_input)
    et_out = et_out[0]

   

    # Handle Dictionary outputs (Common in Timm models)
    if isinstance(torch_out, dict):
        if 'maps' in torch_out:
            torch_tensor = torch_out['maps']
        else:
            torch_tensor = list(torch_out.values())[0]
    else:
        torch_tensor = torch_out

    print(f"PyTorch Output Shape:   {torch_tensor.shape}")
    print(f"ExecuTorch Output Shape: {et_out.shape}")
    
    print(f"PyTorch output:   {torch_tensor}")
    print(f"ExecuTorch output: {et_out}")

    diff = (torch_tensor - et_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n📊 Error Analysis (FP32 CPU vs FP16 GPU):")
    print(f"   Max Difference:  {max_diff:.5f}")
    print(f"   Mean Difference: {mean_diff:.5f}")

    # For Segmentation/Detection tasks, a max error < 0.5 is usually acceptable 
    # because the final output is often a probability map or bounding box coordinate.
    if mean_diff < 0.01: 
        print("✅ Model is NUMERICALLY STABLE for deployment.")
    else:
        print("⚠️ Warning: Large drift detected.")

    if torch_tensor.max().item() == 0:
        print("❌ ERROR: PyTorch output is strictly zero. Check model initialization.")
        return

    # Use relaxed tolerance for GPU (FP16) vs CPU (FP32)
    # Relaxed to 0.1 because mobile GPUs approximate floating point math
    is_correct = torch.allclose(torch_tensor, et_out, atol=atol, rtol=rtol)
    
    if is_correct:
        print("✅ Verification PASSED")
    else:
        print("❌ Verification FAILED (Values mismatch)")


def main(cfg):
    _cfg = cfg.cfg
    logger = get_logger()
    global_config = _cfg['Global']

    export_dir = global_config.get('export_dir', '')

    if _cfg['Architecture']['algorithm'] == 'SVTRv2_mobile':
        from tools.infer_rec import OpenRecognizer
        model = OpenRecognizer(_cfg).model
        dummy_input = torch.randn([1, 3, 48, 320], device='cpu')
        if not export_dir:
            export_dir = os.path.join(
                global_config.get('output_dir', 'output'), 'export_rec')
        save_path = os.path.join(export_dir, 'SVTRv2_cpu_fp32.pte')

    elif _cfg['Architecture']['algorithm'] == 'DB_mobile':
        from tools.infer_det import OpenDetector
        model = OpenDetector(_cfg).model
        torch.manual_seed(42)
        dummy_input = torch.randn([1, 3, 960, 960], device='cpu')
        if not export_dir:
            export_dir = os.path.join(
                global_config.get('output_dir', 'output'), 'export_det')
        save_path = os.path.join(export_dir, 'resnet50_gpu_fp32.pte')

    else:
        raise ValueError(
            f"Unsupported algorithm: {_cfg['Architecture']['algorithm']}"
        )

    os.makedirs(export_dir, exist_ok=True)
    to_torchscript(model, dummy_input, save_path)
    logger.info(f'✅ Finished exporting TorchScript model to {save_path}')


def parse_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg)