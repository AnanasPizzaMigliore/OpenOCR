import os
import sys
import torch
from ai_edge_torch import convert
from ai_edge_torch import quantization as quant

# Local project imports
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from tools.utility import ArgsParser
from tools.utils.logging import get_logger


def export_to_tflite_ai_edge(model, sample_input, save_dir, model_name, quant_type=None):
    """
    Export PyTorch model to TensorFlow Lite (.tflite) using ai-edge-torch.
    Optionally applies quantization.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.tflite")

    print("🔄 Converting PyTorch model to Edge Lite format...")
    model.eval()

    # Apply optional quantization
    quant_cfg = None
    if quant_type == "float16":
        quant_cfg = quant.float16()
        print("⚙️ Applying float16 quantization...")
    elif quant_type == "int8":
        quant_cfg = quant.int8()
        print("⚙️ Applying int8 quantization...")

    # Convert model using ai-edge-torch
    edge_model = convert(
        model,
        (sample_input,),
        quantization=quant_cfg
    )

    # Optional: test inference with the converted model
    with torch.no_grad():
        edge_output = edge_model(sample_input)
        print(f"Edge model output shape: {tuple(edge_output.shape)}")

    # Export to .tflite file
    edge_model.export(save_path)
    print(f"✅ TFLite model saved at: {save_path}")

    return save_path


def main(cfg, quant_type=None):
    _cfg = cfg.cfg
    logger = get_logger()
    global_config = _cfg['Global']
    export_dir = global_config.get('export_dir', '')

    # Load and prepare model
    if _cfg['Architecture']['algorithm'] == 'SVTRv2_mobile':
        from tools.infer_rec import OpenRecognizer
        model = OpenRecognizer(_cfg).model.eval().to('cpu')
        dummy_input = torch.randn([1, 3, 48, 320])
        export_dir = export_dir or os.path.join(global_config.get('output_dir', 'output'), 'export_rec')
        model_name = 'rec_model'

    elif _cfg['Architecture']['algorithm'] == 'DB_mobile':
        from tools.infer_det import OpenDetector
        model = OpenDetector(_cfg).model.eval().to('cpu')
        dummy_input = torch.randn([1, 3, 960, 960])
        export_dir = export_dir or os.path.join(global_config.get('output_dir', 'output'), 'export_det')
        model_name = 'det_model'

    else:
        raise ValueError("Unsupported algorithm. Must be 'SVTRv2_mobile' or 'DB_mobile'.")

    # Verify original output
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Original PyTorch output shape: {tuple(output.shape)}")

    # Export
    tflite_path = export_to_tflite_ai_edge(model, dummy_input, export_dir, model_name, quant_type)
    logger.info(f'✅ Finished exporting model to TFLite: {tflite_path}')


def parse_args():
    parser = ArgsParser()
    parser.add_argument("--quant_type", default=None, choices=["float16", "int8"],
                        help="Optional quantization type for TFLite export.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    quant_type = FLAGS.pop('quant_type', None)
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg, quant_type)
