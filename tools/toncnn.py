import os
import sys
import torch
import pnnx

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from tools.utility import ArgsParser
from tools.utils.logging import get_logger


def to_torchscript(model, dummy_input, save_path='model.pt'):
    """
    Convert a PyTorch model to TorchScript and save it.
    Args:
        model: The PyTorch model instance
        dummy_input: A tensor with the same shape as model input
        save_path: Path to save the .pt file
        method: 'trace' (default) or 'script'
    """
    model = model.to('cpu').eval()
    opt_model = pnnx.export(model, save_path, dummy_input)
    print(f"✅ TorchScript model saved to {save_path}")

class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return list(out.values())[0]
        return out


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
        save_path = os.path.join(export_dir, 'rec_model.pt')

    elif _cfg['Architecture']['algorithm'] == 'DB_mobile':
        from tools.infer_det import OpenDetector
        model = OpenDetector(_cfg).model
        dummy_input = torch.randn([1, 3, 960, 960], device='cpu')
        if not export_dir:
            export_dir = os.path.join(
                global_config.get('output_dir', 'output'), 'export_det')
        save_path = os.path.join(export_dir, 'det_model.pt')
        model = ExportWrapper(model).eval().cpu()

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
