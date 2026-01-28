import os
import sys
import torch
from PIL import Image
import torch
import torchvision.transforms as T
import cv2
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from tools.utility import ArgsParser
from tools.utils.logging import get_logger
from tools.data import build_dataloader

#from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
#from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.quantizer.vulkan_quantizer import get_symmetric_quantization_config,VulkanQuantizer
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

def preprocess_opendet(path):
    img = cv2.imread(path)         # BGR
    h, w = img.shape[:2]

    # Resize keeping aspect ratio
    target_size = 960
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))

    # Pad to 960×960
    pad = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad[:new_h, :new_w] = img

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return transform(pad).unsqueeze(0)


def to_torchscript(model, dummy_input, save_path='model.pth', method='trace'):
    """
    Convert a PyTorch model to TorchScript and save it.
    Args:
        model: The PyTorch model instance
        dummy_input: A tensor with the same shape as model input
        save_path: Path to save the .pt file
        method: 'trace' (default) or 'script'
    """
    model = model.to('cpu').eval()
    example_inputs = (dummy_input,)
    #training_ep = torch.export.export(model, (dummy_input,)).module() # (2)
    #quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    quantizer = VulkanQuantizer().set_global(get_symmetric_quantization_config())
    exported_program = torch.export.export(model, example_inputs)
    graph_module = exported_program.module()
    quantized_module = prepare_pt2e(graph_module, quantizer)

    #prepared_model = prepare_pt2e(training_ep, quantizer) # (3)

    #for cal_sample in [torch.randn_like(dummy_input)]: # Replace with representative model inputs
    #    prepared_model(cal_sample) # (4) Calibrate
    
    folder = '/scratch/penghao/datasets/Products-Real/evaluation/images'
    transform = T.Compose([
    T.Resize((960, 960)),
    T.ToTensor(),            # -> (3, 960, 960)
    ])
    count = 0
    max_samples = 20
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(folder, filename)
            img = preprocess_opendet(path)  # (1,3,960,960)
            quantized_module(img)                # (4) Calibrate
            count += 1
            print(f"Calibrated {count} samples")
            if count >= max_samples:
                break
    


    quantized_module = convert_pt2e(quantized_module) # (5)
    exported_program = torch.export.export(quantized_module, example_inputs)
    et_program = to_edge_transform_and_lower( # (6)
        exported_program,
        partitioner=[VulkanPartitioner()],
    ).to_executorch()

    with open(save_path, "wb") as file:
        et_program.write_to_file(file)
    #quantized_ep = torch.export.export(quantized_model, example_inputs)

    #torch.export.save(quantized_ep, save_path)
    
    print(f"✅ TorchScript model saved to {save_path}")
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    method = runtime.load_program(save_path).load_method("forward")
    et_out = method.execute([img])
    et_out = et_out[0]
    #torch_out = model(img)
    #loaded_quantized_ep = torch.export.load(save_path)
    #loaded_quantized_model = loaded_quantized_ep.module()
    torch_out = model(img)
    #et_out = loaded_quantized_model(img)
    #et_program = to_edge_transform_and_lower(
    #    quantized_ep,
    #    partitioner=[VulkanPartitioner()],
    #).to_executorch()
    #with open(save_path, "wb") as file:
    #    et_program.write_to_file(file)
    #from executorch.runtime import Runtime
    #runtime = Runtime.get()
    #method = runtime.load_program(save_path).load_method("forward")
    #outputs = method.execute([img])
    #print("exe outputs:", outputs[0])

    print("torch_out type:", type(torch_out))
    print("et_out type:", type(et_out))
    print("torch_out keys:", torch_out.keys())
    if isinstance(torch_out, torch.Tensor):
        print("torch_out shape:", torch_out.shape)  
    if isinstance(et_out, torch.Tensor):
        print("et_out:", et_out)
    if isinstance(et_out, dict):
        et_out = et_out['maps']
        print("er_out:", et_out)
    if isinstance(torch_out, dict):
        torch_out = torch_out['maps']
        print("torch_out:", torch_out)
    from torchao.quantization.utils import compute_error
    sqnr = compute_error(torch_out, et_out)
    print(f"SQNR between TorchScript and Executorch outputs: {sqnr} dB")
    #print(torch.allclose(torch_out, et_out, rtol=1e-4, atol=1e-5))



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
        save_path = os.path.join(export_dir, 'rec_model.pte')

    elif _cfg['Architecture']['algorithm'] == 'DB_mobile':
        from tools.infer_det import OpenDetector
        model = OpenDetector(_cfg).model
        dummy_input = torch.randn([1, 3, 960, 960], device='cpu')
        if not export_dir:
            export_dir = os.path.join(
                global_config.get('output_dir', 'output'), 'export_det')
        save_path = os.path.join(export_dir, 'repvit_gpu_int8.pte')

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
