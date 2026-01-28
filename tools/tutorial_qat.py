import torch
import torch.nn as nn
import os
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

# Helper for Mode Switching
try:
    from torchao.quantization.pt2e import move_exported_model_to_train, move_exported_model_to_eval
except ImportError:
    from torchao.quantization.pt2e import allow_exported_model_train_eval
    def move_exported_model_to_train(model):
        allow_exported_model_train_eval(model)
        model.train()
    def move_exported_model_to_eval(model):
        allow_exported_model_train_eval(model)
        model.eval()

# Define Model
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

example_inputs = (torch.randn(1, 3, 224, 224),)
model = M()
model.eval()

print("1. Exporting model...")
# Standard Export
exported_prog = torch.export.export(model, example_inputs)
exported_model = exported_prog.module()

# --- NO MANUAL CLEANING NEEDED IF TORCHAO IS UPDATED ---

print("2. Preparing QAT...")
quantizer = XNNPACKQuantizer().set_global(
    get_symmetric_quantization_config(is_qat=True)
)
# This should now succeed natively
prepared_model = prepare_qat_pt2e(exported_model, quantizer)
print("✅ Fusion Successful!")

print("3. Training...")
optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.01)
move_exported_model_to_train(prepared_model)

for epoch in range(10):
    input_data = torch.randn(1, 3, 224, 224)
    optimizer.zero_grad()
    output = prepared_model(input_data)
    loss = output.mean()
    loss.backward()
    optimizer.step()
    print(f"   Epoch {epoch}: Loss {loss.item():.4f}")

print("4. Converting...")
move_exported_model_to_eval(prepared_model)
quantized_model = convert_pt2e(prepared_model)
print("✅ Done.")