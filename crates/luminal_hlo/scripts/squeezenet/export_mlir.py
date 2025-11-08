from torch.export import export
import torch
import torchvision
import torchax as tx
import torchax.export

# Define a PyTorch model or select one from torchvision.models
squeezenet1_0 = torchvision.models.squeezenet1_0()
squeezenet1_0.eval()

dummy = (torch.randn(1, 3, 224, 224),)
output = squeezenet1_0(*dummy)
exported = export(squeezenet1_0, dummy)

weights, stablehlo = tx.export.exported_program_to_stablehlo(exported)

with open("squeezenet1_0.mlir", "w") as f:
    f.write(stablehlo.mlir_module())