from torchvision.models import squeezenet1_0
from safetensors.torch import save_file

model = squeezenet1_0(weights="IMAGENET1K_V1").eval()

state = {}
for name, param in model.named_parameters():
    state[name] = param.detach().cpu()

save_file(state, "squeezenet1_0.safetensors")
print("Saved squeezenet1_0.safetensors with", len(state), "tensors")
