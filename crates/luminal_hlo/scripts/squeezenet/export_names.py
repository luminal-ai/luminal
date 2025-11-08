import re, json
from torchvision.models import squeezenet1_0

model = squeezenet1_0(weights="IMAGENET1K_V1").eval()
params = list(model.named_parameters())

mlir_file = "squeezenet1_0.mlir"
with open(mlir_file) as f:
    text = f.read()

args = re.findall(r"%arg\d+", text)

if len(args) != len(params):
    print(f"WARNING: arg count {len(args)} != param count {len(params)}")

mapping = {arg: pname for arg, (pname, _) in zip(args, params)}

with open("squeezenet1_0_names.json", "w") as f:
    json.dump(mapping, f, indent=2)

print("Wrote squeezenet1_0_names.json")
