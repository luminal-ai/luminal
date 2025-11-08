# Luminal HLO

The Luminal HLO crate alllows you to convert a model in StableHLO representation into a luminal `Graph`. 
Eventually, this will allows you to import any PyTorch model into Luminal using `import_hlo`. 
Currently, this crate only support a subset of [StableHLO ops](https://openxla.org/stablehlo/spec#ops), but that's sufficient to run many popular models, see [supported ops](#supported-stablehlo-ops) below.

## Exporting

Use the provided code from [Torch Export to StableHLO](https://docs.pytorch.org/xla/master/features/stablehlo.html) to export a model into StableHLO representation:

```py
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

with open("squeezenet.mlir", "w") as f:
    f.write(stablehlo.mlir_module())
```

This will output a StableHLO module, but includes a complicated nested structure, which we'll need to flatten to simplify importing.

Following the [build instructions](https://github.com/openxla/stablehlo?tab=readme-ov-file#build-instructions) on the `openxla/stablehlo` github repository, install the `stablehlo-opt` executable. Once installed you can pass you're model with the following flags to flatten your model.

```
./build/bin/stablehlo-opt squeezenet.mlir \
    --inline \
    --symbol-dce \
    --canonicalize \
    -o squeezenet_flat.mlir
```

## Importing

To import your model into luminal simply using the `import_hlo` function within the `lumanl_hlo` crate.

```rs
use luminal::prelude::*;
use luminal_hlo::import_hlo;

fn main() {
    // Import StableHLO model
    let (mut cx, inputs) = import_hlo("model_flat.mlir");

    // Set model inputs...

    cx.execute()
}
```

In the code above, `cx` is a `Graph`, which defines all the model computation. `inputs` is a map of GraphTensor by their StableHLO argument name.

## Supported Ops

Please view [src/utils.rs](./src/utils.rs) for supported ops.
