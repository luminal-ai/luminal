The code in this folder is a proof of concept showing how Flash Attention can be discovered via dynamic search in the space of valid kernels. We take our egglog rules and show that we can rewrite naive attention as flash attention.

Our handwritten mapping then takes the internal representation we generate and converts that into both a CUDA and AMD kernel.

As far as we are aware, we are the first in the world to achieve this

To run the basic demo:
```sh
cargo run --release
```

To run the demo with a visualization of the graph in the browser after completion:
```sh
cargo run --release --features graphvis
```

To be done:
- writing a full mapping of all possible egglog rules to all kernels
- integrating luminal with more low-level hardware architectures