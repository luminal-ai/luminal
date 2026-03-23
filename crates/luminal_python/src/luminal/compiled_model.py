"""CompiledModel wrapper for the Rust OnnxGraphResult."""

from typing import List

import torch


class CompiledModel:
    """Wrapper around OnnxGraphResult that handles PyTorch tensor conversion."""

    def __init__(self, graph_result):
        """Initialize with a compiled OnnxGraphResult from Rust.

        Args:
            graph_result: The OnnxGraphResult from luminal_python.process_onnx()
        """
        self._graph = graph_result
        self._input_names = graph_result.input_names
        self._output_names = graph_result.output_names
        self._output_shapes = graph_result.output_shapes
        self._has_dynamic_dims = graph_result.has_dynamic_dims

    def set_dim(self, param_name: str, value: int) -> None:
        """Set a dynamic dimension value by its ONNX param name."""
        self._graph.set_dim(param_name, value)

    @property
    def has_dynamic_dims(self) -> bool:
        return self._has_dynamic_dims

    @property
    def dim_params(self) -> List[str]:
        return self._graph.dim_params

    def __call__(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """Execute the compiled model with PyTorch tensor inputs.

        Args:
            *inputs: PyTorch tensors matching the model's input signature

        Returns:
            List of PyTorch tensors containing the model outputs
        """
        if len(inputs) != len(self._input_names):
            raise ValueError(
                f"Expected {len(self._input_names)} inputs, got {len(inputs)}"
            )

        input_device = inputs[0].device if inputs else torch.device("cpu")

        # Set input data
        for name, tensor in zip(self._input_names, inputs):
            # Convert to contiguous float32 numpy array (move to CPU first for CUDA tensors)
            arr = tensor.detach().cpu().contiguous().float().numpy()
            data = arr.flatten().tolist()
            self._graph.set_input(name, data)

        # Run the graph
        self._graph.run()

        # Get output shapes — resolve dynamically if needed
        if self._has_dynamic_dims:
            output_shapes = self._graph.resolve_output_shapes()
        else:
            output_shapes = self._output_shapes

        # Get outputs and convert back to PyTorch tensors on the same device as inputs
        outputs = []
        for name, shape in zip(self._output_names, output_shapes):
            data = self._graph.get_output(name)
            tensor = (
                torch.tensor(data, dtype=torch.float32)
                .reshape(tuple(shape))
                .to(input_device)
            )
            outputs.append(tensor)

        # Return as a tuple (TorchDynamo expects tuple return from backend callables)
        return tuple(outputs)
