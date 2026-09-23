"""Preserve the exported calling convention around native runtime bindings."""

import copy

import torch
from torch.export.graph_signature import ConstantArgument, OutputKind
from torch.utils._pytree import tree_unflatten


class RegionModel:
    def __init__(self, artifact, *, user_indices=None, output_spec=None, **options):
        self.artifact = artifact
        self._native = copy.copy(artifact.native)
        static = options.get("static_outputs", False)
        external = options.get("external_cuda_graph", False)
        if artifact.device_type == "cuda":
            self._native.configure_region(
                static_outputs=static, external_cuda_graph=external
            )
        elif external:
            raise ValueError("external CUDA graph capture requires a CUDA region")
        self._graph = self._native._graph
        self._ep = options.get("program", artifact.program)
        self._user_indices = user_indices
        self._output_spec = output_spec
        self._static_outputs = options.get("static_outputs", False)
        self._saved_outputs = None

    @property
    def writeback_inputs(self):
        return {
            name: mutation
            for name, mutation in zip(
                self._graph.output_names, self._graph.output_mutations
            )
            if mutation is not None
        }

    def serialize_artifact(self):
        return self.artifact.serialize()

    def __call__(self, *args, **kwargs):
        if self._user_indices is not None:
            args = tuple(args[i] for i in self._user_indices)
        # Enforce the exported shape/range and constant-input guards before any
        # memory is bound. This also flattens the user's input pytree.
        flat, _ = self._ep._get_flat_args_with_check(args, kwargs)
        native_results = self._native(*flat)
        if self._output_spec is None:
            return native_results
        values = iter(native_results)
        results = []
        for spec in self._ep.graph_signature.output_specs:
            if spec.kind != OutputKind.USER_OUTPUT:
                continue
            results.append(
                spec.arg.value
                if isinstance(spec.arg, ConstantArgument)
                else next(values)
            )
        sentinel = object()
        if next(values, sentinel) is not sentinel:
            raise RuntimeError(
                "native runtime returned more outputs than the export signature"
            )
        if self._static_outputs and self.artifact.device_type == "cpu":
            if self._saved_outputs is None:
                self._saved_outputs = results
            else:
                for saved, current in zip(self._saved_outputs, results):
                    if isinstance(saved, torch.Tensor):
                        saved.copy_(current)
            results = self._saved_outputs
        return tree_unflatten(results, self._output_spec)
