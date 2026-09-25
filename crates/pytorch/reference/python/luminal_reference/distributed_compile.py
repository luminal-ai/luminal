"""Leader-owned reference search for a complete AOT graph's local regions.

Ranks submit their local regions as one batch before native search. The leader
compiles each distinct program and scatters selected plans to their owners.
"""

import hashlib
import io
import json
from weakref import WeakKeyDictionary
from zipfile import ZipFile

import torch
import torch.distributed as dist

from .backend import _prepare_local_graph, compile_exported

_coordination_groups = WeakKeyDictionary()


def _without_graph_ids(value):
    """Drop only process-local IDs from torch.export's node provenance tree."""
    if isinstance(value, dict):
        return {
            key: _without_graph_ids(item)
            for key, item in value.items()
            if key != "graph_id"
        }
    if isinstance(value, list):
        return [_without_graph_ids(item) for item in value]
    return value


def _canonical_model_json(data):
    model = json.loads(data)

    def normalize(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "from_node" and isinstance(item, str):
                    try:
                        provenance = json.loads(item)
                    except json.JSONDecodeError:
                        result[key] = item
                    else:
                        result[key] = json.dumps(
                            _without_graph_ids(provenance),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                else:
                    result[key] = normalize(item)
            return result
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return json.dumps(normalize(model), sort_keys=True, separators=(",", ":")).encode()


def _semantic_program_digest(program, specs, buckets, options):
    """Hash PT2 contents, excluding derived archive and node provenance IDs."""
    digest = hashlib.sha256()
    with ZipFile(io.BytesIO(program)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("PT2 archive contains duplicate entries")
        model_entries = [name for name in names if name.endswith("/models/model.json")]
        if len(model_entries) != 1:
            raise ValueError("PT2 archive must contain one model.json")
        for name in sorted(names):
            if name.endswith("/.data/serialization_id"):
                # PyTorch derives this ID from the archive, including the
                # nonsemantic graph_id values normalized below.
                continue
            payload = archive.read(name)
            if name == model_entries[0]:
                payload = _canonical_model_json(payload)
            name_bytes = name.encode()
            digest.update(len(name_bytes).to_bytes(8, "big"))
            digest.update(name_bytes)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    digest.update(repr((specs, buckets, options)).encode())
    return digest.digest()


def enabled(group=None):
    return (
        dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1
    )


class DeferredRegion:
    def __init__(self, prepared, options, prepare_error=None):
        self.prepared = prepared
        self.options = options
        self.prepare_error = prepare_error
        self.model = None
        self.record = None

    def __call__(self, *args):
        if self.model is None:
            raise RuntimeError("distributed reference region was not installed")
        return self.model(*args)


class RegionBatch:
    def __init__(self, group=None):
        if dist.get_backend(group) != "gloo":
            raise RuntimeError(
                "leader reference compilation requires a Gloo process group"
            )
        self.jobs = []
        self.compilations = 0
        model_group = group if group is not None else dist.group.WORLD
        if model_group not in _coordination_groups:
            # Keep artifact traffic off the group's collective sequence. Model
            # collectives may still be in flight when a rank starts compiling
            # its next local region.
            _coordination_groups[model_group] = dist.new_group(
                ranks=dist.get_process_group_ranks(model_group),
                backend="gloo",
                use_local_synchronization=True,
            )
        self.group = _coordination_groups[model_group]
        self.leader = dist.get_global_rank(self.group, 0)

    def enqueue(self, gm, examples, **options):
        symbol_buckets = options.pop("symbol_buckets", None)
        try:
            prepared = _prepare_local_graph(gm, examples, symbol_buckets)
            job = DeferredRegion(prepared, options)
        except Exception as exc:  # noqa: BLE001
            # Defer the error until every rank can observe it together.
            job = DeferredRegion(None, options, str(exc))
        self.jobs.append(job)
        return job

    @staticmethod
    def _request(job):
        if job.prepare_error is not None:
            return False, job.prepare_error
        ep, inputs, _scalars, buckets = job.prepared
        output = io.BytesIO()
        torch.export.save(ep, output)
        specs = [(tuple(int(d) for d in value.shape), value.dtype) for value in inputs]
        return True, (output.getvalue(), specs, buckets, job.options)

    @staticmethod
    def _compile(request):
        program, specs, buckets, options = request
        ep = torch.export.load(io.BytesIO(program))
        inputs = [
            torch.ones(shape, dtype=dtype, device="cpu") for shape, dtype in specs
        ]
        model = compile_exported(
            ep,
            inputs,
            options.get("search_iterations"),
            search_log=options.get("search_log", False),
            max_intermediate_bytes=options.get("max_intermediate_bytes"),
            memory_budget_bytes=options.get("memory_budget_bytes"),
            dim_buckets=buckets,
        )
        return model, bytes(model._graph.serialize_compiled())

    def resolve(self):
        # AOT invokes backend compilation under FakeTensor/ShapeEnv tracing.
        # Gloo's object collectives inspect real size tensors and must not be
        # intercepted as symbolic computations.
        from torch._guards import tracing
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily(), tracing(None):
            self._resolve_real()

    def _resolve_real(self):
        local = []
        for job in self.jobs:
            try:
                local.append(self._request(job))
            except Exception as exc:  # noqa: BLE001
                local.append((False, str(exc)))
        all_requests = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(all_requests, local, group=self.group)
        rank = dist.get_rank(self.group)
        responses = None
        if rank == 0:
            responses = []
            cache = {}
            try:
                for owner, requests in enumerate(all_requests):
                    owner_responses = []
                    for index, (ready, request) in enumerate(requests):
                        if not ready:
                            raise RuntimeError(
                                f"rank {owner} region {index} export failed: {request}"
                            )
                        program, specs, buckets, options = request
                        key = _semantic_program_digest(program, specs, buckets, options)
                        if key not in cache:
                            try:
                                _model, artifact = self._compile(request)
                            except Exception as exc:
                                raise RuntimeError(
                                    f"rank {owner} region {index}: {exc}"
                                ) from exc
                            self.compilations += 1
                            cache[key] = artifact
                        owner_responses.append((True, cache[key]))
                    responses.append(owner_responses)
            except Exception as exc:  # noqa: BLE001
                # Every peer must receive a failure; never leave one blocked
                # in scatter while rank zero unwinds from a compiler error.
                responses = [
                    [(False, f"leader compilation failed: {exc}")] for _ in all_requests
                ]
        received = [None]
        dist.scatter_object_list(received, responses, src=self.leader, group=self.group)
        result = received[0]
        if result and not result[0][0]:
            raise RuntimeError(result[0][1])
        installation_error = None
        try:
            if len(result) != len(self.jobs):
                raise RuntimeError(
                    "leader returned a different number of reference regions"
                )
            for index, (job, (_ok, artifact)) in enumerate(zip(self.jobs, result)):
                ep, inputs, scalars, buckets = job.prepared
                job.model = compile_exported(
                    ep,
                    inputs,
                    job.options.get("search_iterations"),
                    scalars,
                    search_log=job.options.get("search_log", False),
                    max_intermediate_bytes=job.options.get("max_intermediate_bytes"),
                    memory_budget_bytes=job.options.get("memory_budget_bytes"),
                    dim_buckets=buckets,
                    artifact=artifact,
                )
                if job.record is not None:
                    job.record.buckets = job.model._graph.dim_buckets
        except Exception as exc:  # noqa: BLE001
            installation_error = f"rank {rank} failed to install leader plan: {exc}"
        installation_errors = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(
            installation_errors, installation_error, group=self.group
        )
        errors = [error for error in installation_errors if error is not None]
        if errors:
            raise RuntimeError("; ".join(errors))
