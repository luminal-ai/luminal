"""Reference plans are shared only when exported program semantics match."""

import io
import json
from zipfile import ZipFile

from luminal_reference.distributed_compile import _semantic_program_digest


def _pt2(*, graph_id, op="aten.mm.default", constant=b"weight", archive_id=b"id"):
    provenance = [
        {
            "name": "mm",
            "target": op,
            "graph_id": graph_id,
            "from_node": [{"name": "matmul", "graph_id": graph_id + 1}],
        }
    ]
    model = {
        "graph": {
            "nodes": [{"target": op, "metadata": {"from_node": json.dumps(provenance)}}]
        }
    }
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("archive/models/model.json", json.dumps(model))
        archive.writestr("archive/data/constants/0", constant)
        archive.writestr("archive/.data/serialization_id", archive_id)
    return output.getvalue()


def test_pt2_digest_ignores_only_node_provenance_ids():
    def digest(program, *, options=None):
        return _semantic_program_digest(
            program, [((4, 4), "float32")], {}, options or {}
        )

    first = digest(_pt2(graph_id=1, archive_id=b"one"))
    assert first == digest(_pt2(graph_id=2, archive_id=b"two"))
    assert first != digest(_pt2(graph_id=1, op="aten.add.Tensor"))
    assert first != digest(_pt2(graph_id=1, constant=b"other"))
    assert first != digest(_pt2(graph_id=1), options={"search_iterations": 2})
