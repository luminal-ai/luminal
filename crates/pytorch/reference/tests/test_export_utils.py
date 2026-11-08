from luminal_reference.export_utils import _run_decompositions_without_guards


class _Program:
    def __init__(self, guards):
        self._guards_code = list(guards)
        self.guards_seen_by_decomposition = None

    def run_decompositions(self, decomposition_table):
        self.guards_seen_by_decomposition = list(self._guards_code)
        return _Program(["__meta_utils_unknown_tensor1.shape[0] == 4"])


def test_each_decomposition_pass_clears_regenerated_input_guards():
    first = _Program(["args[0].shape[0] == 4"])

    second = _run_decompositions_without_guards(first, {})
    third = _run_decompositions_without_guards(second, {"op": "decomposition"})

    assert first.guards_seen_by_decomposition == []
    assert second.guards_seen_by_decomposition == []
    assert second._guards_code == []
    assert third._guards_code == []
