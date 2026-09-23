"""AOT contract tests: gradients, saved tensors, and runtime parameter binding."""

import torch
from luminal_reference import Compiler


def test_forward_backward_and_optimizer_update():
    torch.manual_seed(71)
    backend = Compiler()

    def model(x, w):
        return (x @ w).square().sum()

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    weight = torch.nn.Parameter(torch.randn(4, 3))
    expected_weight = torch.nn.Parameter(weight.detach().clone())
    optimizer = torch.optim.SGD([weight], lr=0.01)
    expected_optimizer = torch.optim.SGD([expected_weight], lr=0.01)
    for _ in range(2):
        inputs = [torch.randn(2, 4, requires_grad=True) for _ in range(2)]
        expected_inputs = [x.detach().clone().requires_grad_() for x in inputs]
        # Two outstanding forwards must retain different saved activations.
        outputs = [compiled(x, weight) for x in inputs]
        expected = [model(x, expected_weight) for x in expected_inputs]
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference)
        sum(outputs).backward()
        sum(expected).backward()
        torch.testing.assert_close(weight.grad, expected_weight.grad)
        for actual, reference in zip(inputs, expected_inputs):
            torch.testing.assert_close(actual.grad, reference.grad)
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(weight, expected_weight)
        optimizer.zero_grad()
        expected_optimizer.zero_grad()

    assert {r.phase for r in backend.regions} == {"forward", "backward"}
    assert all(r.executions >= 4 for r in backend.regions)


def test_output_structure_and_missing_gradient():
    backend = Compiler()

    def model(x, unused):
        return x.square() + unused.detach(), None, x.sum(), 7

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    unused = torch.ones(2, requires_grad=True)
    square, missing, total, constant = compiled(x, unused)
    assert missing is None and constant == 7
    (square.sum() + total).backward()
    torch.testing.assert_close(x.grad, 2 * x.detach() + 1)
    assert unused.grad is None


def test_inference_preserves_forwarded_input_identity():
    backend = Compiler()

    def model(x):
        return {"input": x, "computed": (x + 1, x * 2), "empty": None}

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    x = torch.arange(4.0)
    actual = compiled(x)
    expected = model(x)
    assert actual["input"] is x
    assert actual["empty"] is None
    torch.testing.assert_close(actual["computed"], expected["computed"])
    assert {r.phase for r in backend.regions} == {"inference"}


def test_dynamic_forward_backward_reuses_graphs():
    backend = Compiler()

    def model(x):
        # Backward must retain symbolic dimensions even though this sum's
        # saved state contains no full-sized activation.
        return x.sum()

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
    for size in (3, 7, 5):
        x = torch.randn(size, 4, requires_grad=True)
        actual = compiled(x)
        torch.testing.assert_close(actual, x.sum())
        actual.backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
    assert len(backend.graphs) == 2
    assert {g.phase for g in backend.graphs} == {"forward", "backward"}


def test_compiler_trains_with_dynamic_shapes():
    compiled = torch.compile(
        lambda x: x.square().sum(), backend=Compiler(), fullgraph=True, dynamic=True
    )
    for size in (3, 8):
        x = torch.randn(size, 4, requires_grad=True)
        output = compiled(x)
        torch.testing.assert_close(output, x.square().sum())
        output.backward()
        torch.testing.assert_close(x.grad, 2 * x.detach())


def test_default_backend_dynamic_mutation_and_alias():
    import luminal_reference

    def model(x):
        x.add_(2)
        return x.view(-1), x * 3

    compiled = torch.compile(
        model, backend=luminal_reference.Compiler(), fullgraph=True, dynamic=True
    )
    for size in (3, 7):
        x = torch.randn(size, 4)
        expected_input = x.clone()
        expected = model(expected_input)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x, expected_input)
        assert actual[0].data_ptr() == x.data_ptr()


def test_dynamic_scalar_shape_output():
    import luminal_reference

    compiled = torch.compile(
        lambda x: (x + 1, x.shape[0] // 2),
        backend=luminal_reference.Compiler(),
        fullgraph=True,
        dynamic=True,
    )
    for rows in (6, 9, 12):
        x = torch.randn(rows, 4)
        tensor, count = compiled(x)
        torch.testing.assert_close(tensor, x + 1)
        assert count == rows // 2
