"""Regression tests for Modal workflow admission and concurrency, without GPUs."""

import ast
import itertools
import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
SPECS = {
    "test-cuda.yml": ("modal-ready", True),
    "test-python-cuda.yml": ("modal-ready", True),
    "test-full-cuda.yml": ("full-modal-ready", False),
}


def admits(condition, event_name, action, label, labels):
    """Evaluate the small boolean subset used by these GitHub job predicates."""
    values = {
        "event_name": event_name,
        "action": action,
        "label": label,
        "labels": labels,
        "contains": lambda items, item: item in items,
    }
    for source, target in (
        ("github.event.pull_request.labels.*.name", "labels"),
        ("github.event.label.name", "label"),
        ("github.event.action", "action"),
        ("github.event_name", "event_name"),
    ):
        condition = condition.replace(source, target)
    expression = ast.parse(
        condition.replace("&&", "and").replace("||", "or"), mode="eval"
    )
    allowed = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
    )
    for node in ast.walk(expression):
        if not isinstance(node, allowed):
            raise TypeError(f"Unsupported predicate syntax: {ast.dump(node)}")
        if isinstance(node, ast.Name) and node.id not in values:
            raise AssertionError(f"Unexpected predicate name: {node.id}")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id != "contains"
        ):
            raise AssertionError("Only contains() is supported")
    return bool(
        eval(
            compile(expression, "<workflow condition>", "eval"),
            {"__builtins__": {}},
            values,
        )
    )


class ModalWorkflowTests(unittest.TestCase):
    def test_admission_matrix_and_both_label_orders(self):
        labels = ("modal-ready", "full-modal-ready")
        label_sets = [[], [labels[0]], [labels[1]], list(labels)]
        for filename, (required, allows_push) in SPECS.items():
            workflow = yaml.load(
                (WORKFLOWS / filename).read_text(), Loader=yaml.BaseLoader
            )
            self.assertEqual(
                workflow["on"]["pull_request_target"]["types"],
                ["labeled", "synchronize"],
            )
            self.assertIn("workflow_dispatch", workflow["on"])
            self.assertEqual(
                workflow["on"]["pull_request_target"]["branches"], ["main"]
            )
            if allows_push:
                self.assertEqual(workflow["on"]["push"]["branches"], ["main"])
            else:
                self.assertNotIn("push", workflow["on"])
            for job_name, job in workflow["jobs"].items():
                for action, added, current in itertools.product(
                    ["labeled", "synchronize", "opened", "unlabeled"],
                    [*labels, "bug"],
                    label_sets,
                ):
                    expected = required in current and (
                        action == "synchronize"
                        or (action == "labeled" and added == required)
                    )
                    with self.subTest(
                        workflow=filename,
                        job=job_name,
                        action=action,
                        added=added,
                        labels=current,
                    ):
                        self.assertEqual(
                            admits(
                                job["if"], "pull_request_target", action, added, current
                            ),
                            expected,
                        )
                for event, expected in [
                    ("push", allows_push),
                    ("workflow_dispatch", True),
                    ("pull_request", False),
                ]:
                    self.assertEqual(admits(job["if"], event, "", "", []), expected)
                # GitHub can include both labels in each simultaneous labeled event.
                for order in itertools.permutations(labels):
                    self.assertEqual(
                        sum(
                            admits(
                                job["if"],
                                "pull_request_target",
                                "labeled",
                                added,
                                list(labels),
                            )
                            for added in order
                        ),
                        1,
                    )

    def test_concurrency_is_per_job_and_per_pr(self):
        groups = []
        for filename in SPECS:
            workflow = yaml.load(
                (WORKFLOWS / filename).read_text(), Loader=yaml.BaseLoader
            )
            # A skipped unrelated-label workflow must not cancel its active GPU job.
            self.assertNotIn("concurrency", workflow)
            for job in workflow["jobs"].values():
                concurrency = job["concurrency"]
                self.assertEqual(concurrency["cancel-in-progress"], "true")
                group = concurrency["group"]
                suffix = "${{ github.event.pull_request.number || github.run_id }}"
                self.assertTrue(group.endswith(suffix))
                self.assertNotIn("github.job", group)
                groups.append(group)
        # In particular, both full-suite jobs must run rather than cancel each other.
        self.assertEqual(len(groups), len(set(groups)))


if __name__ == "__main__":
    unittest.main()
