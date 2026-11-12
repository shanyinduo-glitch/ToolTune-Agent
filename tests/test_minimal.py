from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tooltune.environment import OrderToolEnvironment
from tooltune.evaluation import cases, heldout_unseen_cases, unseen_cases, independent_order_cases, independent_unseen_cases, expanded_independent_order_cases, expanded_independent_unseen_cases, challenge_order_cases, challenge_unseen_cases
from tooltune.schema import read_jsonl
from scripts.audit_hermes import parse_calls


def test_seed_jsonl_is_valid():
    examples = read_jsonl(Path(__file__).parents[1] / "data/raw/examples.jsonl")
    assert len(examples) == 1
    assert examples[0].target_action["name"] == "get_order"


def test_confirmation_gate_and_trace():
    env = OrderToolEnvironment()
    assert env.execute("cancel_order", {"order_id": "A1001", "confirmed": True})["error"] == "confirmation_required"
    assert env.execute("cancel_order", {"order_id": "A1001", "confirmed": False})["error"] == "confirmation_required"
    assert env.confirm_cancellation("A1001", env.session_id, True)
    assert env.execute("cancel_order", {"order_id": "A1001", "confirmed": True})["status"] == "cancelled"
    assert len(env.trace) == 3


def test_expanded_evaluation_suites_have_expected_coverage():
    order_cases = cases()
    heldout = [case for case in order_cases if case.case_id.startswith("heldout_")]
    assert len(order_cases) == 14
    assert len(heldout) == 9
    assert len(unseen_cases()) == 8
    assert len(heldout_unseen_cases()) == 6
    assert len(independent_order_cases()) == 6
    assert len(independent_unseen_cases()) == 6
    assert len(expanded_independent_order_cases()) == 18
    assert len(expanded_independent_unseen_cases()) == 18
    assert len(challenge_order_cases()) == 12
    assert len(challenge_unseen_cases()) == 12
    assert not ({x.case_id for x in challenge_order_cases()} & {x.case_id for x in cases()})


def test_balanced_scale_manifests_are_reproducible_and_stratified():
    root = Path(__file__).parents[1] / "data/splits/scales_balanced"
    expected = {
        7: {"confirmation": 3, "multi_step": 1, "single_tool": 3},
        14: {"confirmation": 6, "multi_step": 2, "single_tool": 6},
        28: {"confirmation": 12, "multi_step": 4, "single_tool": 12},
    }
    import json
    for size, counts in expected.items():
        manifest = json.loads((root / f"train_{size}.manifest.json").read_text(encoding="utf-8"))
        assert manifest["seed"] == 42
        assert manifest["requested_size"] == size
        assert manifest["counts_by_task_type"] == counts


def test_hermes_tool_call_parser_handles_nested_json_and_non_tool_responses():
    text = '<tool_call>{"name":"outer","arguments":{"filter":{"tags":["a}","b"]}}}</tool_call>'
    assert parse_calls(text) == [{"name": "outer", "arguments": {"filter": {"tags": ["a}", "b"]}}}]
    import pytest
    with pytest.raises(ValueError, match="no parseable tool calls"):
        parse_calls("Please provide a location first.")


def test_evaluation_separates_exact_and_casefolded_argument_matches():
    from tooltune.evaluation import evaluate_text, heldout_unseen_cases

    case = next(item for item in heldout_unseen_cases() if item.case_id == "unseen_search_webcam")
    result = evaluate_text('<tool_call>{"name":"search_product","arguments":{"query":"USB webcam"}}</tool_call>', case)
    assert result["arguments_correct"] is False
    assert result["normalized_arguments_correct"] is True
    assert result["failure_reason"] == "exact_argument_mismatch"


def test_recovery_suite_covers_stateful_and_multistep_paths():
    import subprocess, json
    root = Path(__file__).parents[1]
    output = root / "outputs/metrics/test_recovery_suite.json"
    subprocess.run([sys.executable, "scripts/evaluate_recovery.py", "--output", str(output)], cwd=root, check=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["num_cases"] == 7
    assert report["passed"] == 7
