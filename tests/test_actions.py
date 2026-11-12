from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tooltune.actions import parse_action
from tooltune.evaluation import cases, unseen_cases, evaluate_text
from tooltune.agent_loop import run_agent


def test_parse_json_and_fenced_actions():
    assert parse_action('{"name":"get_order","arguments":{"order_id":"A1001"}}')["name"] == "get_order"
    assert parse_action('```json\n{"name":"get_order","arguments":"{\\"order_id\\":\\"A1001\\"}"}\n```')["arguments"]["order_id"] == "A1001"


def test_evaluation_marks_valid_action_and_environment_state():
    result = evaluate_text('{"name":"cancel_order","arguments":{"order_id":"A1001","confirmed":false}}', next(case for case in cases() if case.case_id == "cancel_requires_confirmation"))
    assert result["arguments_correct"] is True
    assert result["task_success"] is True


def test_blocked_unsafe_action_is_not_scored_as_task_success():
    case = next(case for case in cases() if case.case_id == "cancel_requires_confirmation")
    result = evaluate_text('{"name":"cancel_order","arguments":{"order_id":"A1001","confirmed":true}}', case)
    assert result["execution"]["error"] == "confirmation_required"
    assert result["task_success"] is False


def test_agent_loop_can_observe_and_continue():
    outputs = iter([
        '{"name":"cancel_order","arguments":{"order_id":"A1001","confirmed":false}}',
        '{"name":"cancel_order","arguments":{"order_id":"A1001","confirmed":true}}',
    ])
    result = run_agent(lambda messages, tools: next(outputs), "取消订单 A1001")
    assert result["status"] == "awaiting_confirmation"
    assert result["steps"] == 1


def test_shipped_order_cannot_be_cancelled():
    from tooltune.environment import OrderToolEnvironment
    env = OrderToolEnvironment()
    assert env.execute("cancel_order", {"order_id": "A1002", "confirmed": False})["error"] == "already_shipped"


def test_unseen_product_environment_is_executable():
    result = evaluate_text('{"name":"check_inventory","arguments":{"product_id":"keyboard"}}', unseen_cases()[1])
    assert result["arguments_correct"] is True
    assert result["task_success"] is True


def test_unseen_search_case_uses_executable_query():
    result = evaluate_text('{"name":"search_product","arguments":{"query":"wireless keyboard"}}', unseen_cases()[0])
    assert result["arguments_correct"] is True
    assert result["task_success"] is True


def test_product_search_does_not_end_multistep_loop():
    from tooltune.agent_loop import run_agent
    from tooltune.environment import ProductToolEnvironment
    outputs = iter([
        '{"name":"search_product","arguments":{"query":"wireless keyboard"}}',
        '{"name":"check_inventory","arguments":{"product_id":"keyboard"}}',
    ])
    result = run_agent(lambda messages, tools: next(outputs), "Find keyboard and check stock", environment=ProductToolEnvironment())
    assert result["status"] == "completed"
    assert result["steps"] == 2


def test_seed_helper_is_callable():
    from tooltune.reproducibility import seed_everything
    seed_everything(42)
