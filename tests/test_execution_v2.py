import json
import pytest
from tooltune.environment import OrderToolEnvironment, ProductToolEnvironment
from tooltune.agent_loop import run_agent
from tooltune.evaluation import unseen_cases, evaluate_text
from tooltune.actions import parse_action


def scripted(actions):
    sequence = iter(actions)
    return lambda messages, tools: json.dumps(next(sequence))


def action(name, **arguments):
    return {"name": name, "arguments": arguments}


def test_user_confirmation_is_bound_consumed_and_required():
    env = OrderToolEnvironment()
    generate = scripted([action("cancel_order", order_id="A1001", confirmed=False)])
    paused = run_agent(generate, "Cancel A1001", environment=env)
    assert paused["status"] == "awaiting_confirmation"
    assert env.execute("cancel_order", {"order_id":"A1001", "confirmed":True})["error"] == "confirmation_required"
    assert not env.confirm_cancellation("A1001", "wrong-session", True)
    assert not env.confirm_cancellation("A2101", env.session_id, True)
    assert env.confirm_cancellation("A1001", env.session_id, True)
    resumed = run_agent(scripted([action("cancel_order",order_id="A1001",confirmed=True)]), "", environment=env, messages=paused["messages"])
    assert resumed["status"] == "completed"
    assert env._authorized_order is None
    assert env.execute("cancel_order", {"order_id":"A1001", "confirmed":True})["error"] == "not_cancellable"


def test_rejection_does_not_cancel():
    env = OrderToolEnvironment()
    env.execute("cancel_order", {"order_id":"A1001", "confirmed":False})
    assert not env.confirm_cancellation("A1001",env.session_id,False)
    assert env.execute("cancel_order", {"order_id":"A1001", "confirmed":True})["error"] == "confirmation_required"
    assert env.orders["A1001"]["status"] == "processing"


@pytest.mark.parametrize("name,args",[("check_inventory",{"product_id":[]}), ("check_inventory",{"product_id":{}}), ("search_product",{"query":123}), ("search_product",{}), ("search_product",{"query":""}), ("search_product",{"query":"mouse","extra":1}), ("search_product",None)])
def test_invalid_arguments_are_structured(name,args):
    assert ProductToolEnvironment().execute(name,args)["error"] == "invalid_arguments"


def test_unknown_and_malformed_calls():
    assert ProductToolEnvironment().execute("no_tool",{})["error"] == "unknown_tool"
    assert parse_action("tool: get_order(order_id=A1001)") is None


def test_normalization_never_executes_gold_action():
    case = next(c for c in unseen_cases() if c.case_id == "unseen_inventory_keyboard")
    score = evaluate_text(json.dumps(action("check_inventory",product_id="KEYBOARD")),case)
    assert score["normalized_arguments_correct"]
    assert not score["normalized_task_success"]
    assert not score["execution_success"]


def test_recovery_after_actual_model_action_error():
    env = ProductToolEnvironment()
    result = run_agent(scripted([action("check_inventory",product_id="wrong"), action("search_product",query="usb webcam"), action("check_inventory",product_id="webcam")]), "Find webcam stock",environment=env)
    assert result["status"] == "completed"
    assert result["trace"][0]["result"]["error"] == "product_not_found"
    assert result["result"]["stock"] == 3


def test_empty_search_then_retry():
    result = run_agent(scripted([action("search_product",query="no-result"),action("search_product",query="wireless mouse"),action("check_inventory",product_id="mouse")]), "mouse",environment=ProductToolEnvironment())
    assert result["trace"][0]["result"]["results"] == []
    assert result["status"] == "completed"


def test_retry_budget_and_terminal_errors():
    result = run_agent(lambda messages,tools: json.dumps(action("check_inventory",product_id="missing")),"stock",environment=ProductToolEnvironment(),max_steps=3)
    assert result["status"] == "max_steps" and result["steps"] == 3
    result = run_agent(scripted([action("cancel_order",order_id="A1002",confirmed=False)]),"cancel",environment=OrderToolEnvironment())
    assert result["status"] == "terminal_error"
