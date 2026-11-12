from tooltune.evaluation import fresh_stress_rollout_specs, fresh_stress_single_cases
from demo_server import metric_payload


def test_fresh_stress_suites_are_split_and_have_valid_utf8_prompts():
    single = fresh_stress_single_cases()
    rollouts = fresh_stress_rollout_specs()
    assert len(single) == 14
    assert len(rollouts) == 4
    assert len({case.case_id for case in single}) == len(single)
    assert len({case_id for case_id, *_ in rollouts}) == len(rollouts)
    assert all(case.user_prompt.encode("utf-8").decode("utf-8") == case.user_prompt for case in single)
    assert all(prompt.encode("utf-8").decode("utf-8") == prompt for _, prompt, *_ in rollouts)
    assert all([tool for tool, _ in steps] == ["search_product", "check_inventory"] for _, _, steps, _ in rollouts)


def test_fresh_stress_contains_intent_and_alias_coverage():
    cases = fresh_stress_single_cases()
    by_id = {case.case_id: case for case in cases}
    assert by_id["stress_search_keyboard_alias"].expected_name == "search_product"
    assert by_id["stress_stock_keyboard_alias"].expected_name == "check_inventory"
    assert by_id["stress_stock_missing"].expected_arguments["product_id"] == "ghost-item-77"


def test_demo_metrics_reflect_recorded_fresh_stress_results():
    payload = metric_payload()
    assert [(row["name"], row["single"], row["multi"]) for row in payload["stress"]] == [
        ("Base", "4/14", "0/4"),
        ("Decision LoRA", "8/14", "2/4"),
        ("500 repair LoRA", "9/14", "2/4"),
    ]
    assert payload["examples"]
