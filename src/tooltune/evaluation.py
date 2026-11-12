"""Deterministic baseline evaluation against the executable order environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .actions import parse_action
from .environment import OrderToolEnvironment, ProductToolEnvironment


@dataclass
class EvaluationCase:
    case_id: str
    user_prompt: str
    expected_name: str
    expected_arguments: dict[str, Any]
    tools: list[dict[str, Any]]
    validate_result: Callable[[OrderToolEnvironment], bool]
    environment_factory: Callable[[], Any] = OrderToolEnvironment


def cases() -> list[EvaluationCase]:
    template = OrderToolEnvironment().tool_definitions()
    return [
        EvaluationCase("status_existing", "查询订单 A1001 的状态。", "get_order", {"order_id": "A1001"}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("ok"))),
        EvaluationCase("status_missing", "查询订单 A9999 的状态。", "get_order", {"order_id": "A9999"}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("error") == "order_not_found")),
        EvaluationCase("cancel_requires_confirmation", "我想取消订单 A1001。", "cancel_order", {"order_id": "A1001", "confirmed": False}, template, lambda env: env.pending_cancellation == "A1001"),
        EvaluationCase("cancel_shipped", "取消已发货订单 A1002。", "cancel_order", {"order_id": "A1002", "confirmed": False}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("error") == "already_shipped")),
        EvaluationCase("status_second_order", "请查一下订单 A1002。", "get_order", {"order_id": "A1002"}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("ok"))),
        EvaluationCase("heldout_processing_mouse", "麻烦确认一下 A2101 订单进度。", "get_order", {"order_id": "A2101"}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("order", {}).get("status") == "processing")),
        EvaluationCase("heldout_shipped_webcam", "Is order A2102 on its way?", "get_order", {"order_id": "A2102"}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("order", {}).get("status") == "shipped")),
        EvaluationCase("heldout_processing_keyboard", "查一下订单号 A2103 的物流/处理状态。", "get_order", {"order_id": "A2103"}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("order", {}).get("item") == "mechanical keyboard")),
        EvaluationCase("heldout_delivered_lamp", "Please look up order A2104.", "get_order", {"order_id": "A2104"}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("order", {}).get("status") == "delivered")),
        EvaluationCase("heldout_processing_stand", "订单 A2105 的状态是什么？", "get_order", {"order_id": "A2105"}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("order", {}).get("status") == "processing")),
        EvaluationCase("heldout_unknown_8888", "能帮我查订单 A8888 吗？", "get_order", {"order_id": "A8888"}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("error") == "order_not_found")),
        EvaluationCase("heldout_cancel_processing", "申请取消 A2101 订单。", "cancel_order", {"order_id": "A2101", "confirmed": False}, template, lambda env: env.pending_cancellation == "A2101"),
        EvaluationCase("heldout_cancel_delivered", "Cancel delivered order A2104.", "cancel_order", {"order_id": "A2104", "confirmed": False}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("error") == "already_shipped")),
        EvaluationCase("heldout_cancel_missing", "请取消订单 A8888。", "cancel_order", {"order_id": "A8888", "confirmed": False}, template, lambda env: bool(env.trace and env.trace[-1]["result"].get("error") == "order_not_found")),
    ]


def unseen_cases() -> list[EvaluationCase]:
    env = ProductToolEnvironment()
    tools = env.tool_definitions()
    return [
        EvaluationCase("unseen_search_keyboard", "Search for the wireless keyboard product.", "search_product", {"query": "wireless keyboard"}, tools, lambda e: bool(e.trace and e.trace[-1]["result"]["results"]), ProductToolEnvironment),
        EvaluationCase("unseen_inventory_keyboard", "查询 keyboard 的库存。", "check_inventory", {"product_id": "keyboard"}, tools, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 12), ProductToolEnvironment),
        EvaluationCase("unseen_inventory_missing", "查询 product-x 的库存。", "check_inventory", {"product_id": "product-x"}, tools, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "product_not_found"), ProductToolEnvironment),
        EvaluationCase("unseen_search_mouse", "Search for the wireless mouse.", "search_product", {"query": "wireless mouse"}, tools, lambda e: bool(e.trace and any(x["product_id"] == "mouse" for x in e.trace[-1]["result"]["results"])), ProductToolEnvironment),
        EvaluationCase("unseen_stock_mouse", "How many wireless mice are in stock? Use product id mouse.", "check_inventory", {"product_id": "mouse"}, tools, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 7), ProductToolEnvironment),
        EvaluationCase("unseen_search_webcam", "Find the USB webcam product.", "search_product", {"query": "usb webcam"}, tools, lambda e: bool(e.trace and any(x["product_id"] == "webcam" for x in e.trace[-1]["result"]["results"])), ProductToolEnvironment),
        EvaluationCase("unseen_stock_zero", "Check inventory for microphone.", "check_inventory", {"product_id": "microphone"}, tools, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0), ProductToolEnvironment),
        EvaluationCase("unseen_search_stand", "Find a laptop stand.", "search_product", {"query": "laptop stand"}, tools, lambda e: bool(e.trace and any(x["product_id"] == "stand" for x in e.trace[-1]["result"]["results"])), ProductToolEnvironment),
    ]


def heldout_unseen_cases() -> list[EvaluationCase]:
    """Return product cases without overlap with the keyboard SFT examples."""
    training_overlap = {"unseen_search_keyboard", "unseen_inventory_keyboard"}
    return [case for case in unseen_cases() if case.case_id not in training_overlap]


def independent_order_cases() -> list[EvaluationCase]:
    template = OrderToolEnvironment().tool_definitions()
    specs = [
        ("independent_status_2101", "Could you tell me whether order A2101 is still being processed?", "get_order", {"order_id": "A2101"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("order", {}).get("status") == "processing")),
        ("independent_status_2102", "Please check the delivery progress for A2102.", "get_order", {"order_id": "A2102"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("order", {}).get("status") == "shipped")),
        ("independent_status_missing", "Look up order A7777 for me.", "get_order", {"order_id": "A7777"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
        ("independent_cancel_processing", "I want to cancel order A2105.", "cancel_order", {"order_id": "A2105", "confirmed": False}, lambda e: e.pending_cancellation == "A2105"),
        ("independent_cancel_delivered", "Can I cancel the delivered order A2104?", "cancel_order", {"order_id": "A2104", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "already_shipped")),
        ("independent_cancel_missing", "Cancel order A7777.", "cancel_order", {"order_id": "A7777", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
    ]
    return [EvaluationCase(case_id, prompt, name, arguments, template, validate) for case_id, prompt, name, arguments, validate in specs]


def independent_unseen_cases() -> list[EvaluationCase]:
    env = ProductToolEnvironment()
    tools = env.tool_definitions()
    specs = [
        ("independent_search_keyboard", "Find the wireless keyboard in the catalog.", "search_product", {"query": "wireless keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("independent_inventory_mouse", "How much stock remains for product mouse?", "check_inventory", {"product_id": "mouse"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 7)),
        ("independent_inventory_unknown", "Check the inventory of product unknown-item.", "check_inventory", {"product_id": "unknown-item"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "product_not_found")),
        ("independent_search_webcam", "Search the catalog for a USB webcam.", "search_product", {"query": "usb webcam"}, lambda e: bool(e.trace and any(x["product_id"] == "webcam" for x in e.trace[-1]["result"]["results"]))),
        ("independent_inventory_stand", "What is the stock count for product stand?", "check_inventory", {"product_id": "stand"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0)),
        ("independent_search_mouse", "Find a wireless mouse product.", "search_product", {"query": "wireless mouse"}, lambda e: bool(e.trace and any(x["product_id"] == "mouse" for x in e.trace[-1]["result"]["results"]))),
    ]
    return [EvaluationCase(case_id, prompt, name, arguments, tools, validate, ProductToolEnvironment) for case_id, prompt, name, arguments, validate in specs]


def expanded_independent_order_cases() -> list[EvaluationCase]:
    cases_out = independent_order_cases()
    template = OrderToolEnvironment().tool_definitions()
    specs = [
        ("expanded_status_1001", "What is happening with order A1001 right now?", "get_order", {"order_id": "A1001"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("ok"))),
        ("expanded_status_1002", "Has order A1002 shipped yet?", "get_order", {"order_id": "A1002"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("ok"))),
        ("expanded_status_2103", "Give me the status of order A2103.", "get_order", {"order_id": "A2103"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("ok"))),
        ("expanded_status_unknown", "Is there any information for order A6666?", "get_order", {"order_id": "A6666"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
        ("expanded_cancel_1001", "Please start a cancellation for A1001.", "cancel_order", {"order_id": "A1001", "confirmed": False}, lambda e: e.pending_cancellation == "A1001"),
        ("expanded_cancel_2101", "I no longer need order A2101; cancel it.", "cancel_order", {"order_id": "A2101", "confirmed": False}, lambda e: e.pending_cancellation == "A2101"),
        ("expanded_cancel_2102", "Try to cancel the shipped A2102 order.", "cancel_order", {"order_id": "A2102", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "already_shipped")),
        ("expanded_cancel_2103", "Can order A2103 be cancelled?", "cancel_order", {"order_id": "A2103", "confirmed": False}, lambda e: e.pending_cancellation == "A2103"),
        ("expanded_cancel_unknown", "Remove order A6666, please.", "cancel_order", {"order_id": "A6666", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
        ("expanded_status_lamp", "Look up the desk lamp order A2104.", "get_order", {"order_id": "A2104"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("ok"))),
        ("expanded_status_stand", "Is A2105 still processing?", "get_order", {"order_id": "A2105"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("ok"))),
        ("expanded_cancel_lamp", "I want to cancel delivered order A2104.", "cancel_order", {"order_id": "A2104", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "already_shipped")),
    ]
    cases_out.extend(EvaluationCase(spec[0], spec[1], spec[2], spec[3], template, spec[4]) for spec in specs)
    return cases_out


def expanded_independent_unseen_cases() -> list[EvaluationCase]:
    cases_out = independent_unseen_cases()
    env = ProductToolEnvironment(); tools = env.tool_definitions()
    specs = [
        ("expanded_search_keyboard", "Search products for a wireless keyboard.", "search_product", {"query": "wireless keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("expanded_search_mouse", "Please locate the wireless mouse.", "search_product", {"query": "wireless mouse"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("expanded_search_webcam", "Find the USB webcam listing.", "search_product", {"query": "usb webcam"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("expanded_search_stand", "Search for laptop stand.", "search_product", {"query": "laptop stand"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("expanded_inventory_keyboard", "How many keyboards are available? Product id keyboard.", "check_inventory", {"product_id": "keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 12)),
        ("expanded_inventory_microphone", "Check microphone inventory.", "check_inventory", {"product_id": "microphone"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0)),
        ("expanded_inventory_webcam", "Check stock for product webcam.", "check_inventory", {"product_id": "webcam"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 3)),
        ("expanded_inventory_unknown", "Check product no-such-product.", "check_inventory", {"product_id": "no-such-product"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "product_not_found")),
        ("expanded_search_keyboard_case", "Find a WIRELESS KEYBOARD.", "search_product", {"query": "wireless keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("expanded_search_mouse_case", "Find a WIRELESS MOUSE.", "search_product", {"query": "wireless mouse"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("expanded_inventory_stand_again", "Tell me the stock of stand.", "check_inventory", {"product_id": "stand"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0)),
        ("expanded_inventory_product_x", "Is product-x in stock?", "check_inventory", {"product_id": "product-x"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "product_not_found")),
    ]
    cases_out.extend(EvaluationCase(spec[0], spec[1], spec[2], spec[3], tools, spec[4], ProductToolEnvironment) for spec in specs)
    return cases_out


def challenge_order_cases() -> list[EvaluationCase]:
    """A fixed stress set with paraphrases and boundary order states.

    These prompts are deliberately disjoint from the training wording and are
    kept separate from the headline held-out results.
    """
    template = OrderToolEnvironment().tool_definitions()
    specs = [
        ("challenge_status_1001_cn", "帮我确认一下订单 A1001 目前到哪一步了？", "get_order", {"order_id": "A1001"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("ok"))),
        ("challenge_status_2103_en", "Could you check whether A2103 is still processing?", "get_order", {"order_id": "A2103"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("order", {}).get("status") == "processing")),
        ("challenge_status_2104_mix", "请查一下 delivered order A2104 的状态", "get_order", {"order_id": "A2104"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("order", {}).get("status") == "delivered")),
        ("challenge_status_unknown", "Do we have any record for order A9090?", "get_order", {"order_id": "A9090"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
        ("challenge_cancel_2101_cn", "我不需要 A2101 了，发起取消。", "cancel_order", {"order_id": "A2101", "confirmed": False}, lambda e: e.pending_cancellation == "A2101"),
        ("challenge_cancel_2103_en", "Please cancel A2103 if it has not shipped.", "cancel_order", {"order_id": "A2103", "confirmed": False}, lambda e: e.pending_cancellation == "A2103"),
        ("challenge_cancel_shipped", "Can you cancel the already shipped A2102?", "cancel_order", {"order_id": "A2102", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "already_shipped")),
        ("challenge_cancel_delivered", "Remove delivered order A2104 from my orders.", "cancel_order", {"order_id": "A2104", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "already_shipped")),
        ("challenge_cancel_unknown", "取消不存在的订单 A9090。", "cancel_order", {"order_id": "A9090", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
        ("challenge_status_1002", "Has A1002 left the warehouse yet?", "get_order", {"order_id": "A1002"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("ok"))),
        ("challenge_status_2105", "告诉我 A2105 的订单进度。", "get_order", {"order_id": "A2105"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("ok"))),
        ("challenge_cancel_1001", "Start cancellation for order A1001.", "cancel_order", {"order_id": "A1001", "confirmed": False}, lambda e: e.pending_cancellation == "A1001"),
    ]
    return [EvaluationCase(case_id, prompt, name, arguments, template, validate) for case_id, prompt, name, arguments, validate in specs]


def challenge_unseen_cases() -> list[EvaluationCase]:
    """A fixed product stress set covering zero stock and query paraphrases."""
    tools = ProductToolEnvironment().tool_definitions()
    specs = [
        ("challenge_search_keyboard_cn", "在商品目录里找无线键盘。", "search_product", {"query": "wireless keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("challenge_search_mouse_en", "Locate the wireless mouse listing.", "search_product", {"query": "wireless mouse"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("challenge_search_webcam_mix", "找一下 USB webcam 商品。", "search_product", {"query": "usb webcam"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("challenge_search_stand", "Search the catalog for a laptop stand.", "search_product", {"query": "laptop stand"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("challenge_inventory_keyboard", "keyboard 还有多少库存？", "check_inventory", {"product_id": "keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 12)),
        ("challenge_inventory_mouse", "What is the stock count for mouse?", "check_inventory", {"product_id": "mouse"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 7)),
        ("challenge_inventory_webcam", "Check inventory for product webcam.", "check_inventory", {"product_id": "webcam"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 3)),
        ("challenge_inventory_zero_microphone", "麦克风现在有货吗？", "check_inventory", {"product_id": "microphone"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0)),
        ("challenge_inventory_zero_stand", "How many laptop stands are available?", "check_inventory", {"product_id": "stand"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0)),
        ("challenge_inventory_unknown", "Check stock for product ghost-item.", "check_inventory", {"product_id": "ghost-item"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "product_not_found")),
        ("challenge_search_keyboard_case", "Find a WIRELESS KEYBOARD product.", "search_product", {"query": "wireless keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("challenge_search_mouse_cn", "请搜索 wireless mouse。", "search_product", {"query": "wireless mouse"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
    ]
    return [EvaluationCase(case_id, prompt, name, arguments, tools, validate, ProductToolEnvironment) for case_id, prompt, name, arguments, validate in specs]


def expanded_external_cases() -> list[EvaluationCase]:
    """Independently authored regression cases added after the 500-record pilot."""
    order_tools = OrderToolEnvironment().tool_definitions()
    product_env = ProductToolEnvironment()
    product_tools = product_env.tool_definitions()
    order_specs = [
        ("external_status_1001", "Could you check the latest update on A1001?", "get_order", {"order_id": "A1001"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("ok"))),
        ("external_status_2104", "订单 A2104 已经送达了吗？", "get_order", {"order_id": "A2104"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("order", {}).get("status") == "delivered")),
        ("external_status_unknown", "Please look for order A5555.", "get_order", {"order_id": "A5555"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
        ("external_cancel_processing", "不要 A2103 了，请先申请取消。", "cancel_order", {"order_id": "A2103", "confirmed": False}, lambda e: e.pending_cancellation == "A2103"),
        ("external_cancel_shipped", "Start a cancellation request for shipped order A1002.", "cancel_order", {"order_id": "A1002", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "already_shipped")),
        ("external_cancel_unknown", "申请取消不存在的订单 A5555。", "cancel_order", {"order_id": "A5555", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
    ]
    product_specs = [
        ("external_search_keyboard_zh", "请在目录中搜索无线键盘。", "search_product", {"query": "wireless keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("external_search_mouse_zh", "查找无线鼠标这个商品。", "search_product", {"query": "wireless mouse"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("external_inventory_mic", "How many USB microphone units are available?", "check_inventory", {"product_id": "microphone"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0)),
        ("external_inventory_stand_zh", "笔记本支架还有库存吗？", "check_inventory", {"product_id": "stand"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0)),
        ("external_inventory_webcam", "Give me the current stock count for the webcam.", "check_inventory", {"product_id": "webcam"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 3)),
        ("external_inventory_missing", "查一下 ghost-item 的库存。", "check_inventory", {"product_id": "ghost-item"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "product_not_found")),
    ]
    return [EvaluationCase(case_id, prompt, name, args, order_tools, validate) for case_id, prompt, name, args, validate in order_specs] + [EvaluationCase(case_id, prompt, name, args, product_tools, validate, ProductToolEnvironment) for case_id, prompt, name, args, validate in product_specs]


def fresh_stress_cases() -> list[EvaluationCase]:
    """Freshly authored phrasing/alias stress cases, separate from repair prompts."""
    order_tools = OrderToolEnvironment().tool_definitions()
    product_tools = ProductToolEnvironment().tool_definitions()
    orders = [
        ("stress_order_status_2101", "\u9ebb\u70e6\u6838\u5b9e A2101 \u8fd9\u7b14\u5355\u5b50\u76ee\u524d\u5361\u5728\u54ea\u4e2a\u73af\u8282\u3002", "get_order", {"order_id": "A2101"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("order", {}).get("status") == "processing")),
        ("stress_order_status_1002", "Do we have a dispatch update for order A1002?", "get_order", {"order_id": "A1002"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("order", {}).get("status") == "shipped")),
        ("stress_order_missing", "\u8bf7\u67e5\u627e\u7f16\u53f7 A5555 \u7684\u8ba2\u5355\u8bb0\u5f55\u3002", "get_order", {"order_id": "A5555"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
        ("stress_cancel_wait", "A2105 \u8fd8\u6ca1\u53d1\u8d27\u7684\u8bdd\uff0c\u5e2e\u6211\u5148\u63d0\u4ea4\u53d6\u6d88\u7533\u8bf7\uff0c\u4e4b\u540e\u6211\u518d\u786e\u8ba4\u3002", "cancel_order", {"order_id": "A2105", "confirmed": False}, lambda e: e.pending_cancellation == "A2105"),
        ("stress_cancel_delivered", "This order was delivered already; try to cancel A2104.", "cancel_order", {"order_id": "A2104", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "already_shipped")),
        ("stress_cancel_unknown", "Please request cancellation of order A5555.", "cancel_order", {"order_id": "A5555", "confirmed": False}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "order_not_found")),
    ]
    products = [
        ("stress_stock_keyboard_alias", "\u9375\u76e4\u7684\u5269\u4f59\u6570\u91cf\u662f\u591a\u5c11\uff1f", "check_inventory", {"product_id": "keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 12)),
        ("stress_stock_mic_alias", "\u9019\u652f USB \u9ea5\u514b\u98a8\u9084\u6709\u6c92\u6709\u73fe\u8ca8\uff0c\u5e6b\u6211\u67e5\u4e0b\u3002", "check_inventory", {"product_id": "microphone"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0)),
        ("stress_stock_mouse_alias", "\u7121\u7dda\u6ed1\u9f20\u76ee\u524d\u53ef\u4f9b\u8cfc\u8cb7\u55ce\uff1f", "check_inventory", {"product_id": "mouse"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 7)),
        ("stress_stock_webcam_alias", "\u651d\u5f71\u982d\u5eab\u5b58\u67e5\u4e00\u4e0b\u3002", "check_inventory", {"product_id": "webcam"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 3)),
        ("stress_stock_stand_alias", "\u7b46\u96fb\u652f\u6490\u67b6\u73fe\u5728\u9084\u6709\u5e7e\u500b\uff1f", "check_inventory", {"product_id": "stand"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("stock") == 0)),
        ("stress_stock_missing", "\u67e5\u4e00\u4e0b\u578b\u865f ghost-item-77 \u662f\u5426\u9084\u6709\u5eab\u5b58\u3002", "check_inventory", {"product_id": "ghost-item-77"}, lambda e: bool(e.trace and e.trace[-1]["result"].get("error") == "product_not_found")),
        ("stress_search_keyboard_alias", "\u628a\u7121\u7dda\u9375\u76e4\u90a3\u6b3e\u5728\u76ee\u9304\u88e1\u6aa2\u7d22\u4e00\u4e0b\u3002", "search_product", {"query": "wireless keyboard"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
        ("stress_search_webcam_alias", "Find the camera for video calls in the product list.", "search_product", {"query": "usb webcam"}, lambda e: bool(e.trace and e.trace[-1]["result"]["results"])),
    ]
    multi = [
        ("stress_multi_keyboard", "\u8bf7\u5148\u641c\u4e00\u4e0b\u65e0\u7ebf\u952e\u76d8\u8fd9\u4e2a\u54c1\u9879\uff0c\u62ff\u5230\u5546\u54c1\u7f16\u53f7\u540e\u518d\u544a\u8bc9\u6211\u5e93\u5b58\u4f59\u91cf\u3002", {"query": "wireless keyboard"}, {"product_id": "keyboard"}, 12),
        ("stress_multi_mic", "\u5728\u6e05\u5355\u91cc\u5b9a\u4f4d USB \u9ea6\u514b\u98ce\uff0c\u7136\u540e\u6839\u636e\u67e5\u5230\u7684\u578b\u53f7\u786e\u8ba4\u5269\u4f59\u91cf\u3002", {"query": "usb microphone"}, {"product_id": "microphone"}, 0),
        ("stress_multi_stand", "Locate the notebook stand first, then use its catalog ID to check stock.", {"query": "laptop stand"}, {"product_id": "stand"}, 0),
        ("stress_multi_mouse", "\u5e6b\u6211\u5728\u5217\u8868\u4e2d\u6838\u5c0d\u7121\u7dda\u6ed1\u9f20\uff0c\u63a5\u8457\u67e5\u770b\u7d50\u679c\u4e2d\u5c0d\u61c9\u5546\u54c1\u9084\u5269\u5e7e\u4ef6\u3002", {"query": "wireless mouse"}, {"product_id": "mouse"}, 7),
    ]
    order_cases = [EvaluationCase(cid, prompt, name, args, order_tools, validate) for cid, prompt, name, args, validate in orders]
    product_cases = [EvaluationCase(cid, prompt, name, args, product_tools, validate, ProductToolEnvironment) for cid, prompt, name, args, validate in products]
    multi_cases = []
    for cid, prompt, search_args, stock_args, expected_stock in multi:
        def validate_sequence(env, sid=stock_args["product_id"], stock=expected_stock):
            return len(env.trace) == 2 and env.trace[0]["tool"] == "search_product" and any(x["product_id"] == sid for x in env.trace[0]["result"].get("results", [])) and env.trace[1]["tool"] == "check_inventory" and env.trace[1]["arguments"].get("product_id") == sid and env.trace[1]["result"].get("stock") == stock
        multi_cases.append(EvaluationCase(cid, prompt, "search_product", search_args, product_tools, validate_sequence, ProductToolEnvironment))
    return order_cases + product_cases + multi_cases


def fresh_stress_single_cases() -> list[EvaluationCase]:
    """Return only the independently authored single-call stress cases."""
    return fresh_stress_cases()[:14]


def fresh_stress_rollout_specs() -> list[tuple[str, str, list[tuple[str, dict[str, Any]]], int]]:
    """Return the four independently authored two-call stress tasks.

    The single-call evaluator cannot score these because success depends on the
    complete search-then-inventory sequence.  The rollout evaluator consumes
    this compact representation and executes each task in a fresh environment.
    """
    return [
        ("stress_multi_keyboard", "\u8bf7\u5148\u641c\u4e00\u4e0b\u65e0\u7ebf\u952e\u76d8\u8fd9\u4e2a\u54c1\u9879\uff0c\u62ff\u5230\u5546\u54c1\u7f16\u53f7\u540e\u518d\u544a\u8bc9\u6211\u5e93\u5b58\u4f59\u91cf\u3002", [("search_product", {"query": "wireless keyboard"}), ("check_inventory", {"product_id": "keyboard"})], 12),
        ("stress_multi_mic", "\u5728\u6e05\u5355\u91cc\u5b9a\u4f4d USB \u9ea6\u514b\u98ce\uff0c\u7136\u540e\u6839\u636e\u67e5\u5230\u7684\u578b\u53f7\u786e\u8ba4\u5269\u4f59\u91cf\u3002", [("search_product", {"query": "usb microphone"}), ("check_inventory", {"product_id": "microphone"})], 0),
        ("stress_multi_stand", "Locate the notebook stand first, then use its catalog ID to check stock.", [("search_product", {"query": "laptop stand"}), ("check_inventory", {"product_id": "stand"})], 0),
        ("stress_multi_mouse", "\u5e6b\u6211\u5728\u5217\u8868\u4e2d\u6838\u5c0d\u7121\u7dda\u6ed1\u9f20\uff0c\u63a5\u8457\u67e5\u770b\u7d50\u679c\u4e2d\u5c0d\u61c9\u5546\u54c1\u9084\u5269\u5e7e\u4ef6\u3002", [("search_product", {"query": "wireless mouse"}), ("check_inventory", {"product_id": "mouse"})], 7),
    ]


def score_action(action: dict[str, Any] | None, case: EvaluationCase) -> dict[str, Any]:
    if not action:
        return {"parsed": False, "tool_correct": False, "arguments_correct": False, "normalized_arguments_correct": False, "failure_reason": "parse_failure"}
    tool_correct = action.get("name") == case.expected_name
    arguments_correct = tool_correct and action.get("arguments") == case.expected_arguments
    normalized_arguments_correct = False
    if tool_correct and isinstance(action.get("arguments"), dict):
        actual = action["arguments"]
        expected = case.expected_arguments
        normalized_arguments_correct = actual.keys() == expected.keys() and all(
            (str(actual[key]).strip().casefold() == str(expected[key]).strip().casefold())
            if key in {"query", "product_id"} and isinstance(actual[key], str) and isinstance(expected[key], str)
            else actual[key] == expected[key]
            for key in expected
        )
    reason = "success" if arguments_correct else ("wrong_tool" if not tool_correct else "exact_argument_mismatch")
    return {"parsed": True, "tool_correct": tool_correct, "arguments_correct": arguments_correct, "normalized_arguments_correct": normalized_arguments_correct, "failure_reason": reason}


def evaluate_text(text: str, case: EvaluationCase) -> dict[str, Any]:
    action = parse_action(text)
    score = score_action(action, case)
    env = case.environment_factory()
    if action:
        execution = env.execute(action["name"], action["arguments"])
        score["task_success"] = bool(score["arguments_correct"] and case.validate_result(env))
        if score["arguments_correct"] and not case.validate_result(env):
            score["failure_reason"] = "execution_result_mismatch"
    else:
        execution = None
        score["task_success"] = False
    score["execution_success"] = bool(action and score['tool_correct'] and case.validate_result(env))
    score["normalized_task_success"] = bool(score["normalized_arguments_correct"] and score['execution_success'])
    score.update({"action": action, "execution": execution})
    return score
