"""Command line smoke checks for the local tool environment."""

from .environment import OrderToolEnvironment


def main() -> None:
    env = OrderToolEnvironment()
    print(env.execute("get_order", {"order_id": "A1001"}))
    print(env.execute("cancel_order", {"order_id": "A1001", "confirmed": False}))
    print({"scripted_confirmation_event": env.confirm_cancellation("A1001", env.session_id, True)})
    print(env.execute("cancel_order", {"order_id": "A1001", "confirmed": True}))


if __name__ == "__main__":
    main()
