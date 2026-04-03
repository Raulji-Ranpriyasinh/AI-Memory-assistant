# tests/test_stm.py
from app.memory.memory_factory import get_memory_manager


def test_stm():
    manager = get_memory_manager()

    session_id = "sess1"

    # 1. Set STM values
    manager.set_short_term(session_id, "topic", "We are discussing AI memory.")
    manager.set_short_term(session_id, "mood", "Excited")

    # 2. Retrieve single key
    topic = manager.get_short_term(session_id, "topic")
    print(f"Retrieved topic: {topic}")

    # 3. Retrieve all STM entries
    all_data = manager.get_all_short_term(session_id)
    print("All STM entries:", all_data)

    # 4. Clear STM
    manager.clear_short_term(session_id)
    print("After clearing:", manager.get_all_short_term(session_id))


if __name__ == "__main__":
    test_stm()
