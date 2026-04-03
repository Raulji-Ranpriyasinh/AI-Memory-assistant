# tests/test_ltm.py
from app.memory.memory_factory import get_memory_manager


def test_ltm():
    manager = get_memory_manager()

    user_id = "user1"

    # 1. Add an LTM entry
    entry_id = manager.add_long_term(
        user_id=user_id,
        text="AI memory systems store short-term and long-term context.",
        metadata={"tag": "AI"},
    )
    print(f"Added LTM entry ID: {entry_id}")

    # 2. Search LTM
    results = manager.search_long_term(query="context", user_id=user_id, top_k=3)
    print("Search results:")
    for r in results:
        print(f" - {r.text} (score={r.importance})")


if __name__ == "__main__":
    test_ltm()
