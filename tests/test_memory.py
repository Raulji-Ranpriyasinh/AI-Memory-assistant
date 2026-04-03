# test_memory_system.py
#!/usr/bin/env python3
"""
Simple test script to demonstrate the memory system
"""

from app.memory.memory_manager import MemoryManager
import time


def test_memory_system():
    """Test both STM and LTM functionality"""
    print("=== Memory System Test ===\n")

    # Initialize memory manager
    memory = MemoryManager(stm_backend="memory")

    # Test STM
    print("1. Testing Short Term Memory...")
    session_id = "test_session_123"

    memory.set_short_term(session_id, "user_name", "John Doe")
    memory.set_short_term(session_id, "current_task", "Writing a report")
    memory.set_short_term(session_id, "mood", "focused and productive")

    print(f"Stored 3 items in STM for session: {session_id}")

    # Retrieve STM data
    all_stm = memory.get_all_short_term(session_id)
    print("STM Contents:")
    for key, value in all_stm.items():
        print(f"  {key}: {value}")

    # Test LTM
    print("\n2. Testing Long Term Memory...")
    user_id = "user_123"
    conversation_id = "conv_456"

    # Add some memories
    memories = [
        "User prefers morning meetings between 9-11 AM",
        "User is working on AI project with memory system",
        "User uses Python and prefers clean, documented code",
        "User asked to remember their coffee preference: black coffee, no sugar",
    ]

    for memory_text in memories:
        memory.add_long_term(
            user_id=user_id,
            text=memory_text,
            conversation_id=conversation_id,
            context={"user_explicitly_asked_to_remember": True},
        )

    print(f"Stored {len(memories)} memories in LTM for user: {user_id}")

    # Test recall
    print("\n3. Testing Memory Recall...")

    # Test different queries
    test_queries = [
        "meeting preferences",
        "coffee",
        "programming language",
        "project details",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = memory.recall(
            user_id=user_id,
            query=query,
            session_id=session_id,
            conversation_id=conversation_id,
        )

        for i, result in enumerate(results, 1):
            print(
                f"  {i}. [{result.source}] {result.text[:80]}{'...' if len(result.text) > 80 else ''}"
            )
            print(f"      Importance: {result.importance:.3f}")

    # Test promotion
    print("\n4. Testing STM to LTM Promotion...")
    promotion_result = memory.promote_stm_to_ltm(
        session_id=session_id, user_id=user_id, conversation_id=conversation_id
    )

    if promotion_result:
        print(f"Successfully promoted STM to LTM with ID: {promotion_result}")
    else:
        print("STM data was not promoted (importance score too low)")

    print("\n=== Test Complete ===")
    print("\nTo view the data, run:")
    print(
        f"  python scripts/memory_viewer.py --session-id {session_id} --user-id {user_id}"
    )
    print("\nTo see statistics, run:")
    print(f"  python scripts/memory_stats.py --session-id {session_id}")


if __name__ == "__main__":
    test_memory_system()
