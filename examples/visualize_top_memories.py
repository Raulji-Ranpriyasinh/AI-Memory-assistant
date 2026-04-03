# examples/visualize_top_memories.py

from app.memory.memory_manager import MemoryManager
import matplotlib.pyplot as plt


def visualize_top_memories(user_id: str, top_k: int = 10):
    memory = MemoryManager()
    results = memory.search_long_term(query="", user_id=user_id, top_k=top_k)

    if not results:
        print("No memories found.")
        return

    sorted_results = sorted(results, key=lambda x: x.importance, reverse=True)

    labels = [
        entry.text[:30] + "..." if len(entry.text) > 30 else entry.text
        for entry in sorted_results
    ]
    scores = [entry.importance for entry in sorted_results]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, scores)
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_k} Important Memories for {user_id}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_top_memories(user_id="test_user", top_k=10)
