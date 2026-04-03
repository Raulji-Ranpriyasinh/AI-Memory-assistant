# -- usage example
# With usage like this this,  memory system can run in FastAPI or any async AI agent without blocking.
import asyncio
from app.memory.backends import AsyncRedisSTMBackend, AsyncQdrantLTMBackend
from app.memory.schema import ShortTermMemoryEntry


async def main():
    stm_backend = AsyncRedisSTMBackend()
    await stm_backend.init()
    await stm_backend.set(
        "session1",
        "topic",
        ShortTermMemoryEntry(session_id="session1", key="topic", value="AI memory"),
    )
    entry = await stm_backend.get("session1", "topic")
    print(entry)

    ltm_backend = AsyncQdrantLTMBackend(
        host="localhost",
        port=6333,
        collection_name="memory_test",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_size=384,
    )
    await ltm_backend.add_entry("user1", "Async LTM test", {"tag": "test"})
    results = await ltm_backend.search("Async test")
    print(results)


asyncio.run(main())
