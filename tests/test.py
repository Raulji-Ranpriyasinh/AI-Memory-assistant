# tests/test_memory.py

from app.memory.memory_manager import MemoryManager
from unittest.mock import MagicMock
from app.memory.long_term import LongTermMemory


def test_memory_system():
    """Test STM + promotion without real Qdrant"""

    # 🔧 Patch LongTermMemory
    memory = MemoryManager(stm_backend="memory")
    memory.ltm = MagicMock(spec=LongTermMemory)
    memory.ltm.add_entry.return_value = "mock-ltm-id"
    memory.ltm.search.return_value = []

    # === STM ===
    session_id = "test-session"
    memory.set_short_term(session_id, "task", "Build memory system")
    assert memory.get_short_term(session_id, "task") == "Build memory system"

    # === Promote STM to LTM ===
    user_id = "user-001"
    ltm_id = memory.promote_stm_to_ltm(session_id, user_id)

    assert ltm_id == "mock-ltm-id"
    memory.ltm.add_entry.assert_called_once()
