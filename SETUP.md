# SETUP.md – AI Memory Building Blocks

This guide explains how to:
1. Set up and run this memory system standalone
2. Integrate it into any other AI project (chatbots, agents, task systems, etc.)

## PART 1: Setting Up This Project (Standalone)

### 1. Prerequisites
* Python 3.10+
* Docker installed and running (for Qdrant vector DB)

### 2. Clone the Repository

```bash
git clone <your-repo-url> ai-memory-building-blocks
cd ai-memory-building-blocks
```

### 3. Set Up Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate # On Windows
# OR
source venv/bin/activate # On macOS/Linux
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Qdrant Locally
Make sure Docker is running, then start Qdrant:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Verify it's running by opening: http://localhost:6333

You should see:

```json
{
  "title": "qdrant - vector search engine",
  "version": "...",
  ...
}
```

### 6. Run Example Code

Save and Recall Memory:

```bash
python examples/memory_demo.py
```

Visualize Top Memories (Optional):

```bash
python examples/visualize_top_memories.py
```

Run Tests:

```bash
pytest tests/
```

## PART 2: Integrating Into Another AI Project

To reuse this memory system in **any AI/LLM/agent project**, follow these steps:

### Step 1: Copy Required Directories
Copy the following folders into your project:

```
your-ai-project/
├── app/
│   ├── config/
│   │   └── settings.py <-- Required
│   └── memory/
│       ├── schema.py
│       ├── scoring.py
│       ├── short_term.py
│       ├── long_term.py
│       └── memory_manager.py <-- Main controller
```

If your project already uses another config system, merge `settings.py` manually.

### Step 2: Install Required Python Libraries
In your project's `requirements.txt` or `pyproject.toml`, include:

```text
qdrant-client
sentence-transformers
matplotlib
```

Install via:

```bash
pip install qdrant-client sentence-transformers matplotlib
```

### Step 3: Start Qdrant (Locally or Remote)

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Or connect to remote Qdrant (set `LTM_QDRANT_HOST` in `settings.py`).

### Step 4: Use in Code
Here's how you integrate memory in your AI app:

```python
from app.memory.memory_manager import MemoryManager

memory = MemoryManager()

# Save STM
memory.set_short_term("session_123", "task", "Write email to client")

# Promote STM to LTM
memory.promote_stm_to_ltm(session_id="session_123", user_id="agent_1")

# Save LTM directly
memory.add_long_term("agent_1", "User wants to reschedule meeting", metadata={"source": "chatbot"})

# Recall memory
results = memory.recall("agent_1", query="meeting")
for item in results:
    print(f"[{item.source}] {item.text} (importance: {item.importance})")
```

### Notes for Production Use
* You can use environment variables or `.env` files for settings (instead of hardcoded config).
* You can wrap this into a microservice using **FastAPI**, **gRPC**, or a custom API.
* Supports multiple users and sessions (great for multi-agent systems).
* Qdrant can be deployed to the cloud for scale.

## You're Ready to Build Smarter AI Systems

This memory module is designed to be:
* **Plug & Play** — no external dependencies on other project logic
* **Modular** — STM and LTM logic separated cleanly
* **AI-Ready** — Scoring and embedding for relevance & filtering
* **Recyclable** — Reuse across agents, planners, chatbots, etc.