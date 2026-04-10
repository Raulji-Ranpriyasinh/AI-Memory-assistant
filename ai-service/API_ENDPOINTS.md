# API Endpoints

All endpoints replace the CLI commands from `app/cli.py`.

## Quick Start

```bash
# Start FastAPI server
uvicorn main:app --reload --port 8000

# Or use CLI mode (original behavior)
python main.py --cli
```

## End Reference

| CLI Command | FastAPI Endpoint | Method |
|-------------|------------------|--------|
| Chat (interactive loop) | `/api/chat` | `POST` |
| `/memories` | `/api/memories` | `GET` |
| `/history` | `/api/history` | `GET` |
| `/summaries` | `/api/summaries` | `GET` |
| `/search <query>` | `/api/memories/search?query=<q>` | `GET` |
| `/metrics` | `/api/metrics` | `GET` |
| `/prune` | `/api/prune` | `POST` |

## Request/Response Examples

### 1. Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "user_id": "default_user"}'
```

**Response:**
```json
{
  "response": "Hello! How can I help you today?",
  "user_id": "default_user"
}
```

---

### 2. Get All Memories

```bash
curl http://localhost:8000/api/memories?user_id=default_user
```

**Response:**
```json
{
  "ltm": [
    {
      "text": "User likes coffee",
      "category": "preference",
      "salience_score": 0.85
    }
  ],
  "summaries": [
    {
      "turn_range": "1-10",
      "key_topics": ["coffee", "morning routine"]
    }
  ]
}
```

---

### 3. Get Conversation History

```bash
curl http://localhost:8000/api/history?user_id=default_user
```

**Response:**
```json
[
  {"role": "user", "content": "What's my name?"},
  {"role": "assistant", "content": "Your name is John."}
]
```

---

### 4. Get Summaries

```bash
curl http://localhost:8000/api/summaries?user_id=default_user
```

**Response:**
```json
[
  {
    "turn_range": "1-10",
    "key_topics": ["coffee", "preferences"],
    "decisions_made": [],
    "action_items": ["Track coffee habits"],
    "important_context": "User mentioned coffee preference multiple times"
  }
]
```

---

### 5. Search Memories

```bash
curl "http://localhost:8000/api/memories/search?query=coffee&top_k=5&user_id=default_user"
```

**Response:**
```json
{
  "query": "coffee",
  "results": [
    {
      "text": "User prefers dark roast coffee",
      "category": "preference",
      "final_score": 0.92,
      "salience_score": 0.85,
      "relevance_score": 0.89,
      "recency_score": 0.95
    }
  ],
  "count": 1
}
```

---

### 6. Get Metrics

```bash
curl http://localhost:8000/api/metrics?user_id=default_user
```

**Response:**
```json
{
  "counts": {
    "total_queries": 150,
    "memories_stored": 45
  },
  "sums": {
    "total_latency_ms": 12500.5
  }
}
```

---

### 7. Prune Memories

```bash
curl -X POST http://localhost:8000/api/prune?user_id=default_user
```

**Response:**
```json
{
  "pruned_count": 3,
  "message": "Successfully pruned 3 stale memories"
}
```

---

### 8. Delete Specific Memory

```bash
curl -X DELETE http://localhost:8000/api/memories/{memory_id}?user_id=default_user
```

---

### 9. Delete ALL Memories (⚠️ Irreversible)

```bash
curl -X DELETE http://localhost:8000/api/memories?user_id=default_user
```

---

## Interactive API Docs

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
