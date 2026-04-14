# API Endpoints

All endpoints are defined in the modular route structure under `app/api/routes/`.

## Quick Start

```bash
# Start FastAPI server
uvicorn main:app --reload --port 8000
```

## Interactive API Docs

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Endpoints Reference

All endpoints use the `/api/v1/` prefix and require authentication unless otherwise noted.

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | `POST` | Send a text message and get an AI response |
| `/api/v1/chat/voice` | `POST` | Send a voice message (base64-encoded audio) and get an AI response |

### Memories

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memories` | `GET` | Get all memories for the authenticated user |
| `/api/v1/memories/search` | `GET` | Search memories by query |
| `/api/v1/memories/{memory_id}` | `DELETE` | Delete a specific memory |
| `/api/v1/memories` | `DELETE` | Delete ALL memories (⚠️ Irreversible) |

### History & Summaries

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/history` | `GET` | Get conversation history |
| `/api/v1/summaries` | `GET` | Get conversation summaries |

### Health Tracking

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/cgm/readings` | `POST` | Log CGM (Continuous Glucose Monitor) readings |
| `/api/v1/cgm/summary` | `GET` | Get CGM summary |
| `/api/v1/mood` | `POST` | Log mood entry |
| `/api/v1/food/log` | `POST` | Log food entry |
| `/api/v1/food/recognize` | `POST` | Recognize food from image |
| `/api/v1/activity` | `POST` | Log activity entry |

### Programs & Nudges

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/programs` | `GET` | Get available programs |
| `/api/v1/programs/{program_id}/progress` | `POST` | Log progress for a program |
| `/api/v1/nudges` | `GET` | Get nudges/reminders |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | `GET` | Health check |
| `/api/v1/prune` | `POST` | Prune stale memories |
| `/api/v1/metrics` | `GET` | Get usage metrics |

---

## Request/Response Examples

### 1. Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"message": "Hello!"}'
```

**Response:**
```json
{
  "response": "Hello! How can I help you today?",
  "user_id": "default_user"
}
```

---

### 2. Voice Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat/voice \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"audio_base64": "<base64-encoded-audio>", "format": "webm"}'
```

**Response:**
```json
{
  "response": "I understood you said: Hello!",
  "user_id": "default_user",
  "transcription": "Hello!"
}
```

---

### 3. Get All Memories

```bash
curl http://localhost:8000/api/v1/memories \
  -H "Authorization: Bearer <token>"
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

### 4. Get Conversation History

```bash
curl http://localhost:8000/api/v1/history \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
[
  {"role": "user", "content": "What's my name?"},
  {"role": "assistant", "content": "Your name is John."}
]
```

---

### 5. Get Summaries

```bash
curl http://localhost:8000/api/v1/summaries \
  -H "Authorization: Bearer <token>"
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

### 6. Search Memories

```bash
curl "http://localhost:8000/api/v1/memories/search?query=coffee&top_k=5" \
  -H "Authorization: Bearer <token>"
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

### 7. Log CGM Reading

```bash
curl -X POST http://localhost:8000/api/v1/cgm/readings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"glucose_value": 95, "timestamp": "2026-04-14T10:00:00Z"}'
```

---

### 8. Log Mood

```bash
curl -X POST http://localhost:8000/api/v1/mood \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"mood_score": 8, "note": "Feeling great today"}'
```

---

### 9. Log Food

```bash
curl -X POST http://localhost:8000/api/v1/food/log \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"meal_type": "lunch", "items": ["Chicken salad", "Rice"], "estimated_calories": 450}'
```

---

### 10. Recognize Food from Image

```bash
POST http://localhost:8000/api/v1/food/recognize
Authorization: Bearer <token>
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="image"; filename="food.jpg"
Content-Type: image/jpeg

< C:\path\to\your\food-image.jpg
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="image_path"

C:/Users/YourName/Pictures/food.jpg
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

**Or with file path only (simpler for REST Client):**

```bash
POST http://localhost:8000/api/v1/food/recognize
Authorization: Bearer <token>
Content-Type: application/x-www-form-urlencoded

image_path=C:/Users/YourName/Pictures/food.jpg
```

**Response:**
```json
{
  "items": ["Chicken breast", "Rice", "Broccoli"],
  "estimated_calories": 450,
  "glycemic_load": "medium",
  "portion_sizes": [{"item": "Chicken breast", "size": "150g"}],
  "confidence": 0.92
}
```

**Requirements:**
- `ENABLE_FOOD_RECOGNITION=true` in `.env`
- Either `NUTRITIONIX_API_KEY` or `GOOGLE_VISION_API_KEY` configured

---

### 11. Get Metrics

```bash
curl http://localhost:8000/api/v1/metrics \
  -H "Authorization: Bearer <token>"
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

### 12. Prune Memories

```bash
curl -X POST http://localhost:8000/api/v1/prune \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "pruned_count": 3,
  "message": "Successfully pruned 3 stale memories"
}
```

---

### 13. Delete Specific Memory

```bash
curl -X DELETE http://localhost:8000/api/v1/memories/{memory_id} \
  -H "Authorization: Bearer <token>"
```

---

### 14. Delete ALL Memories (⚠️ Irreversible)

```bash
curl -X DELETE http://localhost:8000/api/v1/memories \
  -H "Authorization: Bearer <token>"
```
