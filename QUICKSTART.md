# 🚀 Quick Start - Docker

## One Command to Run Everything

```bash
# Start all services (Frontend + Backend + AI Service + Databases)
docker-compose up -d
```

That's it! This will start:
- ✅ **Frontend** (React) → `http://localhost:3000`
- ✅ **Backend** (NestJS API) → `http://localhost:4000`
- ✅ **AI Service** (FastAPI) → `http://localhost:8000`
- ✅ **MongoDB** → Port 27017
- ✅ **PostgreSQL** → Port 5442
- ✅ **Redis** → Port 6379

## Test the Flow

### 1. Open Frontend
```
http://localhost:3000
```

### 2. Register
- Click "Register"
- Fill in: Alice, alice@test.com, password
- Click "Sign Up"

### 3. Chat with AI
- Go to "Assistant" tab
- Say: "My name is Alice"
- Then ask: "What is my name?"
- AI should remember! ✨

## View Logs

```bash
# All services
docker-compose logs -f

# Just backend
docker-compose logs -f backend

# Just AI service
docker-compose logs -f ai-service

# Just frontend
docker-compose logs -f frontend
```

## Stop Everything

```bash
docker-compose down
```

## Troubleshooting

### Port already in use?
Stop any services using ports 3000, 4000, or 8000, or edit `docker-compose.yml` to use different ports.

### AI not remembering?
Check if Pinecone API key is configured in `.env` file.

### Can't register?
Check logs: `docker-compose logs -f backend`
