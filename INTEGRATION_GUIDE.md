# Complete Integration Guide

## Architecture Overview

```
┌─────────────┐         ┌──────────────┐         ┌──────────────┐
│  Frontend   │  ────►  │   Backend    │  ────►  │  AI Service  │
│  (React:3000)│         │ (NestJS:4000) │         │ (FastAPI:8000)│
└─────────────┘         └──────────────┘         └──────────────┘
      │                         │                        │
      │                         │                        │
      └─────────────────────────┼────────────────────────┘
                                │
                         ┌──────────────┐
                         │   MongoDB    │
                         │  (Port 27017)│
                         └──────────────┘
```

## What's Been Synchronized

✅ **JWT Authentication**: Both Backend and AI Service use the same JWT_SECRET
✅ **User Registration**: When you register in Backend, AI Service gets notified
✅ **Chat Flow**: Frontend → Backend → AI Service → Memory (Pinecone)
✅ **Response Format**: All services properly handle `{response, user_id}` format

## Quick Start

### 1. Setup Environment

```bash
# Copy and configure environment
cp .env.example .env

# Update these critical values:
# JWT_SECRET=your-super-secret-key (MUST be same for all services)
# MONGODB_URI=mongodb://delight:delight123@localhost:27017/delight
# AI_SERVICE_URL=http://localhost:8000
```

### 2. Start All Services

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or start services manually:
# Terminal 1 - AI Service
cd ai-service
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 - Backend
cd backend
npm run start:dev

# Terminal 3 - Frontend
cd frontend
npm run dev
```

### 3. Test Complete Flow

#### Step 1: Register a New User

```bash
curl -X POST http://localhost:4000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "password": "Alice123!",
    "firstName": "Alice",
    "lastName": "Smith"
  }'
```

**Expected Response:**
```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "507f1f77bcf86cd799439011",
    "email": "alice@example.com",
    "role": "patient",
    "status": "active",
    "profile": {
      "firstName": "Alice",
      "lastName": "Smith",
      "language": "en"
    }
  }
}
```

#### Step 2: Tell AI Your Name

```bash
curl -X POST http://localhost:4000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_FROM_STEP_1" \
  -d '{
    "message": "My name is Alice"
  }'
```

**Expected Flow:**
```
Frontend (React) 
  → Backend (NestJS: saves to MongoDB if needed)
    → AI Service (FastAPI: processes with memory)
      → Pinecone (stores memory: "User's name is Alice")
        → AI generates response
      ← Returns: {"response": "Nice to meet you, Alice!"}
    ← Backend forwards response
  ← Frontend displays response
```

#### Step 3: Ask What Your Name Is

```bash
curl -X POST http://localhost:4000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_FROM_STEP_1" \
  -d '{
    "message": "What is my name?"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "data": {
    "response": "Your name is Alice! You told me earlier."
  }
}
```

## HTTP File (for REST Client/VS Code)

Copy this into a `.http` file and test directly:

```http
### 1. Register New User
POST http://localhost:4000/api/v1/auth/register
Content-Type: application/json

{
  "email": "alice@example.com",
  "password": "Alice123!",
  "firstName": "Alice",
  "lastName": "Smith"
}

### 2. Login (if already registered)
POST http://localhost:4000/api/v1/auth/login
Content-Type: application/json

{
  "email": "alice@example.com",
  "password": "Alice123!"
}

### 3. Chat - Tell AI your name
# Replace TOKEN with the accessToken from login/register response
POST http://localhost:4000/api/v1/chat
Content-Type: application/json
Authorization: Bearer TOKEN

{
  "message": "My name is Alice"
}

### 4. Chat - Ask what your name is
POST http://localhost:4000/api/v1/chat
Content-Type: application/json
Authorization: Bearer TOKEN

{
  "message": "What is my name?"
}

### 5. Get Chat History
GET http://localhost:4000/api/v1/chat/history
Authorization: Bearer TOKEN

### 6. Get AI Memories (what AI knows about you)
GET http://localhost:4000/api/v1/ai/memories
Authorization: Bearer TOKEN
```

## Verification Checklist

- [ ] User registered successfully in Backend
- [ ] JWT token received in response
- [ ] User synced to AI Service (check AI service logs)
- [ ] Chat message sent and AI responded
- [ ] AI remembered your name in subsequent messages
- [ ] Chat history retrieved correctly
- [ ] All services running without errors

## Common Issues & Solutions

### Issue: "Invalid token" error
**Solution**: Ensure JWT_SECRET is identical in both Backend and AI Service `.env` files

### Issue: AI doesn't remember
**Solution**: Check Pinecone configuration and that PINECONE_API_KEY is set

### Issue: Backend can't reach AI Service
**Solution**: Verify AI_SERVICE_URL is correct and AI Service is running on port 8000

### Issue: CORS errors
**Solution**: Check CORS_ORIGINS in AI Service includes your Frontend URL

## Testing with Frontend UI

1. Open `http://localhost:3000` in your browser
2. Click "Register" and create an account
3. After registration, you'll be redirected to Dashboard
4. Click "Assistant" in the sidebar
5. Type: "My name is Alice"
6. Then ask: "What is my name?"
7. AI should remember your name!

## Logs to Watch

### Backend (NestJS)
```bash
# Watch for user registration and AI proxy calls
tail -f backend/logs/*.log
```

### AI Service (FastAPI)
```bash
# Watch for user sync and chat requests
tail -f ai-service/logs/*.log
```

### Docker
```bash
# View all service logs
docker-compose logs -f
```

## Next Steps

- [ ] Add personality assessment endpoint
- [ ] Configure Pinecone for production
- [ ] Set up Firebase for push notifications
- [ ] Enable CGM integration
- [ ] Add food recognition
