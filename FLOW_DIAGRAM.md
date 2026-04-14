# Complete Request Flow: Registration to AI Chat

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  USER REGISTRATION & AI CHAT - COMPLETE FLOW                          │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────┐         ┌──────────┐         ┌──────────┐         ┌──────────┐
│Frontend  │         │ Backend  │         │   AI     │         │ Pinecone │
│ React:3k │         │ NestJS:4k│         │ FastAPI  │         │ Memory   │
│          │         │          │         │   :8000  │         │          │
└────┬─────┘         └────┬─────┘         └────┬─────┘         └────┬─────┘
     │                    │                    │                    │
     │  1. REGISTER       │                    │                    │
     │───────────────────►│                    │                    │
     │ POST /auth/register│                    │                    │
     │ {email, pwd, name} │                    │                    │
     │                    │                    │                    │
     │                    │ [Save to MongoDB]  │                    │
     │                    │                    │                    │
     │                    │ 2. SYNC USER       │                    │
     │                    │───────────────────►│                    │
     │                    │ POST /users/sync   │                    │
     │                    │ {user_id, email}   │                    │
     │                    │                    │ [Prepare memory]   │
     │                    │                    │───────────────────►│
     │                    │                    │                    │
     │                    │ 3. SYNC RESPONSE   │                    │
     │                    │◄───────────────────│                    │
     │                    │                    │                    │
     │ 4. REGISTRATION OK │                    │                    │
     │◄───────────────────│                    │                    │
     │ {token, user}      │                    │                    │
     │                    │                    │                    │
     │                    │                    │                    │
     │ 5. CHAT MESSAGE    │                    │                    │
     │───────────────────►│                    │                    │
     │ POST /chat         │                    │                    │
     │ {message}          │                    │                    │
     │ Header: Bearer JWT │                    │                    │
     │                    │                    │                    │
     │                    │ 6. PROXY TO AI     │                    │
     │                    │───────────────────►│                    │
     │                    │ POST /api/v1/chat  │                    │
     │                    │ {message, user_id} │                    │
     │                    │ Header: Bearer JWT │                    │
     │                    │                    │                    │
     │                    │                    │ 7. RETRIEVE MEMORY │
     │                    │                    │───────────────────►│
     │                    │                    │ "What do I know?"  │
     │                    │                    │                    │
     │                    │                    │ 8. MEMORY DATA     │
     │                    │                    │◄───────────────────│
     │                    │                    │                    │
     │                    │                    │ 9. AI RESPONSE     │
     │                    │                    │ [Generate with context]
     │                    │                    │                    │
     │                    │ 10. RESPONSE       │                    │
     │                    │◄───────────────────│                    │
     │                    │ {response}         │                    │
     │                    │                    │                    │
     │ 11. RESPONSE       │                    │                    │
     │◄───────────────────│                    │                    │
     │ {success, data}    │                    │                    │
     │                    │                    │                    │
     │                    │                    │                    │
     │ 12. SECOND MESSAGE │                    │                    │
     │───────────────────►│                    │                    │
     │ "What is my name?" │                    │                    │
     │                    │                    │                    │
     │                    │ 13. PROXY TO AI    │                    │
     │                    │───────────────────►│                    │
     │                    │                    │                    │
     │                    │                    │ 14. RETRIEVE MEMORY│
     │                    │                    │───────────────────►│
     │                    │                    │ "User's name?"     │
     │                    │                    │                    │
     │                    │                    │ 15. FOUND: "Alice" │
     │                    │                    │◄───────────────────│
     │                    │                    │                    │
     │                    │                    │ 16. "Your name is Alice!"
     │                    │                    │                    │
     │                    │ 17. RESPONSE       │                    │
     │                    │◄───────────────────│                    │
     │                    │                    │                    │
     │ 18. DISPLAY        │                    │                    │
     │◄───────────────────│                    │                    │
     │ "Your name is Alice!"                   │                    │
     │                    │                    │                    │
```

## Step-by-Step Breakdown

### Phase 1: Registration (Steps 1-4)
1. **Frontend** sends POST to `/api/v1/auth/register` with user details
2. **Backend** saves user to MongoDB with hashed password
3. **Backend** asynchronously notifies AI Service via `/api/v1/users/sync`
4. **Frontend** receives JWT token and user data

### Phase 2: First Chat Message (Steps 5-11)
5. **Frontend** sends message "My name is Alice" with JWT token
6. **Backend** forwards to AI Service at `/api/v1/chat`
7. **AI Service** retrieves any existing memories from Pinecone
8. **Pinecone** returns empty (first interaction)
9. **AI Service** generates response and stores memory: "User's name is Alice"
10. **AI Service** returns response to Backend
11. **Frontend** displays AI response

### Phase 3: Memory Test (Steps 12-18)
12. **Frontend** asks "What is my name?"
13. **Backend** forwards to AI Service
14. **AI Service** queries Pinecone for memories about user
15. **Pinecone** returns: "User's name is Alice" (stored in step 9)
16. **AI Service** generates contextual response using memory
17. **AI Service** returns response to Backend
18. **Frontend** displays: "Your name is Alice!"

## Key Points

### JWT Token Flow
- **Generated by**: Backend during login/register
- **Contains**: `{sub: userId, email, role, exp}`
- **Used by**: AI Service to identify user (extracts `sub` as `user_id`)
- **Secret**: Must be identical in both Backend and AI Service

### Memory Storage
- **Where**: Pinecone vector database
- **When**: Every chat message that contains personal info
- **How**: AI extracts facts and stores as embeddings
- **Retrieval**: Automatically queried before generating responses

### Error Handling
- If AI Service is down during registration: User still registers (sync fails silently)
- If AI Service is down during chat: Returns 503 "AI service temporarily unavailable"
- If Pinecone is down: AI responds without memory context

## Environment Variables Required

### Backend (.env)
```env
JWT_SECRET=your-secret-key        # ← MUST match AI Service
AI_SERVICE_URL=http://localhost:8000
MONGODB_URI=mongodb://...
```

### AI Service (.env)
```env
JWT_SECRET=your-secret-key        # ← MUST match Backend
PINECONE_API_KEY=your-key
```

### Frontend (.env)
```env
VITE_API_URL=http://localhost:4000/api/v1
```
