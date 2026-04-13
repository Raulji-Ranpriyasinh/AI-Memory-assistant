# Personal AI Assistant

A full-stack AI assistant application built with a modern microservices architecture.

## Tech Stack

### Backend
- **NestJS** - Node.js framework
- **MongoDB** - Primary database
- **PostgreSQL** - Relational database
- **Redis** - Caching & session management
- **JWT** - Authentication

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Zustand** - State management
- **React Query** - Data fetching
- **i18next** - Internationalization

### AI Service
- **Python** - Pre-existing AI service (see `ai-service/`)

### Mobile (Planned)
- **React Native** - Cross-platform mobile app

## Project Structure

```
├── backend/          # NestJS API server (Port 4000)
├── frontend/         # React web app (Port 3000)
├── mobile/           # React Native app (scaffolded)
├── ai-service/       # Pre-existing Python AI service (Port 8000)
├── docker-compose.yml
└── .env
```

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Node.js 20+ (for local development)
- Python 3.11+ (for AI service)

### Running with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Running Locally (Development)

#### Backend
```bash
cd backend
npm install
cp .env.example .env  # if needed
npm run start:dev
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

The backend will be available at `http://localhost:4000`
The frontend will be available at `http://localhost:3000`

## API Endpoints

### Health Check
```
GET http://localhost:4000/api/v1/health
Response: { "status": "ok", "timestamp": "..." }
```

## Environment Variables

See `.env` file at the project root for all configuration options.

## Development Phases

- [x] **Phase 1**: Project Scaffolding & Infrastructure
- [ ] **Phase 2**: Authentication & User Management
- [ ] **Phase 3**: AI Service Integration
- [ ] **Phase 4**: Payment Integration
- [ ] **Phase 5**: Mobile App Development

## License

Private - All rights reserved
