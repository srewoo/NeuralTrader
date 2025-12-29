#!/bin/bash

# NeuralTrader - Start Script
# This script starts both backend and frontend servers

set -e

# Get the directory where this script is located (works on any machine)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NeuralTrader${NC}"
echo -e "${BLUE}  Starting All Services...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to check if Redis is installed
check_redis() {
    if ! command -v redis-server &> /dev/null; then
        echo -e "${YELLOW}⚠️  Redis not found. Installing via Homebrew...${NC}"
        if command -v brew &> /dev/null; then
            brew install redis
        else
            echo -e "${RED}❌ Homebrew not found. Please install Redis manually:${NC}"
            echo -e "${YELLOW}   brew install redis${NC}"
            echo -e "${YELLOW}   OR download from: https://redis.io/download${NC}"
            exit 1
        fi
    fi
}

# Check Redis installation
check_redis

# Start Redis Server
echo -e "${GREEN}🔴 Starting Redis Server...${NC}"
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Redis is already running!${NC}"
else
    nohup redis-server > logs/redis.log 2>&1 &
    REDIS_PID=$!
    echo $REDIS_PID > logs/redis.pid
    echo -e "${GREEN}✅ Redis started (PID: $REDIS_PID)${NC}"
    echo -e "${GREEN}   Logs: logs/redis.log${NC}"
    sleep 2
fi
echo ""

# Check if backend servers are already running
if pgrep -f "uvicorn server:app" > /dev/null; then
    echo -e "${YELLOW}⚠️  Backend server is already running!${NC}"
    echo -e "${YELLOW}   Run ./stop.sh first to stop existing servers${NC}"
    echo ""
else
    # Start Backend Server
    echo -e "${GREEN}📡 Starting Backend Server...${NC}"
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
        python3 -m venv venv
        source venv/bin/activate
        echo -e "${YELLOW}📦 Installing backend dependencies...${NC}"
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Start backend in background (using asyncio loop instead of uvloop to prevent macOS crash)
    nohup uvicorn server:app --host 0.0.0.0 --port 8005 --loop asyncio > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../logs/backend.pid
    echo -e "${GREEN}✅ Backend FastAPI started on http://localhost:8005 (PID: $BACKEND_PID)${NC}"
    echo -e "${GREEN}   Logs: logs/backend.log${NC}"
    echo ""

    # Start Celery Worker (must be run from backend directory with PYTHONPATH set)
    echo -e "${GREEN}⚙️  Starting Celery Worker...${NC}"
    if pgrep -f "celery.*worker" > /dev/null; then
        echo -e "${YELLOW}⚠️  Celery worker is already running!${NC}"
    else
        # Set PYTHONPATH to include backend directory for imports
        PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH" \
        nohup celery -A celery_app worker --loglevel=info --pool=solo > ../logs/celery_worker.log 2>&1 &
        CELERY_WORKER_PID=$!
        echo $CELERY_WORKER_PID > ../logs/celery_worker.pid
        echo -e "${GREEN}✅ Celery Worker started (PID: $CELERY_WORKER_PID)${NC}"
        echo -e "${GREEN}   Logs: logs/celery_worker.log${NC}"
    fi
    echo ""

    # Start Celery Beat (Scheduler)
    echo -e "${GREEN}⏰ Starting Celery Beat Scheduler...${NC}"
    if pgrep -f "celery.*beat" > /dev/null; then
        echo -e "${YELLOW}⚠️  Celery beat is already running!${NC}"
    else
        # Set PYTHONPATH to include backend directory
        PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH" \
        nohup celery -A celery_app beat --loglevel=info > ../logs/celery_beat.log 2>&1 &
        CELERY_BEAT_PID=$!
        echo $CELERY_BEAT_PID > ../logs/celery_beat.pid
        echo -e "${GREEN}✅ Celery Beat started (PID: $CELERY_BEAT_PID)${NC}"
        echo -e "${GREEN}   Logs: logs/celery_beat.log${NC}"
    fi
    echo ""

    cd ..
fi

# Check if frontend is already running
if pgrep -f "react-scripts start" > /dev/null || lsof -ti:3005 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Frontend server is already running!${NC}"
    echo ""
else
    # Start Frontend Server
    echo -e "${GREEN}🎨 Starting Frontend Server...${NC}"
    cd frontend

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}⚠️  Node modules not found. Installing...${NC}"
        npm install
    fi

    # Start frontend in background with custom port
    PORT=3005 nohup npm start > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../logs/frontend.pid
    echo -e "${GREEN}✅ Frontend started on http://localhost:3005 (PID: $FRONTEND_PID)${NC}"
    echo -e "${GREEN}   Logs: logs/frontend.log${NC}"
    echo ""
    
    cd ..
fi

# Wait a moment for servers to start
echo -e "${BLUE}⏳ Waiting for all services to initialize...${NC}"
sleep 8

# Check if all services are running
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Service Status${NC}"
echo -e "${BLUE}========================================${NC}"

# Check Redis (use redis-cli ping for accurate detection)
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis: Running${NC}"
else
    echo -e "${RED}❌ Redis: Not running${NC}"
fi

# Check Backend
if pgrep -f "uvicorn server:app" > /dev/null; then
    echo -e "${GREEN}✅ Backend API: Running on http://localhost:8005${NC}"
    echo -e "${GREEN}   API Docs: http://localhost:8005/docs${NC}"
else
    echo -e "${RED}❌ Backend API: Not running${NC}"
fi

# Check Celery Worker
if pgrep -f "celery.*worker" > /dev/null; then
    echo -e "${GREEN}✅ Celery Worker: Running (background tasks)${NC}"
else
    echo -e "${RED}❌ Celery Worker: Not running${NC}"
fi

# Check Celery Beat
if pgrep -f "celery.*beat" > /dev/null; then
    echo -e "${GREEN}✅ Celery Beat: Running (scheduler)${NC}"
else
    echo -e "${RED}❌ Celery Beat: Not running${NC}"
fi

# Check Frontend
if pgrep -f "react-scripts start" > /dev/null || lsof -ti:3005 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend: Running on http://localhost:3005${NC}"
else
    echo -e "${RED}❌ Frontend: Not running${NC}"
    echo -e "${YELLOW}   Waiting for React to start...${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🚀 NeuralTrader is Running!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}📝 Useful Commands:${NC}"
echo -e "   ${GREEN}./stop.sh${NC}                      - Stop all services"
echo -e "   ${GREEN}tail -f logs/backend.log${NC}       - View FastAPI logs"
echo -e "   ${GREEN}tail -f logs/celery_worker.log${NC} - View Celery worker logs"
echo -e "   ${GREEN}tail -f logs/celery_beat.log${NC}   - View Celery beat logs"
echo -e "   ${GREEN}tail -f logs/frontend.log${NC}      - View React logs"
echo -e "   ${GREEN}tail -f logs/redis.log${NC}         - View Redis logs"
echo ""
echo -e "${YELLOW}🌐 Access Points:${NC}"
echo -e "   Frontend:   ${GREEN}http://localhost:3005${NC}"
echo -e "   Backend:    ${GREEN}http://localhost:8005${NC}"
echo -e "   API Docs:   ${GREEN}http://localhost:8005/docs${NC}"
echo ""
echo -e "${YELLOW}📊 Background Services:${NC}"
echo -e "   Redis:      ${GREEN}Running${NC} (Cache & Message Broker)"
echo -e "   Celery:     ${GREEN}Running${NC} (AI Tasks, News, Market Data)"
echo -e "   Beat:       ${GREEN}Running${NC} (Scheduled Jobs)"
echo ""
echo -e "${BLUE}========================================${NC}"

