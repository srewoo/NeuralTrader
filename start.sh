#!/bin/bash

# NeuralTrader - Start Script
# This script starts both backend and frontend servers

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NeuralTrader${NC}"
echo -e "${BLUE}  Starting Servers...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if servers are already running
if pgrep -f "uvicorn.*backend.server:app" > /dev/null; then
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
    
    # Start backend in background
    nohup uvicorn server:app --reload --host 0.0.0.0 --port 8005 > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../logs/backend.pid
    echo -e "${GREEN}✅ Backend started on http://localhost:8005 (PID: $BACKEND_PID)${NC}"
    echo -e "${GREEN}   Logs: logs/backend.log${NC}"
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
echo -e "${BLUE}⏳ Waiting for servers to initialize...${NC}"
sleep 5

# Check if servers are running
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Server Status${NC}"
echo -e "${BLUE}========================================${NC}"

if pgrep -f "uvicorn.*backend.server:app" > /dev/null; then
    echo -e "${GREEN}✅ Backend: Running on http://localhost:8005${NC}"
    echo -e "${GREEN}   API Docs: http://localhost:8005/docs${NC}"
else
    echo -e "${RED}❌ Backend: Not running${NC}"
fi

if pgrep -f "react-scripts start" > /dev/null || lsof -ti:3005 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend: Running on http://localhost:3005${NC}"
else
    echo -e "${RED}❌ Frontend: Not running${NC}"
    echo -e "${YELLOW}   Waiting for React to start...${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🚀 Application is starting!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}📝 Useful Commands:${NC}"
echo -e "   ${GREEN}./stop.sh${NC}          - Stop all servers"
echo -e "   ${GREEN}tail -f logs/backend.log${NC}   - View backend logs"
echo -e "   ${GREEN}tail -f logs/frontend.log${NC}  - View frontend logs"
echo ""
echo -e "${YELLOW}🌐 Access Points:${NC}"
echo -e "   Frontend:  ${GREEN}http://localhost:3005${NC}"
echo -e "   Backend:   ${GREEN}http://localhost:8005${NC}"
echo -e "   API Docs:  ${GREEN}http://localhost:8005/docs${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"

