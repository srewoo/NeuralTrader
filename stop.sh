#!/bin/bash

# NeuralTrader - Stop Script
# This script stops both backend and frontend servers

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NeuralTrader${NC}"
echo -e "${BLUE}  Stopping Servers...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

STOPPED_ANY=false

# Stop Backend Server
echo -e "${YELLOW}🛑 Stopping Backend Server...${NC}"

# Try to kill using PID file first
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        kill $BACKEND_PID 2>/dev/null
        echo -e "${GREEN}✅ Backend server stopped (PID: $BACKEND_PID)${NC}"
        STOPPED_ANY=true
    fi
    rm -f logs/backend.pid
fi

# Also kill any uvicorn processes running the backend
UVICORN_PIDS=$(pgrep -f "uvicorn.*backend.server:app" || true)
if [ ! -z "$UVICORN_PIDS" ]; then
    echo "$UVICORN_PIDS" | while read pid; do
        kill $pid 2>/dev/null && echo -e "${GREEN}✅ Killed backend process (PID: $pid)${NC}"
        STOPPED_ANY=true
    done
fi

# Stop processes on port 8005
PORT_8005_PIDS=$(lsof -ti:8005 2>/dev/null || true)
if [ ! -z "$PORT_8005_PIDS" ]; then
    echo "$PORT_8005_PIDS" | while read pid; do
        kill $pid 2>/dev/null && echo -e "${GREEN}✅ Killed process on port 8005 (PID: $pid)${NC}"
        STOPPED_ANY=true
    done
fi

if ! pgrep -f "uvicorn.*backend.server:app" > /dev/null && [ -z "$(lsof -ti:8005 2>/dev/null || true)" ]; then
    if [ "$STOPPED_ANY" = false ]; then
        echo -e "${YELLOW}ℹ️  Backend server was not running${NC}"
    fi
fi

echo ""

# Stop Frontend Server
echo -e "${YELLOW}🛑 Stopping Frontend Server...${NC}"

# Try to kill using PID file first
if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        kill $FRONTEND_PID 2>/dev/null
        echo -e "${GREEN}✅ Frontend server stopped (PID: $FRONTEND_PID)${NC}"
        STOPPED_ANY=true
    fi
    rm -f logs/frontend.pid
fi

# Kill any react-scripts processes
REACT_PIDS=$(pgrep -f "react-scripts start" || true)
if [ ! -z "$REACT_PIDS" ]; then
    echo "$REACT_PIDS" | while read pid; do
        kill $pid 2>/dev/null && echo -e "${GREEN}✅ Killed frontend process (PID: $pid)${NC}"
        STOPPED_ANY=true
    done
fi

# Kill any node processes on port 3005
PORT_3005_PIDS=$(lsof -ti:3005 2>/dev/null || true)
if [ ! -z "$PORT_3005_PIDS" ]; then
    echo "$PORT_3005_PIDS" | while read pid; do
        kill $pid 2>/dev/null && echo -e "${GREEN}✅ Killed process on port 3005 (PID: $pid)${NC}"
        STOPPED_ANY=true
    done
fi

if ! pgrep -f "react-scripts start" > /dev/null && [ -z "$(lsof -ti:3005 2>/dev/null || true)" ]; then
    if [ "$STOPPED_ANY" = false ]; then
        echo -e "${YELLOW}ℹ️  Frontend server was not running${NC}"
    fi
fi

echo ""

# Wait a moment for processes to fully terminate
sleep 2

# Verify servers are stopped
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Verification${NC}"
echo -e "${BLUE}========================================${NC}"

BACKEND_RUNNING=false
FRONTEND_RUNNING=false

if pgrep -f "uvicorn.*backend.server:app" > /dev/null || [ ! -z "$(lsof -ti:8005 2>/dev/null || true)" ]; then
    echo -e "${RED}⚠️  Backend may still be running${NC}"
    BACKEND_RUNNING=true
else
    echo -e "${GREEN}✅ Backend: Stopped${NC}"
fi

if pgrep -f "react-scripts start" > /dev/null || [ ! -z "$(lsof -ti:3005 2>/dev/null || true)" ]; then
    echo -e "${RED}⚠️  Frontend may still be running${NC}"
    FRONTEND_RUNNING=true
else
    echo -e "${GREEN}✅ Frontend: Stopped${NC}"
fi

echo ""

if [ "$BACKEND_RUNNING" = true ] || [ "$FRONTEND_RUNNING" = true ]; then
    echo -e "${YELLOW}⚠️  Some processes may still be running.${NC}"
    echo -e "${YELLOW}   Try running with force: kill -9${NC}"
    echo ""
    echo -e "${YELLOW}Manual cleanup commands:${NC}"
    [ "$BACKEND_RUNNING" = true ] && echo -e "   ${RED}pkill -9 -f 'uvicorn.*backend.server:app'${NC}"
    [ "$FRONTEND_RUNNING" = true ] && echo -e "   ${RED}pkill -9 -f 'react-scripts start'${NC}"
    echo ""
    exit 1
else
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}✅ All servers stopped successfully!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${YELLOW}To start servers again, run: ${GREEN}./start.sh${NC}"
    echo ""
fi

