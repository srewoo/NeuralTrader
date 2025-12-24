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
echo -e "${BLUE}  Stopping All Services...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

STOPPED_ANY=false

# Stop Redis Server
echo -e "${YELLOW}🛑 Stopping Redis Server...${NC}"

if [ -f "logs/redis.pid" ]; then
    REDIS_PID=$(cat logs/redis.pid)
    if ps -p $REDIS_PID > /dev/null 2>&1; then
        kill $REDIS_PID 2>/dev/null
        echo -e "${GREEN}✅ Redis server stopped (PID: $REDIS_PID)${NC}"
        STOPPED_ANY=true
    fi
    rm -f logs/redis.pid
fi

# Kill any redis-server processes
REDIS_PIDS=$(pgrep -x "redis-server" || true)
if [ ! -z "$REDIS_PIDS" ]; then
    echo "$REDIS_PIDS" | while read pid; do
        kill $pid 2>/dev/null && echo -e "${GREEN}✅ Killed Redis process (PID: $pid)${NC}"
        STOPPED_ANY=true
    done
fi

if ! pgrep -x "redis-server" > /dev/null; then
    if [ "$STOPPED_ANY" = false ]; then
        echo -e "${YELLOW}ℹ️  Redis was not running${NC}"
    fi
fi

echo ""

# Stop Celery Worker
echo -e "${YELLOW}🛑 Stopping Celery Worker...${NC}"

if [ -f "logs/celery_worker.pid" ]; then
    CELERY_WORKER_PID=$(cat logs/celery_worker.pid)
    if ps -p $CELERY_WORKER_PID > /dev/null 2>&1; then
        kill $CELERY_WORKER_PID 2>/dev/null
        echo -e "${GREEN}✅ Celery worker stopped (PID: $CELERY_WORKER_PID)${NC}"
        STOPPED_ANY=true
    fi
    rm -f logs/celery_worker.pid
fi

# Kill any celery worker processes
CELERY_WORKER_PIDS=$(pgrep -f "celery.*worker" || true)
if [ ! -z "$CELERY_WORKER_PIDS" ]; then
    echo "$CELERY_WORKER_PIDS" | while read pid; do
        kill $pid 2>/dev/null && echo -e "${GREEN}✅ Killed Celery worker process (PID: $pid)${NC}"
        STOPPED_ANY=true
    done
fi

if ! pgrep -f "celery.*worker" > /dev/null; then
    if [ "$STOPPED_ANY" = false ]; then
        echo -e "${YELLOW}ℹ️  Celery worker was not running${NC}"
    fi
fi

echo ""

# Stop Celery Beat
echo -e "${YELLOW}🛑 Stopping Celery Beat...${NC}"

if [ -f "logs/celery_beat.pid" ]; then
    CELERY_BEAT_PID=$(cat logs/celery_beat.pid)
    if ps -p $CELERY_BEAT_PID > /dev/null 2>&1; then
        kill $CELERY_BEAT_PID 2>/dev/null
        echo -e "${GREEN}✅ Celery beat stopped (PID: $CELERY_BEAT_PID)${NC}"
        STOPPED_ANY=true
    fi
    rm -f logs/celery_beat.pid
fi

# Kill any celery beat processes
CELERY_BEAT_PIDS=$(pgrep -f "celery.*beat" || true)
if [ ! -z "$CELERY_BEAT_PIDS" ]; then
    echo "$CELERY_BEAT_PIDS" | while read pid; do
        kill $pid 2>/dev/null && echo -e "${GREEN}✅ Killed Celery beat process (PID: $pid)${NC}"
        STOPPED_ANY=true
    done
fi

if ! pgrep -f "celery.*beat" > /dev/null; then
    if [ "$STOPPED_ANY" = false ]; then
        echo -e "${YELLOW}ℹ️  Celery beat was not running${NC}"
    fi
fi

echo ""

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

# Verify all services are stopped
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Verification${NC}"
echo -e "${BLUE}========================================${NC}"

REDIS_RUNNING=false
BACKEND_RUNNING=false
CELERY_WORKER_RUNNING=false
CELERY_BEAT_RUNNING=false
FRONTEND_RUNNING=false

if pgrep -x "redis-server" > /dev/null; then
    echo -e "${RED}⚠️  Redis may still be running${NC}"
    REDIS_RUNNING=true
else
    echo -e "${GREEN}✅ Redis: Stopped${NC}"
fi

if pgrep -f "uvicorn.*backend.server:app" > /dev/null || [ ! -z "$(lsof -ti:8005 2>/dev/null || true)" ]; then
    echo -e "${RED}⚠️  Backend API may still be running${NC}"
    BACKEND_RUNNING=true
else
    echo -e "${GREEN}✅ Backend API: Stopped${NC}"
fi

if pgrep -f "celery.*worker" > /dev/null; then
    echo -e "${RED}⚠️  Celery worker may still be running${NC}"
    CELERY_WORKER_RUNNING=true
else
    echo -e "${GREEN}✅ Celery Worker: Stopped${NC}"
fi

if pgrep -f "celery.*beat" > /dev/null; then
    echo -e "${RED}⚠️  Celery beat may still be running${NC}"
    CELERY_BEAT_RUNNING=true
else
    echo -e "${GREEN}✅ Celery Beat: Stopped${NC}"
fi

if pgrep -f "react-scripts start" > /dev/null || [ ! -z "$(lsof -ti:3005 2>/dev/null || true)" ]; then
    echo -e "${RED}⚠️  Frontend may still be running${NC}"
    FRONTEND_RUNNING=true
else
    echo -e "${GREEN}✅ Frontend: Stopped${NC}"
fi

echo ""

if [ "$REDIS_RUNNING" = true ] || [ "$BACKEND_RUNNING" = true ] || [ "$CELERY_WORKER_RUNNING" = true ] || [ "$CELERY_BEAT_RUNNING" = true ] || [ "$FRONTEND_RUNNING" = true ]; then
    echo -e "${YELLOW}⚠️  Some services may still be running.${NC}"
    echo -e "${YELLOW}   Try running with force: kill -9${NC}"
    echo ""
    echo -e "${YELLOW}Manual cleanup commands:${NC}"
    [ "$REDIS_RUNNING" = true ] && echo -e "   ${RED}pkill -9 -x redis-server${NC}"
    [ "$BACKEND_RUNNING" = true ] && echo -e "   ${RED}pkill -9 -f 'uvicorn.*backend.server:app'${NC}"
    [ "$CELERY_WORKER_RUNNING" = true ] && echo -e "   ${RED}pkill -9 -f 'celery.*worker'${NC}"
    [ "$CELERY_BEAT_RUNNING" = true ] && echo -e "   ${RED}pkill -9 -f 'celery.*beat'${NC}"
    [ "$FRONTEND_RUNNING" = true ] && echo -e "   ${RED}pkill -9 -f 'react-scripts start'${NC}"
    echo ""
    exit 1
else
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}✅ All services stopped successfully!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Services Stopped:${NC}"
    echo -e "   ${GREEN}✅${NC} Redis Server"
    echo -e "   ${GREEN}✅${NC} Backend API (FastAPI)"
    echo -e "   ${GREEN}✅${NC} Celery Worker"
    echo -e "   ${GREEN}✅${NC} Celery Beat Scheduler"
    echo -e "   ${GREEN}✅${NC} Frontend (React)"
    echo ""
    echo -e "${YELLOW}To start all services again, run: ${GREEN}./start.sh${NC}"
    echo ""
fi

