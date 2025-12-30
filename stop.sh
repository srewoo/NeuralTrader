#!/bin/bash

# NeuralTrader - Force Stop Script
# This script force kills ALL backend and frontend processes

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NeuralTrader${NC}"
echo -e "${RED}  FORCE STOPPING All Services...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check if a PID is valid and running
is_pid_valid() {
    local pid="$1"
    if [ -z "$pid" ] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then
        return 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Function to kill process from PID file with stale PID handling
kill_from_pidfile() {
    local pidfile="$1"
    local name="$2"

    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile" 2>/dev/null)
        if is_pid_valid "$pid"; then
            kill -9 "$pid" 2>/dev/null && echo -e "${GREEN}✅ Killed $name (PID: $pid)${NC}"
        else
            echo -e "${YELLOW}ℹ️  $name PID file exists but process not running (stale PID: $pid)${NC}"
        fi
        rm -f "$pidfile"
    fi
}

# Function to force kill processes
force_kill() {
    local pattern="$1"
    local name="$2"
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)

    if [ ! -z "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null
        echo -e "${GREEN}✅ Force killed $name${NC}"
        return 0
    fi
    return 1
}

# Function to force kill by port
force_kill_port() {
    local port="$1"
    local name="$2"
    local pids=$(lsof -ti:$port 2>/dev/null || true)

    if [ ! -z "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null
        echo -e "${GREEN}✅ Force killed $name (port $port)${NC}"
        return 0
    fi
    return 1
}

# ============================================
# BACKEND SERVICES
# ============================================
echo -e "${YELLOW}🔴 Stopping Backend Services...${NC}"

# Force kill uvicorn/FastAPI (multiple patterns)
force_kill "uvicorn server:app" "Backend API (uvicorn)"
force_kill "uvicorn.*server:app" "Backend API (uvicorn)"
force_kill_port 8005 "Backend API"

# Force kill Celery Worker
force_kill "celery.*worker" "Celery Worker"
force_kill "celery -A celery_app worker" "Celery Worker"

# Force kill Celery Beat
force_kill "celery.*beat" "Celery Beat Scheduler"
force_kill "celery -A celery_app beat" "Celery Beat Scheduler"

echo ""

# ============================================
# FRONTEND SERVICES
# ============================================
echo -e "${YELLOW}🎨 Stopping Frontend Services...${NC}"

# Force kill React dev server
force_kill "react-scripts start" "React Dev Server"
force_kill "node.*react-scripts" "React Dev Server (Node)"
force_kill_port 3005 "Frontend"

# Kill any stray node processes for this project
force_kill "node.*NeuralTrader/frontend" "Frontend Node processes"

echo ""

# ============================================
# DATABASE / CACHE SERVICES
# ============================================
echo -e "${YELLOW}💾 Stopping Database/Cache Services...${NC}"

# Force kill Redis (only if started by this app, not Homebrew service)
# Check if Redis was started by our start.sh (has PID file)
if [ -f "logs/redis.pid" ]; then
    REDIS_PID=$(cat logs/redis.pid 2>/dev/null)
    if [ ! -z "$REDIS_PID" ]; then
        kill -9 $REDIS_PID 2>/dev/null && echo -e "${GREEN}✅ Force killed Redis (PID: $REDIS_PID)${NC}"
    fi
    rm -f logs/redis.pid
fi

# Also kill redis-server if running standalone (not Homebrew managed)
# Note: We don't kill Homebrew Redis service - only standalone redis-server
if ! brew services list 2>/dev/null | grep -q "redis.*started"; then
    force_kill "redis-server" "Redis Server (standalone)"
fi

echo ""

# ============================================
# CLEANUP PID FILES (with stale PID detection)
# ============================================
echo -e "${YELLOW}🧹 Cleaning up PID files...${NC}"

# Try to kill from PID files first (handles stale PIDs gracefully)
kill_from_pidfile "logs/backend.pid" "Backend API"
kill_from_pidfile "logs/frontend.pid" "Frontend"
kill_from_pidfile "logs/celery_worker.pid" "Celery Worker"
kill_from_pidfile "logs/celery_beat.pid" "Celery Beat"

# Clean up any remaining stale PID files
for pidfile in logs/*.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile" 2>/dev/null)
        if ! is_pid_valid "$pid"; then
            rm -f "$pidfile"
            echo -e "${YELLOW}ℹ️  Removed stale PID file: $pidfile${NC}"
        fi
    fi
done

echo -e "${GREEN}✅ PID cleanup complete${NC}"

echo ""

# Wait for processes to terminate
sleep 1

# ============================================
# VERIFICATION
# ============================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Verification${NC}"
echo -e "${BLUE}========================================${NC}"

ALL_STOPPED=true

# Check Backend
if lsof -ti:8005 > /dev/null 2>&1; then
    echo -e "${RED}❌ Backend still running on port 8005${NC}"
    ALL_STOPPED=false
else
    echo -e "${GREEN}✅ Backend: Stopped${NC}"
fi

# Check Frontend
if lsof -ti:3005 > /dev/null 2>&1; then
    echo -e "${RED}❌ Frontend still running on port 3005${NC}"
    ALL_STOPPED=false
else
    echo -e "${GREEN}✅ Frontend: Stopped${NC}"
fi

# Check Celery Worker
if pgrep -f "celery.*worker" > /dev/null 2>&1; then
    echo -e "${RED}❌ Celery Worker still running${NC}"
    ALL_STOPPED=false
else
    echo -e "${GREEN}✅ Celery Worker: Stopped${NC}"
fi

# Check Celery Beat
if pgrep -f "celery.*beat" > /dev/null 2>&1; then
    echo -e "${RED}❌ Celery Beat still running${NC}"
    ALL_STOPPED=false
else
    echo -e "${GREEN}✅ Celery Beat: Stopped${NC}"
fi

# Check Redis (just report status, don't treat Homebrew Redis as error)
if redis-cli ping > /dev/null 2>&1; then
    if brew services list 2>/dev/null | grep -q "redis.*started"; then
        echo -e "${YELLOW}ℹ️  Redis: Running (Homebrew service - not stopped)${NC}"
    else
        echo -e "${GREEN}✅ Redis: Running (standalone)${NC}"
    fi
else
    echo -e "${GREEN}✅ Redis: Stopped${NC}"
fi

echo ""

if [ "$ALL_STOPPED" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}🎉 All NeuralTrader services stopped!${NC}"
    echo -e "${BLUE}========================================${NC}"
else
    echo -e "${BLUE}========================================${NC}"
    echo -e "${RED}⚠️  Some services may still be running${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Nuclear option (kill everything):${NC}"
    echo -e "   ${RED}lsof -ti:8005,3005 | xargs kill -9${NC}"
    echo -e "   ${RED}pkill -9 -f 'celery'${NC}"
    echo ""
fi

echo -e "${YELLOW}To restart: ${GREEN}./start.sh${NC}"
echo ""
