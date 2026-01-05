#!/usr/bin/env bash

# Clear caches (Redis/Celery) and logs (frontend/backend) without stopping services.
# Designed to be safe to run multiple times.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
BACKEND_DIR="${SCRIPT_DIR}/backend"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

echo -e "${GREEN}=== NeuralTrader Cleanup ===${NC}"

# Clear application logs (root logs folder)
if [ -d "${LOG_DIR}" ]; then
  echo -e "${GREEN}• Clearing logs in ${LOG_DIR}${NC}"
  # Remove log and pid files, keep the folder structure
  find "${LOG_DIR}" -type f \( -name "*.log" -o -name "*.pid" \) -print -delete || true
else
  echo -e "${YELLOW}• Logs directory not found at ${LOG_DIR}${NC}"
fi

# Clear backend logs folder if present (ignored in git)
if [ -d "${BACKEND_DIR}/logs" ]; then
  echo -e "${GREEN}• Clearing backend logs in ${BACKEND_DIR}/logs${NC}"
  find "${BACKEND_DIR}/logs" -type f -print -delete || true
fi

# Clear Redis cache
if command -v redis-cli >/dev/null 2>&1; then
  if redis-cli ping >/dev/null 2>&1; then
    echo -e "${GREEN}• Flushing Redis (all databases)${NC}"
    redis-cli FLUSHALL || echo -e "${RED}  ⚠️ Redis flush failed${NC}"
  else
    echo -e "${YELLOW}• Redis not running; skipped flush${NC}"
  fi
else
  echo -e "${YELLOW}• redis-cli not found; skipped Redis flush${NC}"
fi

# Clear Celery beat schedule cache files (local state)
echo -e "${GREEN}• Removing Celery beat schedule cache${NC}"
shopt -s nullglob
for schedule_file in "${BACKEND_DIR}"/celerybeat-schedule*; do
  rm -f "${schedule_file}" && echo "  removed ${schedule_file}"
done
shopt -u nullglob

echo -e "${GREEN}✅ Cleanup complete.${NC}"

