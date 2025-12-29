#!/bin/bash

# NeuralTrader - Integration Tests Runner
# Runs all API integration tests and generates reports

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo -e "  NeuralTrader - Integration Tests"
echo -e "========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create reports directory
mkdir -p reports

# Set environment variables
export DISABLE_CHROMADB=true
export PYTHONPATH="$SCRIPT_DIR"

# Timestamp for report files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${YELLOW}Running integration tests...${NC}"
echo ""

# Run integration tests with coverage and HTML report
./venv/bin/pytest tests/integration/ \
    -v \
    --tb=short \
    --cov=. \
    --cov-report=html:reports/integration_coverage_${TIMESTAMP} \
    --cov-report=term-missing \
    --html=reports/integration_report_${TIMESTAMP}.html \
    --self-contained-html \
    2>&1 | tee reports/integration_output_${TIMESTAMP}.log

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo -e "${BLUE}========================================"
echo -e "  Test Results Summary"
echo -e "========================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All integration tests PASSED${NC}"
else
    echo -e "${RED}Some integration tests FAILED${NC}"
fi

echo ""
echo -e "${YELLOW}Reports generated:${NC}"
echo -e "  - HTML Report: reports/integration_report_${TIMESTAMP}.html"
echo -e "  - Coverage: reports/integration_coverage_${TIMESTAMP}/index.html"
echo -e "  - Log: reports/integration_output_${TIMESTAMP}.log"
echo ""

# Create symlink to latest report
ln -sf "integration_report_${TIMESTAMP}.html" reports/integration_report_latest.html
ln -sf "integration_coverage_${TIMESTAMP}" reports/integration_coverage_latest

echo -e "${GREEN}Latest report symlinks created:${NC}"
echo -e "  - reports/integration_report_latest.html"
echo -e "  - reports/integration_coverage_latest/"
echo ""

exit $EXIT_CODE
