#!/bin/bash
#
# YES/NO.EVENTS Test Runner
# =========================
# Run all tests for the terminal and web applications
#

cd "$(dirname "$0")"
source .venv/bin/activate

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           YES/NO.EVENTS - Test Suite Runner               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check pytest installed
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}Installing pytest...${NC}"
    pip install pytest pytest-timeout > /dev/null
fi

# Test categories
run_test_suite() {
    local name="$1"
    local pattern="$2"

    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  Running: ${name}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

    if python -m pytest "$pattern" -v --tb=short 2>/dev/null; then
        echo -e "\n${GREEN}✓ ${name} PASSED${NC}"
        return 0
    else
        echo -e "\n${RED}✗ ${name} FAILED${NC}"
        return 1
    fi
}

# Help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: ./run_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  all        Run all tests (default)"
    echo "  quant      Run quant model tests only"
    echo "  search     Run search module tests only"
    echo "  trading    Run trading API tests only"
    echo "  dashboard  Run dashboard API tests only"
    echo "  fast       Run fast tests (skip network tests)"
    echo "  -v         Verbose output"
    echo "  -h, --help Show this help"
    echo ""
    echo "Examples:"
    echo "  ./run_tests.sh           # Run all tests"
    echo "  ./run_tests.sh quant     # Run quant tests only"
    echo "  ./run_tests.sh fast      # Skip slow network tests"
    exit 0
fi

# Track results
TOTAL=0
PASSED=0
FAILED=0

case "$1" in
    quant)
        run_test_suite "Quant Models" "tests/test_quant_models.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        ;;
    search)
        run_test_suite "Search Module" "tests/test_search_module.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        ;;
    trading)
        run_test_suite "Trading API" "tests/test_trading_api.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        ;;
    dashboard)
        run_test_suite "Dashboard API" "tests/test_dashboard_api.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        ;;
    fast)
        echo -e "${YELLOW}Running fast tests (skipping network-dependent tests)...${NC}"
        run_test_suite "Quant Models" "tests/test_quant_models.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        run_test_suite "Search Module" "tests/test_search_module.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        ;;
    all|"")
        run_test_suite "Quant Models" "tests/test_quant_models.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        run_test_suite "Search Module" "tests/test_search_module.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        run_test_suite "Trading API" "tests/test_trading_api.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        run_test_suite "Dashboard API" "tests/test_dashboard_api.py" && ((PASSED++)) || ((FAILED++))
        ((TOTAL++))
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Run ./run_tests.sh --help for usage"
        exit 1
        ;;
esac

# Summary
echo -e "\n${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  TEST SUMMARY${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "  Total Suites: ${TOTAL}"
echo -e "  ${GREEN}Passed: ${PASSED}${NC}"
echo -e "  ${RED}Failed: ${FAILED}${NC}"

if [[ $FAILED -eq 0 ]]; then
    echo -e "\n${GREEN}✓ ALL TESTS PASSED${NC}\n"
    exit 0
else
    echo -e "\n${RED}✗ SOME TESTS FAILED${NC}\n"
    exit 1
fi
