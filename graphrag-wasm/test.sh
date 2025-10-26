#!/bin/bash
# GraphRAG WASM Test Runner
#
# Simplified test execution for GraphRAG WASM components.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BROWSER="chrome"
HEADLESS="--headless"
TEST_FILE=""
TEST_NAME=""

# Help message
show_help() {
    cat << EOF
GraphRAG WASM Test Runner

USAGE:
    ./test.sh [OPTIONS]

OPTIONS:
    -b, --browser <BROWSER>    Browser to use: chrome, firefox, safari (default: chrome)
    -v, --visual               Run with visible browser (no headless)
    -t, --test <FILE>          Run specific test file (e.g., end_to_end)
    -n, --name <NAME>          Run specific test function
    -h, --help                 Show this help message

EXAMPLES:
    # Run all tests in Chrome headless
    ./test.sh

    # Run tests in Firefox with visible browser
    ./test.sh --browser firefox --visual

    # Run only end-to-end tests
    ./test.sh --test end_to_end

    # Run specific test function
    ./test.sh --test end_to_end --name test_create_graphrag

    # Debug mode (visible browser, storage tests)
    ./test.sh --visual --test storage_tests

TEST FILES:
    - end_to_end          Complete integration tests
    - webllm_tests        WebLLM integration (most tests disabled by default)
    - persistence_tests   Save/load functionality
    - webgpu_tests        GPU detection
    - voy_tests          Vector search with Voy
    - storage_tests       IndexedDB and Cache API

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--browser)
            BROWSER="$2"
            shift 2
            ;;
        -v|--visual)
            HEADLESS=""
            shift
            ;;
        -t|--test)
            TEST_FILE="$2"
            shift 2
            ;;
        -n|--name)
            TEST_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo -e "${RED}Error: wasm-pack is not installed${NC}"
    echo "Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Build command
CMD="wasm-pack test $HEADLESS --$BROWSER"

if [ -n "$TEST_FILE" ]; then
    CMD="$CMD --test $TEST_FILE"
fi

if [ -n "$TEST_NAME" ]; then
    CMD="$CMD -- $TEST_NAME"
fi

# Print info
echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}  GraphRAG WASM Test Suite${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""
echo -e "${YELLOW}Browser:${NC} $BROWSER"
echo -e "${YELLOW}Mode:${NC}    $([ -z "$HEADLESS" ] && echo "Visual" || echo "Headless")"
if [ -n "$TEST_FILE" ]; then
    echo -e "${YELLOW}Test File:${NC} $TEST_FILE"
fi
if [ -n "$TEST_NAME" ]; then
    echo -e "${YELLOW}Test Name:${NC} $TEST_NAME"
fi
echo ""
echo -e "${YELLOW}Command:${NC} $CMD"
echo ""

# Run tests
echo -e "${GREEN}Running tests...${NC}"
echo ""

if $CMD; then
    echo ""
    echo -e "${GREEN}===========================================${NC}"
    echo -e "${GREEN}  ✅ All tests passed!${NC}"
    echo -e "${GREEN}===========================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}===========================================${NC}"
    echo -e "${RED}  ❌ Tests failed${NC}"
    echo -e "${RED}===========================================${NC}"
    exit 1
fi
