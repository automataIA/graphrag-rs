#!/bin/bash
#
# GraphRAG Leptos Demo - ONNX Model Setup Script
#
# Scarica automaticamente il modello all-MiniLM-L6-v2 in formato ONNX
# da HuggingFace e lo prepara per l'uso con ONNX Runtime Web.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_DIR="models"
MODEL_NAME="all-MiniLM-L6-v2"
MODEL_FILE="${MODEL_NAME}.onnx"
HUGGINGFACE_REPO="Xenova/all-MiniLM-L6-v2"
HUGGINGFACE_URL="https://huggingface.co/${HUGGINGFACE_REPO}/resolve/main"

# Functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  GraphRAG Leptos Demo - ONNX Model Setup              ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_dependencies() {
    print_step "Checking dependencies..."

    # Check for curl or wget
    if command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl"
        print_success "curl found"
    elif command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget"
        print_success "wget found"
    else
        print_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    # Check for git (optional, for git lfs method)
    if command -v git &> /dev/null; then
        HAS_GIT=true
        if git lfs version &> /dev/null 2>&1; then
            HAS_GIT_LFS=true
            print_success "git with LFS support found"
        else
            HAS_GIT_LFS=false
            print_warning "git found but LFS not installed (will use direct download)"
        fi
    else
        HAS_GIT=false
        HAS_GIT_LFS=false
        print_warning "git not found (will use direct download)"
    fi

    echo ""
}

create_model_dir() {
    print_step "Creating model directory..."

    if [ ! -d "$MODEL_DIR" ]; then
        mkdir -p "$MODEL_DIR"
        print_success "Created $MODEL_DIR/"
    else
        print_success "$MODEL_DIR/ already exists"
    fi

    echo ""
}

download_model_git_lfs() {
    print_step "Downloading model using git lfs..."

    local TEMP_DIR="${MODEL_DIR}/${MODEL_NAME}-temp"

    if [ -d "$TEMP_DIR" ]; then
        print_warning "Temp directory exists, removing..."
        rm -rf "$TEMP_DIR"
    fi

    # Clone with git lfs
    git clone "https://huggingface.co/${HUGGINGFACE_REPO}" "$TEMP_DIR"

    # Copy ONNX model
    if [ -f "${TEMP_DIR}/onnx/model.onnx" ]; then
        cp "${TEMP_DIR}/onnx/model.onnx" "${MODEL_DIR}/${MODEL_FILE}"
        print_success "Copied model file"
    else
        print_error "ONNX model not found in repository"
        rm -rf "$TEMP_DIR"
        return 1
    fi

    # Copy tokenizer
    if [ -f "${TEMP_DIR}/tokenizer.json" ]; then
        cp "${TEMP_DIR}/tokenizer.json" "${MODEL_DIR}/"
        print_success "Copied tokenizer"
    else
        print_warning "tokenizer.json not found (may not be needed)"
    fi

    # Cleanup
    rm -rf "$TEMP_DIR"
    print_success "Cleaned up temp files"

    return 0
}

download_model_direct() {
    print_step "Downloading model directly from HuggingFace..."

    local MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
    local MODEL_URL="${HUGGINGFACE_URL}/onnx/model.onnx"

    if [ "$DOWNLOAD_CMD" = "curl" ]; then
        curl -L -o "$MODEL_PATH" "$MODEL_URL" --progress-bar
    else
        wget -O "$MODEL_PATH" "$MODEL_URL" --show-progress
    fi

    if [ $? -eq 0 ] && [ -f "$MODEL_PATH" ]; then
        print_success "Downloaded ONNX model"

        # Download tokenizer (optional)
        local TOKENIZER_URL="${HUGGINGFACE_URL}/tokenizer.json"
        if [ "$DOWNLOAD_CMD" = "curl" ]; then
            curl -L -o "${MODEL_DIR}/tokenizer.json" "$TOKENIZER_URL" --progress-bar 2>/dev/null || true
        else
            wget -O "${MODEL_DIR}/tokenizer.json" "$TOKENIZER_URL" --show-progress 2>/dev/null || true
        fi

        return 0
    else
        print_error "Download failed"
        return 1
    fi
}

verify_model() {
    print_step "Verifying downloaded model..."

    local MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"

    if [ ! -f "$MODEL_PATH" ]; then
        print_error "Model file not found: $MODEL_PATH"
        return 1
    fi

    # Check file size (should be around 90MB)
    local FILE_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)
    local FILE_SIZE_MB=$((FILE_SIZE / 1024 / 1024))

    if [ $FILE_SIZE_MB -lt 50 ]; then
        print_error "Model file too small (${FILE_SIZE_MB}MB), download may be incomplete"
        return 1
    fi

    print_success "Model file verified: ${FILE_SIZE_MB}MB"

    # Check for tokenizer
    if [ -f "${MODEL_DIR}/tokenizer.json" ]; then
        print_success "Tokenizer file found"
    else
        print_warning "Tokenizer file not found (may not be required)"
    fi

    return 0
}

print_next_steps() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  Setup Complete! Next Steps:                           ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "1. Start the development server:"
    echo -e "   ${GREEN}trunk serve --open${NC}"
    echo ""
    echo "2. Open browser console (F12) and verify:"
    echo -e "   ${GREEN}window.ort !== undefined${NC}  // ONNX Runtime loaded"
    echo -e "   ${GREEN}navigator.gpu !== undefined${NC}  // WebGPU available"
    echo ""
    echo "3. Test model loading in browser console:"
    echo -e "   ${GREEN}// Copy test-helpers.js to console, then run:${NC}"
    echo -e "   ${GREEN}await benchmarkModelLoading()${NC}"
    echo ""
    echo "4. For detailed testing instructions, see:"
    echo -e "   ${GREEN}RUNTIME_TESTING_GUIDE.md${NC}"
    echo ""
}

# Main execution
main() {
    print_header

    # Check if model already exists
    if [ -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
        print_warning "Model already exists at ${MODEL_DIR}/${MODEL_FILE}"
        read -p "Do you want to re-download? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping download"
            verify_model
            print_next_steps
            exit 0
        fi
    fi

    check_dependencies
    create_model_dir

    # Try git lfs first, fallback to direct download
    if [ "$HAS_GIT_LFS" = true ]; then
        if download_model_git_lfs; then
            echo ""
            verify_model && print_success "✨ Model setup complete!" || exit 1
            print_next_steps
            exit 0
        else
            print_warning "Git LFS method failed, trying direct download..."
        fi
    fi

    # Direct download
    if download_model_direct; then
        echo ""
        verify_model && print_success "✨ Model setup complete!" || exit 1
        print_next_steps
        exit 0
    else
        print_error "All download methods failed"
        echo ""
        echo "Manual download instructions:"
        echo "1. Go to: https://huggingface.co/${HUGGINGFACE_REPO}/tree/main"
        echo "2. Download 'onnx/model.onnx' to ${MODEL_DIR}/${MODEL_FILE}"
        echo "3. (Optional) Download 'tokenizer.json' to ${MODEL_DIR}/"
        exit 1
    fi
}

# Run main function
main
