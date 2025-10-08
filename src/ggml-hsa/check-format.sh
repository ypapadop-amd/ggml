#!/bin/bash
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Check formatting of Python and C++ files in src/ggml-hsa
# This script can be run locally to verify formatting before pushing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "=== Checking code formatting in src/ggml-hsa ==="
echo ""

# Check if tools are available
MISSING_TOOLS=()

if ! command -v black &> /dev/null; then
    MISSING_TOOLS+=("black")
fi

if ! command -v clang-format &> /dev/null; then
    MISSING_TOOLS+=("clang-format")
fi

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo "❌ Missing required tools: ${MISSING_TOOLS[*]}"
    echo ""
    echo "To install missing tools:"
    echo "  - black: pip install black"
    echo "  - clang-format: Install LLVM 19+ (https://apt.llvm.org/)"
    echo ""
    exit 1
fi

# Check clang-format version
CLANG_FORMAT_VERSION=$(clang-format --version | grep -oP '\d+\.\d+\.\d+' | head -1)
CLANG_FORMAT_MAJOR=$(echo "$CLANG_FORMAT_VERSION" | cut -d. -f1)

if [ "$CLANG_FORMAT_MAJOR" -lt 19 ]; then
    echo "⚠️  Warning: clang-format version $CLANG_FORMAT_VERSION is older than 19.x"
    echo "   Some .clang-format options may not be supported."
    echo "   Install clang-format 19+ for best results."
    echo ""
fi

EXIT_CODE=0

# Check Python files
echo "--- Python Files (black) ---"
PYTHON_FILES=$(find src/ggml-hsa -name "*.py" 2>/dev/null || true)

if [ -n "$PYTHON_FILES" ]; then
    if black --check --diff $PYTHON_FILES; then
        echo "✅ All Python files are properly formatted!"
    else
        echo ""
        echo "❌ Python formatting issues found!"
        echo ""
        echo "To fix, run:"
        echo "  black src/ggml-hsa"
        echo ""
        EXIT_CODE=1
    fi
else
    echo "No Python files found"
fi

echo ""
echo "--- C++ Files (clang-format) ---"

CPP_FILES=$(find src/ggml-hsa -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) ! -path "*/kernels/*" 2>/dev/null || true)

if [ -n "$CPP_FILES" ]; then
    FORMAT_ISSUES=0
    
    for file in $CPP_FILES; do
        # Generate formatted output and compare with original
        FORMATTED=$(clang-format --style=file:src/ggml-hsa/.clang-format "$file" 2>/dev/null || echo "")
        
        if [ -z "$FORMATTED" ]; then
            echo "⚠️  Warning: Could not format $file (clang-format error)"
            continue
        fi
        
        ORIGINAL=$(cat "$file")
        
        if [ "$FORMATTED" != "$ORIGINAL" ]; then
            echo "❌ Formatting issues in $file:"
            echo "$FORMATTED" | diff -u "$file" - || true
            echo ""
            FORMAT_ISSUES=1
        fi
    done
    
    if [ $FORMAT_ISSUES -eq 1 ]; then
        echo ""
        echo "To fix, run:"
        echo "  clang-format -i --style=file:src/ggml-hsa/.clang-format <file>"
        echo ""
        echo "Or to format all C++ files:"
        echo "  find src/ggml-hsa -type f \\( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \\) ! -path '*/kernels/*' -exec clang-format -i --style=file:src/ggml-hsa/.clang-format {} +"
        EXIT_CODE=1
    else
        echo "✅ All C++ files are properly formatted!"
    fi
else
    echo "No C++ files found"
fi

echo ""
echo "=== Format check complete ==="

exit $EXIT_CODE
