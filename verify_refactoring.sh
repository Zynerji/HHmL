#!/bin/bash
# HHmL Production Refactoring Verification Script

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=================================================="
echo "HHmL Production Refactoring Verification"
echo "=================================================="
echo ""

# Check directory structure
echo -e "${YELLOW}Checking directory structure...${NC}"
required_dirs=(
    "src/hhml"
    "src/hhml/core"
    "src/hhml/ml"
    "src/hhml/analysis"
    "src/hhml/monitoring"
    "tests"
    "examples"
    "docker"
    "docs"
    "tools"
    "configs"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "  ${GREEN}[OK]${NC} $dir"
    else
        echo -e "  ${RED}[MISSING]${NC} $dir"
    fi
done
echo ""

# Check production files
echo -e "${YELLOW}Checking production files...${NC}"
required_files=(
    "README.md"
    "LICENSE"
    "CONTRIBUTING.md"
    "CHANGELOG.md"
    "MIGRATION_GUIDE.md"
    "pyproject.toml"
    "setup.py"
    ".gitignore"
    ".dockerignore"
    ".editorconfig"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}[OK]${NC} $file"
    else
        echo -e "  ${RED}[MISSING]${NC} $file"
    fi
done
echo ""

# Check Docker files
echo -e "${YELLOW}Checking Docker infrastructure...${NC}"
docker_files=(
    "docker/Dockerfile.cpu"
    "docker/Dockerfile.cuda"
    "docker/Dockerfile.dev"
    "docker/docker-compose.yml"
    "docker/docker-compose.dev.yml"
    "docker/scripts/build.sh"
    "docker/scripts/run.sh"
)

for file in "${docker_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}[OK]${NC} $file"
    else
        echo -e "  ${RED}[MISSING]${NC} $file"
    fi
done
echo ""

# Check package structure
echo -e "${YELLOW}Checking package structure...${NC}"
package_dirs=(
    "src/hhml/__init__.py"
    "src/hhml/core/__init__.py"
    "src/hhml/core/mobius/__init__.py"
    "src/hhml/core/resonance/__init__.py"
    "src/hhml/core/gft/__init__.py"
    "src/hhml/core/tensor_networks/__init__.py"
    "src/hhml/ml/__init__.py"
    "src/hhml/ml/rl/__init__.py"
    "src/hhml/ml/training/__init__.py"
    "src/hhml/analysis/__init__.py"
    "src/hhml/analysis/dark_matter/__init__.py"
    "src/hhml/monitoring/__init__.py"
)

for file in "${package_dirs[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}[OK]${NC} $file"
    else
        echo -e "  ${RED}[MISSING]${NC} $file"
    fi
done
echo ""

# Check README content
echo -e "${YELLOW}Checking README.md content...${NC}"
if grep -q "@Conceptual1" README.md; then
    echo -e "  ${GREEN}[OK]${NC} @Conceptual1 contact present"
else
    echo -e "  ${RED}[MISSING]${NC} @Conceptual1 contact"
fi

if grep -q "Docker" README.md; then
    echo -e "  ${GREEN}[OK]${NC} Docker documentation present"
else
    echo -e "  ${RED}[MISSING]${NC} Docker documentation"
fi

if grep -q "LaTeX" README.md || grep -q "Mathematical Framework" README.md; then
    echo -e "  ${GREEN}[OK]${NC} Mathematical framework present"
else
    echo -e "  ${RED}[MISSING]${NC} Mathematical framework"
fi
echo ""

# Check pyproject.toml
echo -e "${YELLOW}Checking pyproject.toml configuration...${NC}"
if grep -q "name = \"hhml\"" pyproject.toml; then
    echo -e "  ${GREEN}[OK]${NC} Package name configured"
else
    echo -e "  ${RED}[MISSING]${NC} Package name"
fi

if grep -q "version = \"0.1.0\"" pyproject.toml; then
    echo -e "  ${GREEN}[OK]${NC} Version configured"
else
    echo -e "  ${RED}[MISSING]${NC} Version"
fi

if grep -q "@Conceptual1" pyproject.toml || grep -q "Conceptual1" pyproject.toml; then
    echo -e "  ${GREEN}[OK]${NC} Contact information present"
else
    echo -e "  ${YELLOW}[WARNING]${NC} Contact information may be missing"
fi
echo ""

# Test Python import
echo -e "${YELLOW}Testing Python package installation...${NC}"
if python3 -c "import sys; sys.path.insert(0, 'src'); import hhml" 2>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} Package imports successfully"
else
    echo -e "  ${RED}[FAIL]${NC} Package import failed"
    echo -e "  ${YELLOW}Run: pip install -e .${NC}"
fi
echo ""

# Summary
echo "=================================================="
echo -e "${GREEN}Verification Complete!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Review REFACTORING_SUMMARY.md"
echo "  2. Read MIGRATION_GUIDE.md"
echo "  3. Install package: pip install -e ."
echo "  4. Test Docker: cd docker && ./scripts/build.sh cpu"
echo "  5. Run example: python examples/training/train_mobius_basic.py"
echo ""
echo "Questions? Contact @Conceptual1 or open a GitHub issue."
echo ""
