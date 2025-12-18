#!/bin/bash
# Build HHmL Docker images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$DOCKER_DIR")"

echo -e "${GREEN}Building HHmL Docker images...${NC}"

# Parse arguments
BUILD_TARGET=${1:-all}

build_image() {
    local target=$1
    local dockerfile=$2
    local tag=$3

    echo -e "\n${YELLOW}Building ${target} image...${NC}"
    docker build \
        -f "${DOCKER_DIR}/${dockerfile}" \
        -t "hhml:${tag}" \
        "${ROOT_DIR}"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] Built hhml:${tag}${NC}"
    else
        echo -e "${RED}[FAIL] Failed to build hhml:${tag}${NC}"
        exit 1
    fi
}

# Build images based on target
case $BUILD_TARGET in
    cpu)
        build_image "CPU" "Dockerfile.cpu" "cpu-latest"
        ;;
    cuda|gpu)
        build_image "CUDA" "Dockerfile.cuda" "cuda-latest"
        ;;
    dev)
        build_image "Development" "Dockerfile.dev" "dev-latest"
        ;;
    all)
        build_image "CPU" "Dockerfile.cpu" "cpu-latest"
        build_image "CUDA" "Dockerfile.cuda" "cuda-latest"
        build_image "Development" "Dockerfile.dev" "dev-latest"
        ;;
    *)
        echo -e "${RED}Unknown build target: ${BUILD_TARGET}${NC}"
        echo "Usage: $0 [cpu|cuda|dev|all]"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Build complete!${NC}"
echo -e "Available images:"
docker images | grep hhml
