#!/bin/bash
# Run HHmL Docker containers

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}Starting HHmL Docker containers...${NC}"

# Parse arguments
MODE=${1:-production}

case $MODE in
    production|prod)
        echo -e "${YELLOW}Starting production environment...${NC}"
        cd "$DOCKER_DIR"
        docker-compose up -d
        echo -e "\n${GREEN}Production environment started!${NC}"
        echo -e "Monitoring dashboard: http://localhost:8000"
        ;;

    development|dev)
        echo -e "${YELLOW}Starting development environment...${NC}"
        cd "$DOCKER_DIR"
        docker-compose -f docker-compose.dev.yml up -d
        echo -e "\n${GREEN}Development environment started!${NC}"
        echo -e "JupyterLab: http://localhost:8888"
        echo -e "Monitoring: http://localhost:8000"
        echo -e "TensorBoard: http://localhost:6006"
        ;;

    whitepaper)
        echo -e "${YELLOW}Running whitepaper generator...${NC}"
        cd "$DOCKER_DIR"
        docker-compose --profile tools run --rm hhml-whitepaper
        ;;

    stop)
        echo -e "${YELLOW}Stopping all containers...${NC}"
        cd "$DOCKER_DIR"
        docker-compose down
        docker-compose -f docker-compose.dev.yml down
        echo -e "${GREEN}All containers stopped${NC}"
        ;;

    logs)
        cd "$DOCKER_DIR"
        docker-compose logs -f
        ;;

    *)
        echo "Usage: $0 [production|development|whitepaper|stop|logs]"
        echo ""
        echo "Modes:"
        echo "  production   - Start production training + monitoring"
        echo "  development  - Start JupyterLab development environment"
        echo "  whitepaper   - Generate whitepaper from results"
        echo "  stop         - Stop all containers"
        echo "  logs         - View container logs"
        exit 1
        ;;
esac
