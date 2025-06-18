#!/bin/bash

# Insurance AI Agent System - Local Setup Helper
# Ensures Docker environment is ready for local development

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

log() {
  echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

info() {
  echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

error() {
  echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
  exit 1
}

check_command() {
  if ! command -v "$1" &> /dev/null; then
    error "$1 is not installed. Please install it before continuing."
  fi
}

check_command docker
check_command docker-compose

cd "$PROJECT_DIR"

if [[ ! -f .env ]]; then
  info "Creating .env from .env.example"
  cp .env.example .env
fi

log "Pulling Docker images..."
docker-compose pull

log "Starting Docker services..."
docker-compose up -d

log "Setup complete. Access the application at http://localhost:80"
