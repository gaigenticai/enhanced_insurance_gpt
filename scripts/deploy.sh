#!/bin/bash

# Insurance AI Agent System - Deployment Script
# Production-ready deployment automation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"
BACKUP_DIR="$PROJECT_DIR/backups"
LOG_FILE="$PROJECT_DIR/deployment.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if .env file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        warn ".env file not found. Creating from template..."
        cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
        warn "Please edit .env file with your configuration before running deployment."
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then
        warn "Less than 10GB disk space available. Deployment may fail."
    fi
    
    # Check available memory (minimum 4GB)
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [[ $available_memory -lt 4096 ]]; then
        warn "Less than 4GB memory available. Performance may be affected."
    fi
    
    log "Prerequisites check completed."
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/uploads"
    mkdir -p "$PROJECT_DIR/docker/grafana/provisioning/dashboards"
    mkdir -p "$PROJECT_DIR/docker/grafana/provisioning/datasources"
    mkdir -p "$PROJECT_DIR/docker/logstash/pipeline"
    mkdir -p "$PROJECT_DIR/docker/logstash/config"
    
    log "Directories created successfully."
}

# Generate secrets if not present
generate_secrets() {
    log "Checking and generating secrets..."
    
    # Source environment file
    source "$ENV_FILE"
    
    # Generate JWT secret if not set
    if [[ -z "${JWT_SECRET:-}" ]] || [[ "$JWT_SECRET" == "your-super-secret-jwt-key-change-in-production-min-32-chars" ]]; then
        JWT_SECRET=$(openssl rand -base64 32)
        sed -i "s/JWT_SECRET=.*/JWT_SECRET=$JWT_SECRET/" "$ENV_FILE"
        log "Generated new JWT secret."
    fi
    
    # Generate encryption key if not set
    if [[ -z "${ENCRYPTION_KEY:-}" ]] || [[ "$ENCRYPTION_KEY" == "your-encryption-key-exactly-32-chars" ]]; then
        ENCRYPTION_KEY=$(openssl rand -base64 32 | head -c 32)
        sed -i "s/ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$ENCRYPTION_KEY/" "$ENV_FILE"
        log "Generated new encryption key."
    fi
    
    # Generate password salt if not set
    if [[ -z "${PASSWORD_SALT:-}" ]] || [[ "$PASSWORD_SALT" == "your-password-salt-for-hashing" ]]; then
        PASSWORD_SALT=$(openssl rand -base64 16)
        sed -i "s/PASSWORD_SALT=.*/PASSWORD_SALT=$PASSWORD_SALT/" "$ENV_FILE"
        log "Generated new password salt."
    fi
    
    log "Secrets check completed."
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build backend image
    info "Building backend image..."
    docker build -f Dockerfile.backend -t insurance-ai-backend:latest .
    
    # Build frontend image
    info "Building frontend image..."
    docker build -f Dockerfile.frontend -t insurance-ai-frontend:latest .
    
    log "Docker images built successfully."
}

# Initialize database
init_database() {
    log "Initializing database..."
    
    cd "$PROJECT_DIR"
    
    # Start only PostgreSQL first
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    info "Waiting for PostgreSQL to be ready..."
    timeout=60
    while ! docker-compose exec -T postgres pg_isready -U postgres; do
        sleep 2
        timeout=$((timeout - 2))
        if [[ $timeout -le 0 ]]; then
            error "PostgreSQL failed to start within 60 seconds."
        fi
    done
    
    # Run database migrations
    info "Running database migrations..."
    docker-compose exec -T postgres psql -U postgres -d insurance_ai -f /docker-entrypoint-initdb.d/01-schema.sql
    
    log "Database initialized successfully."
}

# Start all services
start_services() {
    log "Starting all services..."
    
    cd "$PROJECT_DIR"
    
    # Start all services
    docker-compose up -d
    
    # Wait for services to be healthy
    info "Waiting for services to be healthy..."
    
    services=("postgres" "redis" "backend" "frontend")
    for service in "${services[@]}"; do
        info "Checking health of $service..."
        timeout=120
        while ! docker-compose ps "$service" | grep -q "healthy"; do
            sleep 5
            timeout=$((timeout - 5))
            if [[ $timeout -le 0 ]]; then
                warn "$service health check timeout. Continuing anyway..."
                break
            fi
        done
    done
    
    log "All services started successfully."
}

# Run health checks
health_check() {
    log "Running health checks..."
    
    cd "$PROJECT_DIR"
    
    # Check backend health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log "Backend health check: PASSED"
    else
        warn "Backend health check: FAILED"
    fi
    
    # Check frontend health
    if curl -f http://localhost:80/health &> /dev/null; then
        log "Frontend health check: PASSED"
    else
        warn "Frontend health check: FAILED"
    fi
    
    # Check database connection
    if docker-compose exec -T postgres pg_isready -U postgres &> /dev/null; then
        log "Database health check: PASSED"
    else
        warn "Database health check: FAILED"
    fi
    
    # Check Redis connection
    if docker-compose exec -T redis redis-cli ping &> /dev/null; then
        log "Redis health check: PASSED"
    else
        warn "Redis health check: FAILED"
    fi
    
    log "Health checks completed."
}

# Display deployment information
show_deployment_info() {
    log "Deployment completed successfully!"
    echo ""
    echo -e "${GREEN}=== Insurance AI Agent System ===${NC}"
    echo -e "${BLUE}Frontend URL:${NC} http://localhost"
    echo -e "${BLUE}Backend API:${NC} http://localhost:8000"
    echo -e "${BLUE}API Documentation:${NC} http://localhost:8000/docs"
    echo -e "${BLUE}Grafana Dashboard:${NC} http://localhost:3000"
    echo -e "${BLUE}Prometheus Metrics:${NC} http://localhost:9090"
    echo -e "${BLUE}Kibana Logs:${NC} http://localhost:5601"
    echo -e "${BLUE}MinIO Console:${NC} http://localhost:9001"
    echo ""
    echo -e "${YELLOW}Default Credentials:${NC}"
    echo -e "${BLUE}Application:${NC} admin@zurich.com / password123"
    echo -e "${BLUE}Grafana:${NC} admin / admin"
    echo -e "${BLUE}MinIO:${NC} minioadmin / minioadmin"
    echo ""
    echo -e "${YELLOW}Important:${NC}"
    echo "- Change default passwords in production"
    echo "- Configure SSL certificates for HTTPS"
    echo "- Set up proper backup procedures"
    echo "- Review security settings in .env file"
    echo ""
    echo -e "${GREEN}Logs location:${NC} $LOG_FILE"
    echo -e "${GREEN}Configuration:${NC} $ENV_FILE"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    # Add cleanup tasks here if needed
}

# Main deployment function
deploy() {
    log "Starting Insurance AI Agent System deployment..."
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    check_prerequisites
    create_directories
    generate_secrets
    build_images
    init_database
    start_services
    health_check
    show_deployment_info
    
    log "Deployment process completed!"
}

# Command line options
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "start")
        log "Starting services..."
        cd "$PROJECT_DIR"
        docker-compose up -d
        health_check
        ;;
    "stop")
        log "Stopping services..."
        cd "$PROJECT_DIR"
        docker-compose down
        ;;
    "restart")
        log "Restarting services..."
        cd "$PROJECT_DIR"
        docker-compose restart
        health_check
        ;;
    "logs")
        cd "$PROJECT_DIR"
        docker-compose logs -f "${2:-}"
        ;;
    "status")
        cd "$PROJECT_DIR"
        docker-compose ps
        ;;
    "backup")
        log "Creating backup..."
        cd "$PROJECT_DIR"
        ./scripts/backup.sh
        ;;
    "update")
        log "Updating system..."
        cd "$PROJECT_DIR"
        git pull
        docker-compose build
        docker-compose up -d
        health_check
        ;;
    "clean")
        log "Cleaning up Docker resources..."
        cd "$PROJECT_DIR"
        docker-compose down -v
        docker system prune -f
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy    - Full deployment (default)"
        echo "  start     - Start services"
        echo "  stop      - Stop services"
        echo "  restart   - Restart services"
        echo "  logs      - Show logs (optionally for specific service)"
        echo "  status    - Show service status"
        echo "  backup    - Create backup"
        echo "  update    - Update and restart system"
        echo "  clean     - Clean up Docker resources"
        echo "  help      - Show this help"
        ;;
    *)
        error "Unknown command: $1. Use 'help' for available commands."
        ;;
esac

