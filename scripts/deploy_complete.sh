#!/bin/bash

# Insurance AI Agent System - Complete System Integration and Deployment
# Final integration script for production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOYMENT_MODE=${1:-"development"}
SKIP_TESTS=${2:-false}

# Utility functions
log_header() {
  echo -e "\n${PURPLE}================================${NC}"
  echo -e "${PURPLE}$1${NC}"
  echo -e "${PURPLE}================================${NC}\n"
}

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
  echo -e "${CYAN}[STEP]${NC} $1"
}

# System requirements check
check_system_requirements() {
  log_header "CHECKING SYSTEM REQUIREMENTS"
  
  log_step "Checking operating system..."
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    log_success "Linux operating system detected"
  else
    log_warning "Non-Linux OS detected. Some features may not work correctly."
  fi
  
  log_step "Checking Docker installation..."
  if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    log_success "Docker $DOCKER_VERSION is installed"
  else
    log_error "Docker is not installed. Please install Docker 20.10+ before proceeding."
    exit 1
  fi
  
  log_step "Checking Docker Compose installation..."
  if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
    log_success "Docker Compose $COMPOSE_VERSION is installed"
  else
    log_error "Docker Compose is not installed. Please install Docker Compose 2.0+ before proceeding."
    exit 1
  fi
  
  log_step "Checking Node.js installation..."
  if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    log_success "Node.js $NODE_VERSION is installed"
  else
    log_error "Node.js is not installed. Please install Node.js 18+ before proceeding."
    exit 1
  fi
  
  log_step "Checking Python installation..."
  if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    log_success "$PYTHON_VERSION is installed"
  else
    log_error "Python 3 is not installed. Please install Python 3.11+ before proceeding."
    exit 1
  fi
  
  log_step "Checking available disk space..."
  AVAILABLE_SPACE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
  if (( $(echo "$AVAILABLE_SPACE > 10" | bc -l) )); then
    log_success "Sufficient disk space available (${AVAILABLE_SPACE}GB)"
  else
    log_warning "Low disk space detected (${AVAILABLE_SPACE}GB). Recommend at least 10GB free space."
  fi
  
  log_step "Checking available memory..."
  AVAILABLE_MEMORY=$(free -g | awk 'NR==2{printf "%.1f", $7}')
  if (( $(echo "$AVAILABLE_MEMORY > 2" | bc -l) )); then
    log_success "Sufficient memory available (${AVAILABLE_MEMORY}GB)"
  else
    log_warning "Low memory detected (${AVAILABLE_MEMORY}GB). Recommend at least 4GB RAM."
  fi
}

# Environment setup
setup_environment() {
  log_header "SETTING UP ENVIRONMENT"
  
  cd "$PROJECT_ROOT"
  
  log_step "Creating environment configuration..."
  if [ ! -f ".env" ]; then
    if [ "$DEPLOYMENT_MODE" == "production" ]; then
      if [ -f ".env.production" ]; then
        cp .env.production .env
        log_success "Production environment configuration copied"
      else
        cp .env.example .env
        log_warning "Production config not found. Using example config. Please update .env file."
      fi
    else
      cp .env.example .env
      log_success "Development environment configuration created"
    fi
  else
    log_info "Environment configuration already exists"
  fi
  
  log_step "Creating required directories..."
  mkdir -p logs
  mkdir -p uploads
  mkdir -p backups
  mkdir -p monitoring/data
  mkdir -p ssl
  log_success "Required directories created"
  
  log_step "Setting up file permissions..."
  chmod +x scripts/*.sh
  chmod 755 uploads
  chmod 755 logs
  chmod 755 backups
  log_success "File permissions configured"
}

# Database initialization
initialize_database() {
  log_header "INITIALIZING DATABASE"
  
  log_step "Starting PostgreSQL container..."
  docker-compose up -d postgres
  
  log_step "Waiting for PostgreSQL to be ready..."
  for i in {1..30}; do
    if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
      log_success "PostgreSQL is ready"
      break
    fi
    if [ $i -eq 30 ]; then
      log_error "PostgreSQL failed to start within timeout"
      exit 1
    fi
    sleep 2
  done
  
  log_step "Running database migrations..."
  docker-compose exec -T postgres psql -U postgres -c "CREATE DATABASE IF NOT EXISTS insurance_ai_system;"
  
  # Wait for backend to be built and run migrations
  log_step "Building backend container..."
  docker-compose build backend
  
  log_step "Running database schema creation..."
  docker-compose run --rm backend python database/migrate.py
  log_success "Database initialized successfully"
}

# Build and test services
build_and_test_services() {
  log_header "BUILDING AND TESTING SERVICES"
  
  log_step "Building all Docker images..."
  docker-compose build --parallel
  log_success "All Docker images built successfully"
  
  if [ "$SKIP_TESTS" != "true" ]; then
    log_step "Running test suite..."
    
    # Install test dependencies
    log_info "Installing test dependencies..."
    cd "$PROJECT_ROOT/tests"
    npm install
    cd "$PROJECT_ROOT"
    
    # Run backend tests
    log_info "Running backend tests..."
    docker-compose run --rm backend python -m pytest tests/test_backend.py -v --tb=short
    
    # Run frontend tests
    log_info "Running frontend tests..."
    cd "$PROJECT_ROOT/tests"
    npm test -- --watchAll=false
    cd "$PROJECT_ROOT"
    
    log_success "All tests passed successfully"
  else
    log_warning "Skipping tests as requested"
  fi
}

# Start all services
start_services() {
  log_header "STARTING ALL SERVICES"
  
  log_step "Starting infrastructure services..."
  docker-compose up -d postgres redis
  
  log_step "Waiting for infrastructure services..."
  sleep 10
  
  log_step "Starting backend services..."
  docker-compose up -d backend
  
  log_step "Waiting for backend services..."
  for i in {1..60}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
      log_success "Backend services are ready"
      break
    fi
    if [ $i -eq 60 ]; then
      log_error "Backend services failed to start within timeout"
      docker-compose logs backend
      exit 1
    fi
    sleep 2
  done
  
  log_step "Starting frontend services..."
  docker-compose up -d frontend
  
  log_step "Waiting for frontend services..."
  for i in {1..60}; do
    if curl -f http://localhost:80 > /dev/null 2>&1; then
      log_success "Frontend services are ready"
      break
    fi
    if [ $i -eq 60 ]; then
      log_error "Frontend services failed to start within timeout"
      docker-compose logs frontend
      exit 1
    fi
    sleep 2
  done
  
  if [ "$DEPLOYMENT_MODE" == "production" ]; then
    log_step "Starting monitoring services..."
    docker-compose up -d prometheus grafana
    log_success "Monitoring services started"
  fi
  
  log_success "All services started successfully"
}

# Health checks
perform_health_checks() {
  log_header "PERFORMING HEALTH CHECKS"
  
  log_step "Checking backend health..."
  BACKEND_HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "error")
  if [ "$BACKEND_HEALTH" == "healthy" ]; then
    log_success "Backend health check passed"
  else
    log_error "Backend health check failed"
    exit 1
  fi
  
  log_step "Checking frontend accessibility..."
  if curl -f http://localhost:80 > /dev/null 2>&1; then
    log_success "Frontend accessibility check passed"
  else
    log_error "Frontend accessibility check failed"
    exit 1
  fi
  
  log_step "Checking database connectivity..."
  DB_STATUS=$(docker-compose exec -T postgres pg_isready -U postgres 2>/dev/null && echo "ready" || echo "error")
  if [ "$DB_STATUS" == "ready" ]; then
    log_success "Database connectivity check passed"
  else
    log_error "Database connectivity check failed"
    exit 1
  fi
  
  log_step "Checking Redis connectivity..."
  REDIS_STATUS=$(docker-compose exec -T redis redis-cli ping 2>/dev/null || echo "error")
  if [ "$REDIS_STATUS" == "PONG" ]; then
    log_success "Redis connectivity check passed"
  else
    log_error "Redis connectivity check failed"
    exit 1
  fi
  
  log_step "Checking API endpoints..."
  API_ENDPOINTS=(
    "/api/v1/health"
    "/api/v1/auth/status"
    "/api/v1/policies"
    "/api/v1/claims"
    "/api/v1/agents/status"
  )
  
  for endpoint in "${API_ENDPOINTS[@]}"; do
    if curl -f "http://localhost:8000$endpoint" > /dev/null 2>&1; then
      log_info "✓ $endpoint"
    else
      log_warning "✗ $endpoint (may require authentication)"
    fi
  done
  
  log_success "Health checks completed"
}

# Create sample data
create_sample_data() {
  log_header "CREATING SAMPLE DATA"
  
  log_step "Creating sample users..."
  
  # Create admin user
  ADMIN_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/auth/register \
    -H "Content-Type: application/json" \
    -d '{
      "email": "admin@zurich.com",
      "password": "admin123",
      "first_name": "System",
      "last_name": "Administrator",
      "role": "admin",
      "department": "management"
    }' || echo "error")
  
  if [[ "$ADMIN_RESPONSE" == *"email"* ]]; then
    log_success "Admin user created successfully"
  else
    log_info "Admin user may already exist"
  fi
  
  # Create sample agent user
  AGENT_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/auth/register \
    -H "Content-Type: application/json" \
    -d '{
      "email": "agent@zurich.com",
      "password": "agent123",
      "first_name": "John",
      "last_name": "Agent",
      "role": "agent",
      "department": "underwriting"
    }' || echo "error")
  
  if [[ "$AGENT_RESPONSE" == *"email"* ]]; then
    log_success "Sample agent user created successfully"
  else
    log_info "Sample agent user may already exist"
  fi
  
  log_step "Sample data creation completed"
}

# Generate deployment report
generate_deployment_report() {
  log_header "GENERATING DEPLOYMENT REPORT"
  
  REPORT_FILE="$PROJECT_ROOT/deployment_report_$(date +%Y%m%d_%H%M%S).txt"
  
  cat > "$REPORT_FILE" << EOF
Insurance AI Agent System - Deployment Report
Generated: $(date)
Deployment Mode: $DEPLOYMENT_MODE

SYSTEM INFORMATION
==================
Operating System: $(uname -a)
Docker Version: $(docker --version)
Docker Compose Version: $(docker-compose --version)
Node.js Version: $(node --version)
Python Version: $(python3 --version)

DEPLOYMENT CONFIGURATION
========================
Project Root: $PROJECT_ROOT
Environment File: $([ -f "$PROJECT_ROOT/.env" ] && echo "Present" || echo "Missing")
SSL Certificates: $([ -d "$PROJECT_ROOT/ssl" ] && echo "Directory exists" || echo "Not configured")

RUNNING SERVICES
================
$(docker-compose ps)

SERVICE HEALTH STATUS
====================
Backend Health: $(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "Unable to check")
Frontend Status: $(curl -s -o /dev/null -w "%{http_code}" http://localhost:80 2>/dev/null || echo "Unable to check")
Database Status: $(docker-compose exec -T postgres pg_isready -U postgres 2>/dev/null || echo "Unable to check")
Redis Status: $(docker-compose exec -T redis redis-cli ping 2>/dev/null || echo "Unable to check")

RESOURCE USAGE
==============
Disk Usage: $(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')
Memory Usage: $(free -h | awk 'NR==2{printf "%.1f/%.1fGB (%.1f%% used)", $3/1024, $2/1024, $3*100/$2}')
CPU Load: $(uptime | awk -F'load average:' '{print $2}')

DOCKER IMAGES
=============
$(docker images | grep insurance-ai)

NETWORK CONFIGURATION
====================
$(docker network ls | grep insurance)

ACCESS INFORMATION
==================
Frontend URL: http://localhost:80
Backend API: http://localhost:8000
API Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/health

DEFAULT CREDENTIALS
==================
Admin User: admin@zurich.com / admin123
Agent User: agent@zurich.com / agent123

IMPORTANT NOTES
===============
- Change default passwords before production use
- Configure SSL certificates for production deployment
- Set up proper backup procedures
- Configure monitoring and alerting
- Review security settings in .env file

NEXT STEPS
==========
1. Access the application at http://localhost:80
2. Login with the provided credentials
3. Configure additional users and permissions
4. Upload sample documents for testing
5. Review monitoring dashboards (if enabled)
6. Set up automated backups
7. Configure SSL/TLS for production

For support and documentation, see: $PROJECT_ROOT/docs/README.md
EOF

  log_success "Deployment report generated: $REPORT_FILE"
}

# Display final status
display_final_status() {
  log_header "DEPLOYMENT COMPLETE"
  
  echo -e "${GREEN}"
  cat << "EOF"
  ___                                        _    ___ 
 |_ _|_ __  ___ _   _ _ __ __ _ _ __   ___ ___| |  |_ _|
  | || '_ \/ __| | | | '__/ _` | '_ \ / __/ _ \ |   | | 
  | || | | \__ \ |_| | | | (_| | | | | (_|  __/ |   | | 
 |___|_| |_|___/\__,_|_|  \__,_|_| |_|\___\___|_|  |___|
                                                       
     _    ___      _                    _   
    / \  |_ _|    / \   __ _  ___ _ __ | |_ 
   / _ \  | |    / _ \ / _` |/ _ \ '_ \| __|
  / ___ \ | |   / ___ \ (_| |  __/ | | | |_ 
 /_/   \_\___|_/_/   \_\__, |\___|_| |_|\__|
                       |___/                
  ____            _                 
 / ___| _   _ ___| |_ ___ _ __ ___   
 \___ \| | | / __| __/ _ \ '_ ` _ \  
  ___) | |_| \__ \ ||  __/ | | | | | 
 |____/ \__, |___/\__\___|_| |_| |_| 
        |___/                       
EOF
  echo -e "${NC}"
  
  echo -e "${CYAN}🎉 Insurance AI Agent System has been successfully deployed!${NC}\n"
  
  echo -e "${BLUE}📋 Quick Access Information:${NC}"
  echo -e "   🌐 Frontend Application: ${GREEN}http://localhost:80${NC}"
  echo -e "   🔧 Backend API: ${GREEN}http://localhost:8000${NC}"
  echo -e "   📚 API Documentation: ${GREEN}http://localhost:8000/docs${NC}"
  echo -e "   ❤️  Health Check: ${GREEN}http://localhost:8000/health${NC}"
  
  echo -e "\n${BLUE}👤 Default Login Credentials:${NC}"
  echo -e "   👨‍💼 Admin: ${YELLOW}admin@zurich.com${NC} / ${YELLOW}admin123${NC}"
  echo -e "   👨‍💻 Agent: ${YELLOW}agent@zurich.com${NC} / ${YELLOW}agent123${NC}"
  
  echo -e "\n${BLUE}🛠️  Management Commands:${NC}"
  echo -e "   📊 View logs: ${CYAN}docker-compose logs -f${NC}"
  echo -e "   🔄 Restart services: ${CYAN}docker-compose restart${NC}"
  echo -e "   🛑 Stop services: ${CYAN}docker-compose down${NC}"
  echo -e "   🧪 Run tests: ${CYAN}./scripts/run_tests.sh${NC}"
  echo -e "   💾 Create backup: ${CYAN}./scripts/backup.sh${NC}"
  
  echo -e "\n${BLUE}📖 Documentation:${NC}"
  echo -e "   📋 Complete Guide: ${CYAN}$PROJECT_ROOT/docs/README.md${NC}"
  echo -e "   📊 Deployment Report: ${CYAN}$(ls -t deployment_report_*.txt | head -1)${NC}"
  
  echo -e "\n${YELLOW}⚠️  Important Security Notes:${NC}"
  echo -e "   🔐 Change default passwords before production use"
  echo -e "   🛡️  Configure SSL certificates for production"
  echo -e "   🔒 Review and update security settings in .env file"
  echo -e "   📝 Set up proper backup and monitoring procedures"
  
  echo -e "\n${GREEN}✅ System is ready for use!${NC}"
  echo -e "${BLUE}For support, refer to the troubleshooting section in the documentation.${NC}\n"
}

# Main execution function
main() {
  log_header "INSURANCE AI AGENT SYSTEM - COMPLETE DEPLOYMENT"
  
  echo -e "${BLUE}Deployment Mode: ${YELLOW}$DEPLOYMENT_MODE${NC}"
  echo -e "${BLUE}Skip Tests: ${YELLOW}$SKIP_TESTS${NC}"
  echo -e "${BLUE}Project Root: ${YELLOW}$PROJECT_ROOT${NC}\n"
  
  # Execution steps
  check_system_requirements
  setup_environment
  initialize_database
  build_and_test_services
  start_services
  perform_health_checks
  create_sample_data
  generate_deployment_report
  display_final_status
  
  log_success "Deployment completed successfully!"
}

# Handle script arguments
case "$1" in
  "production")
    DEPLOYMENT_MODE="production"
    ;;
  "development"|"dev"|"")
    DEPLOYMENT_MODE="development"
    ;;
  "help"|"--help"|"-h")
    echo "Usage: $0 [deployment_mode] [skip_tests]"
    echo ""
    echo "Arguments:"
    echo "  deployment_mode    'production' or 'development' (default: development)"
    echo "  skip_tests         'true' to skip tests (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Development deployment with tests"
    echo "  $0 development              # Development deployment with tests"
    echo "  $0 production               # Production deployment with tests"
    echo "  $0 production true          # Production deployment without tests"
    echo ""
    exit 0
    ;;
  *)
    log_error "Invalid deployment mode: $1"
    echo "Use '$0 help' for usage information"
    exit 1
    ;;
esac

# Trap to handle interruption
trap 'echo -e "\n${RED}Deployment interrupted by user${NC}"; exit 1' INT

# Run main function
main "$@"

