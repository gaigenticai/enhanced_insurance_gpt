#!/bin/bash

# Insurance AI Agent System - Test Runner
# Comprehensive testing script for all components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
TESTS_DIR="$PROJECT_ROOT/tests"

# Default values
RUN_BACKEND_TESTS=true
RUN_FRONTEND_TESTS=true
RUN_E2E_TESTS=true
RUN_SECURITY_TESTS=true
RUN_PERFORMANCE_TESTS=true
GENERATE_COVERAGE=true
PARALLEL_EXECUTION=true
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --backend-only)
      RUN_FRONTEND_TESTS=false
      RUN_E2E_TESTS=false
      shift
      ;;
    --frontend-only)
      RUN_BACKEND_TESTS=false
      RUN_E2E_TESTS=false
      shift
      ;;
    --e2e-only)
      RUN_BACKEND_TESTS=false
      RUN_FRONTEND_TESTS=false
      shift
      ;;
    --no-coverage)
      GENERATE_COVERAGE=false
      shift
      ;;
    --sequential)
      PARALLEL_EXECUTION=false
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --backend-only     Run only backend tests"
      echo "  --frontend-only    Run only frontend tests"
      echo "  --e2e-only         Run only end-to-end tests"
      echo "  --no-coverage      Skip coverage generation"
      echo "  --sequential       Run tests sequentially instead of parallel"
      echo "  --verbose          Enable verbose output"
      echo "  --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Utility functions
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

check_dependencies() {
  log_info "Checking dependencies..."
  
  # Check Python and pip
  if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed"
    exit 1
  fi
  
  if ! command -v pip3 &> /dev/null; then
    log_error "pip3 is not installed"
    exit 1
  fi
  
  # Check Node.js and npm
  if ! command -v node &> /dev/null; then
    log_error "Node.js is not installed"
    exit 1
  fi
  
  if ! command -v npm &> /dev/null; then
    log_error "npm is not installed"
    exit 1
  fi
  
  # Check Docker (for integration tests)
  if ! command -v docker &> /dev/null; then
    log_warning "Docker is not installed - some integration tests may fail"
  fi
  
  log_success "Dependencies check completed"
}

setup_test_environment() {
  log_info "Setting up test environment..."
  
  # Create test results directory
  mkdir -p "$PROJECT_ROOT/test-results"
  mkdir -p "$PROJECT_ROOT/coverage"
  
  # Set environment variables for testing
  export NODE_ENV=test
  export ENVIRONMENT=testing
  export DATABASE_URL="sqlite:///test.db"
  export REDIS_URL="redis://localhost:6379/15"
  
  log_success "Test environment setup completed"
}

install_backend_dependencies() {
  log_info "Installing backend dependencies..."
  
  cd "$BACKEND_DIR"
  
  # Install Python dependencies
  if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
  fi
  
  # Install test dependencies
  pip3 install pytest pytest-asyncio pytest-cov httpx fakeredis
  
  log_success "Backend dependencies installed"
}

install_frontend_dependencies() {
  log_info "Installing frontend dependencies..."
  
  cd "$FRONTEND_DIR"
  
  # Install npm dependencies
  npm install
  
  # Install test dependencies
  cd "$TESTS_DIR"
  npm install
  
  log_success "Frontend dependencies installed"
}

run_backend_tests() {
  if [ "$RUN_BACKEND_TESTS" = false ]; then
    return 0
  fi
  
  log_info "Running backend tests..."
  
  cd "$PROJECT_ROOT"
  
  # Set test environment variables
  export PYTHONPATH="$BACKEND_DIR:$PYTHONPATH"
  
  # Run pytest with coverage
  if [ "$GENERATE_COVERAGE" = true ]; then
    if [ "$VERBOSE" = true ]; then
      python3 -m pytest tests/test_backend.py -v --cov=backend --cov-report=html:coverage/backend --cov-report=xml:coverage/backend.xml --cov-report=term-missing
    else
      python3 -m pytest tests/test_backend.py --cov=backend --cov-report=html:coverage/backend --cov-report=xml:coverage/backend.xml --cov-report=term-missing
    fi
  else
    if [ "$VERBOSE" = true ]; then
      python3 -m pytest tests/test_backend.py -v
    else
      python3 -m pytest tests/test_backend.py
    fi
  fi
  
  if [ $? -eq 0 ]; then
    log_success "Backend tests passed"
  else
    log_error "Backend tests failed"
    return 1
  fi
}

run_frontend_tests() {
  if [ "$RUN_FRONTEND_TESTS" = false ]; then
    return 0
  fi
  
  log_info "Running frontend tests..."
  
  cd "$TESTS_DIR"
  
  # Run Jest tests
  if [ "$GENERATE_COVERAGE" = true ]; then
    if [ "$VERBOSE" = true ]; then
      npm run test:ci -- --verbose
    else
      npm run test:ci
    fi
  else
    if [ "$VERBOSE" = true ]; then
      npm test -- --verbose --watchAll=false
    else
      npm test -- --watchAll=false
    fi
  fi
  
  if [ $? -eq 0 ]; then
    log_success "Frontend tests passed"
  else
    log_error "Frontend tests failed"
    return 1
  fi
}

start_services() {
  log_info "Starting services for E2E tests..."
  
  # Start backend service
  cd "$BACKEND_DIR"
  python3 -m uvicorn api_gateway:app --host 0.0.0.0 --port 8000 &
  BACKEND_PID=$!
  
  # Start frontend service
  cd "$FRONTEND_DIR"
  npm run dev -- --host &
  FRONTEND_PID=$!
  
  # Wait for services to start
  log_info "Waiting for services to start..."
  sleep 10
  
  # Check if services are running
  if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    log_error "Backend service failed to start"
    cleanup_services
    return 1
  fi
  
  if ! curl -f http://localhost:5173 > /dev/null 2>&1; then
    log_error "Frontend service failed to start"
    cleanup_services
    return 1
  fi
  
  log_success "Services started successfully"
}

cleanup_services() {
  log_info "Stopping services..."
  
  if [ ! -z "$BACKEND_PID" ]; then
    kill $BACKEND_PID 2>/dev/null || true
  fi
  
  if [ ! -z "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID 2>/dev/null || true
  fi
  
  # Kill any remaining processes
  pkill -f "uvicorn" 2>/dev/null || true
  pkill -f "vite" 2>/dev/null || true
  
  log_success "Services stopped"
}

run_e2e_tests() {
  if [ "$RUN_E2E_TESTS" = false ]; then
    return 0
  fi
  
  log_info "Running end-to-end tests..."
  
  # Start services
  start_services
  
  cd "$PROJECT_ROOT"
  
  # Run Playwright tests
  if [ "$VERBOSE" = true ]; then
    npx playwright test --reporter=list
  else
    npx playwright test
  fi
  
  local exit_code=$?
  
  # Cleanup services
  cleanup_services
  
  if [ $exit_code -eq 0 ]; then
    log_success "End-to-end tests passed"
  else
    log_error "End-to-end tests failed"
    return 1
  fi
}

run_security_tests() {
  if [ "$RUN_SECURITY_TESTS" = false ]; then
    return 0
  fi
  
  log_info "Running security tests..."
  
  cd "$PROJECT_ROOT"
  
  # Run security-specific tests
  python3 -m pytest tests/test_backend.py::TestSecurity -v
  
  if [ $? -eq 0 ]; then
    log_success "Security tests passed"
  else
    log_error "Security tests failed"
    return 1
  fi
}

run_performance_tests() {
  if [ "$RUN_PERFORMANCE_TESTS" = false ]; then
    return 0
  fi
  
  log_info "Running performance tests..."
  
  cd "$PROJECT_ROOT"
  
  # Run performance-specific tests
  python3 -m pytest tests/test_backend.py::TestPerformance -v
  
  if [ $? -eq 0 ]; then
    log_success "Performance tests passed"
  else
    log_error "Performance tests failed"
    return 1
  fi
}

generate_test_report() {
  log_info "Generating test report..."
  
  cd "$PROJECT_ROOT"
  
  # Create test report directory
  mkdir -p test-results/reports
  
  # Generate combined coverage report
  if [ "$GENERATE_COVERAGE" = true ]; then
    log_info "Generating coverage report..."
    
    # Backend coverage is already generated
    # Frontend coverage is already generated
    
    # Create a simple HTML report combining both
    cat > test-results/reports/index.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Insurance AI Agent System - Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { background: #d4edda; border-color: #c3e6cb; }
        .warning { background: #fff3cd; border-color: #ffeaa7; }
        .error { background: #f8d7da; border-color: #f5c6cb; }
        .link { color: #007bff; text-decoration: none; }
        .link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Insurance AI Agent System - Test Report</h1>
        <p>Generated on: $(date)</p>
    </div>
    
    <div class="section success">
        <h2>Test Results Summary</h2>
        <p>All test suites have been executed. Check individual reports for detailed results.</p>
    </div>
    
    <div class="section">
        <h2>Coverage Reports</h2>
        <ul>
            <li><a href="../coverage/backend/index.html" class="link">Backend Coverage Report</a></li>
            <li><a href="../coverage/lcov-report/index.html" class="link">Frontend Coverage Report</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Test Artifacts</h2>
        <ul>
            <li><a href="../playwright-report/index.html" class="link">Playwright Test Report</a></li>
            <li><a href="../results.xml" class="link">JUnit Test Results</a></li>
        </ul>
    </div>
</body>
</html>
EOF
    
    log_success "Test report generated: test-results/reports/index.html"
  fi
}

# Main execution
main() {
  log_info "Starting Insurance AI Agent System test suite..."
  
  # Check dependencies
  check_dependencies
  
  # Setup test environment
  setup_test_environment
  
  # Install dependencies
  install_backend_dependencies
  install_frontend_dependencies
  
  # Track test results
  local backend_result=0
  local frontend_result=0
  local e2e_result=0
  local security_result=0
  local performance_result=0
  
  # Run tests
  if [ "$PARALLEL_EXECUTION" = true ] && [ "$RUN_BACKEND_TESTS" = true ] && [ "$RUN_FRONTEND_TESTS" = true ]; then
    log_info "Running backend and frontend tests in parallel..."
    
    # Run backend tests in background
    (run_backend_tests) &
    local backend_pid=$!
    
    # Run frontend tests in background
    (run_frontend_tests) &
    local frontend_pid=$!
    
    # Wait for both to complete
    wait $backend_pid
    backend_result=$?
    
    wait $frontend_pid
    frontend_result=$?
  else
    # Run tests sequentially
    run_backend_tests
    backend_result=$?
    
    run_frontend_tests
    frontend_result=$?
  fi
  
  # Run E2E tests (always sequential)
  run_e2e_tests
  e2e_result=$?
  
  # Run security tests
  run_security_tests
  security_result=$?
  
  # Run performance tests
  run_performance_tests
  performance_result=$?
  
  # Generate test report
  generate_test_report
  
  # Summary
  log_info "Test execution summary:"
  
  if [ $backend_result -eq 0 ]; then
    log_success "✓ Backend tests: PASSED"
  else
    log_error "✗ Backend tests: FAILED"
  fi
  
  if [ $frontend_result -eq 0 ]; then
    log_success "✓ Frontend tests: PASSED"
  else
    log_error "✗ Frontend tests: FAILED"
  fi
  
  if [ $e2e_result -eq 0 ]; then
    log_success "✓ End-to-end tests: PASSED"
  else
    log_error "✗ End-to-end tests: FAILED"
  fi
  
  if [ $security_result -eq 0 ]; then
    log_success "✓ Security tests: PASSED"
  else
    log_error "✗ Security tests: FAILED"
  fi
  
  if [ $performance_result -eq 0 ]; then
    log_success "✓ Performance tests: PASSED"
  else
    log_error "✗ Performance tests: FAILED"
  fi
  
  # Overall result
  local overall_result=$((backend_result + frontend_result + e2e_result + security_result + performance_result))
  
  if [ $overall_result -eq 0 ]; then
    log_success "🎉 All tests passed successfully!"
    exit 0
  else
    log_error "❌ Some tests failed. Check the logs above for details."
    exit 1
  fi
}

# Trap to cleanup on exit
trap cleanup_services EXIT

# Run main function
main "$@"

