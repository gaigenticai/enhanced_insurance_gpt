# Insurance AI Agent System - Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying the Insurance AI Agent System with the simplified authentication system using user ID/password and registration functionality.

## Prerequisites

### System Requirements
- Docker and Docker Compose
- PostgreSQL 15+ with pgvector extension
- Node.js 18+ and npm/pnpm
- Python 3.11+
- At least 4GB RAM and 20GB disk space

### Environment Setup
1. Clone the repository
2. Ensure all required ports are available:
   - Frontend: 3000
   - Backend API Gateway: 8000
   - Auth Service: 8001
   - PostgreSQL: 5432
   - Redis: 6379

## Quick Start (Docker Compose)

### 1. Environment Configuration
Create a `.env` file in the root directory:

```bash
# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_DB=insurance_ai
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_PORT=5432

# Authentication
JWT_SECRET=your_super_secure_jwt_secret_change_this_in_production
ACCESS_TOKEN_EXPIRE_HOURS=24

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
AUTH_SERVICE_PORT=8001

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_AUTH_URL=http://localhost:8001

# Environment
NODE_ENV=production
ENVIRONMENT=production
```

### 2. Database Initialization
The database schema will be automatically created when the containers start. The system includes:
- Complete user management with all fields exposed
- Organization management
- Audit logging
- Security features

### 3. Start the Application
```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Access the Application
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- Auth Service Documentation: http://localhost:8001/docs

## Manual Deployment

### 1. Database Setup
```bash
# Install PostgreSQL with pgvector
sudo apt update
sudo apt install postgresql-15 postgresql-15-pgvector

# Create database and user
sudo -u postgres psql
CREATE DATABASE insurance_ai;
CREATE USER insurance_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE insurance_ai TO insurance_user;

# Enable extensions
\c insurance_ai
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

# Run schema
\i database/schema.sql
```

### 2. Backend Services

#### Auth Service
```bash
cd auth_service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_DB=insurance_ai
export POSTGRES_USER=insurance_user
export POSTGRES_PASSWORD=your_password
export JWT_SECRET=your_jwt_secret

# Start the service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

#### Main Backend (if needed)
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set environment variables and start
uvicorn api_gateway:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend
```bash
cd frontend
npm install  # or pnpm install

# Set environment variables
export REACT_APP_API_URL=http://localhost:8000
export REACT_APP_AUTH_URL=http://localhost:8001

# Build and start
npm run build
npm run preview  # or serve the build directory
```

## Production Deployment

### 1. Security Considerations
- Change all default passwords and secrets
- Use HTTPS with proper SSL certificates
- Configure firewall rules
- Enable database encryption
- Set up proper backup procedures
- Configure log rotation and monitoring

### 2. Environment Variables for Production
```bash
# Use strong, unique values for production
JWT_SECRET=generate_a_very_long_random_string_here
POSTGRES_PASSWORD=use_a_strong_database_password
REDIS_PASSWORD=use_a_strong_redis_password

# Use production URLs
REACT_APP_API_URL=https://your-domain.com/api
REACT_APP_AUTH_URL=https://your-domain.com/auth

# Production settings
NODE_ENV=production
ENVIRONMENT=production
```

### 3. Nginx Configuration (Optional)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    # Frontend
    location / {
        root /path/to/frontend/build;
        try_files $uri $uri/ /index.html;
    }

    # API Gateway
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Auth Service
    location /auth/ {
        proxy_pass http://localhost:8001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## User Management

### Default Admin User Creation
The system allows registration with automatic admin privileges. The first user to register will have admin access.

### User Registration Process
1. Navigate to the application URL
2. Click "Create Account" on the login page
3. Fill in all required fields:
   - User ID (unique identifier)
   - Email address
   - Password (with strength requirements)
   - First and last name
   - Phone number (optional)
   - Organization name (optional)
4. Submit the form to create an account with admin privileges

### User Profile Management
All user table fields are exposed in the UI:
- Basic Information: User ID, email, name, phone
- System Information: Role, status, organization
- Security Information: Account active/verified status
- Audit Information: Creation date, last login, failed attempts
- Advanced Settings: User settings, lock status

## API Endpoints

### Authentication Service (Port 8001)
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/token` - OAuth2 compatible login
- `GET /api/v1/auth/profile` - Get user profile
- `PUT /api/v1/auth/profile` - Update user profile
- `POST /api/v1/auth/change-password` - Change password
- `POST /api/v1/auth/verify-token` - Verify token validity
- `GET /health` - Health check

### API Documentation
- Auth Service: http://localhost:8001/docs
- Main API: http://localhost:8000/docs

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check database connectivity
psql -h localhost -U insurance_user -d insurance_ai

# Verify extensions
SELECT * FROM pg_extension;
```

#### Authentication Issues
```bash
# Check auth service logs
docker-compose logs auth_service

# Verify JWT secret is set
echo $JWT_SECRET

# Test auth endpoint
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "password": "test"}'
```

#### Frontend Issues
```bash
# Check frontend build
cd frontend && npm run build

# Verify environment variables
echo $REACT_APP_AUTH_URL

# Check network connectivity
curl http://localhost:8001/health
```

### Log Locations
- Docker logs: `docker-compose logs [service_name]`
- Auth service: Check container logs
- Database: PostgreSQL logs in `/var/log/postgresql/`
- Frontend: Browser developer console

## Monitoring and Maintenance

### Health Checks
- Auth Service: `GET /health`
- Database: Connection monitoring
- Frontend: Application availability

### Backup Procedures
```bash
# Database backup
pg_dump -h localhost -U insurance_user insurance_ai > backup_$(date +%Y%m%d).sql

# Restore database
psql -h localhost -U insurance_user insurance_ai < backup_file.sql
```

### Updates and Maintenance
1. Always backup before updates
2. Test in staging environment first
3. Use rolling updates for zero-downtime deployment
4. Monitor logs during and after deployment

## Support and Documentation

### Additional Resources
- API Documentation: Available at `/docs` endpoints
- Database Schema: `database/schema.sql`
- Frontend Components: `frontend/src/components/`
- Authentication Logic: `auth_service/main.py`

### Getting Help
1. Check logs for error messages
2. Verify environment variables
3. Test individual components
4. Review API documentation
5. Check database connectivity and schema

This deployment guide ensures a production-ready setup with all security considerations and user management features properly configured.

