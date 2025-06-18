# Insurance AI Agent System - Troubleshooting Guide

## Common Issues and Solutions

### 🔐 Authentication Issues

#### Problem: Cannot login with correct credentials
**Symptoms:**
- Login form shows "Invalid credentials" error
- User account exists but login fails
- Password reset doesn't work

**Solutions:**
1. **Check Account Status**
   ```bash
   # Check if user account is active
   docker-compose exec backend python -c "
   from shared.database import get_db
   from shared.models import User
   db = next(get_db())
   user = db.query(User).filter(User.email == 'user@example.com').first()
   print(f'User status: {user.status if user else \"Not found\"}')"
   ```

2. **Reset Password**
   ```bash
   # Reset user password
   docker-compose exec backend python -c "
   from shared.database import get_db
   from shared.models import User
   from shared.utils import hash_password
   db = next(get_db())
   user = db.query(User).filter(User.email == 'user@example.com').first()
   user.password_hash = hash_password('newpassword123')
   db.commit()"
   ```

3. **Check JWT Configuration**
   - Verify `JWT_SECRET_KEY` in environment variables
   - Ensure system time is synchronized
   - Check token expiration settings

#### Problem: Session expires too quickly
**Solutions:**
1. **Adjust Session Timeout**
   ```bash
   # Update environment variable
   echo "JWT_EXPIRATION_HOURS=8" >> .env
   docker-compose restart backend
   ```

2. **Check Browser Settings**
   - Clear browser cache and cookies
   - Disable browser extensions that might interfere
   - Check if cookies are enabled

### 📁 File Upload Issues

#### Problem: Files fail to upload
**Symptoms:**
- Upload progress bar gets stuck
- "File too large" errors
- Unsupported file format errors

**Solutions:**
1. **Check File Size Limits**
   ```bash
   # Increase file size limit
   echo "MAX_FILE_SIZE=20971520" >> .env  # 20MB
   docker-compose restart backend nginx
   ```

2. **Verify Supported Formats**
   - PDF, DOC, DOCX: Documents
   - JPG, PNG, GIF: Images
   - MP4, AVI: Videos (claims evidence)
   - Maximum file size: 10MB (configurable)

3. **Check Disk Space**
   ```bash
   # Check available disk space
   df -h
   
   # Clean up old uploads if needed
   docker-compose exec backend python -c "
   import os
   from datetime import datetime, timedelta
   upload_dir = '/app/uploads'
   cutoff = datetime.now() - timedelta(days=30)
   for file in os.listdir(upload_dir):
       file_path = os.path.join(upload_dir, file)
       if os.path.getctime(file_path) < cutoff.timestamp():
           os.remove(file_path)"
   ```

#### Problem: Document processing fails
**Solutions:**
1. **Check OCR Service**
   ```bash
   # Restart document analysis agent
   docker-compose restart backend
   
   # Check agent logs
   docker-compose logs backend | grep "DocumentAnalysisAgent"
   ```

2. **Verify Document Quality**
   - Ensure documents are clear and readable
   - Check if document is password-protected
   - Verify document is not corrupted

#### Problem: `ImportError: libGL.so.1`
**Symptoms:**
- Backend container exits with `ImportError: libGL.so.1: cannot open shared object file`
- Document processing tasks fail to start

**Solution:**
1. **Install missing library**
   Add `libgl1` to the backend Docker image:
   ```bash
   # Dockerfile.backend
   RUN apt-get update && apt-get install -y libgl1
   ```
   Rebuild the backend container:
   ```bash
docker-compose build backend
docker-compose up -d backend
   ```

#### Problem: `ModuleNotFoundError: No module named 'tensorflow'`
**Symptoms:**
- Backend logs show this error on startup
- Evidence processing features fail to initialize

**Solution:**
1. **Install TensorFlow**
   Add `tensorflow==2.16.1` to `backend/requirements.txt`:
   ```bash
   # backend/requirements.txt
   tensorflow==2.16.1
   ```
   Rebuild the backend container:
   ```bash
   docker-compose build backend
   docker-compose up -d backend
   ```

#### Problem: `ImportError: libgthread-2.0.so.0`
**Symptoms:**
- Backend or local run fails with `ImportError: libgthread-2.0.so.0: cannot open shared object file`
- Computer vision features (e.g. OpenCV) do not start

**Solution:**
1. **Install missing glib library**
   Add `libglib2.0-0` to the backend Docker image or install it on your system:
   ```bash
   # Dockerfile.backend
   RUN apt-get update && apt-get install -y libglib2.0-0
   ```
   On macOS, install via Homebrew:
   ```bash
   brew install glib
   ```
   Rebuild the backend container:
   ```bash
    docker-compose build backend
    docker-compose up -d backend
    ```

#### Problem: `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`
**Symptoms:**
- Backend container fails to start with this error in the logs
- Occurs when using a newer `huggingface-hub` release

**Solution:**
1. **Pin an older compatible version**
   ```bash
   # backend/requirements.txt
   huggingface-hub==0.21.0
   ```
   Rebuild the backend container:
   ```bash
  docker-compose build backend
  docker-compose up -d backend
  ```

#### Problem: `ModuleNotFoundError: No module named 'face_recognition'`
**Symptoms:**
- Backend startup fails or evidence processing endpoints return errors
- Logs contain `ModuleNotFoundError: No module named 'face_recognition'`

**Solution:**
1. **Install the missing dependency**
   Add `face_recognition` to `backend/requirements.txt` or install it manually:
   ```bash
   pip install face_recognition
   ```
   Rebuild the backend container:
   ```bash
   docker-compose build backend
  docker-compose up -d backend
   ```

#### Problem: `ERROR: Failed building wheel for dlib`
**Symptoms:**
- Docker build or local install fails with `ERROR: Failed building wheel for dlib`
- Output mentions missing CMake or Python headers

**Solution:**
1. **Install required build tools**
   Add `cmake` and `python3-dev` to the backend Docker image:
   ```bash
   # Dockerfile.backend
   RUN apt-get update && apt-get install -y cmake python3-dev
   ```
2. **Pin a compatible dlib version**
   ```bash
   # backend/requirements.txt
   dlib>=19.24.2
   ```
   Then rebuild the backend container:
   ```bash
   docker-compose build backend
   docker-compose up -d backend
   ```

### 🚀 Performance Issues

#### Problem: Application loads slowly
**Symptoms:**
- Long page load times
- Slow API responses
- Timeouts during operations

**Solutions:**
1. **Check System Resources**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Check database performance
   docker-compose exec postgres psql -U postgres -d insurance_ai -c "
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC LIMIT 10;"
   ```

2. **Optimize Database**
   ```bash
   # Run database maintenance
   docker-compose exec postgres psql -U postgres -d insurance_ai -c "
   VACUUM ANALYZE;
   REINDEX DATABASE insurance_ai;"
   ```

3. **Scale Services**
   ```bash
   # Scale backend services
   docker-compose up -d --scale backend=3
   
   # Scale specific agents
   docker-compose up -d --scale document-agent=2
   ```

#### Problem: High memory usage
**Solutions:**
1. **Adjust Memory Limits**
   ```yaml
   # In docker-compose.yml
   services:
     backend:
       deploy:
         resources:
           limits:
             memory: 2G
           reservations:
             memory: 1G
   ```

2. **Clear Cache**
   ```bash
   # Clear Redis cache
   docker-compose exec redis redis-cli FLUSHALL
   
   # Restart services
   docker-compose restart
   ```

### 🤖 Agent Issues

#### Problem: Agents showing as offline
**Symptoms:**
- Agent status shows "offline" or "error"
- Processing queues backing up
- No agent heartbeats

**Solutions:**
1. **Check Agent Health**
   ```bash
   # Check agent status
   curl -H "Authorization: Bearer $TOKEN" \
        http://localhost:8000/api/v1/agents
   
   # Restart specific agent
   curl -X POST \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"action": "restart"}' \
        http://localhost:8000/api/v1/agents/doc-analysis-001/control
   ```

2. **Check Agent Logs**
   ```bash
   # View agent logs
   docker-compose logs backend | grep "Agent"
   
   # Check for specific errors
   docker-compose logs backend | grep "ERROR"
   ```

3. **Reset Agent Configuration**
   ```bash
   # Reset to default configuration
   curl -X PUT \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{
          "processing_timeout": 300,
          "max_queue_size": 100,
          "retry_attempts": 3
        }' \
        http://localhost:8000/api/v1/agents/doc-analysis-001/config
   ```

#### Problem: Poor agent performance
**Solutions:**
1. **Adjust Performance Settings**
   ```bash
   # Increase processing timeout
   curl -X PUT \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"processing_timeout": 600}' \
        http://localhost:8000/api/v1/agents/doc-analysis-001/config
   ```

2. **Scale Agent Instances**
   ```bash
   # Add more agent instances
   docker-compose up -d --scale backend=3
   ```

### 🗄️ Database Issues

#### Problem: Database connection errors
**Symptoms:**
- "Connection refused" errors
- "Too many connections" errors
- Slow database queries

**Solutions:**
1. **Check Database Status**
   ```bash
   # Check if PostgreSQL is running
   docker-compose ps postgres
   
   # Check database logs
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec postgres psql -U postgres -d insurance_ai -c "SELECT 1;"
   ```

2. **Increase Connection Pool**
   ```bash
   # Update environment variables
   echo "DATABASE_POOL_SIZE=20" >> .env
   echo "DATABASE_MAX_OVERFLOW=30" >> .env
   docker-compose restart backend
   ```

3. **Optimize Database Configuration**
   ```bash
   # Increase PostgreSQL connections
   docker-compose exec postgres psql -U postgres -c "
   ALTER SYSTEM SET max_connections = 200;
   ALTER SYSTEM SET shared_buffers = '256MB';
   SELECT pg_reload_conf();"
   ```

#### Problem: Database performance issues
**Solutions:**
1. **Analyze Slow Queries**
   ```bash
   # Enable query logging
   docker-compose exec postgres psql -U postgres -c "
   ALTER SYSTEM SET log_min_duration_statement = 1000;
   SELECT pg_reload_conf();"
   
   # Check slow queries
   docker-compose logs postgres | grep "duration:"
   ```

2. **Update Statistics**
   ```bash
   # Update table statistics
   docker-compose exec postgres psql -U postgres -d insurance_ai -c "
   ANALYZE;
   VACUUM ANALYZE;"
   ```

### 🌐 Network Issues

#### Problem: API requests failing
**Symptoms:**
- 502 Bad Gateway errors
- Connection timeouts
- CORS errors

**Solutions:**
1. **Check Service Health**
   ```bash
   # Check all services
   docker-compose ps
   
   # Test API health
   curl http://localhost:8000/health
   
   # Check nginx configuration
   docker-compose exec nginx nginx -t
   ```

2. **Fix CORS Issues**
   ```bash
   # Update CORS settings
   echo "CORS_ORIGINS=http://localhost:3000,http://localhost:80" >> .env
   docker-compose restart backend
   ```

3. **Check Load Balancer**
   ```bash
   # Restart nginx
   docker-compose restart nginx
   
   # Check nginx logs
   docker-compose logs nginx
   ```

### 📊 Frontend Issues

#### Problem: React application not loading
**Symptoms:**
- Blank white screen
- JavaScript errors in console
- Build failures

**Solutions:**
1. **Check Build Process**
   ```bash
   # Rebuild frontend
   cd frontend
   npm install
   npm run build
   
   # Check for build errors
   npm run build 2>&1 | grep ERROR
   ```

2. **Clear Browser Cache**
   - Hard refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
   - Clear browser cache and cookies
   - Try incognito/private browsing mode

3. **Check Console Errors**
   ```javascript
   // Open browser console (F12) and look for errors
   // Common issues:
   // - Missing environment variables
   // - API endpoint not accessible
   // - JavaScript syntax errors
   ```

#### Problem: Charts not displaying
**Solutions:**
1. **Check Data API**
   ```bash
   # Test dashboard API
   curl -H "Authorization: Bearer $TOKEN" \
        http://localhost:8000/api/v1/dashboard/metrics
   ```

2. **Verify Chart Libraries**
   ```bash
   # Reinstall chart dependencies
   cd frontend
   npm install recharts chart.js react-chartjs-2
   ```

### 🔧 System Maintenance

#### Daily Maintenance Tasks
```bash
#!/bin/bash
# Daily maintenance script

# Check disk space
df -h | awk '$5 > 80 {print "Warning: " $1 " is " $5 " full"}'

# Clean up old logs
find /var/log -name "*.log" -mtime +7 -delete

# Backup database
docker-compose exec postgres pg_dump -U postgres insurance_ai > backup_$(date +%Y%m%d).sql

# Check service health
docker-compose ps | grep -v "Up" && echo "Some services are down!"

# Update statistics
docker-compose exec postgres psql -U postgres -d insurance_ai -c "ANALYZE;"
```

#### Weekly Maintenance Tasks
```bash
#!/bin/bash
# Weekly maintenance script

# Full database backup
docker-compose exec postgres pg_dump -U postgres insurance_ai | gzip > weekly_backup_$(date +%Y%m%d).sql.gz

# Clean up old uploads
find uploads/ -mtime +30 -delete

# Restart services for fresh start
docker-compose restart

# Check for updates
docker-compose pull
```

### 🚨 Emergency Procedures

#### System Recovery
1. **Complete System Failure**
   ```bash
   # Stop all services
   docker-compose down
   
   # Check system resources
   free -h
   df -h
   
   # Restart with fresh containers
   docker-compose up -d --force-recreate
   ```

2. **Database Recovery**
   ```bash
   # Restore from backup
   docker-compose exec postgres psql -U postgres -c "DROP DATABASE insurance_ai;"
   docker-compose exec postgres psql -U postgres -c "CREATE DATABASE insurance_ai;"
   cat backup_20240115.sql | docker-compose exec -T postgres psql -U postgres insurance_ai
   ```

3. **Data Corruption**
   ```bash
   # Check database integrity
   docker-compose exec postgres psql -U postgres -d insurance_ai -c "
   SELECT datname, pg_database_size(datname) FROM pg_database WHERE datname='insurance_ai';"
   
   # Repair if needed
   docker-compose exec postgres psql -U postgres -d insurance_ai -c "REINDEX DATABASE insurance_ai;"
   ```

### 📞 Getting Help

#### Log Collection
Before contacting support, collect these logs:
```bash
# Collect all logs
mkdir support_logs_$(date +%Y%m%d)
cd support_logs_$(date +%Y%m%d)

# System information
docker --version > system_info.txt
docker-compose --version >> system_info.txt
uname -a >> system_info.txt
free -h >> system_info.txt
df -h >> system_info.txt

# Service logs
docker-compose logs backend > backend.log
docker-compose logs frontend > frontend.log
docker-compose logs postgres > postgres.log
docker-compose logs redis > redis.log
docker-compose logs nginx > nginx.log

# Configuration
cp ../.env env_config.txt
cp ../docker-compose.yml docker_compose.yml

# Create archive
cd ..
tar -czf support_logs_$(date +%Y%m%d).tar.gz support_logs_$(date +%Y%m%d)/
```

#### Support Channels
- **Email**: support@insurance-ai-system.com
- **Emergency**: +1-800-SUPPORT
- **Documentation**: `/docs` endpoint
- **GitHub Issues**: For bug reports and feature requests

#### Information to Include
When reporting issues, include:
- System information (OS, Docker version)
- Steps to reproduce the issue
- Error messages and logs
- Screenshots if applicable
- Configuration files (without sensitive data)

---

*This troubleshooting guide covers the most common issues. For additional help, consult the complete documentation or contact support.*

