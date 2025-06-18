# Insurance AI Agent System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Insurance AI Agent System in production environments. The system supports multiple deployment options including Docker Compose, Kubernetes, and cloud-native deployments.

## Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: 100 Mbps

#### Recommended Requirements (Production)
- **CPU**: 16+ cores
- **RAM**: 32+ GB
- **Storage**: 500+ GB SSD with IOPS 3000+
- **Network**: 1 Gbps+
- **Load Balancer**: Application Load Balancer
- **Database**: Managed PostgreSQL with read replicas

#### Software Dependencies
- Docker 24.0+
- Docker Compose 2.20+
- Kubernetes 1.28+ (for K8s deployment)
- kubectl configured with cluster access
- Helm 3.12+ (for K8s deployment)
- Terraform 1.5+ (for infrastructure)

### Environment Setup

#### 1. Clone Repository
```bash
git clone https://github.com/zurich/insurance-ai-system.git
cd insurance-ai-system
```

#### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

#### 3. Required Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/insurance_ai
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=insurance_ai
DATABASE_USER=postgres
DATABASE_PASSWORD=secure_password

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password

# Security Configuration
SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=jwt-secret-key
ENCRYPTION_KEY=32-byte-encryption-key

# AI Configuration
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACE_API_KEY=your-huggingface-key

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# File Storage
STORAGE_TYPE=local  # or s3
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_BUCKET_NAME=insurance-ai-documents
AWS_REGION=us-east-1

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin_password

# External APIs
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
FIREBASE_SERVER_KEY=your-firebase-key
```

## Deployment Options

### Option 1: Docker Compose (Recommended for Development/Testing)

#### 1. Quick Start
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### 2. Production Docker Compose
```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale backend=3 --scale worker=2
```

#### 3. Service Health Checks
```bash
# Check API health
curl http://localhost:8000/health

# Check database connection
docker-compose exec backend python -c "from backend.shared.database import test_connection; test_connection()"

# Check Redis connection
docker-compose exec backend python -c "import redis; r=redis.Redis(host='redis'); print(r.ping())"
```

### Option 2: Kubernetes Deployment

#### 1. Namespace Setup
```bash
# Create namespace
kubectl create namespace insurance-ai

# Set default namespace
kubectl config set-context --current --namespace=insurance-ai
```

#### 2. Secrets Configuration
```bash
# Create database secret
kubectl create secret generic database-secret \
  --from-literal=username=postgres \
  --from-literal=password=secure_password \
  --from-literal=database=insurance_ai

# Create API keys secret
kubectl create secret generic api-keys \
  --from-literal=openai-key=your-openai-key \
  --from-literal=jwt-secret=your-jwt-secret

# Create TLS secret (if using HTTPS)
kubectl create secret tls insurance-ai-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

#### 3. Deploy Base Infrastructure
```bash
# Deploy PostgreSQL
kubectl apply -f k8s/base/postgres.yaml

# Deploy Redis
kubectl apply -f k8s/base/redis.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s
```

#### 4. Deploy Application Services
```bash
# Deploy backend services
kubectl apply -f k8s/base/backend-frontend.yaml

# Deploy monitoring stack
kubectl apply -f k8s/base/monitoring.yaml

# Deploy ingress
kubectl apply -f k8s/base/nginx-ingress.yaml
```

#### 5. Verify Deployment
```bash
# Check pod status
kubectl get pods

# Check services
kubectl get services

# Check ingress
kubectl get ingress

# View logs
kubectl logs -f deployment/backend
```

### Option 3: Helm Deployment

#### 1. Add Helm Repository
```bash
# Add custom helm repository (if available)
helm repo add insurance-ai https://charts.insurance-ai.zurich.com
helm repo update
```

#### 2. Install with Helm
```bash
# Install with default values
helm install insurance-ai insurance-ai/insurance-ai-system

# Install with custom values
helm install insurance-ai insurance-ai/insurance-ai-system -f values.yaml

# Upgrade deployment
helm upgrade insurance-ai insurance-ai/insurance-ai-system
```

#### 3. Custom Values File (values.yaml)
```yaml
# Application configuration
app:
  name: insurance-ai-system
  version: "1.0.0"
  environment: production

# Database configuration
postgresql:
  enabled: true
  auth:
    postgresPassword: "secure_password"
    database: "insurance_ai"
  primary:
    persistence:
      size: 100Gi
      storageClass: "gp3"

# Redis configuration
redis:
  enabled: true
  auth:
    password: "redis_password"
  master:
    persistence:
      size: 10Gi

# Backend configuration
backend:
  replicaCount: 3
  image:
    repository: insurance-ai/backend
    tag: "1.0.0"
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi

# Frontend configuration
frontend:
  replicaCount: 2
  image:
    repository: insurance-ai/frontend
    tag: "1.0.0"

# Ingress configuration
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: insurance-ai.zurich.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: insurance-ai-tls
      hosts:
        - insurance-ai.zurich.com

# Monitoring
monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "admin_password"
```

### Option 4: Cloud-Native Deployment (AWS)

#### 1. Infrastructure with Terraform
```bash
# Initialize Terraform
cd deployment/terraform
terraform init

# Plan deployment
terraform plan -var-file="production.tfvars"

# Apply infrastructure
terraform apply -var-file="production.tfvars"
```

#### 2. EKS Cluster Setup
```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name insurance-ai-cluster

# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

# Install cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.12.0/cert-manager.yaml
```

#### 3. Deploy to EKS
```bash
# Deploy using Kustomize
kubectl apply -k k8s/overlays/production

# Monitor deployment
kubectl get pods -w
```

## Database Setup

### 1. Database Initialization
```bash
# Run database migrations
docker-compose exec backend python database/migrations/migrate.py

# Seed initial data
docker-compose exec backend python database/seeds/seed_data.py

# Verify database setup
docker-compose exec backend python -c "
from backend.shared.database import get_db_session
from backend.shared.models import User
session = get_db_session()
print(f'Users count: {session.query(User).count()}')
"
```

### 2. Database Backup Setup
```bash
# Create backup script
cat > scripts/backup_db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="insurance_ai_backup_${TIMESTAMP}.sql"

# Create backup
pg_dump $DATABASE_URL > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Upload to S3 (optional)
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" s3://insurance-ai-backups/

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
EOF

chmod +x scripts/backup_db.sh

# Setup cron job for daily backups
echo "0 2 * * * /path/to/scripts/backup_db.sh" | crontab -
```

## SSL/TLS Configuration

### 1. Let's Encrypt with Cert-Manager
```bash
# Install cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer
cat << EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@zurich.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 2. Custom SSL Certificate
```bash
# Create TLS secret with custom certificate
kubectl create secret tls insurance-ai-tls \
  --cert=path/to/certificate.crt \
  --key=path/to/private.key \
  --namespace=insurance-ai
```

## Monitoring Setup

### 1. Prometheus Configuration
```bash
# Deploy Prometheus
kubectl apply -f monitoring/prometheus/

# Access Prometheus UI
kubectl port-forward svc/prometheus 9090:9090
# Open http://localhost:9090
```

### 2. Grafana Setup
```bash
# Deploy Grafana
kubectl apply -f monitoring/grafana/

# Get Grafana admin password
kubectl get secret grafana-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Access Grafana UI
kubectl port-forward svc/grafana 3000:3000
# Open http://localhost:3000
```

### 3. Log Aggregation with ELK
```bash
# Deploy Elasticsearch
kubectl apply -f monitoring/elasticsearch/

# Deploy Logstash
kubectl apply -f monitoring/logstash/

# Deploy Kibana
kubectl apply -f monitoring/kibana/

# Access Kibana
kubectl port-forward svc/kibana 5601:5601
```

## Performance Tuning

### 1. Database Optimization
```sql
-- PostgreSQL configuration optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Reload configuration
SELECT pg_reload_conf();
```

### 2. Redis Optimization
```bash
# Redis configuration
cat > redis.conf << EOF
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
tcp-keepalive 300
timeout 0
EOF
```

### 3. Application Performance
```python
# Backend optimization settings
UVICORN_WORKERS = 4
UVICORN_WORKER_CONNECTIONS = 1000
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
REDIS_CONNECTION_POOL_SIZE = 50
```

## Security Hardening

### 1. Network Security
```bash
# Create network policies
cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: insurance-ai-network-policy
spec:
  podSelector:
    matchLabels:
      app: insurance-ai
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
EOF
```

### 2. Pod Security Standards
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: backend
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: backend
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

### 3. Secrets Management
```bash
# Use external secrets operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace

# Configure AWS Secrets Manager integration
cat << EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
EOF
```

## Backup and Recovery

### 1. Database Backup Strategy
```bash
# Automated backup script
#!/bin/bash
set -e

BACKUP_DIR="/backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump $DATABASE_URL | gzip > "${BACKUP_DIR}/db_backup_${TIMESTAMP}.sql.gz"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/db_backup_${TIMESTAMP}.sql.gz" s3://insurance-ai-backups/database/

# Clean old backups
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete

# Verify backup integrity
gunzip -t "${BACKUP_DIR}/db_backup_${TIMESTAMP}.sql.gz"

echo "Backup completed successfully: db_backup_${TIMESTAMP}.sql.gz"
```

### 2. Application Data Backup
```bash
# Backup uploaded files
rsync -av /app/uploads/ s3://insurance-ai-backups/uploads/

# Backup configuration
kubectl get configmaps -o yaml > configmaps_backup.yaml
kubectl get secrets -o yaml > secrets_backup.yaml
```

### 3. Disaster Recovery Procedure
```bash
# 1. Restore database
gunzip -c db_backup_20240614_120000.sql.gz | psql $DATABASE_URL

# 2. Restore application files
aws s3 sync s3://insurance-ai-backups/uploads/ /app/uploads/

# 3. Restore Kubernetes resources
kubectl apply -f configmaps_backup.yaml
kubectl apply -f secrets_backup.yaml

# 4. Restart services
kubectl rollout restart deployment/backend
kubectl rollout restart deployment/frontend
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check database connectivity
kubectl exec -it deployment/backend -- python -c "
from backend.shared.database import test_connection
test_connection()
"

# Check database logs
kubectl logs deployment/postgres

# Verify database credentials
kubectl get secret database-secret -o yaml
```

#### 2. Redis Connection Issues
```bash
# Test Redis connection
kubectl exec -it deployment/backend -- python -c "
import redis
r = redis.Redis(host='redis', port=6379)
print(r.ping())
"

# Check Redis logs
kubectl logs deployment/redis
```

#### 3. AI Agent Issues
```bash
# Check AI agent status
curl http://localhost:8000/ai/agents

# View AI processing logs
kubectl logs -f deployment/backend | grep "AI_AGENT"

# Check model loading
kubectl exec -it deployment/backend -- python -c "
from backend.agents.document_analysis.document_processor import DocumentProcessor
processor = DocumentProcessor()
print('Models loaded successfully')
"
```

#### 4. Performance Issues
```bash
# Check resource usage
kubectl top pods
kubectl top nodes

# Monitor database performance
kubectl exec -it deployment/postgres -- psql -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
"

# Check application metrics
curl http://localhost:8000/metrics
```

### Log Analysis
```bash
# View application logs
kubectl logs -f deployment/backend --tail=100

# Search for errors
kubectl logs deployment/backend | grep ERROR

# Monitor real-time logs
kubectl logs -f deployment/backend | grep -E "(ERROR|WARNING|CRITICAL)"
```

### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/database

# Redis health
curl http://localhost:8000/health/redis

# AI agents health
curl http://localhost:8000/health/ai-agents
```

## Maintenance

### 1. Regular Updates
```bash
# Update Docker images
docker-compose pull
docker-compose up -d

# Update Kubernetes deployments
kubectl set image deployment/backend backend=insurance-ai/backend:1.1.0
kubectl rollout status deployment/backend
```

### 2. Database Maintenance
```bash
# Vacuum and analyze
kubectl exec -it deployment/postgres -- psql -c "VACUUM ANALYZE;"

# Update statistics
kubectl exec -it deployment/postgres -- psql -c "ANALYZE;"

# Check database size
kubectl exec -it deployment/postgres -- psql -c "
SELECT pg_size_pretty(pg_database_size('insurance_ai'));
"
```

### 3. Log Rotation
```bash
# Configure log rotation
cat > /etc/logrotate.d/insurance-ai << EOF
/var/log/insurance-ai/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 app app
    postrotate
        systemctl reload insurance-ai
    endscript
}
EOF
```

This deployment guide provides comprehensive instructions for deploying the Insurance AI Agent System in various environments with proper security, monitoring, and maintenance procedures.

