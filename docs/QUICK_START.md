# Insurance AI Agent System - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Docker and Docker Compose installed
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space

### Step 1: Download and Extract
```bash
# Extract the system files
unzip insurance-ai-system-complete.zip
cd insurance-ai-system
```

### Step 2: Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional for demo)
nano .env
```

### Step 3: Run the Setup Script
```bash
./scripts/setup_local.sh
```

### Step 4: View Logs
```bash
docker-compose logs -f
```

### Step 5: Access the Application
- **Frontend**: http://localhost:80
- **API Documentation**: http://localhost:8000/docs
- **Default Login**: admin@zurich.com / admin123

### Step 6: Verify Installation
1. Open http://localhost:80 in your browser
2. Login with default credentials
3. Check dashboard loads with sample data
4. Test each module (Underwriting, Claims, Agents, Admin)

## 📱 Screenshots Guide

### Login Screen
![Login Screen](screenshots/01-login.png)
- Clean, professional login interface
- Zurich Insurance branding
- Secure authentication

### Main Dashboard
![Dashboard](screenshots/02-dashboard.png)
- Real-time KPI cards
- Interactive charts and analytics
- System status indicators

### Underwriting Module
![Underwriting](screenshots/03-underwriting.png)
- File upload interface
- Risk assessment display
- Workflow management

### Claims Processing
![Claims](screenshots/04-claims.png)
- Evidence upload and analysis
- Damage assessment tools
- Fraud detection indicators

### Agent Management
![Agent Management](screenshots/05-agents.png)
- Real-time agent monitoring
- Performance metrics
- Configuration controls

### Administration Panel
![Administration](screenshots/06-admin.png)
- User management interface
- System configuration
- Security settings

## 🔧 Production Deployment

### Kubernetes Deployment
```bash
# Deploy to production
kubectl apply -k k8s/overlays/production

# Check status
kubectl get pods -n insurance-ai-production
```

### Environment Configuration
```bash
# Production environment variables
DATABASE_URL=postgresql://user:pass@prod-db:5432/insurance_ai
REDIS_URL=redis://prod-redis:6379/0
JWT_SECRET_KEY=your-production-secret-key
ENCRYPTION_KEY=your-production-encryption-key
```

### Health Checks
```bash
# Check system health
curl http://localhost:8000/health

# Check individual services
docker-compose ps
```

## 📞 Support

- **Documentation**: See docs/USER_GUIDE.md for complete instructions
- **API Reference**: http://localhost:8000/docs
- **Troubleshooting**: See docs/TROUBLESHOOTING.md
- **Support Email**: support@insurance-ai-system.com

## 🔐 Security Notes

- Change default passwords immediately
- Use HTTPS in production
- Configure firewall rules
- Enable audit logging
- Regular security updates

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 20GB
- **Network**: 100 Mbps

### Recommended Requirements
- **CPU**: 8 cores
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **Network**: 1 Gbps

## 🎯 Next Steps

1. **Customize Configuration**: Modify settings for your environment
2. **Import Data**: Load your existing policies and claims data
3. **Train Models**: Customize AI models with your data
4. **User Training**: Train your team on the system
5. **Go Live**: Deploy to production environment

---

*For detailed instructions, see the complete USER_GUIDE.md*

