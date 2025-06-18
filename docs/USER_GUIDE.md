# Insurance AI Agent System - Complete User Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Installation Guide](#installation-guide)
3. [User Authentication](#user-authentication)
4. [Dashboard Overview](#dashboard-overview)
5. [Underwriting Module](#underwriting-module)
6. [Claims Processing Module](#claims-processing-module)
7. [Agent Management Interface](#agent-management-interface)
8. [Administration Panel](#administration-panel)
9. [API Documentation](#api-documentation)
10. [Troubleshooting](#troubleshooting)

---

## System Overview

The Insurance AI Agent System is a comprehensive platform designed for Zurich Insurance to automate and streamline insurance operations through AI-powered agents and orchestrators.

### Key Features
- **AI-Powered Underwriting**: Automated risk assessment and policy evaluation
- **Claims Processing**: Intelligent claims handling with evidence analysis
- **Agent Management**: Monitor and manage AI agents in real-time
- **Administration**: Complete system administration and user management
- **Real-time Dashboard**: Live KPIs and system metrics
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### System Architecture
- **Frontend**: React.js application with responsive design
- **Backend**: FastAPI with microservices architecture
- **Database**: PostgreSQL with Redis caching
- **AI Agents**: 5 specialized agents for different tasks
- **Orchestrators**: 2 master orchestrators for workflow management
- **Deployment**: Docker containers with Kubernetes support

---

## Installation Guide

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for development)
- Python 3.11+ (for development)
- Kubernetes cluster (for production deployment)

### Quick Start - Docker Compose

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd insurance-ai-system
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the system**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Frontend: http://localhost:80
   - Backend API: http://localhost:8000/docs
   - Admin Panel: Integrated in frontend

### Production Deployment - Kubernetes

1. **Deploy to Kubernetes**
   ```bash
   kubectl apply -k k8s/overlays/production
   ```

2. **Check deployment status**
   ```bash
   kubectl get pods -n insurance-ai-production
   ```

3. **Access the application**
   ```bash
   kubectl port-forward svc/frontend-service 8080:80 -n insurance-ai-production
   ```

### Development Setup

1. **Backend Development**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn api_gateway:app --reload
   ```

2. **Frontend Development**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

---

## User Authentication

### Login Process

1. **Access the Login Page**
   - Navigate to the application URL
   - You'll be automatically redirected to the login page if not authenticated

2. **Enter Credentials**
   - **Email**: Enter your registered email address
   - **Password**: Enter your password
   - Click "Sign In" button

3. **Default Admin Credentials**
   - Email: `admin@zurich.com`
   - Password: `admin123`

### User Roles
- **Admin**: Full system access including user management
- **Underwriter**: Access to underwriting module and dashboard
- **Claims Adjuster**: Access to claims processing and dashboard
- **Agent Manager**: Access to agent management interface
- **Viewer**: Read-only access to dashboard and reports

### Security Features
- JWT token-based authentication
- Session timeout after inactivity
- Password encryption
- Role-based access control
- Audit logging for all actions

---

## Dashboard Overview

The main dashboard provides a comprehensive view of system performance and key metrics.

### Key Performance Indicators (KPIs)

1. **Total Policies**
   - Displays the total number of active policies
   - Shows percentage change from previous period
   - Color-coded indicators (green for positive, red for negative)

2. **Claims Processed**
   - Number of claims processed in the current period
   - Processing efficiency metrics
   - Average processing time

3. **Revenue**
   - Total revenue generated
   - Revenue trends and projections
   - Breakdown by policy types

4. **Active Agents**
   - Number of AI agents currently active
   - Agent performance metrics
   - System health indicators

### Charts and Analytics

1. **Policy Trends Chart**
   - Line chart showing policy creation trends over time
   - Interactive tooltips with detailed information
   - Filterable by date range and policy type

2. **Claims Processing Chart**
   - Bar chart showing claims processing volumes
   - Breakdown by claim types and status
   - Processing time analytics

3. **Revenue Analytics**
   - Area chart showing revenue trends
   - Comparison with previous periods
   - Forecasting capabilities

4. **Agent Performance Chart**
   - Real-time agent performance metrics
   - Success rates and processing times
   - Error rates and system health

### Real-time Updates
- Dashboard updates automatically every 30 seconds
- WebSocket connections for real-time data
- Push notifications for critical alerts
- Live status indicators for all systems

---

## Underwriting Module

The Underwriting Module automates the policy evaluation and risk assessment process using AI-powered analysis.

### Features Overview
- **Document Upload**: Drag-and-drop file upload with progress tracking
- **Risk Assessment**: AI-powered risk analysis and scoring
- **Policy Evaluation**: Automated policy recommendation engine
- **Workflow Management**: Complete underwriting workflow automation
- **Decision Support**: AI recommendations with human oversight

### Step-by-Step Usage

#### 1. Creating a New Underwriting Case

1. **Navigate to Underwriting Module**
   - Click "Underwriting" in the main navigation menu
   - You'll see the underwriting dashboard with active cases

2. **Start New Case**
   - Click the "New Underwriting Case" button
   - Fill in the basic information form:
     - **Policy Type**: Select from dropdown (Auto, Home, Life, etc.)
     - **Applicant Name**: Enter full name
     - **Application ID**: Auto-generated or manual entry
     - **Priority Level**: High, Medium, or Low

3. **Upload Documents**
   - Use the drag-and-drop area to upload documents
   - Supported formats: PDF, DOC, DOCX, JPG, PNG
   - Maximum file size: 10MB per file
   - Multiple files can be uploaded simultaneously

#### 2. Document Processing

1. **Automatic Processing**
   - Documents are automatically processed using OCR
   - AI extracts key information and data points
   - Progress bar shows processing status

2. **Review Extracted Data**
   - Review automatically extracted information
   - Edit or correct any inaccuracies
   - Add additional notes or comments

3. **Document Validation**
   - System validates document authenticity
   - Checks for required documents
   - Flags any missing or incomplete information

#### 3. Risk Assessment

1. **AI Risk Analysis**
   - Automated risk scoring based on multiple factors
   - Machine learning models analyze historical data
   - Real-time risk calculation and updates

2. **Risk Factors Review**
   - View detailed breakdown of risk factors
   - Understand scoring methodology
   - Add manual adjustments if needed

3. **Risk Score Interpretation**
   - **Low Risk (0-30)**: Green indicator, automatic approval recommended
   - **Medium Risk (31-70)**: Yellow indicator, manual review required
   - **High Risk (71-100)**: Red indicator, detailed investigation needed

#### 4. Policy Recommendations

1. **AI Recommendations**
   - System provides policy recommendations
   - Suggested premium calculations
   - Coverage options and limits

2. **Manual Override**
   - Underwriters can override AI recommendations
   - Add justification for manual decisions
   - Document reasoning for audit trail

3. **Approval Workflow**
   - Submit for approval based on risk level
   - Automatic routing to appropriate approvers
   - Email notifications to stakeholders

#### 5. Case Management

1. **Case Tracking**
   - View all cases in various stages
   - Filter by status, priority, or date
   - Search functionality for quick access

2. **Status Updates**
   - Real-time status updates
   - Automated notifications for status changes
   - Integration with external systems

3. **Reporting**
   - Generate detailed reports
   - Export data in multiple formats
   - Performance analytics and metrics

### Advanced Features

#### Bulk Processing
- Upload multiple applications simultaneously
- Batch processing for efficiency
- Progress tracking for bulk operations

#### Integration Capabilities
- Connect with external data sources
- API integration with third-party services
- Real-time data validation

#### Compliance Monitoring
- Regulatory compliance checking
- Audit trail maintenance
- Automated compliance reporting

---

## Claims Processing Module

The Claims Processing Module handles the complete claims lifecycle from submission to settlement using AI-powered analysis and automation.

### Features Overview
- **Claims Intake**: Multi-channel claim submission
- **Evidence Processing**: AI analysis of photos, videos, and documents
- **Fraud Detection**: Advanced fraud detection algorithms
- **Damage Assessment**: Automated damage evaluation
- **Settlement Processing**: Streamlined settlement workflows

### Step-by-Step Usage

#### 1. Creating a New Claim

1. **Navigate to Claims Module**
   - Click "Claims" in the main navigation menu
   - View the claims dashboard with active claims

2. **New Claim Submission**
   - Click "New Claim" button
   - Fill in claim details:
     - **Policy Number**: Enter or search for policy
     - **Claim Type**: Auto, Property, Liability, etc.
     - **Date of Loss**: Select date from calendar
     - **Description**: Detailed incident description

3. **Claimant Information**
   - Verify policyholder details
   - Add additional claimants if applicable
   - Contact information validation

#### 2. Evidence Upload and Processing

1. **Upload Evidence**
   - Drag-and-drop interface for files
   - Support for photos, videos, documents
   - Real-time upload progress tracking

2. **AI Evidence Analysis**
   - Automatic image analysis for damage assessment
   - Video processing for incident reconstruction
   - Document OCR for information extraction

3. **Evidence Validation**
   - Metadata analysis for authenticity
   - Timestamp and location verification
   - Fraud indicators detection

#### 3. Damage Assessment

1. **Automated Assessment**
   - AI-powered damage evaluation
   - Cost estimation algorithms
   - Repair vs. replacement analysis

2. **Expert Review**
   - Flag cases requiring human expert review
   - Integration with external assessors
   - Collaborative assessment tools

3. **Assessment Reports**
   - Detailed damage assessment reports
   - Photo annotations and markups
   - Cost breakdown and justifications

#### 4. Fraud Detection

1. **AI Fraud Analysis**
   - Pattern recognition algorithms
   - Historical data comparison
   - Risk scoring for fraud probability

2. **Red Flag Indicators**
   - Suspicious patterns detection
   - Inconsistency identification
   - External database cross-referencing

3. **Investigation Workflow**
   - Automatic routing to investigation team
   - Case escalation procedures
   - Investigation tracking and reporting

#### 5. Settlement Processing

1. **Settlement Calculation**
   - Automated settlement amount calculation
   - Policy coverage verification
   - Deductible application

2. **Approval Workflow**
   - Multi-level approval process
   - Authority limits enforcement
   - Approval tracking and notifications

3. **Payment Processing**
   - Integration with payment systems
   - Multiple payment methods support
   - Payment tracking and confirmation

### Advanced Features

#### Real-time Collaboration
- Multi-user case collaboration
- Real-time comments and notes
- Activity timeline tracking

#### External Integrations
- Third-party repair networks
- Medical provider networks
- Legal service integrations

#### Analytics and Reporting
- Claims processing metrics
- Fraud detection statistics
- Settlement analysis reports

---

## Agent Management Interface

The Agent Management Interface provides comprehensive monitoring and control of all AI agents in the system.

### Features Overview
- **Agent Monitoring**: Real-time agent status and performance
- **Configuration Management**: Agent settings and parameters
- **Performance Analytics**: Detailed performance metrics and trends
- **Health Monitoring**: System health and error tracking
- **Workflow Management**: Agent workflow configuration

### Step-by-Step Usage

#### 1. Agent Overview Dashboard

1. **Navigate to Agent Management**
   - Click "Agent Management" in the main navigation
   - View the agent overview dashboard

2. **Agent Status Grid**
   - See all agents with current status
   - Color-coded status indicators:
     - **Green**: Active and healthy
     - **Yellow**: Warning or degraded performance
     - **Red**: Error or offline
     - **Blue**: Maintenance mode

3. **Quick Actions**
   - Start/stop agents with one click
   - Restart agents experiencing issues
   - View detailed agent information

#### 2. Individual Agent Management

1. **Select an Agent**
   - Click on any agent card to view details
   - Access agent-specific dashboard

2. **Agent Configuration**
   - Modify agent parameters and settings
   - Update processing thresholds
   - Configure integration endpoints

3. **Performance Monitoring**
   - View real-time performance metrics
   - Monitor processing queues
   - Track success/failure rates

#### 3. Agent Types and Functions

##### Document Analysis Agent
- **Function**: OCR, document parsing, data extraction
- **Metrics**: Processing speed, accuracy rate, error count
- **Configuration**: OCR settings, supported formats, quality thresholds

##### Risk Assessment Agent
- **Function**: Risk scoring, pattern analysis, decision support
- **Metrics**: Assessment accuracy, processing time, model performance
- **Configuration**: Risk models, scoring parameters, threshold settings

##### Communication Agent
- **Function**: Email, SMS, notifications, customer communication
- **Metrics**: Delivery rates, response times, engagement metrics
- **Configuration**: Templates, channels, delivery settings

##### Evidence Processing Agent
- **Function**: Image/video analysis, damage assessment, fraud detection
- **Metrics**: Analysis accuracy, processing speed, detection rates
- **Configuration**: Analysis models, quality settings, detection thresholds

##### Compliance Agent
- **Function**: Regulatory compliance, audit trails, reporting
- **Metrics**: Compliance rate, audit findings, report generation
- **Configuration**: Regulatory rules, reporting schedules, audit settings

#### 4. Performance Analytics

1. **Performance Metrics**
   - Processing volume and throughput
   - Success and error rates
   - Response time analytics
   - Resource utilization

2. **Trend Analysis**
   - Historical performance trends
   - Comparative analysis between agents
   - Performance forecasting

3. **Alert Management**
   - Configure performance alerts
   - Set threshold-based notifications
   - Escalation procedures

#### 5. Workflow Management

1. **Workflow Configuration**
   - Define agent interaction workflows
   - Set up processing pipelines
   - Configure routing rules

2. **Queue Management**
   - Monitor processing queues
   - Prioritize urgent tasks
   - Load balancing configuration

3. **Integration Management**
   - External system integrations
   - API endpoint configuration
   - Data flow monitoring

### Advanced Features

#### Automated Scaling
- Auto-scaling based on load
- Resource optimization
- Performance-based scaling

#### Machine Learning Model Management
- Model versioning and deployment
- A/B testing capabilities
- Performance comparison

#### Audit and Compliance
- Complete audit trails
- Compliance reporting
- Security monitoring

---

## Administration Panel

The Administration Panel provides comprehensive system administration capabilities for managing users, system settings, security, and maintenance.

### Features Overview
- **User Management**: Create, modify, and manage user accounts
- **System Configuration**: Global system settings and parameters
- **Security Management**: Security policies and access controls
- **Backup and Recovery**: Data backup and system recovery
- **Monitoring and Alerts**: System monitoring and alert configuration

### Step-by-Step Usage

#### 1. User Management

1. **Navigate to Administration**
   - Click "Administration" in the main navigation
   - Access the admin dashboard

2. **User Account Management**
   - View all user accounts in a searchable table
   - Filter users by role, status, or department
   - Sort by various criteria

3. **Creating New Users**
   - Click "Add New User" button
   - Fill in user details:
     - **Full Name**: User's complete name
     - **Email Address**: Login email (must be unique)
     - **Role**: Select from available roles
     - **Department**: User's department
     - **Phone Number**: Contact information
   - Set initial password or send invitation email

4. **Modifying User Accounts**
   - Click on any user to edit details
   - Update roles and permissions
   - Reset passwords or disable accounts
   - View user activity logs

5. **Role Management**
   - Define custom roles and permissions
   - Assign specific module access
   - Set data access restrictions

#### 2. System Configuration

1. **General Settings**
   - System name and branding
   - Default language and timezone
   - Session timeout settings
   - Email configuration

2. **Business Rules Configuration**
   - Underwriting rules and thresholds
   - Claims processing parameters
   - Approval workflows and limits
   - Escalation procedures

3. **Integration Settings**
   - External API configurations
   - Database connection settings
   - Third-party service integrations
   - Webhook configurations

#### 3. Security Management

1. **Security Policies**
   - Password complexity requirements
   - Account lockout policies
   - Session management settings
   - Two-factor authentication

2. **Access Control**
   - IP address restrictions
   - Time-based access controls
   - Geographic restrictions
   - Device management

3. **Audit and Compliance**
   - Audit log configuration
   - Compliance reporting settings
   - Data retention policies
   - Privacy controls

#### 4. System Monitoring

1. **System Health Dashboard**
   - Server status and performance
   - Database health metrics
   - Application performance indicators
   - Resource utilization

2. **Alert Configuration**
   - Set up system alerts and notifications
   - Configure alert thresholds
   - Define escalation procedures
   - Notification channels

3. **Performance Monitoring**
   - Real-time performance metrics
   - Historical performance data
   - Capacity planning tools
   - Optimization recommendations

#### 5. Backup and Recovery

1. **Backup Configuration**
   - Schedule automated backups
   - Configure backup retention
   - Set backup destinations
   - Verify backup integrity

2. **Recovery Procedures**
   - System recovery options
   - Data restoration procedures
   - Disaster recovery planning
   - Recovery testing

3. **Maintenance Windows**
   - Schedule system maintenance
   - Notify users of downtime
   - Coordinate updates and patches
   - Maintenance history tracking

### Advanced Features

#### System Analytics
- Usage analytics and reporting
- Performance trend analysis
- Capacity planning tools
- Cost optimization insights

#### Automation
- Automated system tasks
- Scheduled maintenance procedures
- Automated reporting
- Self-healing capabilities

#### Compliance and Governance
- Regulatory compliance monitoring
- Data governance policies
- Privacy protection measures
- Audit trail management

---

## API Documentation

The Insurance AI Agent System provides a comprehensive REST API for integration with external systems and custom applications.

### API Overview
- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: JWT Bearer tokens
- **Format**: JSON request/response
- **Rate Limiting**: 1000 requests per hour per user

### Authentication Endpoints

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": 1,
    "email": "user@example.com",
    "role": "underwriter"
  }
}
```

#### Refresh Token
```http
POST /auth/refresh
Authorization: Bearer <access_token>
```

### Underwriting API

#### Create Underwriting Case
```http
POST /underwriting/cases
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "policy_type": "auto",
  "applicant_name": "John Doe",
  "application_id": "APP-2024-001",
  "priority": "medium"
}
```

#### Upload Documents
```http
POST /underwriting/cases/{case_id}/documents
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

file: <binary_data>
document_type: "application"
```

#### Get Risk Assessment
```http
GET /underwriting/cases/{case_id}/risk-assessment
Authorization: Bearer <access_token>
```

### Claims API

#### Create Claim
```http
POST /claims
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "policy_number": "POL-2024-001",
  "claim_type": "auto",
  "date_of_loss": "2024-01-15",
  "description": "Vehicle collision on Highway 101"
}
```

#### Upload Evidence
```http
POST /claims/{claim_id}/evidence
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

file: <binary_data>
evidence_type: "photo"
description: "Front bumper damage"
```

#### Get Damage Assessment
```http
GET /claims/{claim_id}/assessment
Authorization: Bearer <access_token>
```

### Agent Management API

#### Get Agent Status
```http
GET /agents
Authorization: Bearer <access_token>
```

#### Update Agent Configuration
```http
PUT /agents/{agent_id}/config
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "processing_threshold": 0.85,
  "max_queue_size": 100,
  "timeout_seconds": 300
}
```

### WebSocket API

#### Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Real-time update:', data);
};
```

### Error Handling

All API endpoints return standardized error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    }
  }
}
```

Common HTTP status codes:
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Rate Limited
- `500`: Internal Server Error

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Login Issues

**Problem**: Cannot log in with correct credentials
**Solutions**:
- Check if Caps Lock is enabled
- Verify email address format
- Try password reset if available
- Contact administrator for account status
- Clear browser cache and cookies

**Problem**: Session expires too quickly
**Solutions**:
- Check session timeout settings in admin panel
- Ensure stable internet connection
- Avoid opening multiple browser tabs
- Contact administrator to adjust timeout settings

#### 2. File Upload Issues

**Problem**: Files fail to upload
**Solutions**:
- Check file size (max 10MB per file)
- Verify file format is supported
- Ensure stable internet connection
- Try uploading one file at a time
- Check browser console for error messages

**Problem**: Upload progress stuck
**Solutions**:
- Refresh the page and try again
- Check internet connection stability
- Try using a different browser
- Contact support if issue persists

#### 3. Performance Issues

**Problem**: Application loads slowly
**Solutions**:
- Check internet connection speed
- Clear browser cache and cookies
- Disable browser extensions temporarily
- Try using a different browser
- Contact administrator about server performance

**Problem**: Charts and data not loading
**Solutions**:
- Refresh the page
- Check if JavaScript is enabled
- Disable ad blockers temporarily
- Try using a different browser
- Check browser console for errors

#### 4. Agent Management Issues

**Problem**: Agents showing as offline
**Solutions**:
- Check system status in admin panel
- Restart affected agents
- Verify network connectivity
- Check agent configuration settings
- Contact system administrator

**Problem**: Poor agent performance
**Solutions**:
- Review agent configuration settings
- Check system resource usage
- Analyze processing queues
- Adjust performance thresholds
- Consider scaling resources

#### 5. Data Synchronization Issues

**Problem**: Data not updating in real-time
**Solutions**:
- Check WebSocket connection status
- Refresh the page manually
- Verify network connectivity
- Check browser WebSocket support
- Contact support for server issues

### Getting Help

#### Support Channels
- **Email Support**: support@insurance-ai-system.com
- **Phone Support**: +1-800-SUPPORT (24/7)
- **Online Documentation**: Available in the help section
- **Community Forum**: Access through the help menu

#### Reporting Issues
When reporting issues, please include:
- Browser type and version
- Operating system
- Steps to reproduce the issue
- Error messages or screenshots
- User account information (without passwords)

#### System Requirements
- **Browsers**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Internet**: Broadband connection recommended
- **JavaScript**: Must be enabled
- **Cookies**: Must be enabled for authentication

---

## Appendix

### Keyboard Shortcuts
- `Ctrl + /` (or `Cmd + /`): Open help
- `Ctrl + K` (or `Cmd + K`): Global search
- `Esc`: Close modals and dialogs
- `Tab`: Navigate between form fields
- `Enter`: Submit forms or confirm actions

### Browser Compatibility
- Chrome 90 and later
- Firefox 88 and later
- Safari 14 and later
- Microsoft Edge 90 and later

### Mobile Support
- iOS Safari 14+
- Android Chrome 90+
- Responsive design for tablets and phones
- Touch-friendly interface elements

### Security Best Practices
- Use strong, unique passwords
- Enable two-factor authentication when available
- Log out when finished using the system
- Don't share login credentials
- Report suspicious activity immediately

### Data Privacy
- All data is encrypted in transit and at rest
- Personal information is protected according to privacy policies
- Users can request data deletion
- Audit logs track all data access
- Compliance with GDPR and other regulations

---

*This user guide is version 1.0 and is subject to updates as the system evolves. For the latest version, please check the online documentation.*

