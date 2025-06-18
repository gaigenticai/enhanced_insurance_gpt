# Insurance AI Agent System - API Documentation

## Overview

The Insurance AI Agent System provides a comprehensive RESTful API for managing insurance operations, including policies, claims, documents, evidence processing, and AI agent interactions. This API is designed for production use with enterprise-grade security, monitoring, and scalability.

## Base URL

```
Production: https://api.insurance-ai.zurich.com
Development: http://localhost:8000
```

## Authentication

The API uses JWT (JSON Web Tokens) for authentication with OAuth 2.0 integration.

### Authentication Endpoints

#### POST /auth/login
Authenticate user and receive access token.

**Request Body:**
```json
{
  "email": "user@zurich.com",
  "password": "password123",
  "remember_me": false
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "id": "uuid",
    "email": "user@zurich.com",
    "first_name": "John",
    "last_name": "Doe",
    "roles": ["underwriter"]
  }
}
```

#### POST /auth/refresh
Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### POST /auth/logout
Invalidate current session.

**Headers:**
```
Authorization: Bearer <access_token>
```

### OAuth 2.0 Integration

#### GET /auth/oauth/authorize
Initiate OAuth flow with external providers.

**Query Parameters:**
- `provider`: oauth provider (google, microsoft, okta)
- `redirect_uri`: callback URL after authentication

#### POST /auth/oauth/callback
Handle OAuth callback and exchange code for tokens.

## User Management

### GET /users
List users with pagination and filtering.

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)
- `search`: Search term for name or email
- `role`: Filter by role
- `status`: Filter by status (active, inactive, suspended)

**Response:**
```json
{
  "users": [
    {
      "id": "uuid",
      "email": "user@zurich.com",
      "first_name": "John",
      "last_name": "Doe",
      "status": "active",
      "roles": ["underwriter"],
      "last_login": "2024-06-14T10:30:00Z",
      "created_at": "2024-01-15T09:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8
  }
}
```

### POST /users
Create new user.

**Request Body:**
```json
{
  "email": "newuser@zurich.com",
  "first_name": "Jane",
  "last_name": "Smith",
  "phone": "+1-555-0123",
  "roles": ["claims_adjuster"],
  "send_invitation": true
}
```

### GET /users/{user_id}
Get user details by ID.

### PUT /users/{user_id}
Update user information.

### DELETE /users/{user_id}
Deactivate user (soft delete).

## Customer Management

### GET /customers
List customers with advanced filtering.

**Query Parameters:**
- `page`, `limit`: Pagination
- `search`: Search name, email, or customer number
- `risk_profile`: Filter by risk level
- `customer_since`: Filter by registration date range
- `sort`: Sort field (name, customer_since, lifetime_value)
- `order`: Sort order (asc, desc)

### POST /customers
Create new customer.

**Request Body:**
```json
{
  "first_name": "John",
  "last_name": "Doe",
  "email": "john.doe@email.com",
  "phone": "+1-555-0123",
  "date_of_birth": "1985-03-15",
  "address": {
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "zip_code": "10001",
    "country": "US"
  },
  "preferences": {
    "communication_method": "email",
    "language": "en",
    "marketing_consent": true
  }
}
```

### GET /customers/{customer_id}
Get customer details including policies and claims.

### PUT /customers/{customer_id}
Update customer information.

### GET /customers/{customer_id}/policies
Get all policies for a customer.

### GET /customers/{customer_id}/claims
Get all claims for a customer.

## Policy Management

### GET /policies
List policies with comprehensive filtering.

**Query Parameters:**
- `page`, `limit`: Pagination
- `customer_id`: Filter by customer
- `policy_type`: Filter by policy type
- `status`: Filter by status
- `effective_date_from`, `effective_date_to`: Date range
- `expiring_soon`: Boolean for policies expiring within 30 days
- `agent_id`: Filter by agent
- `underwriter_id`: Filter by underwriter

### POST /policies
Create new policy.

**Request Body:**
```json
{
  "customer_id": "uuid",
  "policy_type_id": 1,
  "effective_date": "2024-07-01",
  "expiration_date": "2025-07-01",
  "premium_amount": 1200.00,
  "deductible": 500.00,
  "coverage_limits": {
    "bodily_injury": 100000,
    "property_damage": 50000,
    "comprehensive": 25000
  },
  "policy_terms": {
    "payment_frequency": "monthly",
    "auto_renewal": true
  },
  "agent_id": "uuid"
}
```

### GET /policies/{policy_id}
Get policy details with full coverage information.

### PUT /policies/{policy_id}
Update policy (creates endorsement for significant changes).

### POST /policies/{policy_id}/endorsements
Create policy endorsement.

### GET /policies/{policy_id}/documents
Get all documents associated with policy.

### POST /policies/{policy_id}/renew
Initiate policy renewal process.

### POST /policies/{policy_id}/cancel
Cancel policy with reason and effective date.

## Claims Management

### GET /claims
List claims with advanced filtering and sorting.

**Query Parameters:**
- `page`, `limit`: Pagination
- `policy_id`: Filter by policy
- `customer_id`: Filter by customer
- `status`: Filter by status
- `claim_type`: Filter by claim type
- `incident_date_from`, `incident_date_to`: Date range
- `adjuster_id`: Filter by adjuster
- `priority`: Filter by priority level
- `fraud_score_min`, `fraud_score_max`: Fraud score range
- `amount_min`, `amount_max`: Claim amount range

### POST /claims
Create new claim.

**Request Body:**
```json
{
  "policy_id": "uuid",
  "claim_type_id": 1,
  "incident_date": "2024-06-10",
  "description": "Vehicle collision at intersection of Main St and Oak Ave",
  "location": {
    "address": "Main St & Oak Ave, New York, NY",
    "coordinates": {
      "latitude": 40.7128,
      "longitude": -74.0060
    }
  },
  "estimated_amount": 15000.00,
  "priority": "normal",
  "participants": [
    {
      "type": "claimant",
      "first_name": "John",
      "last_name": "Doe",
      "contact_info": {
        "phone": "+1-555-0123",
        "email": "john.doe@email.com"
      },
      "role_description": "Driver of insured vehicle"
    }
  ]
}
```

### GET /claims/{claim_id}
Get claim details with all related information.

### PUT /claims/{claim_id}
Update claim information.

### POST /claims/{claim_id}/assign
Assign claim to adjuster.

**Request Body:**
```json
{
  "adjuster_id": "uuid",
  "priority": "high",
  "notes": "Complex case requiring immediate attention"
}
```

### POST /claims/{claim_id}/reserve
Set or update claim reserve amount.

### POST /claims/{claim_id}/payment
Process claim payment.

**Request Body:**
```json
{
  "amount": 12500.00,
  "payment_type": "settlement",
  "payee": "John Doe",
  "description": "Final settlement for vehicle damage",
  "supporting_documents": ["uuid1", "uuid2"]
}
```

### GET /claims/{claim_id}/timeline
Get claim activity timeline.

### POST /claims/{claim_id}/notes
Add note to claim.

### GET /claims/{claim_id}/documents
Get all claim documents.

### GET /claims/{claim_id}/evidence
Get all evidence items for claim.

## Document Management

### POST /documents/upload
Upload document with metadata.

**Request (multipart/form-data):**
- `file`: Document file
- `document_type_id`: Document type ID
- `title`: Document title
- `description`: Optional description
- `related_entity_type`: Entity type (policy, claim, customer)
- `related_entity_id`: Entity ID
- `tags`: Comma-separated tags

**Response:**
```json
{
  "id": "uuid",
  "title": "Policy Application",
  "file_name": "policy_app_12345.pdf",
  "file_size": 2048576,
  "mime_type": "application/pdf",
  "status": "uploaded",
  "upload_url": "https://storage.zurich.com/documents/uuid",
  "processing_status": "queued"
}
```

### GET /documents
List documents with filtering.

### GET /documents/{document_id}
Get document metadata and download URL.

### PUT /documents/{document_id}
Update document metadata.

### DELETE /documents/{document_id}
Delete document (soft delete with retention policy).

### GET /documents/{document_id}/download
Download document file.

### POST /documents/{document_id}/process
Trigger AI processing for document.

### GET /documents/{document_id}/analysis
Get AI analysis results for document.

## Evidence Management

### POST /evidence
Submit evidence item.

**Request Body:**
```json
{
  "claim_id": "uuid",
  "evidence_type_id": 1,
  "title": "Accident Scene Photos",
  "description": "Photos taken immediately after collision",
  "source": "Mobile phone camera",
  "collection_date": "2024-06-10",
  "location": {
    "address": "Main St & Oak Ave",
    "coordinates": {
      "latitude": 40.7128,
      "longitude": -74.0060
    }
  },
  "chain_of_custody": [
    {
      "timestamp": "2024-06-10T14:30:00Z",
      "action": "collected",
      "person": "John Doe",
      "location": "Accident scene"
    }
  ]
}
```

### GET /evidence
List evidence items with filtering.

### GET /evidence/{evidence_id}
Get evidence details and analysis results.

### PUT /evidence/{evidence_id}
Update evidence metadata.

### POST /evidence/{evidence_id}/analyze
Trigger AI analysis for evidence.

### GET /evidence/{evidence_id}/analysis
Get AI analysis results.

### POST /evidence/{evidence_id}/verify
Verify evidence integrity and authenticity.

## AI Agent Operations

### GET /ai/agents
List available AI agents and their capabilities.

**Response:**
```json
{
  "agents": [
    {
      "id": "uuid",
      "name": "Document Analysis Agent",
      "type": "document_processor",
      "status": "active",
      "capabilities": [
        "ocr",
        "text_extraction",
        "document_classification",
        "entity_extraction"
      ],
      "performance_metrics": {
        "accuracy": 0.95,
        "processing_time_avg": 2.3,
        "uptime": 0.999
      }
    }
  ]
}
```

### POST /ai/agents/{agent_id}/process
Submit processing request to AI agent.

**Request Body:**
```json
{
  "operation_type": "document_analysis",
  "input_data": {
    "document_id": "uuid",
    "analysis_type": "full",
    "extract_entities": true
  },
  "priority": "normal",
  "callback_url": "https://api.client.com/webhook/ai-results"
}
```

### GET /ai/operations/{operation_id}
Get AI operation status and results.

### GET /ai/agents/{agent_id}/metrics
Get agent performance metrics.

### POST /ai/agents/{agent_id}/train
Initiate agent training with new data.

## Workflow Management

### GET /workflows/definitions
List workflow definitions.

### POST /workflows/definitions
Create new workflow definition.

### GET /workflows/instances
List workflow instances with filtering.

### POST /workflows/instances
Start new workflow instance.

**Request Body:**
```json
{
  "workflow_definition_id": "uuid",
  "entity_type": "claim",
  "entity_id": "uuid",
  "input_data": {
    "claim_amount": 15000,
    "claim_type": "auto_collision",
    "priority": "normal"
  },
  "priority": "normal"
}
```

### GET /workflows/instances/{instance_id}
Get workflow instance details and progress.

### POST /workflows/instances/{instance_id}/pause
Pause workflow execution.

### POST /workflows/instances/{instance_id}/resume
Resume paused workflow.

### POST /workflows/instances/{instance_id}/cancel
Cancel workflow execution.

## Notification Management

### GET /notifications
List notifications for current user.

### POST /notifications
Send notification.

**Request Body:**
```json
{
  "recipient_type": "user",
  "recipient_id": "uuid",
  "type": "email",
  "template_id": "uuid",
  "subject": "Claim Update",
  "content": "Your claim has been approved",
  "priority": "normal",
  "scheduled_for": "2024-06-14T15:00:00Z"
}
```

### PUT /notifications/{notification_id}/read
Mark notification as read.

### GET /notifications/templates
List notification templates.

### POST /notifications/templates
Create notification template.

## Reporting and Analytics

### GET /reports/dashboard
Get dashboard metrics and KPIs.

**Response:**
```json
{
  "metrics": {
    "total_policies": 15420,
    "active_claims": 342,
    "pending_documents": 89,
    "ai_processing_queue": 23
  },
  "trends": {
    "new_policies_30d": 156,
    "claims_ratio": 0.023,
    "avg_settlement_time": 14.5,
    "customer_satisfaction": 4.2
  },
  "alerts": [
    {
      "type": "high_fraud_score",
      "message": "Claim CLM-2024-001234 has high fraud score",
      "severity": "warning"
    }
  ]
}
```

### GET /reports/policies
Generate policy reports with filters.

### GET /reports/claims
Generate claims reports with analytics.

### GET /reports/financial
Generate financial reports and summaries.

### POST /reports/custom
Generate custom report with specified parameters.

### GET /reports/{report_id}/download
Download generated report file.

## System Administration

### GET /admin/system/health
Get system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-06-14T12:00:00Z",
  "services": {
    "database": {
      "status": "healthy",
      "response_time": 12,
      "connections": 45
    },
    "redis": {
      "status": "healthy",
      "memory_usage": "256MB",
      "hit_ratio": 0.95
    },
    "ai_agents": {
      "status": "healthy",
      "active_agents": 8,
      "queue_size": 23
    }
  }
}
```

### GET /admin/system/metrics
Get system performance metrics.

### GET /admin/audit/logs
Get audit log entries with filtering.

### GET /admin/config
Get system configuration.

### PUT /admin/config
Update system configuration.

### POST /admin/backup
Initiate system backup.

### GET /admin/backup/status
Get backup status and history.

## Error Handling

All API endpoints return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ],
    "request_id": "req_12345",
    "timestamp": "2024-06-14T12:00:00Z"
  }
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

API requests are rate limited per user:
- 60 requests per minute
- 1000 requests per hour
- 10000 requests per day

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1623672000
```

## Webhooks

The system supports webhooks for real-time notifications:

### Webhook Events

- `policy.created`
- `policy.updated`
- `policy.cancelled`
- `claim.created`
- `claim.updated`
- `claim.approved`
- `claim.denied`
- `document.processed`
- `evidence.analyzed`
- `workflow.completed`
- `ai.operation.completed`

### Webhook Payload

```json
{
  "event": "claim.approved",
  "timestamp": "2024-06-14T12:00:00Z",
  "data": {
    "claim_id": "uuid",
    "policy_id": "uuid",
    "approved_amount": 12500.00,
    "approver_id": "uuid"
  },
  "webhook_id": "uuid"
}
```

## SDK and Libraries

Official SDKs are available for:
- Python: `pip install zurich-insurance-api`
- JavaScript/Node.js: `npm install @zurich/insurance-api`
- Java: Maven/Gradle dependency
- C#/.NET: NuGet package

## Support and Resources

- API Documentation: https://docs.api.insurance-ai.zurich.com
- Developer Portal: https://developers.zurich.com
- Support: api-support@zurich.com
- Status Page: https://status.api.insurance-ai.zurich.com

