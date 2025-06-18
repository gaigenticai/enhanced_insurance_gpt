# Insurance AI Agent System - API Documentation

## Overview

The Insurance AI Agent System provides a comprehensive REST API for integration with external systems, mobile applications, and third-party services.

## Base Information

- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: JWT Bearer tokens
- **Content Type**: `application/json`
- **Rate Limiting**: 1000 requests per hour per user
- **API Version**: v1.0

## Authentication

### Login
Authenticate user and receive access token.

**Endpoint**: `POST /auth/login`

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "full_name": "John Doe",
    "role": "underwriter",
    "permissions": ["underwriting:read", "underwriting:write"]
  }
}
```

### Refresh Token
Refresh access token using refresh token.

**Endpoint**: `POST /auth/refresh`

**Headers**: `Authorization: Bearer <refresh_token>`

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "expires_in": 3600
}
```

### Logout
Invalidate current session.

**Endpoint**: `POST /auth/logout`

**Headers**: `Authorization: Bearer <access_token>`

## Underwriting API

### List Underwriting Cases
Get list of underwriting cases with pagination and filtering.

**Endpoint**: `GET /underwriting/cases`

**Query Parameters**:
- `page` (integer): Page number (default: 1)
- `limit` (integer): Items per page (default: 20, max: 100)
- `status` (string): Filter by status (pending, approved, rejected)
- `policy_type` (string): Filter by policy type
- `priority` (string): Filter by priority (low, medium, high)
- `search` (string): Search in applicant name or application ID

**Response**:
```json
{
  "cases": [
    {
      "id": 1,
      "application_id": "APP-2024-001",
      "applicant_name": "John Doe",
      "policy_type": "auto",
      "status": "pending",
      "priority": "medium",
      "risk_score": 65,
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T14:20:00Z"
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

### Create Underwriting Case
Create a new underwriting case.

**Endpoint**: `POST /underwriting/cases`

**Request Body**:
```json
{
  "application_id": "APP-2024-002",
  "applicant_name": "Jane Smith",
  "policy_type": "home",
  "priority": "high",
  "applicant_details": {
    "email": "jane.smith@email.com",
    "phone": "+1-555-0123",
    "address": "123 Main St, City, State 12345",
    "date_of_birth": "1985-03-15"
  },
  "policy_details": {
    "coverage_amount": 500000,
    "deductible": 1000,
    "property_value": 450000
  }
}
```

**Response**:
```json
{
  "id": 2,
  "application_id": "APP-2024-002",
  "status": "pending",
  "created_at": "2024-01-15T15:30:00Z",
  "workflow_id": "wf-uw-001"
}
```

### Get Underwriting Case
Get detailed information about a specific case.

**Endpoint**: `GET /underwriting/cases/{case_id}`

**Response**:
```json
{
  "id": 1,
  "application_id": "APP-2024-001",
  "applicant_name": "John Doe",
  "policy_type": "auto",
  "status": "pending",
  "priority": "medium",
  "risk_score": 65,
  "risk_factors": [
    {
      "factor": "age",
      "value": 25,
      "weight": 0.3,
      "score": 70
    },
    {
      "factor": "driving_history",
      "value": "clean",
      "weight": 0.4,
      "score": 90
    }
  ],
  "documents": [
    {
      "id": 1,
      "filename": "application.pdf",
      "type": "application",
      "status": "processed",
      "uploaded_at": "2024-01-15T10:35:00Z"
    }
  ],
  "workflow_status": {
    "current_step": "risk_assessment",
    "completed_steps": ["document_upload", "data_extraction"],
    "next_steps": ["manual_review", "decision"]
  }
}
```

### Upload Documents
Upload documents for an underwriting case.

**Endpoint**: `POST /underwriting/cases/{case_id}/documents`

**Content Type**: `multipart/form-data`

**Form Data**:
- `file`: Document file (PDF, DOC, DOCX, JPG, PNG)
- `document_type`: Type of document (application, identity, financial, etc.)
- `description`: Optional description

**Response**:
```json
{
  "id": 2,
  "filename": "drivers_license.jpg",
  "type": "identity",
  "size": 1024000,
  "status": "uploaded",
  "processing_status": "queued",
  "uploaded_at": "2024-01-15T11:00:00Z"
}
```

### Get Risk Assessment
Get AI-generated risk assessment for a case.

**Endpoint**: `GET /underwriting/cases/{case_id}/risk-assessment`

**Response**:
```json
{
  "case_id": 1,
  "risk_score": 65,
  "risk_level": "medium",
  "confidence": 0.87,
  "factors": [
    {
      "category": "demographic",
      "factors": [
        {
          "name": "age",
          "value": 25,
          "impact": "negative",
          "weight": 0.3,
          "score": 70
        }
      ]
    },
    {
      "category": "behavioral",
      "factors": [
        {
          "name": "driving_history",
          "value": "clean",
          "impact": "positive",
          "weight": 0.4,
          "score": 90
        }
      ]
    }
  ],
  "recommendations": [
    {
      "type": "premium_adjustment",
      "description": "Consider 10% premium increase due to age factor",
      "confidence": 0.75
    }
  ],
  "generated_at": "2024-01-15T12:00:00Z"
}
```

## Claims API

### List Claims
Get list of claims with pagination and filtering.

**Endpoint**: `GET /claims`

**Query Parameters**:
- `page` (integer): Page number
- `limit` (integer): Items per page
- `status` (string): Filter by status
- `claim_type` (string): Filter by claim type
- `policy_number` (string): Filter by policy number
- `date_from` (string): Filter claims from date (ISO 8601)
- `date_to` (string): Filter claims to date (ISO 8601)

**Response**:
```json
{
  "claims": [
    {
      "id": 1,
      "claim_number": "CLM-2024-001",
      "policy_number": "POL-2024-001",
      "claim_type": "auto",
      "status": "investigating",
      "amount_claimed": 15000,
      "amount_approved": null,
      "date_of_loss": "2024-01-10",
      "created_at": "2024-01-12T09:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 75,
    "pages": 4
  }
}
```

### Create Claim
Create a new insurance claim.

**Endpoint**: `POST /claims`

**Request Body**:
```json
{
  "policy_number": "POL-2024-001",
  "claim_type": "auto",
  "date_of_loss": "2024-01-10",
  "location_of_loss": "Highway 101, San Francisco, CA",
  "description": "Rear-end collision during morning traffic",
  "claimant_details": {
    "name": "John Doe",
    "phone": "+1-555-0123",
    "email": "john.doe@email.com"
  },
  "incident_details": {
    "weather_conditions": "clear",
    "road_conditions": "dry",
    "police_report": true,
    "police_report_number": "PR-2024-001"
  }
}
```

**Response**:
```json
{
  "id": 2,
  "claim_number": "CLM-2024-002",
  "status": "submitted",
  "workflow_id": "wf-claim-002",
  "created_at": "2024-01-12T10:30:00Z"
}
```

### Upload Evidence
Upload evidence files for a claim.

**Endpoint**: `POST /claims/{claim_id}/evidence`

**Content Type**: `multipart/form-data`

**Form Data**:
- `file`: Evidence file (images, videos, documents)
- `evidence_type`: Type of evidence (photo, video, document, audio)
- `description`: Description of the evidence
- `location`: GPS coordinates (optional)
- `timestamp`: When evidence was captured (optional)

**Response**:
```json
{
  "id": 1,
  "filename": "damage_front.jpg",
  "evidence_type": "photo",
  "size": 2048000,
  "status": "uploaded",
  "analysis_status": "queued",
  "metadata": {
    "dimensions": "1920x1080",
    "format": "JPEG",
    "location": "37.7749,-122.4194",
    "timestamp": "2024-01-10T08:30:00Z"
  },
  "uploaded_at": "2024-01-12T11:00:00Z"
}
```

### Get Damage Assessment
Get AI-generated damage assessment for a claim.

**Endpoint**: `GET /claims/{claim_id}/assessment`

**Response**:
```json
{
  "claim_id": 1,
  "assessment_id": "assess-001",
  "total_damage_estimate": 12500,
  "confidence": 0.92,
  "damage_categories": [
    {
      "category": "front_bumper",
      "severity": "moderate",
      "repair_cost": 3500,
      "replace_cost": 5000,
      "recommendation": "repair",
      "confidence": 0.89
    },
    {
      "category": "headlight",
      "severity": "severe",
      "repair_cost": 0,
      "replace_cost": 800,
      "recommendation": "replace",
      "confidence": 0.95
    }
  ],
  "fraud_indicators": {
    "risk_score": 15,
    "risk_level": "low",
    "indicators": []
  },
  "generated_at": "2024-01-12T13:00:00Z"
}
```

## Agent Management API

### List Agents
Get list of all AI agents and their status.

**Endpoint**: `GET /agents`

**Response**:
```json
{
  "agents": [
    {
      "id": "doc-analysis-001",
      "name": "Document Analysis Agent",
      "type": "document_analysis",
      "status": "active",
      "health": "healthy",
      "performance": {
        "success_rate": 0.95,
        "avg_processing_time": 2.3,
        "queue_size": 5,
        "processed_today": 150
      },
      "last_heartbeat": "2024-01-15T14:59:30Z"
    },
    {
      "id": "risk-assess-001",
      "name": "Risk Assessment Agent",
      "type": "risk_assessment",
      "status": "active",
      "health": "healthy",
      "performance": {
        "success_rate": 0.98,
        "avg_processing_time": 1.8,
        "queue_size": 2,
        "processed_today": 89
      },
      "last_heartbeat": "2024-01-15T14:59:45Z"
    }
  ]
}
```

### Get Agent Details
Get detailed information about a specific agent.

**Endpoint**: `GET /agents/{agent_id}`

**Response**:
```json
{
  "id": "doc-analysis-001",
  "name": "Document Analysis Agent",
  "type": "document_analysis",
  "status": "active",
  "health": "healthy",
  "configuration": {
    "max_file_size": 10485760,
    "supported_formats": ["pdf", "doc", "docx", "jpg", "png"],
    "ocr_confidence_threshold": 0.8,
    "processing_timeout": 300
  },
  "performance_metrics": {
    "success_rate": 0.95,
    "error_rate": 0.05,
    "avg_processing_time": 2.3,
    "queue_size": 5,
    "processed_today": 150,
    "processed_this_week": 1050,
    "processed_this_month": 4200
  },
  "recent_activity": [
    {
      "timestamp": "2024-01-15T14:55:00Z",
      "action": "document_processed",
      "document_id": "doc-123",
      "processing_time": 2.1,
      "status": "success"
    }
  ],
  "last_heartbeat": "2024-01-15T14:59:30Z"
}
```

### Update Agent Configuration
Update configuration settings for an agent.

**Endpoint**: `PUT /agents/{agent_id}/config`

**Request Body**:
```json
{
  "max_file_size": 15728640,
  "ocr_confidence_threshold": 0.85,
  "processing_timeout": 600,
  "enable_advanced_analysis": true
}
```

**Response**:
```json
{
  "message": "Agent configuration updated successfully",
  "agent_id": "doc-analysis-001",
  "updated_at": "2024-01-15T15:00:00Z",
  "restart_required": false
}
```

### Control Agent
Start, stop, or restart an agent.

**Endpoint**: `POST /agents/{agent_id}/control`

**Request Body**:
```json
{
  "action": "restart",
  "reason": "Configuration update"
}
```

**Response**:
```json
{
  "message": "Agent restart initiated",
  "agent_id": "doc-analysis-001",
  "action": "restart",
  "estimated_downtime": 30,
  "initiated_at": "2024-01-15T15:05:00Z"
}
```

## Administration API

### User Management

#### List Users
**Endpoint**: `GET /admin/users`

**Query Parameters**:
- `page`, `limit`: Pagination
- `role`: Filter by role
- `status`: Filter by status (active, inactive, suspended)
- `search`: Search in name or email

**Response**:
```json
{
  "users": [
    {
      "id": 1,
      "email": "john.doe@company.com",
      "full_name": "John Doe",
      "role": "underwriter",
      "status": "active",
      "last_login": "2024-01-15T14:30:00Z",
      "created_at": "2024-01-01T09:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 25,
    "pages": 2
  }
}
```

#### Create User
**Endpoint**: `POST /admin/users`

**Request Body**:
```json
{
  "email": "new.user@company.com",
  "full_name": "New User",
  "role": "claims_adjuster",
  "department": "Claims",
  "phone": "+1-555-0199",
  "send_invitation": true
}
```

### System Configuration

#### Get System Settings
**Endpoint**: `GET /admin/settings`

**Response**:
```json
{
  "general": {
    "system_name": "Insurance AI Agent System",
    "timezone": "UTC",
    "session_timeout": 3600,
    "max_file_upload_size": 10485760
  },
  "security": {
    "password_min_length": 8,
    "password_require_special": true,
    "max_login_attempts": 5,
    "account_lockout_duration": 900
  },
  "notifications": {
    "email_enabled": true,
    "sms_enabled": false,
    "webhook_enabled": true
  }
}
```

#### Update System Settings
**Endpoint**: `PUT /admin/settings`

**Request Body**:
```json
{
  "general": {
    "session_timeout": 7200,
    "max_file_upload_size": 20971520
  },
  "security": {
    "max_login_attempts": 3
  }
}
```

## WebSocket API

### Real-time Updates
Connect to WebSocket for real-time updates.

**Endpoint**: `ws://localhost:8000/ws`

**Authentication**: Send JWT token in connection query parameter
`ws://localhost:8000/ws?token=<access_token>`

**Message Types**:

#### Subscribe to Updates
```json
{
  "type": "subscribe",
  "channels": ["dashboard", "underwriting", "claims", "agents"]
}
```

#### Dashboard Updates
```json
{
  "type": "dashboard_update",
  "data": {
    "total_policies": 1250,
    "claims_processed": 89,
    "revenue": 2500000,
    "active_agents": 5
  },
  "timestamp": "2024-01-15T15:00:00Z"
}
```

#### Agent Status Updates
```json
{
  "type": "agent_status",
  "data": {
    "agent_id": "doc-analysis-001",
    "status": "active",
    "health": "healthy",
    "queue_size": 3
  },
  "timestamp": "2024-01-15T15:00:00Z"
}
```

#### Case Updates
```json
{
  "type": "case_update",
  "data": {
    "case_id": 123,
    "type": "underwriting",
    "status": "approved",
    "risk_score": 65
  },
  "timestamp": "2024-01-15T15:00:00Z"
}
```

## Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    },
    "request_id": "req-123456789"
  }
}
```

### HTTP Status Codes
- `200`: Success
- `201`: Created
- `204`: No Content
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `409`: Conflict
- `422`: Unprocessable Entity
- `429`: Too Many Requests
- `500`: Internal Server Error
- `503`: Service Unavailable

### Error Codes
- `VALIDATION_ERROR`: Input validation failed
- `AUTHENTICATION_ERROR`: Authentication failed
- `AUTHORIZATION_ERROR`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `CONFLICT`: Resource conflict
- `RATE_LIMITED`: Rate limit exceeded
- `INTERNAL_ERROR`: Internal server error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

## Rate Limiting

### Limits
- **Default**: 1000 requests per hour per user
- **Authentication**: 10 requests per minute
- **File Upload**: 50 requests per hour
- **WebSocket**: 1000 messages per hour

### Headers
Response includes rate limiting headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642262400
```

## SDK and Examples

### Python SDK Example
```python
import requests

class InsuranceAIClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def create_underwriting_case(self, data):
        response = requests.post(
            f"{self.base_url}/underwriting/cases",
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def upload_document(self, case_id, file_path, doc_type):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'document_type': doc_type}
            response = requests.post(
                f"{self.base_url}/underwriting/cases/{case_id}/documents",
                files=files,
                data=data,
                headers=self.headers
            )
        return response.json()

# Usage
client = InsuranceAIClient("http://localhost:8000/api/v1", "your-token")
case = client.create_underwriting_case({
    "applicant_name": "John Doe",
    "policy_type": "auto",
    "priority": "medium"
})
```

### JavaScript SDK Example
```javascript
class InsuranceAIClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async createClaim(data) {
        const response = await fetch(`${this.baseUrl}/claims`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        return response.json();
    }
    
    async uploadEvidence(claimId, file, evidenceType) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('evidence_type', evidenceType);
        
        const response = await fetch(
            `${this.baseUrl}/claims/${claimId}/evidence`,
            {
                method: 'POST',
                headers: {
                    'Authorization': this.headers.Authorization
                },
                body: formData
            }
        );
        return response.json();
    }
}

// Usage
const client = new InsuranceAIClient('http://localhost:8000/api/v1', 'your-token');
const claim = await client.createClaim({
    policy_number: 'POL-2024-001',
    claim_type: 'auto',
    date_of_loss: '2024-01-10'
});
```

## Testing

### Postman Collection
A complete Postman collection is available at `/docs/postman/Insurance_AI_API.postman_collection.json`

### Test Environment
- **Base URL**: `http://localhost:8000/api/v1`
- **Test User**: `test@example.com` / `test123`
- **Admin User**: `admin@zurich.com` / `admin123`

---

*This API documentation is version 1.0. For the latest updates, check the interactive documentation at `/docs` endpoint.*

