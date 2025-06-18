"""
Insurance AI Agent System - Pydantic Schemas
Production-ready schemas for API request/response validation
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from enum import Enum
import uuid

from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from pydantic.types import UUID4, PositiveFloat, PositiveInt

# Import additional underwriting submission schemas defined in the single-module
# version of this package. These are referenced throughout the codebase but were
# missing from this package's namespace which caused ``ImportError`` when other
# modules attempted ``from backend.shared.schemas import UnderwritingSubmissionCreate``.
try:  # pragma: no cover - import guard for environments without dependencies
    from .schemas import (
        UnderwritingDecision,
        AgentExecutionStatus,
        UnderwritingSubmissionBase,
        UnderwritingSubmissionCreate,
        UnderwritingSubmissionUpdate,
        UnderwritingSubmissionResponse,
        WorkflowStatus,
        EvidenceCreate,
        EvidenceUpdate,
        ClaimActivityCreate,
        ClaimStatus,
        ClaimPriority,
        # Communication related schemas and enums from the single module
        CommunicationType,
        CommunicationDirection,
        CommunicationStatus,
        CommunicationChannel,
        CommunicationPriority,
        CommunicationCreate,
        CommunicationUpdate,
        CommunicationResponse,
        CommunicationTemplateCreate,
        CommunicationTemplateUpdate,
        CommunicationTemplateResponse,
    )
except Exception:
    pass

# Enums
class UserRoleEnum(str, Enum):
    ADMIN = "admin"
    UNDERWRITER = "underwriter"
    CLAIMS_ADJUSTER = "claims_adjuster"
    AGENT = "agent"
    VIEWER = "viewer"

class PolicyStatusEnum(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"

class ClaimStatusEnum(str, Enum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    INVESTIGATING = "investigating"
    APPROVED = "approved"
    DENIED = "denied"
    SETTLED = "settled"
    CLOSED = "closed"

class WorkflowStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DocumentTypeEnum(str, Enum):
    APPLICATION = "application"
    POLICY = "policy"
    CLAIM = "claim"
    EVIDENCE = "evidence"
    REPORT = "report"
    CORRESPONDENCE = "correspondence"

class AgentTypeEnum(str, Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COMMUNICATION = "communication"
    EVIDENCE_PROCESSING = "evidence_processing"
    COMPLIANCE = "compliance"

# Evidence related enums
class EvidenceType(str, Enum):
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    METADATA = "metadata"
    PHYSICAL_OBJECT = "physical_object"
    WITNESS_STATEMENT = "witness_statement"
    POLICE_REPORT = "police_report"
    MEDICAL_REPORT = "medical_report"
    FINANCIAL_RECORD = "financial_record"
    OTHER = "other"

class EvidenceStatus(str, Enum):
    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"
    INVALID = "invalid"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    REQUIRES_REVIEW = "requires_review"
    ARCHIVED = "archived"
    FAILED = "failed"

class EvidenceFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    TIFF = "tiff"
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MP3 = "mp3"
    WAV = "wav"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"
    XML = "xml"
    UNKNOWN = "unknown"

# Compliance related enums
class ComplianceStatus(str, Enum):
    """Possible compliance status values."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"

class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks for compliance checks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NAIC = "naic"
    STATE_INSURANCE = "state_insurance"

class ComplianceViolationType(str, Enum):
    """Types of compliance violations."""
    DATA_RETENTION = "data_retention"
    CONSENT = "consent"
    SECURITY = "security"
    PRIVACY = "privacy"
    OTHER = "other"

# Base Schemas
class BaseSchema(BaseModel):
    class Config:
        orm_mode = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            uuid.UUID: lambda v: str(v)
        }

class TimestampMixin(BaseModel):
    created_at: datetime
    updated_at: datetime

# Authentication Schemas
class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)

class LoginResponse(BaseSchema):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: 'UserResponse'

class UserLogin(BaseSchema):
    """Schema for user login credentials."""
    email: EmailStr
    password: str

class UserLoginResponse(BaseSchema):
    """Schema for the login response containing tokens and user info."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: 'UserResponse'


class TokenResponse(BaseSchema):
    """Schema for access and optional refresh tokens."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class TokenRefreshRequest(BaseModel):
    refresh_token: str

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)

# Backwards compatibility alias
class PasswordReset(PasswordResetRequest):
    """Deprecated: use PasswordResetRequest instead."""

# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRoleEnum = UserRoleEnum.VIEWER
    phone: Optional[str] = None
    department: Optional[str] = None
    employee_id: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    organization_ids: List[UUID4] = []

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    role: Optional[UserRoleEnum] = None
    is_active: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None

class UserResponse(UserBase, TimestampMixin):
    id: UUID4
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime] = None
    full_name: str
    preferences: Dict[str, Any] = {}
    notification_settings: Dict[str, Any] = {}

# Organization Schemas
class OrganizationBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    code: str = Field(..., min_length=2, max_length=50)
    description: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = Field(None, min_length=2, max_length=2)
    timezone: str = "UTC"

class OrganizationCreate(OrganizationBase):
    settings: Dict[str, Any] = {}

class OrganizationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = Field(None, min_length=2, max_length=2)
    timezone: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class OrganizationResponse(OrganizationBase, TimestampMixin):
    id: UUID4
    is_active: bool
    settings: Dict[str, Any] = {}

# Policy Schemas
class PolicyBase(BaseModel):
    policy_type: str = Field(..., min_length=1, max_length=100)
    product_line: Optional[str] = None
    insured_name: str = Field(..., min_length=1, max_length=255)
    insured_address: Optional[str] = None
    insured_phone: Optional[str] = None
    insured_email: Optional[EmailStr] = None
    coverage_amount: Optional[PositiveFloat] = None
    deductible: Optional[PositiveFloat] = None
    premium_amount: Optional[PositiveFloat] = None
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None

class PolicyCreate(PolicyBase):
    organization_id: UUID4
    policy_data: Dict[str, Any] = {}
    underwriting_notes: Optional[str] = None

    @validator('expiration_date')
    def validate_expiration_date(cls, v, values):
        if v and values.get('effective_date') and v <= values['effective_date']:
            raise ValueError('Expiration date must be after effective date')
        return v

class PolicyUpdate(BaseModel):
    policy_type: Optional[str] = Field(None, min_length=1, max_length=100)
    product_line: Optional[str] = None
    insured_name: Optional[str] = Field(None, min_length=1, max_length=255)
    insured_address: Optional[str] = None
    insured_phone: Optional[str] = None
    insured_email: Optional[EmailStr] = None
    coverage_amount: Optional[PositiveFloat] = None
    deductible: Optional[PositiveFloat] = None
    premium_amount: Optional[PositiveFloat] = None
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    status: Optional[PolicyStatusEnum] = None
    risk_score: Optional[float] = Field(None, ge=0, le=1)
    risk_factors: Optional[List[str]] = None
    policy_data: Optional[Dict[str, Any]] = None
    underwriting_notes: Optional[str] = None

class PolicyResponse(PolicyBase, TimestampMixin):
    id: UUID4
    policy_number: str
    organization_id: UUID4
    status: PolicyStatusEnum
    risk_score: Optional[float] = None
    risk_factors: List[str] = []
    policy_data: Dict[str, Any] = {}
    underwriting_notes: Optional[str] = None
    created_by: Optional[UUID4] = None

# Claim Schemas
class ClaimBase(BaseModel):
    incident_date: datetime
    incident_description: str = Field(..., min_length=10)
    incident_location: Optional[str] = None
    claimed_amount: Optional[PositiveFloat] = None
    priority: int = Field(3, ge=1, le=5)

class ClaimCreate(ClaimBase):
    policy_id: UUID4
    organization_id: UUID4
    claim_data: Dict[str, Any] = {}

class ClaimUpdate(BaseModel):
    incident_description: Optional[str] = Field(None, min_length=10)
    incident_location: Optional[str] = None
    claimed_amount: Optional[PositiveFloat] = None
    estimated_amount: Optional[PositiveFloat] = None
    approved_amount: Optional[PositiveFloat] = None
    status: Optional[ClaimStatusEnum] = None
    priority: Optional[int] = Field(None, ge=1, le=5)
    assigned_to: Optional[UUID4] = None
    investigation_notes: Optional[str] = None
    fraud_score: Optional[float] = Field(None, ge=0, le=1)
    fraud_indicators: Optional[List[str]] = None
    claim_data: Optional[Dict[str, Any]] = None
    settlement_details: Optional[Dict[str, Any]] = None

class ClaimResponse(ClaimBase, TimestampMixin):
    id: UUID4
    claim_number: str
    organization_id: UUID4
    policy_id: UUID4
    reported_date: datetime
    estimated_amount: Optional[float] = None
    approved_amount: Optional[float] = None
    paid_amount: float = 0
    status: ClaimStatusEnum
    assigned_to: Optional[UUID4] = None
    created_by: Optional[UUID4] = None
    investigation_notes: Optional[str] = None
    fraud_score: Optional[float] = None
    fraud_indicators: List[str] = []
    claim_data: Dict[str, Any] = {}
    settlement_details: Dict[str, Any] = {}

# Document Schemas
class DocumentBase(BaseModel):
    original_filename: str
    document_type: DocumentTypeEnum
    category: Optional[str] = None
    tags: List[str] = []

class DocumentCreate(DocumentBase):
    organization_id: UUID4
    policy_id: Optional[UUID4] = None
    claim_id: Optional[UUID4] = None
    access_level: str = "internal"

class DocumentUpdate(BaseModel):
    document_type: Optional[DocumentTypeEnum] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    access_level: Optional[str] = None

class DocumentResponse(DocumentBase, TimestampMixin):
    id: UUID4
    filename: str
    file_path: str
    file_size: int
    mime_type: str
    file_hash: Optional[str] = None
    organization_id: UUID4
    policy_id: Optional[UUID4] = None
    claim_id: Optional[UUID4] = None
    uploaded_by: Optional[UUID4] = None
    ocr_text: Optional[str] = None
    extracted_data: Dict[str, Any] = {}
    confidence_score: Optional[float] = None
    is_encrypted: bool = False
    access_level: str = "internal"

class DocumentUploadResponse(BaseModel):
    document_id: UUID4
    upload_url: str
    expires_at: datetime

# Workflow Schemas
class WorkflowBase(BaseModel):
    workflow_type: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    priority: int = Field(3, ge=1, le=5)

class WorkflowCreate(WorkflowBase):
    organization_id: UUID4
    input_data: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class WorkflowUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=5)
    status: Optional[WorkflowStatusEnum] = None
    current_step: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class WorkflowResponse(WorkflowBase, TimestampMixin):
    id: UUID4
    organization_id: UUID4
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    status: WorkflowStatusEnum
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    completed_steps: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = {}

# Agent Execution Schemas
class AgentExecutionCreate(BaseModel):
    workflow_id: UUID4
    agent_type: AgentTypeEnum
    agent_name: str
    step_number: int
    input_data: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class AgentExecutionUpdate(BaseModel):
    output_data: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    metadata: Optional[Dict[str, Any]] = None

class AgentExecutionResponse(TimestampMixin):
    id: UUID4
    workflow_id: UUID4
    agent_type: AgentTypeEnum
    agent_name: str
    step_number: int
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    status: str = "pending"
    success: Optional[bool] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = {}

# Analysis Schemas
class DocumentAnalysisCreate(BaseModel):
    document_id: UUID4
    agent_type: AgentTypeEnum
    model_version: Optional[str] = None

class EvidenceAnalysisCreate(BaseModel):
    """Schema for creating evidence analysis records."""
    evidence_id: UUID4
    analysis_type: str
    analysis_data: Dict[str, Any] = {}
    quality_score: Optional[float] = None
    authenticity_score: Optional[float] = None
    fraud_score: Optional[float] = None
    status: str = "completed"
    error_message: Optional[str] = None

class DocumentAnalysisResponse(TimestampMixin):
    id: UUID4
    document_id: UUID4
    agent_type: AgentTypeEnum
    extracted_text: Optional[str] = None
    extracted_entities: Dict[str, Any] = {}
    key_value_pairs: Dict[str, Any] = {}
    confidence_scores: Dict[str, Any] = {}
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    status: str = "completed"
    error_message: Optional[str] = None

# Risk Assessment Schemas
class RiskAssessmentRequest(BaseModel):
    policy_id: Optional[UUID4] = None
    claim_id: Optional[UUID4] = None
    assessment_type: str
    input_data: Dict[str, Any]

class RiskAssessmentResponse(BaseModel):
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: str  # low, medium, high
    risk_factors: List[str]
    recommendations: List[str]
    confidence: float = Field(..., ge=0, le=1)
    model_version: str
    assessment_date: datetime

# Compliance Schemas
class ComplianceCheckCreate(BaseModel):
    """Schema for creating a compliance check record."""
    entity_type: str
    entity_id: UUID4
    compliance_status: ComplianceStatus
    overall_score: float = Field(..., ge=0, le=100)
    frameworks_checked: List[RegulatoryFramework] = []
    violations: List[Dict[str, Any]] = []
    check_data: Dict[str, Any] = {}

class ComplianceCheckUpdate(BaseModel):
    """Schema for updating a compliance check record."""
    compliance_status: Optional[ComplianceStatus] = None
    overall_score: Optional[float] = Field(None, ge=0, le=100)
    violations: Optional[List[Dict[str, Any]]] = None
    check_data: Optional[Dict[str, Any]] = None

class ComplianceCheckResponse(TimestampMixin):
    """Response schema representing a compliance check."""
    id: UUID4
    entity_type: str
    entity_id: UUID4
    check_timestamp: datetime
    compliance_status: ComplianceStatus
    overall_score: float
    frameworks_checked: List[RegulatoryFramework]
    violations_count: int
    check_data: Dict[str, Any] = {}

# Communication Schemas
class CommunicationRequest(BaseModel):
    recipient_type: str  # email, sms, notification
    recipient: str
    template_id: Optional[str] = None
    subject: Optional[str] = None
    message: str
    priority: int = Field(3, ge=1, le=5)
    metadata: Dict[str, Any] = {}

class CommunicationResponse(BaseModel):
    id: UUID4
    status: str
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None

# Search and Filter Schemas
class PaginationParams(BaseModel):
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1, le=100)

class SortParams(BaseModel):
    sort_by: str = "created_at"
    sort_order: str = Field("desc", pattern="^(asc|desc)$")

class PolicyFilter(BaseModel):
    organization_id: Optional[UUID4] = None
    policy_type: Optional[str] = None
    status: Optional[PolicyStatusEnum] = None
    insured_name: Optional[str] = None
    effective_date_from: Optional[date] = None
    effective_date_to: Optional[date] = None
    risk_score_min: Optional[float] = Field(None, ge=0, le=1)
    risk_score_max: Optional[float] = Field(None, ge=0, le=1)

class ClaimFilter(BaseModel):
    organization_id: Optional[UUID4] = None
    policy_id: Optional[UUID4] = None
    status: Optional[ClaimStatusEnum] = None
    assigned_to: Optional[UUID4] = None
    incident_date_from: Optional[date] = None
    incident_date_to: Optional[date] = None
    claimed_amount_min: Optional[float] = Field(None, ge=0)
    claimed_amount_max: Optional[float] = Field(None, ge=0)
    priority: Optional[int] = Field(None, ge=1, le=5)

# Response Wrappers
class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

class APIResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    errors: Optional[List[str]] = None

class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    errors: List[str] = []
    error_code: Optional[str] = None

# Health Check Schemas
class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]
    uptime: float

class MetricsResponse(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    request_count: int
    error_rate: float
    response_time_avg: float

# Update forward references
LoginResponse.update_forward_refs()

UserLoginResponse.update_forward_refs()


TokenResponse.update_forward_refs()
UserResponse.update_forward_refs()
PolicyResponse.update_forward_refs()
ClaimResponse.update_forward_refs()
DocumentResponse.update_forward_refs()
WorkflowResponse.update_forward_refs()
AgentExecutionResponse.update_forward_refs()
DocumentAnalysisResponse.update_forward_refs()
ComplianceCheckResponse.update_forward_refs()
PaginatedResponse.update_forward_refs()
APIResponse.update_forward_refs()

