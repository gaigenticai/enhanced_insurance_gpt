"""
Insurance AI Agent System - Pydantic Schemas
Production-ready API schemas for request/response validation and serialization
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from enum import Enum
import uuid
from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from pydantic.types import UUID4, Json
import structlog

logger = structlog.get_logger(__name__)

# =============================================================================
# BASE SCHEMAS
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    
    class Config:
        orm_mode = True
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            uuid.UUID: lambda v: str(v)
        }

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime
    updated_at: datetime

class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Page size")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field("desc", pattern="^(asc|desc)$", description="Sort order")

class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

# =============================================================================
# ENUMS
# =============================================================================

class UserRole(str, Enum):
    ADMIN = "admin"
    UNDERWRITER = "underwriter"
    CLAIMS_ADJUSTER = "claims_adjuster"
    BROKER = "broker"
    VIEWER = "viewer"

class WorkflowType(str, Enum):
    UNDERWRITING = "underwriting"
    CLAIMS = "claims"
    VALIDATION = "validation"
    ANALYSIS = "analysis"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class UnderwritingDecision(str, Enum):
    ACCEPT = "accept"
    DECLINE = "decline"
    REFER = "refer"
    PENDING = "pending"

class PolicyStatus(str, Enum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"

class ClaimStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    SETTLED = "settled"
    CLOSED = "closed"
    DENIED = "denied"

class ClaimPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class CommunicationType(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PHONE = "phone"
    LETTER = "letter"
    PORTAL = "portal"

class CommunicationDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"

class CommunicationStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"

# =============================================================================
# ORGANIZATION SCHEMAS
# =============================================================================

class OrganizationBase(BaseSchema):
    """Base organization schema"""
    name: str = Field(..., min_length=1, max_length=255)
    settings: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

class OrganizationCreate(OrganizationBase):
    """Organization creation schema"""
    pass

class OrganizationUpdate(BaseSchema):
    """Organization update schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class OrganizationResponse(OrganizationBase, TimestampMixin):
    """Organization response schema"""
    id: UUID4

# =============================================================================
# USER SCHEMAS
# =============================================================================

class UserBase(BaseSchema):
    """Base user schema"""
    email: EmailStr
    role: UserRole
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    settings: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8, max_length=128)
    organization_id: UUID4

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseSchema):
    """User update schema"""
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase, TimestampMixin):
    """User response schema"""
    id: UUID4
    organization_id: UUID4
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

class UserLogin(BaseSchema):
    """User login schema"""
    email: EmailStr
    password: str

class UserLoginResponse(BaseSchema):
    """User login response schema"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class PasswordChange(BaseSchema):
    """Password change schema"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)

    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

# =============================================================================
# WORKFLOW SCHEMAS
# =============================================================================

class WorkflowBase(BaseSchema):
    """Base workflow schema"""
    type: WorkflowType
    priority: int = Field(5, ge=1, le=10)
    input_data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int = Field(3, ge=0, le=10)

class WorkflowCreate(WorkflowBase):
    """Workflow creation schema"""
    organization_id: UUID4

class WorkflowUpdate(BaseSchema):
    """Workflow update schema"""
    status: Optional[WorkflowStatus] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class WorkflowResponse(WorkflowBase, TimestampMixin):
    """Workflow response schema"""
    id: UUID4
    status: WorkflowStatus
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_at: Optional[datetime] = None
    retry_count: int = 0
    created_by: Optional[UUID4] = None
    organization_id: UUID4

class AgentExecutionBase(BaseSchema):
    """Base agent execution schema"""
    agent_name: str = Field(..., max_length=100)
    agent_version: str = Field("1.0.0", max_length=20)
    input_data: Dict[str, Any]

class AgentExecutionCreate(AgentExecutionBase):
    """Agent execution creation schema"""
    workflow_id: UUID4

class AgentExecutionUpdate(BaseSchema):
    """Agent execution update schema"""
    output_data: Optional[Dict[str, Any]] = None
    status: Optional[AgentExecutionStatus] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    memory_usage_mb: Optional[int] = None
    cpu_usage_percent: Optional[Decimal] = None

class AgentExecutionResponse(AgentExecutionBase, TimestampMixin):
    """Agent execution response schema"""
    id: UUID4
    workflow_id: UUID4
    output_data: Optional[Dict[str, Any]] = None
    status: AgentExecutionStatus
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    memory_usage_mb: Optional[int] = None
    cpu_usage_percent: Optional[Decimal] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

# =============================================================================
# UNDERWRITING SCHEMAS
# =============================================================================

class UnderwritingSubmissionBase(BaseSchema):
    """Base underwriting submission schema"""
    submission_number: str = Field(..., max_length=100)
    broker_id: Optional[str] = Field(None, max_length=255)
    broker_name: Optional[str] = Field(None, max_length=255)
    insured_name: str = Field(..., max_length=255)
    insured_industry: Optional[str] = Field(None, max_length=100)
    policy_type: str = Field(..., max_length=100)
    coverage_amount: Optional[Decimal] = Field(None, ge=0)
    premium_amount: Optional[Decimal] = Field(None, ge=0)
    submission_data: Dict[str, Any]
    underwriter_notes: Optional[str] = None
    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None

class UnderwritingSubmissionCreate(UnderwritingSubmissionBase):
    """Underwriting submission creation schema"""
    workflow_id: UUID4

class UnderwritingSubmissionUpdate(BaseSchema):
    """Underwriting submission update schema"""
    risk_score: Optional[Decimal] = Field(None, ge=0, le=100)
    decision: Optional[UnderwritingDecision] = None
    decision_reasons: Optional[Dict[str, Any]] = None
    decision_confidence: Optional[Decimal] = Field(None, ge=0, le=100)
    underwriter_notes: Optional[str] = None

class UnderwritingSubmissionResponse(UnderwritingSubmissionBase, TimestampMixin):
    """Underwriting submission response schema"""
    id: UUID4
    workflow_id: UUID4
    risk_score: Optional[Decimal] = None
    decision: Optional[UnderwritingDecision] = None
    decision_reasons: Optional[Dict[str, Any]] = None
    decision_confidence: Optional[Decimal] = None

class PolicyBase(BaseSchema):
    """Base policy schema"""
    policy_number: str = Field(..., max_length=100)
    policy_data: Dict[str, Any]
    terms_and_conditions: Optional[Dict[str, Any]] = None
    coverage_details: Optional[Dict[str, Any]] = None
    exclusions: Optional[Dict[str, Any]] = None
    limits: Optional[Dict[str, Any]] = None
    deductibles: Optional[Dict[str, Any]] = None
    effective_date: date
    expiry_date: date
    renewal_date: Optional[date] = None

class PolicyCreate(PolicyBase):
    """Policy creation schema"""
    submission_id: UUID4

class PolicyUpdate(BaseSchema):
    """Policy update schema"""
    status: Optional[PolicyStatus] = None
    policy_data: Optional[Dict[str, Any]] = None
    terms_and_conditions: Optional[Dict[str, Any]] = None
    coverage_details: Optional[Dict[str, Any]] = None
    exclusions: Optional[Dict[str, Any]] = None
    limits: Optional[Dict[str, Any]] = None
    deductibles: Optional[Dict[str, Any]] = None
    renewal_date: Optional[date] = None

class PolicyResponse(PolicyBase, TimestampMixin):
    """Policy response schema"""
    id: UUID4
    submission_id: UUID4
    status: PolicyStatus

# =============================================================================
# CLAIMS SCHEMAS
# =============================================================================

class ClaimBase(BaseSchema):
    """Base claim schema"""
    claim_number: str = Field(..., max_length=100)
    policy_number: Optional[str] = Field(None, max_length=100)
    claim_type: str = Field(..., max_length=100)
    incident_date: Optional[date] = None
    reported_date: date = Field(default_factory=date.today)
    claim_amount: Optional[Decimal] = Field(None, ge=0)
    reserve_amount: Optional[Decimal] = Field(None, ge=0)
    claim_data: Dict[str, Any]
    adjuster_notes: Optional[str] = None

class ClaimCreate(ClaimBase):
    """Claim creation schema"""
    workflow_id: UUID4
    policy_id: Optional[UUID4] = None

class ClaimUpdate(BaseSchema):
    """Claim update schema"""
    status: Optional[ClaimStatus] = None
    priority: Optional[ClaimPriority] = None
    paid_amount: Optional[Decimal] = Field(None, ge=0)
    liability_assessment: Optional[Dict[str, Any]] = None
    settlement_amount: Optional[Decimal] = Field(None, ge=0)
    settlement_date: Optional[date] = None
    adjuster_id: Optional[UUID4] = None
    adjuster_notes: Optional[str] = None
    fraud_indicators: Optional[Dict[str, Any]] = None
    fraud_score: Optional[Decimal] = Field(None, ge=0, le=100)
    stp_eligible: Optional[bool] = None

class ClaimResponse(ClaimBase, TimestampMixin):
    """Claim response schema"""
    id: UUID4
    workflow_id: UUID4
    policy_id: Optional[UUID4] = None
    status: ClaimStatus
    priority: ClaimPriority
    paid_amount: Decimal = 0
    liability_assessment: Optional[Dict[str, Any]] = None
    settlement_amount: Optional[Decimal] = None
    settlement_date: Optional[date] = None
    adjuster_id: Optional[UUID4] = None
    fraud_indicators: Optional[Dict[str, Any]] = None
    fraud_score: Optional[Decimal] = None
    stp_eligible: bool = False

class EvidenceBase(BaseSchema):
    """Base evidence schema"""
    file_name: str = Field(..., max_length=255)
    file_type: str = Field(..., max_length=50)
    file_size: Optional[int] = None
    mime_type: Optional[str] = Field(None, max_length=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_sensitive: bool = False

class EvidenceCreate(EvidenceBase):
    """Evidence creation schema"""
    claim_id: UUID4
    file_key: str = Field(..., max_length=500)

class EvidenceUpdate(BaseSchema):
    """Evidence update schema"""
    analysis_results: Optional[Dict[str, Any]] = None
    analysis_confidence: Optional[Decimal] = Field(None, ge=0, le=100)
    tags: Optional[List[str]] = None
    is_sensitive: Optional[bool] = None

class EvidenceResponse(EvidenceBase, TimestampMixin):
    """Evidence response schema"""
    id: UUID4
    claim_id: UUID4
    file_key: str
    analysis_results: Optional[Dict[str, Any]] = None
    analysis_confidence: Optional[Decimal] = None
    uploaded_by: Optional[UUID4] = None

class ClaimActivityBase(BaseSchema):
    """Base claim activity schema"""
    activity_type: str = Field(..., max_length=100)
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    performed_by_agent: Optional[str] = Field(None, max_length=100)

class ClaimActivityCreate(ClaimActivityBase):
    """Claim activity creation schema"""
    claim_id: UUID4
    performed_by: Optional[UUID4] = None

class ClaimActivityResponse(ClaimActivityBase, TimestampMixin):
    """Claim activity response schema"""
    id: UUID4
    claim_id: UUID4
    performed_by: Optional[UUID4] = None

# =============================================================================
# KNOWLEDGE BASE SCHEMAS
# =============================================================================

class KnowledgeBaseBase(BaseSchema):
    """Base knowledge base schema"""
    category: str = Field(..., max_length=100)
    title: str = Field(..., max_length=500)
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    source_url: Optional[str] = Field(None, max_length=500)
    source_type: Optional[str] = Field(None, max_length=50)
    confidence_score: Optional[Decimal] = Field(None, ge=0, le=100)

class KnowledgeBaseCreate(KnowledgeBaseBase):
    """Knowledge base creation schema"""
    organization_id: UUID4

class KnowledgeBaseUpdate(BaseSchema):
    """Knowledge base update schema"""
    category: Optional[str] = Field(None, max_length=100)
    title: Optional[str] = Field(None, max_length=500)
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None
    confidence_score: Optional[Decimal] = Field(None, ge=0, le=100)

class KnowledgeBaseResponse(KnowledgeBaseBase, TimestampMixin):
    """Knowledge base response schema"""
    id: UUID4
    version: int = 1
    is_active: bool = True
    created_by: Optional[UUID4] = None
    organization_id: UUID4

class KnowledgeBaseSearch(BaseSchema):
    """Knowledge base search schema"""
    query: str = Field(..., min_length=1, max_length=1000)
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=50)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)

class KnowledgeBaseSearchResult(BaseSchema):
    """Knowledge base search result schema"""
    id: UUID4
    title: str
    content: str
    category: str
    similarity_score: float
    metadata: Dict[str, Any]

# =============================================================================
# PROMPT SCHEMAS
# =============================================================================

class PromptBase(BaseSchema):
    """Base prompt schema"""
    name: str = Field(..., max_length=255)
    prompt_text: str
    variables: List[str] = Field(default_factory=list)
    agent_name: Optional[str] = Field(None, max_length=100)
    model_name: Optional[str] = Field(None, max_length=100)
    temperature: Decimal = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4000, ge=1, le=100000)

class PromptCreate(PromptBase):
    """Prompt creation schema"""
    organization_id: UUID4

class PromptUpdate(BaseSchema):
    """Prompt update schema"""
    prompt_text: Optional[str] = None
    variables: Optional[List[str]] = None
    agent_name: Optional[str] = Field(None, max_length=100)
    model_name: Optional[str] = Field(None, max_length=100)
    temperature: Optional[Decimal] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)
    is_active: Optional[bool] = None

class PromptResponse(PromptBase, TimestampMixin):
    """Prompt response schema"""
    id: UUID4
    version: int = 1
    is_active: bool = True
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[UUID4] = None
    organization_id: UUID4

# =============================================================================
# COMMUNICATION SCHEMAS
# =============================================================================

class CommunicationBase(BaseSchema):
    """Base communication schema"""
    communication_type: CommunicationType
    direction: CommunicationDirection
    recipient_email: Optional[EmailStr] = None
    recipient_phone: Optional[str] = Field(None, max_length=20)
    sender_email: Optional[EmailStr] = None
    subject: Optional[str] = Field(None, max_length=500)
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CommunicationCreate(CommunicationBase):
    """Communication creation schema"""
    workflow_id: Optional[UUID4] = None
    claim_id: Optional[UUID4] = None
    submission_id: Optional[UUID4] = None
    template_id: Optional[UUID4] = None

class CommunicationUpdate(BaseSchema):
    """Communication update schema"""
    status: Optional[CommunicationStatus] = None
    external_id: Optional[str] = Field(None, max_length=255)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

class CommunicationResponse(CommunicationBase, TimestampMixin):
    """Communication response schema"""
    id: UUID4
    workflow_id: Optional[UUID4] = None
    claim_id: Optional[UUID4] = None
    submission_id: Optional[UUID4] = None
    template_id: Optional[UUID4] = None
    status: CommunicationStatus
    external_id: Optional[str] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    created_by: Optional[UUID4] = None

class CommunicationTemplateBase(BaseSchema):
    """Base communication template schema"""
    name: str = Field(..., max_length=255)
    template_type: str = Field(..., max_length=50)
    subject_template: Optional[str] = Field(None, max_length=500)
    content_template: str
    variables: List[str] = Field(default_factory=list)
    language: str = Field("en", max_length=10)

class CommunicationTemplateCreate(CommunicationTemplateBase):
    """Communication template creation schema"""
    organization_id: UUID4

class CommunicationTemplateUpdate(BaseSchema):
    """Communication template update schema"""
    name: Optional[str] = Field(None, max_length=255)
    template_type: Optional[str] = Field(None, max_length=50)
    subject_template: Optional[str] = Field(None, max_length=500)
    content_template: Optional[str] = None
    variables: Optional[List[str]] = None
    language: Optional[str] = Field(None, max_length=10)
    is_active: Optional[bool] = None

class CommunicationTemplateResponse(CommunicationTemplateBase, TimestampMixin):
    """Communication template response schema"""
    id: UUID4
    is_active: bool = True
    organization_id: UUID4
    created_by: Optional[UUID4] = None

# =============================================================================
# SYSTEM SCHEMAS
# =============================================================================

class SystemConfigBase(BaseSchema):
    """Base system config schema"""
    key: str = Field(..., max_length=255)
    value: Dict[str, Any]
    description: Optional[str] = None
    is_sensitive: bool = False

class SystemConfigCreate(SystemConfigBase):
    """System config creation schema"""
    organization_id: Optional[UUID4] = None

class SystemConfigUpdate(BaseSchema):
    """System config update schema"""
    value: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    is_sensitive: Optional[bool] = None

class SystemConfigResponse(SystemConfigBase, TimestampMixin):
    """System config response schema"""
    id: UUID4
    organization_id: Optional[UUID4] = None
    created_by: Optional[UUID4] = None

class FeatureFlagBase(BaseSchema):
    """Base feature flag schema"""
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    is_enabled: bool = False
    conditions: Dict[str, Any] = Field(default_factory=dict)
    rollout_percentage: int = Field(0, ge=0, le=100)

class FeatureFlagCreate(FeatureFlagBase):
    """Feature flag creation schema"""
    organization_id: Optional[UUID4] = None

class FeatureFlagUpdate(BaseSchema):
    """Feature flag update schema"""
    description: Optional[str] = None
    is_enabled: Optional[bool] = None
    conditions: Optional[Dict[str, Any]] = None
    rollout_percentage: Optional[int] = Field(None, ge=0, le=100)

class FeatureFlagResponse(FeatureFlagBase, TimestampMixin):
    """Feature flag response schema"""
    id: UUID4
    organization_id: Optional[UUID4] = None
    created_by: Optional[UUID4] = None

# =============================================================================
# HEALTH AND MONITORING SCHEMAS
# =============================================================================

class HealthCheckResponse(BaseSchema):
    """Health check response schema"""
    status: str
    timestamp: datetime
    services: Dict[str, Dict[str, Any]]
    version: str

class AgentMetricBase(BaseSchema):
    """Base agent metric schema"""
    agent_name: str = Field(..., max_length=100)
    metric_name: str = Field(..., max_length=100)
    metric_value: Decimal
    metric_unit: Optional[str] = Field(None, max_length=50)
    tags: Dict[str, Any] = Field(default_factory=dict)

class AgentMetricCreate(AgentMetricBase):
    """Agent metric creation schema"""
    organization_id: Optional[UUID4] = None

class AgentMetricResponse(AgentMetricBase):
    """Agent metric response schema"""
    id: UUID4
    timestamp: datetime
    organization_id: Optional[UUID4] = None

class SystemStatsResponse(BaseSchema):
    """System statistics response schema"""
    database_stats: Dict[str, Any]
    redis_stats: Dict[str, Any]
    agent_stats: Dict[str, Any]
    workflow_stats: Dict[str, Any]
    timestamp: datetime

# =============================================================================
# ERROR SCHEMAS
# =============================================================================

class ErrorDetail(BaseSchema):
    """Error detail schema"""
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseSchema):
    """Error response schema"""
    success: bool = False
    data: Optional[Any] = None
    meta: Dict[str, Any]
    errors: List[ErrorDetail]

class SuccessResponse(BaseSchema):
    """Success response schema"""
    success: bool = True
    data: Any
    meta: Dict[str, Any]
    errors: List[ErrorDetail] = Field(default_factory=list)

# =============================================================================
# AGENT SPECIFIC SCHEMAS
# =============================================================================

class DocumentAnalysisRequest(BaseSchema):
    """Document analysis request schema"""
    file_key: str
    file_type: str
    analysis_type: str = Field("full", pattern="^(full|basic|ocr_only)$")
    language: str = Field("en", max_length=10)
    extract_tables: bool = True
    extract_images: bool = True

class DocumentAnalysisResponse(BaseSchema):
    """Document analysis response schema"""
    extracted_text: str
    document_type: str
    confidence_score: float
    extracted_data: Dict[str, Any]
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any]

class DecisionRequest(BaseSchema):
    """Decision request schema"""
    decision_type: str = Field(..., pattern="^(underwriting|claims|validation)$")
    input_data: Dict[str, Any]
    context: Dict[str, Any] = Field(default_factory=dict)
    explain: bool = True

class DecisionResponse(BaseSchema):
    """Decision response schema"""
    decision: str
    confidence_score: float
    reasoning: List[str]
    risk_factors: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]

class EvidenceAnalysisRequest(BaseSchema):
    """Evidence analysis request schema"""
    evidence_ids: List[UUID4]
    analysis_type: str = Field("comprehensive", pattern="^(basic|comprehensive|fraud_detection)$")
    include_fraud_indicators: bool = True

class EvidenceAnalysisResponse(BaseSchema):
    """Evidence analysis response schema"""
    overall_assessment: str
    confidence_score: float
    fraud_indicators: List[Dict[str, Any]]
    damage_assessment: Optional[Dict[str, Any]] = None
    inconsistencies: List[str]
    recommendations: List[str]
    evidence_summary: List[Dict[str, Any]]

