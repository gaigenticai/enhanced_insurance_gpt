"""
Insurance AI Agent System - Pydantic Schemas
Production-ready API schemas for request/response validation and serialization
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from enum import Enum, IntEnum
import uuid
from pydantic import BaseModel, Field, EmailStr, validator, root_validator, ConfigDict
from pydantic.types import UUID4, Json
import structlog

logger = structlog.get_logger(__name__)

# =============================================================================
# BASE SCHEMAS
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration"""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            uuid.UUID: lambda v: str(v),
        },
    )

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

class CommunicationChannel(str, Enum):
    """Supported communication channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"

class CommunicationPriority(IntEnum):
    """Priority levels for communications"""
    LOW = 1
    MEDIUM = 3
    HIGH = 7
    URGENT = 10

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

    @validator("password")
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
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

    @validator("new_password")
    def validate_new_password(cls, v):
        """Validate new password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
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
    title: str = Field(..., min_length=1, max_length=255)
    content: str
    content_type: str = Field(..., max_length=50)
    tags: List[str] = Field(default_factory=list)
    source_url: Optional[str] = Field(None, max_length=500)
    author: Optional[str] = Field(None, max_length=100)
    version: str = Field("1.0", max_length=20)
    is_active: bool = True

class KnowledgeBaseCreate(KnowledgeBaseBase):
    """Knowledge base creation schema"""
    pass

class KnowledgeBaseUpdate(BaseSchema):
    """Knowledge base update schema"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = None
    content_type: Optional[str] = Field(None, max_length=50)
    tags: Optional[List[str]] = None
    source_url: Optional[str] = Field(None, max_length=500)
    author: Optional[str] = Field(None, max_length=100)
    version: Optional[str] = Field(None, max_length=20)
    is_active: Optional[bool] = None

class KnowledgeBaseResponse(KnowledgeBaseBase, TimestampMixin):
    """Knowledge base response schema"""
    id: UUID4

class KnowledgeBaseSearch(BaseSchema):
    """Knowledge base search request schema"""
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=50)
    offset: int = Field(0, ge=0)
    tags: Optional[List[str]] = None
    content_type: Optional[str] = None

class KnowledgeBaseSearchResult(BaseSchema):
    """Knowledge base search result item schema"""
    id: UUID4
    title: str
    content_snippet: str
    score: float
    source_url: Optional[str] = None
    tags: List[str]

# =============================================================================
# PROMPT MANAGEMENT SCHEMAS
# =============================================================================

class PromptBase(BaseSchema):
    """Base prompt schema"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    template: str = Field(..., min_length=1)
    version: str = Field("1.0", max_length=20)
    is_active: bool = True
    tags: List[str] = Field(default_factory=list)

class PromptCreate(PromptBase):
    """Prompt creation schema"""
    pass

class PromptUpdate(BaseSchema):
    """Prompt update schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    template: Optional[str] = Field(None, min_length=1)
    version: Optional[str] = Field(None, max_length=20)
    is_active: Optional[bool] = None
    tags: Optional[List[str]] = None

class PromptResponse(PromptBase, TimestampMixin):
    """Prompt response schema"""
    id: UUID4

# =============================================================================
# COMMUNICATION SCHEMAS
# =============================================================================

class CommunicationBase(BaseSchema):
    """Base communication schema"""
    type: CommunicationType
    direction: CommunicationDirection
    subject: Optional[str] = Field(None, max_length=500)
    body: str
    sender: str = Field(..., max_length=255)
    recipients: List[str] = Field(..., min_items=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    related_claim_id: Optional[UUID4] = None
    related_policy_id: Optional[UUID4] = None

class CommunicationCreate(CommunicationBase):
    """Communication creation schema"""
    pass

class CommunicationUpdate(BaseSchema):
    """Communication update schema"""
    status: Optional[CommunicationStatus] = None
    delivery_receipt: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class CommunicationResponse(CommunicationBase, TimestampMixin):
    """Communication response schema"""
    id: UUID4
    status: CommunicationStatus
    delivery_receipt: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class CommunicationTemplateBase(BaseSchema):
    """Base communication template schema"""
    name: str = Field(..., min_length=1, max_length=100)
    type: CommunicationType
    subject_template: Optional[str] = Field(None, max_length=500)
    body_template: str
    is_active: bool = True
    tags: List[str] = Field(default_factory=list)

class CommunicationTemplateCreate(CommunicationTemplateBase):
    """Communication template creation schema"""
    pass

class CommunicationTemplateUpdate(BaseSchema):
    """Communication template update schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    type: Optional[CommunicationType] = None
    subject_template: Optional[str] = Field(None, max_length=500)
    body_template: Optional[str] = None
    is_active: Optional[bool] = None
    tags: Optional[List[str]] = None

class CommunicationTemplateResponse(CommunicationTemplateBase, TimestampMixin):
    """Communication template response schema"""
    id: UUID4

# =============================================================================
# SYSTEM CONFIGURATION SCHEMAS
# =============================================================================

class SystemConfigBase(BaseSchema):
    """Base system configuration schema"""
    key: str = Field(..., min_length=1, max_length=100)
    value: Any
    value_type: str = Field(..., max_length=50)
    description: Optional[str] = None
    is_sensitive: bool = False

class SystemConfigCreate(SystemConfigBase):
    """System configuration creation schema"""
    pass

class SystemConfigUpdate(BaseSchema):
    """System configuration update schema"""
    value: Optional[Any] = None
    value_type: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    is_sensitive: Optional[bool] = None

class SystemConfigResponse(SystemConfigBase, TimestampMixin):
    """System configuration response schema"""
    id: UUID4

# =============================================================================
# FEATURE FLAG SCHEMAS
# =============================================================================

class FeatureFlagBase(BaseSchema):
    """Base feature flag schema"""
    name: str = Field(..., min_length=1, max_length=100)
    is_enabled: bool = False
    description: Optional[str] = None
    rollout_percentage: int = Field(0, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FeatureFlagCreate(FeatureFlagBase):
    """Feature flag creation schema"""
    pass

class FeatureFlagUpdate(BaseSchema):
    """Feature flag update schema"""
    is_enabled: Optional[bool] = None
    description: Optional[str] = None
    rollout_percentage: Optional[int] = Field(None, ge=0, le=100)
    metadata: Optional[Dict[str, Any]] = None

class FeatureFlagResponse(FeatureFlagBase, TimestampMixin):
    """Feature flag response schema"""
    id: UUID4

# =============================================================================
# HEALTH AND METRICS SCHEMAS
# =============================================================================

class HealthCheckResponse(BaseSchema):
    """Health check response schema"""
    status: str
    timestamp: datetime
    service_name: str
    version: str
    dependencies: Dict[str, str]
    messages: List[str]

class AgentMetricBase(BaseSchema):
    """Base agent metric schema"""
    metric_name: str = Field(..., max_length=100)
    value: Union[int, float, Decimal]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = Field(default_factory=dict)

class AgentMetricCreate(AgentMetricBase):
    """Agent metric creation schema"""
    pass

class AgentMetricResponse(AgentMetricBase, TimestampMixin):
    """Agent metric response schema"""
    id: UUID4

class SystemStatsResponse(BaseSchema):
    """System statistics response schema"""
    cpu_usage_percent: Decimal
    memory_usage_mb: Decimal
    disk_usage_gb: Decimal
    network_io_mbps: Decimal
    active_users: int
    active_workflows: int
    pending_workflows: int
    failed_workflows: int
    timestamp: datetime

# =============================================================================
# ERROR HANDLING SCHEMAS
# =============================================================================

class ErrorDetail(BaseSchema):
    """Detailed error information"""
    loc: Optional[List[Union[str, int]]] = None
    msg: str
    type: str

class ErrorResponse(BaseSchema):
    """Standard error response schema"""
    detail: Union[str, List[ErrorDetail]]
    status_code: int = 400

class SuccessResponse(BaseSchema):
    """Standard success response schema"""
    message: str = "Operation successful"
    status_code: int = 200
    data: Optional[Dict[str, Any]] = None

# =============================================================================
# DOCUMENT PROCESSING SCHEMAS
# =============================================================================

class DocumentAnalysisRequest(BaseSchema):
    """Request schema for document analysis"""
    document_url: str = Field(..., description="URL of the document to analyze")
    document_type: str = Field(..., max_length=50, description="Type of document (e.g., 'policy', 'claim_form')")
    analysis_options: Dict[str, Any] = Field(default_factory=dict, description="Specific options for analysis")
    callback_url: Optional[str] = Field(None, description="URL to send analysis results")

class DocumentAnalysisResponse(BaseSchema):
    """Response schema for document analysis"""
    analysis_id: UUID4
    status: str
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

# =============================================================================
# DECISIONING SCHEMAS
# =============================================================================

class DecisionRequest(BaseSchema):
    """Request schema for a decision"""
    decision_type: str = Field(..., max_length=100, description="Type of decision to make (e.g., 'underwriting_accept', 'claim_approve')")
    input_data: Dict[str, Any] = Field(..., description="Input data required for the decision")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the decision")
    callback_url: Optional[str] = Field(None, description="URL to send decision results")

class DecisionResponse(BaseSchema):
    """Response schema for a decision"""
    decision_id: UUID4
    status: str
    decision: Optional[str] = None
    reasons: Optional[List[str]] = None
    confidence: Optional[Decimal] = None
    recommendations: Optional[List[str]] = None
    error_message: Optional[str] = None

# =============================================================================
# EVIDENCE ANALYSIS SCHEMAS
# =============================================================================

class EvidenceAnalysisRequest(BaseSchema):
    """Request schema for evidence analysis"""
    evidence_id: UUID4 = Field(..., description="ID of the evidence to analyze")
    analysis_type: str = Field(..., max_length=50, description="Type of analysis to perform (e.g., 'fraud_detection', 'damage_assessment')")
    analysis_options: Dict[str, Any] = Field(default_factory=dict, description="Specific options for analysis")
    callback_url: Optional[str] = Field(None, description="URL to send analysis results")

class EvidenceAnalysisResponse(BaseSchema):
    """Response schema for evidence analysis"""
    evidence_id: UUID4
    analysis_id: UUID4
    status: str
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

