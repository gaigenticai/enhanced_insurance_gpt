"""
Insurance AI Agent System - SQLAlchemy Models
Production-ready database models with relationships, validations, and serialization
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from decimal import Decimal
import uuid
from sqlalchemy import (
    Column, String, Integer, DateTime, Date, Boolean, Text, DECIMAL, 
    ForeignKey, JSON, ARRAY, CheckConstraint, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import structlog

logger = structlog.get_logger(__name__)

Base = declarative_base()

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class UUIDMixin:
    """Mixin for UUID primary key"""
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)

# =============================================================================
# CORE MODELS
# =============================================================================

class Organization(Base, UUIDMixin, TimestampMixin):
    """Organization model for multi-tenancy"""
    __tablename__ = 'organizations'
    
    name = Column(String(255), nullable=False)
    settings = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    workflows = relationship("Workflow", back_populates="organization", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="organization", cascade="all, delete-orphan")
    knowledge_base = relationship("KnowledgeBase", back_populates="organization", cascade="all, delete-orphan")
    prompts = relationship("Prompt", back_populates="organization", cascade="all, delete-orphan")
    system_configs = relationship("SystemConfig", back_populates="organization", cascade="all, delete-orphan")
    feature_flags = relationship("FeatureFlag", back_populates="organization", cascade="all, delete-orphan")
    communication_templates = relationship("CommunicationTemplate", back_populates="organization", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Organization(id={self.id}, name='{self.name}')>"

class User(Base, UUIDMixin, TimestampMixin):
    """User model with role-based access control"""
    __tablename__ = 'users'
    
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    phone = Column(String(20))
    department = Column(String(100))
    employee_id = Column(String(50))
    is_verified = Column(Boolean, default=False, nullable=False)
    preferences = Column(JSON, default=dict)
    notification_settings = Column(JSON, default=dict)
    settings = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'underwriter', 'claims_adjuster', 'agent', 'viewer')", name='valid_role'),
        Index('idx_users_email', 'email'),
        Index('idx_users_organization', 'organization_id'),
        Index('idx_users_role', 'role'),
        Index('idx_users_active', 'is_active'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    created_workflows = relationship("Workflow", back_populates="created_by_user", foreign_keys="Workflow.created_by")
    created_api_keys = relationship("APIKey", back_populates="created_by_user")
    created_knowledge_base = relationship("KnowledgeBase", back_populates="created_by_user")
    created_prompts = relationship("Prompt", back_populates="created_by_user")
    created_system_configs = relationship("SystemConfig", back_populates="created_by_user")
    created_feature_flags = relationship("FeatureFlag", back_populates="created_by_user")
    created_communication_templates = relationship("CommunicationTemplate", back_populates="created_by_user")
    audit_logs = relationship("AuditLog", back_populates="user")
    security_events = relationship("SecurityEvent", back_populates="user")
    communications = relationship("Communication", back_populates="created_by_user")
    evidence_uploads = relationship("Evidence", back_populates="uploaded_by_user")
    assigned_claims = relationship("Claim", back_populates="adjuster")
    websocket_connections = relationship("WebSocketConnection", back_populates="user")
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format"""
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.email
    
    @property
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        if self.locked_until:
            return datetime.utcnow() < self.locked_until
        return False
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"

class APIKey(Base, UUIDMixin, TimestampMixin):
    """API Key model for external integrations"""
    __tablename__ = 'api_keys'
    
    key_hash = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    permissions = Column(JSON, default=list)
    rate_limit = Column(Integer, default=1000)
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True))
    last_used = Column(DateTime(timezone=True))
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    organization = relationship("Organization", back_populates="api_keys")
    created_by_user = relationship("User", back_populates="created_api_keys")
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}')>"

class Workflow(Base, UUIDMixin, TimestampMixin):
    """Workflow model for orchestrating agent processes"""
    __tablename__ = 'workflows'
    
    type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default='pending')
    priority = Column(Integer, default=5)
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON)
    model_metadata = Column(JSON, default=dict)
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    timeout_at = Column(DateTime(timezone=True))
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("type IN ('underwriting', 'claims', 'validation', 'analysis')", name='valid_workflow_type'),
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name='valid_workflow_status'),
        CheckConstraint("priority BETWEEN 1 AND 10", name='valid_priority'),
        Index('idx_workflows_status', 'status'),
        Index('idx_workflows_type', 'type'),
        Index('idx_workflows_created_at', 'created_at'),
        Index('idx_workflows_organization', 'organization_id'),
        Index('idx_workflows_priority', 'priority'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="workflows")
    created_by_user = relationship("User", back_populates="created_workflows")
    agent_executions = relationship("AgentExecution", back_populates="workflow", cascade="all, delete-orphan")
    underwriting_submission = relationship("UnderwritingSubmission", back_populates="workflow", uselist=False)
    claim = relationship("Claim", back_populates="workflow", uselist=False)
    communications = relationship("Communication", back_populates="workflow")
    
    def __repr__(self):
        return f"<Workflow(id={self.id}, type='{self.type}', status='{self.status}')>"

class AgentExecution(Base, UUIDMixin, TimestampMixin):
    """Agent execution model for tracking individual agent runs"""
    __tablename__ = 'agent_executions'
    
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id', ondelete='CASCADE'), nullable=False)
    agent_name = Column(String(100), nullable=False)
    agent_version = Column(String(20), default='1.0.0')
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON)
    status = Column(String(50), nullable=False, default='pending')
    error_message = Column(Text)
    execution_time_ms = Column(Integer)
    memory_usage_mb = Column(Integer)
    cpu_usage_percent = Column(DECIMAL(5, 2))
    started_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name='valid_execution_status'),
        Index('idx_agent_executions_workflow', 'workflow_id'),
        Index('idx_agent_executions_agent', 'agent_name'),
        Index('idx_agent_executions_status', 'status'),
        Index('idx_agent_executions_created_at', 'created_at'),
    )
    
    # Relationships
    workflow = relationship("Workflow", back_populates="agent_executions")
    
    def __repr__(self):
        return f"<AgentExecution(id={self.id}, agent='{self.agent_name}', status='{self.status}')>"

# =============================================================================
# UNDERWRITING MODELS
# =============================================================================

class UnderwritingSubmission(Base, UUIDMixin, TimestampMixin):
    """Underwriting submission model"""
    __tablename__ = 'underwriting_submissions'
    
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id', ondelete='CASCADE'), nullable=False)
    submission_number = Column(String(100), unique=True, nullable=False)
    broker_id = Column(String(255))
    broker_name = Column(String(255))
    insured_name = Column(String(255), nullable=False)
    insured_industry = Column(String(100))
    policy_type = Column(String(100), nullable=False)
    coverage_amount = Column(DECIMAL(15, 2))
    premium_amount = Column(DECIMAL(12, 2))
    submission_data = Column(JSON, nullable=False)
    risk_score = Column(DECIMAL(5, 2))
    decision = Column(String(50))
    decision_reasons = Column(JSON)
    decision_confidence = Column(DECIMAL(5, 2))
    underwriter_notes = Column(Text)
    effective_date = Column(Date)
    expiry_date = Column(Date)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("risk_score BETWEEN 0 AND 100", name='valid_risk_score'),
        CheckConstraint("decision IN ('accept', 'decline', 'refer', 'pending')", name='valid_decision'),
        CheckConstraint("decision_confidence BETWEEN 0 AND 100", name='valid_confidence'),
        Index('idx_underwriting_submissions_number', 'submission_number'),
        Index('idx_underwriting_submissions_broker', 'broker_id'),
        Index('idx_underwriting_submissions_decision', 'decision'),
        Index('idx_underwriting_submissions_created_at', 'created_at'),
    )
    
    # Relationships
    workflow = relationship("Workflow", back_populates="underwriting_submission")
    policies = relationship("Policy", back_populates="submission", cascade="all, delete-orphan")
    communications = relationship("Communication", back_populates="submission")
    
    def __repr__(self):
        return f"<UnderwritingSubmission(id={self.id}, number='{self.submission_number}')>"

class Policy(Base, UUIDMixin, TimestampMixin):
    """Policy model"""
    __tablename__ = 'policies'
    
    submission_id = Column(UUID(as_uuid=True), ForeignKey('underwriting_submissions.id', ondelete='CASCADE'), nullable=False)
    policy_number = Column(String(100), unique=True, nullable=False)
    policy_data = Column(JSON, nullable=False)
    terms_and_conditions = Column(JSON)
    coverage_details = Column(JSON)
    exclusions = Column(JSON)
    limits = Column(JSON)
    deductibles = Column(JSON)
    status = Column(String(50), default='active', nullable=False)
    effective_date = Column(Date, nullable=False)
    expiry_date = Column(Date, nullable=False)
    renewal_date = Column(Date)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('active', 'cancelled', 'expired', 'suspended')", name='valid_policy_status'),
        Index('idx_policies_number', 'policy_number'),
        Index('idx_policies_status', 'status'),
        Index('idx_policies_effective_date', 'effective_date'),
        Index('idx_policies_expiry_date', 'expiry_date'),
    )
    
    # Relationships
    submission = relationship("UnderwritingSubmission", back_populates="policies")
    claims = relationship("Claim", back_populates="policy", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Policy(id={self.id}, number='{self.policy_number}')>"

# =============================================================================
# CLAIMS MODELS
# =============================================================================

class Claim(Base, UUIDMixin, TimestampMixin):
    """Claim model"""
    __tablename__ = 'claims'
    
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id', ondelete='CASCADE'), nullable=False)
    claim_number = Column(String(100), unique=True, nullable=False)
    policy_id = Column(UUID(as_uuid=True), ForeignKey('policies.id'))
    policy_number = Column(String(100))
    claim_type = Column(String(100), nullable=False)
    incident_date = Column(Date)
    reported_date = Column(Date, default=func.current_date())
    claim_amount = Column(DECIMAL(15, 2))
    reserve_amount = Column(DECIMAL(15, 2))
    paid_amount = Column(DECIMAL(15, 2), default=0)
    claim_data = Column(JSON, nullable=False)
    liability_assessment = Column(JSON)
    settlement_amount = Column(DECIMAL(12, 2))
    settlement_date = Column(Date)
    status = Column(String(50), default='open', nullable=False)
    priority = Column(String(20), default='medium', nullable=False)
    adjuster_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    adjuster_notes = Column(Text)
    fraud_indicators = Column(JSON)
    fraud_score = Column(DECIMAL(5, 2))
    stp_eligible = Column(Boolean, default=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('open', 'investigating', 'settled', 'closed', 'denied')", name='valid_claim_status'),
        CheckConstraint("priority IN ('low', 'medium', 'high', 'urgent')", name='valid_claim_priority'),
        CheckConstraint("fraud_score BETWEEN 0 AND 100", name='valid_fraud_score'),
        Index('idx_claims_number', 'claim_number'),
        Index('idx_claims_policy', 'policy_id'),
        Index('idx_claims_status', 'status'),
        Index('idx_claims_priority', 'priority'),
        Index('idx_claims_created_at', 'created_at'),
        Index('idx_claims_incident_date', 'incident_date'),
        Index('idx_claims_adjuster', 'adjuster_id'),
    )
    
    # Relationships
    workflow = relationship("Workflow", back_populates="claim")
    policy = relationship("Policy", back_populates="claims")
    adjuster = relationship("User", back_populates="assigned_claims")
    evidence = relationship("Evidence", back_populates="claim", cascade="all, delete-orphan")
    activities = relationship("ClaimActivity", back_populates="claim", cascade="all, delete-orphan")
    communications = relationship("Communication", back_populates="claim")
    
    def __repr__(self):
        return f"<Claim(id={self.id}, number='{self.claim_number}')>"

class Evidence(Base, UUIDMixin, TimestampMixin):
    """Evidence model for claim documentation"""
    __tablename__ = 'evidence'
    
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id', ondelete='CASCADE'), nullable=False)
    file_key = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    model_metadata = Column(JSON, default=dict)
    analysis_results = Column(JSON)
    analysis_confidence = Column(DECIMAL(5, 2))
    tags = Column(ARRAY(String))
    is_sensitive = Column(Boolean, default=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        Index('idx_evidence_claim', 'claim_id'),
        Index('idx_evidence_file_type', 'file_type'),
        Index('idx_evidence_created_at', 'created_at'),
    )
    
    # Relationships
    claim = relationship("Claim", back_populates="evidence")
    uploaded_by_user = relationship("User", back_populates="evidence_uploads")
    
    def __repr__(self):
        return f"<Evidence(id={self.id}, file_name='{self.file_name}')>"

class ClaimActivity(Base, UUIDMixin, TimestampMixin):
    """Claim activity model for timeline tracking"""
    __tablename__ = 'claim_activities'
    
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id', ondelete='CASCADE'), nullable=False)
    activity_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    details = Column(JSON, default=dict)
    performed_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    performed_by_agent = Column(String(100))
    
    # Relationships
    claim = relationship("Claim", back_populates="activities")
    performed_by_user = relationship("User")
    
    def __repr__(self):
        return f"<ClaimActivity(id={self.id}, type='{self.activity_type}')>"

# =============================================================================
# KNOWLEDGE BASE AND AI MODELS
# =============================================================================

class KnowledgeBase(Base, UUIDMixin, TimestampMixin):
    """Knowledge base model for RAG"""
    __tablename__ = 'knowledge_base'
    
    category = Column(String(100), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))
    model_metadata = Column(JSON, default=dict)
    tags = Column(ARRAY(String))
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True, nullable=False)
    source_url = Column(String(500))
    source_type = Column(String(50))
    confidence_score = Column(DECIMAL(5, 2))
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    
    # Constraints
    __table_args__ = (
        Index('idx_knowledge_base_category', 'category'),
        Index('idx_knowledge_base_active', 'is_active'),
        Index('idx_knowledge_base_embedding', 'embedding', postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'}),
        Index('idx_knowledge_base_tags', 'tags', postgresql_using='gin'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="knowledge_base")
    created_by_user = relationship("User", back_populates="created_knowledge_base")
    
    def __repr__(self):
        return f"<KnowledgeBase(id={self.id}, title='{self.title[:50]}...')>"

class Prompt(Base, UUIDMixin, TimestampMixin):
    """Prompt model for AI agent management"""
    __tablename__ = 'prompts'
    
    name = Column(String(255), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    prompt_text = Column(Text, nullable=False)
    variables = Column(JSON, default=list)
    agent_name = Column(String(100))
    model_name = Column(String(100))
    temperature = Column(DECIMAL(3, 2), default=0.7)
    max_tokens = Column(Integer, default=4000)
    is_active = Column(Boolean, default=True, nullable=False)
    performance_metrics = Column(JSON, default=dict)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('name', 'version', name='unique_prompt_version'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="prompts")
    created_by_user = relationship("User", back_populates="created_prompts")
    
    def __repr__(self):
        return f"<Prompt(id={self.id}, name='{self.name}', version={self.version})>"

class ModelPerformance(Base, UUIDMixin, TimestampMixin):
    """Model performance tracking"""
    __tablename__ = 'model_performance'
    
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    agent_name = Column(String(100), nullable=False)
    metrics = Column(JSON, nullable=False)
    evaluation_date = Column(Date, default=func.current_date())
    dataset_size = Column(Integer)
    accuracy = Column(DECIMAL(5, 4))
    precision_score = Column(DECIMAL(5, 4))
    recall = Column(DECIMAL(5, 4))
    f1_score = Column(DECIMAL(5, 4))
    
    def __repr__(self):
        return f"<ModelPerformance(id={self.id}, model='{self.model_name}')>"

# =============================================================================
# COMMUNICATION MODELS
# =============================================================================

class Communication(Base, UUIDMixin, TimestampMixin):
    """Communication model for tracking all communications"""
    __tablename__ = 'communications'
    
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id'))
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id'))
    submission_id = Column(UUID(as_uuid=True), ForeignKey('underwriting_submissions.id'))
    communication_type = Column(String(50), nullable=False)
    direction = Column(String(20), nullable=False)
    recipient_email = Column(String(255))
    recipient_phone = Column(String(20))
    sender_email = Column(String(255))
    subject = Column(String(500))
    content = Column(Text)
    template_id = Column(UUID(as_uuid=True))
    status = Column(String(50), default='pending', nullable=False)
    external_id = Column(String(255))
    model_metadata = Column(JSON, default=dict)
    sent_at = Column(DateTime(timezone=True))
    delivered_at = Column(DateTime(timezone=True))
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("communication_type IN ('email', 'sms', 'phone', 'letter', 'portal')", name='valid_communication_type'),
        CheckConstraint("direction IN ('inbound', 'outbound')", name='valid_direction'),
        CheckConstraint("status IN ('pending', 'sent', 'delivered', 'failed', 'bounced')", name='valid_communication_status'),
        Index('idx_communications_workflow', 'workflow_id'),
        Index('idx_communications_claim', 'claim_id'),
        Index('idx_communications_type', 'communication_type'),
        Index('idx_communications_status', 'status'),
    )
    
    # Relationships
    workflow = relationship("Workflow", back_populates="communications")
    claim = relationship("Claim", back_populates="communications")
    submission = relationship("UnderwritingSubmission", back_populates="communications")
    created_by_user = relationship("User", back_populates="communications")
    
    def __repr__(self):
        return f"<Communication(id={self.id}, type='{self.communication_type}')>"

class CommunicationTemplate(Base, UUIDMixin, TimestampMixin):
    """Communication template model"""
    __tablename__ = 'communication_templates'
    
    name = Column(String(255), nullable=False)
    template_type = Column(String(50), nullable=False)
    subject_template = Column(String(500))
    content_template = Column(Text, nullable=False)
    variables = Column(JSON, default=list)
    language = Column(String(10), default='en')
    is_active = Column(Boolean, default=True, nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    organization = relationship("Organization", back_populates="communication_templates")
    created_by_user = relationship("User", back_populates="created_communication_templates")
    
    def __repr__(self):
        return f"<CommunicationTemplate(id={self.id}, name='{self.name}')>"

# =============================================================================
# AUDIT AND SECURITY MODELS
# =============================================================================

class AuditLog(Base, UUIDMixin):
    """Audit log model for comprehensive tracking"""
    __tablename__ = 'audit_logs'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(255))
    old_values = Column(JSON)
    new_values = Column(JSON)
    details = Column(JSON, default=dict)
    ip_address = Column(INET)
    user_agent = Column(Text)
    session_id = Column(String(255))
    request_id = Column(String(255))
    severity = Column(String(20), default='info', nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("severity IN ('debug', 'info', 'warning', 'error', 'critical')", name='valid_severity'),
        Index('idx_audit_logs_user_action', 'user_id', 'action', 'created_at'),
        Index('idx_audit_logs_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_logs_created_at', 'created_at'),
        Index('idx_audit_logs_organization', 'organization_id'),
    )
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action='{self.action}')>"

class SecurityEvent(Base, UUIDMixin, TimestampMixin):
    """Security event model"""
    __tablename__ = 'security_events'
    
    event_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    ip_address = Column(INET)
    details = Column(JSON, nullable=False)
    resolved = Column(Boolean, default=False)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    resolved_at = Column(DateTime(timezone=True))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name='valid_security_severity'),
    )
    
    # Relationships
    user = relationship("User", back_populates="security_events", foreign_keys=[user_id])
    resolved_by_user = relationship("User", foreign_keys=[resolved_by])

    def __repr__(self):
        return f"<SecurityEvent(id={self.id}, type='{self.event_type}')>"

class WebSocketConnection(Base, UUIDMixin, TimestampMixin):
    """WebSocket connection record"""
    __tablename__ = 'websocket_connections'

    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'))
    connection_id = Column(String(100), unique=True, nullable=False, index=True)
    connected_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    disconnected_at = Column(DateTime(timezone=True))
    client_ip = Column(INET)
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True, nullable=False)
    model_metadata = Column(JSON, default=dict)

    user = relationship("User", back_populates="websocket_connections")
    organization = relationship("Organization")

    __table_args__ = (
        Index('idx_ws_conn_user', 'user_id'),
        Index('idx_ws_conn_active', 'is_active'),
    )

    def __repr__(self):
        return f"<WebSocketConnection(id={self.id}, user_id={self.user_id})>"

class SystemEvent(Base, UUIDMixin, TimestampMixin):
    """System event log"""
    __tablename__ = 'system_events'

    event_type = Column(String(100), nullable=False)
    message = Column(Text)
    level = Column(String(20), default='info')
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'))
    model_metadata = Column(JSON, default=dict)

    user = relationship("User")
    organization = relationship("Organization")

    __table_args__ = (
        Index('idx_system_events_type', 'event_type'),
        Index('idx_system_events_level', 'level'),
    )

    def __repr__(self):
        return f"<SystemEvent(id={self.id}, type='{self.event_type}')>"

# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class SystemConfig(Base, UUIDMixin, TimestampMixin):
    """System configuration model"""
    __tablename__ = 'system_config'
    
    key = Column(String(255), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(Text)
    is_sensitive = Column(Boolean, default=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'))
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    organization = relationship("Organization", back_populates="system_configs")
    created_by_user = relationship("User", back_populates="created_system_configs")
    
    def __repr__(self):
        return f"<SystemConfig(id={self.id}, key='{self.key}')>"

class FeatureFlag(Base, UUIDMixin, TimestampMixin):
    """Feature flag model"""
    __tablename__ = 'feature_flags'
    
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    is_enabled = Column(Boolean, default=False)
    conditions = Column(JSON, default=dict)
    rollout_percentage = Column(Integer, default=0)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'))
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("rollout_percentage BETWEEN 0 AND 100", name='valid_rollout_percentage'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="feature_flags")
    created_by_user = relationship("User", back_populates="created_feature_flags")
    
    def __repr__(self):
        return f"<FeatureFlag(id={self.id}, name='{self.name}')>"

# =============================================================================
# PERFORMANCE AND MONITORING MODELS
# =============================================================================

class AgentMetric(Base, UUIDMixin):
    """Agent performance metrics model"""
    __tablename__ = 'agent_metrics'
    
    agent_name = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(DECIMAL(15, 4), nullable=False)
    metric_unit = Column(String(50))
    tags = Column(JSON, default=dict)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'))
    
    # Constraints
    __table_args__ = (
        Index('idx_agent_metrics_agent_timestamp', 'agent_name', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<AgentMetric(id={self.id}, agent='{self.agent_name}', metric='{self.metric_name}')>"

class HealthCheck(Base, UUIDMixin):
    """Health check model"""
    __tablename__ = 'health_checks'
    
    service_name = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)
    response_time_ms = Column(Integer)
    details = Column(JSON, default=dict)
    checked_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('healthy', 'degraded', 'unhealthy')", name='valid_health_status'),
        Index('idx_health_checks_service_timestamp', 'service_name', 'checked_at'),
    )
    
    def __repr__(self):
        return f"<HealthCheck(id={self.id}, service='{self.service_name}', status='{self.status}')>"

