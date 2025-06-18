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
    settings = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'underwriter', 'claims_adjuster', 'broker', 'viewer')", name='valid_role'),
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
    metadata_json = Column(JSON, default=dict)
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
    evidences = relationship("Evidence", back_populates="claim", cascade="all, delete-orphan")
    activities = relationship("ClaimActivity", back_populates="claim", cascade="all, delete-orphan")
    communications = relationship("Communication", back_populates="claim")
    adjuster = relationship("User", back_populates="assigned_claims", foreign_keys=[adjuster_id])
    
    def __repr__(self):
        return f"<Claim(id={self.id}, number='{self.claim_number}', status='{self.status}')>"

class Evidence(Base, UUIDMixin, TimestampMixin):
    """Evidence model for claims"""
    __tablename__ = 'evidences'
    
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id', ondelete='CASCADE'), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_key = Column(String(500), nullable=False) # S3 key or similar
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    metadata_json = Column(JSON, default=dict)
    analysis_results = Column(JSON)
    analysis_confidence = Column(DECIMAL(5, 2))
    tags = Column(ARRAY(String), default=list)
    is_sensitive = Column(Boolean, default=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        Index('idx_evidences_claim', 'claim_id'),
        Index('idx_evidences_file_type', 'file_type'),
        Index('idx_evidences_sensitive', 'is_sensitive'),
    )
    
    # Relationships
    claim = relationship("Claim", back_populates="evidences")
    uploaded_by_user = relationship("User", back_populates="evidence_uploads")

    def __repr__(self):
        return f"<Evidence(id={self.id}, file_name='{self.file_name}')>"

class EvidenceAnalysis(Base, UUIDMixin, TimestampMixin):
    """Detailed analysis results for a single evidence item"""
    __tablename__ = 'evidence_analyses'

    evidence_id = Column(UUID(as_uuid=True), ForeignKey('evidences.id', ondelete='CASCADE'), nullable=False)
    analysis_type = Column(String(100), nullable=False)
    analysis_data = Column(JSON, default=dict)
    quality_score = Column(DECIMAL(5, 2))
    authenticity_score = Column(DECIMAL(5, 2))
    fraud_score = Column(DECIMAL(5, 2))
    status = Column(String(50), default='completed')
    error_message = Column(Text)

    # Relationships
    evidence = relationship("Evidence")

    __table_args__ = (
        Index('idx_evidence_analysis_evidence', 'evidence_id'),
        Index('idx_evidence_analysis_type', 'analysis_type'),
    )

    def __repr__(self):
        return (
            f"<EvidenceAnalysis(id={self.id}, evidence_id={self.evidence_id},"
            f" type='{self.analysis_type}')>"
        )

class ClaimActivity(Base, UUIDMixin, TimestampMixin):
    """Claim activity log model"""
    __tablename__ = 'claim_activities'
    
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id', ondelete='CASCADE'), nullable=False)
    activity_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    details = Column(JSON, default=dict)
    performed_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    performed_by_agent = Column(String(100)) # Name of the agent if automated
    
    # Constraints
    __table_args__ = (
        Index('idx_claim_activities_claim', 'claim_id'),
        Index('idx_claim_activities_type', 'activity_type'),
        Index('idx_claim_activities_performed_by', 'performed_by'),
    )
    
    # Relationships
    claim = relationship("Claim", back_populates="activities")
    performed_by_user = relationship("User")
    
    def __repr__(self):
        return f"<ClaimActivity(id={self.id}, type='{self.activity_type}')>"

# =============================================================================
# KNOWLEDGE BASE MODELS
# =============================================================================

class KnowledgeBase(Base, UUIDMixin, TimestampMixin):
    """Knowledge Base model for storing articles, FAQs, etc."""
    __tablename__ = 'knowledge_base'
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'))
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False)
    tags = Column(ARRAY(String), default=list)
    source_url = Column(String(500))
    author = Column(String(100))
    version = Column(String(20), default='1.0')
    is_active = Column(Boolean, default=True)
    embedding = Column(Vector(1536)) # For vector similarity search
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        Index('idx_knowledge_base_organization', 'organization_id'),
        Index('idx_knowledge_base_title', 'title'),
        Index('idx_knowledge_base_content_type', 'content_type'),
        Index('idx_knowledge_base_tags', 'tags', postgresql_using='gin'),
        Index('idx_knowledge_base_active', 'is_active'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="knowledge_base")
    created_by_user = relationship("User", back_populates="created_knowledge_base")
    
    def __repr__(self):
        return f"<KnowledgeBase(id={self.id}, title='{self.title}')>"

# =============================================================================
# PROMPT MANAGEMENT MODELS
# =============================================================================

class Prompt(Base, UUIDMixin, TimestampMixin):
    """Prompt model for managing AI prompts"""
    __tablename__ = 'prompts'
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'))
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    template = Column(Text, nullable=False)
    version = Column(String(20), default='1.0')
    is_active = Column(Boolean, default=True)
    tags = Column(ARRAY(String), default=list)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        Index('idx_prompts_organization', 'organization_id'),
        Index('idx_prompts_name', 'name'),
        Index('idx_prompts_active', 'is_active'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="prompts")
    created_by_user = relationship("User", back_populates="created_prompts")
    
    def __repr__(self):
        return f"<Prompt(id={self.id}, name='{self.name}')>"

# =============================================================================
# COMMUNICATION MODELS
# =============================================================================

class Communication(Base, UUIDMixin, TimestampMixin):
    """Communication log model"""
    __tablename__ = 'communications'
    
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id'))
    submission_id = Column(UUID(as_uuid=True), ForeignKey('underwriting_submissions.id'))
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id'))
    policy_id = Column(UUID(as_uuid=True), ForeignKey('policies.id'))
    type = Column(String(50), nullable=False)
    direction = Column(String(50), nullable=False)
    subject = Column(String(500))
    body = Column(Text, nullable=False)
    sender = Column(String(255), nullable=False)
    recipients = Column(ARRAY(String), nullable=False)
    status = Column(String(50), default='pending')
    delivery_receipt = Column(JSON)
    error_message = Column(Text)
    metadata_json = Column(JSON, default=dict)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("type IN ('email', 'sms', 'phone', 'letter', 'portal')", name='valid_communication_type'),
        CheckConstraint("direction IN ('inbound', 'outbound')", name='valid_communication_direction'),
        CheckConstraint("status IN ('pending', 'sent', 'delivered', 'failed', 'bounced')", name='valid_communication_status'),
        Index('idx_communications_workflow', 'workflow_id'),
        Index('idx_communications_submission', 'submission_id'),
        Index('idx_communications_claim', 'claim_id'),
        Index('idx_communications_policy', 'policy_id'),
        Index('idx_communications_type', 'type'),
        Index('idx_communications_direction', 'direction'),
        Index('idx_communications_status', 'status'),
    )
    
    # Relationships
    workflow = relationship("Workflow", back_populates="communications")
    submission = relationship("UnderwritingSubmission", back_populates="communications")
    claim = relationship("Claim", back_populates="communications")
    policy = relationship("Policy") # No back_populates for policy as it's a generic link
    created_by_user = relationship("User", back_populates="communications")
    
    def __repr__(self):
        return f"<Communication(id={self.id}, type='{self.type}', subject='{self.subject}')>"

class CommunicationTemplate(Base, UUIDMixin, TimestampMixin):
    """Communication Template model"""
    __tablename__ = 'communication_templates'
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'))
    name = Column(String(100), nullable=False, unique=True)
    type = Column(String(50), nullable=False)
    subject_template = Column(String(500))
    body_template = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    tags = Column(ARRAY(String), default=list)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("type IN ('email', 'sms', 'phone', 'letter', 'portal')", name='valid_template_type'),
        Index('idx_communication_templates_organization', 'organization_id'),
        Index('idx_communication_templates_name', 'name'),
        Index('idx_communication_templates_type', 'type'),
        Index('idx_communication_templates_active', 'is_active'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="communication_templates")
    created_by_user = relationship("User", back_populates="created_communication_templates")
    
    def __repr__(self):
        return f"<CommunicationTemplate(id={self.id}, name='{self.name}')>"

# =============================================================================
# SYSTEM CONFIGURATION MODELS
# =============================================================================

class SystemConfig(Base, UUIDMixin, TimestampMixin):
    """System Configuration model"""
    __tablename__ = 'system_configs'
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'))
    key = Column(String(100), nullable=False, unique=True)
    value = Column(JSON, nullable=False)
    value_type = Column(String(50), nullable=False)
    description = Column(Text)
    is_sensitive = Column(Boolean, default=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        Index('idx_system_configs_organization', 'organization_id'),
        Index('idx_system_configs_key', 'key'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="system_configs")
    created_by_user = relationship("User", back_populates="created_system_configs")
    
    def __repr__(self):
        return f"<SystemConfig(id={self.id}, key='{self.key}')>"

# =============================================================================
# FEATURE FLAG MODELS
# =============================================================================

class FeatureFlag(Base, UUIDMixin, TimestampMixin):
    """Feature Flag model"""
    __tablename__ = 'feature_flags'
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'))
    name = Column(String(100), nullable=False, unique=True)
    is_enabled = Column(Boolean, default=False)
    description = Column(Text)
    rollout_percentage = Column(Integer, default=0)
    metadata_json = Column(JSON, default=dict)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("rollout_percentage BETWEEN 0 AND 100", name='valid_rollout_percentage'),
        Index('idx_feature_flags_organization', 'organization_id'),
        Index('idx_feature_flags_name', 'name'),
        Index('idx_feature_flags_enabled', 'is_enabled'),
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="feature_flags")
    created_by_user = relationship("User", back_populates="created_feature_flags")
    
    def __repr__(self):
        return f"<FeatureFlag(id={self.id}, name='{self.name}', enabled={self.is_enabled})>"

# =============================================================================
# AUDIT AND SECURITY MODELS
# =============================================================================

class AuditLog(Base, UUIDMixin, TimestampMixin):
    """Audit Log model"""
    __tablename__ = 'audit_logs'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'))
    action = Column(String(255), nullable=False)
    entity_type = Column(String(100))
    entity_id = Column(UUID(as_uuid=True))
    details = Column(JSON)
    ip_address = Column(INET)
    user_agent = Column(Text)
    
    # Constraints
    __table_args__ = (
        Index('idx_audit_logs_user', 'user_id'),
        Index('idx_audit_logs_organization', 'organization_id'),
        Index('idx_audit_logs_action', 'action'),
        Index('idx_audit_logs_entity', 'entity_type', 'entity_id'),
        Index('idx_audit_logs_created_at', 'created_at'),
    )
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    organization = relationship("Organization") # No back_populates for organization as it's a generic link
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action='{self.action}', user_id={self.user_id})>"

class SecurityEvent(Base, UUIDMixin, TimestampMixin):
    """Security Event model"""
    __tablename__ = 'security_events'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'))
    event_type = Column(String(100), nullable=False)
    description = Column(Text)
    severity = Column(String(50))
    details = Column(JSON)
    ip_address = Column(INET)
    user_agent = Column(Text)
    resolved = Column(Boolean, default=False)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    resolved_at = Column(DateTime(timezone=True))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name='valid_severity'),
        Index('idx_security_events_user', 'user_id'),
        Index('idx_security_events_organization', 'organization_id'),
        Index('idx_security_events_type', 'event_type'),
        Index('idx_security_events_severity', 'severity'),
        Index('idx_security_events_created_at', 'created_at'),
    )
    
    # Relationships
    user = relationship("User", back_populates="security_events", foreign_keys=[user_id])
    organization = relationship("Organization")
    resolver = relationship("User", foreign_keys=[resolved_by])
    
    def __repr__(self):
        return f"<SecurityEvent(id={self.id}, type='{self.event_type}', severity='{self.severity}')>"

# =============================================================================
# METRICS AND MONITORING MODELS
# =============================================================================

class AgentMetric(Base, UUIDMixin, TimestampMixin):
    """Agent Metric model for performance monitoring"""
    __tablename__ = 'agent_metrics'
    
    metric_name = Column(String(100), nullable=False)
    value = Column(DECIMAL(15, 4), nullable=False)
    tags = Column(JSON, default=dict)
    
    # Constraints
    __table_args__ = (
        Index('idx_agent_metrics_name', 'metric_name'),
        Index('idx_agent_metrics_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<AgentMetric(id={self.id}, name='{self.metric_name}', value={self.value})>"

# =============================================================================
# DOCUMENT PROCESSING MODELS
# =============================================================================

class Document(Base, UUIDMixin, TimestampMixin):
    """Document model for storing processed documents"""
    __tablename__ = 'documents'
    
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id'))
    file_name = Column(String(255), nullable=False)
    file_key = Column(String(500), nullable=False) # S3 key or similar
    file_type = Column(String(50), nullable=False)
    mime_type = Column(String(100))
    file_size = Column(Integer)
    content_hash = Column(String(64)) # SHA256 hash of content
    extracted_text = Column(Text)
    extracted_data = Column(JSON)
    analysis_results = Column(JSON)
    status = Column(String(50), default='pending')
    tags = Column(ARRAY(String), default=list)
    is_sensitive = Column(Boolean, default=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        Index('idx_documents_workflow', 'workflow_id'),
        Index('idx_documents_file_type', 'file_type'),
        Index('idx_documents_status', 'status'),
        Index('idx_documents_sensitive', 'is_sensitive'),
    )
    
    # Relationships
    workflow = relationship("Workflow")
    uploaded_by_user = relationship("User")
    
    def __repr__(self):
        return f"<Document(id={self.id}, file_name='{self.file_name}')>"

# =============================================================================
# DECISIONING MODELS
# =============================================================================

class Decision(Base, UUIDMixin, TimestampMixin):
    """Decision log model"""
    __tablename__ = 'decisions'
    
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id'))
    decision_type = Column(String(100), nullable=False)
    input_data = Column(JSON, nullable=False)
    output_decision = Column(String(100))
    reasons = Column(JSON)
    confidence = Column(DECIMAL(5, 2))
    recommendations = Column(JSON)
    context = Column(JSON)
    made_by_agent = Column(String(100)) # Name of the agent if automated
    made_by_user = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Constraints
    __table_args__ = (
        Index('idx_decisions_workflow', 'workflow_id'),
        Index('idx_decisions_type', 'decision_type'),
        Index('idx_decisions_agent', 'made_by_agent'),
        Index('idx_decisions_user', 'made_by_user'),
    )
    
    # Relationships
    workflow = relationship("Workflow")
    user = relationship("User")
    
    def __repr__(self):
        return f"<Decision(id={self.id}, type='{self.decision_type}', decision='{self.output_decision}')>"

