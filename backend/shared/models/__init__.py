"""
Insurance AI Agent System - Database Models
Production-ready SQLAlchemy models for all system entities
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from decimal import Decimal
import json

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint, Enum as SQLEnum,
    DECIMAL, LargeBinary, Table
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property
import enum

Base = declarative_base()

# Enums
class UserRole(enum.Enum):
    ADMIN = "admin"
    UNDERWRITER = "underwriter"
    CLAIMS_ADJUSTER = "claims_adjuster"
    AGENT = "agent"
    VIEWER = "viewer"

class PolicyStatus(enum.Enum):
    DRAFT = "draft"
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"

class ClaimStatus(enum.Enum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    INVESTIGATING = "investigating"
    APPROVED = "approved"
    DENIED = "denied"
    SETTLED = "settled"
    CLOSED = "closed"

class WorkflowStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DocumentType(enum.Enum):
    APPLICATION = "application"
    POLICY = "policy"
    CLAIM = "claim"
    EVIDENCE = "evidence"
    REPORT = "report"
    CORRESPONDENCE = "correspondence"

class AgentType(enum.Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COMMUNICATION = "communication"
    EVIDENCE_PROCESSING = "evidence_processing"
    COMPLIANCE = "compliance"

# Mixins
class UUIDMixin:
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class SoftDeleteMixin:
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)

# Association Tables
user_organization_association = Table(
    'user_organizations',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('organization_id', UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE')),
    Column('role', String(50), nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)

# Core Models
class Organization(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = 'organizations'
    
    name = Column(String(255), nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    industry = Column(String(100))
    country = Column(String(2))  # ISO country code
    timezone = Column(String(50), default='UTC')
    settings = Column(JSONB, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    users = relationship("User", secondary=user_organization_association, back_populates="organizations")
    policies = relationship("Policy", back_populates="organization", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="organization", cascade="all, delete-orphan")
    workflows = relationship("Workflow", back_populates="organization", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_organization_code', 'code'),
        Index('idx_organization_active', 'is_active'),
    )

class User(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = 'users'
    
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    role = Column(
        SQLEnum(
            UserRole,
            name="userrole",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        default=UserRole.VIEWER,
    )
    phone = Column(String(20))
    department = Column(String(100))
    employee_id = Column(String(50))
    
    # Authentication
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    
    # Preferences
    preferences = Column(JSONB, default=dict)
    notification_settings = Column(JSONB, default=dict)
    
    # Relationships
    organizations = relationship("Organization", secondary=user_organization_association, back_populates="users")
    created_policies = relationship("Policy", foreign_keys="Policy.created_by", back_populates="creator")
    created_claims = relationship("Claim", foreign_keys="Claim.created_by", back_populates="creator")
    assigned_claims = relationship("Claim", foreign_keys="Claim.assigned_to", back_populates="assignee")
    
    @validates('email')
    def validate_email(self, key, email):
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    @hybrid_property
    def full_name(self):
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.email
    
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_role', 'role'),
        Index('idx_user_active', 'is_active'),
    )

class Policy(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = 'policies'
    
    policy_number = Column(String(50), unique=True, nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    
    # Policy Details
    policy_type = Column(String(100), nullable=False)
    product_line = Column(String(100))
    insured_name = Column(String(255), nullable=False)
    insured_address = Column(Text)
    insured_phone = Column(String(20))
    insured_email = Column(String(255))
    
    # Coverage
    coverage_amount = Column(DECIMAL(15, 2))
    deductible = Column(DECIMAL(10, 2))
    premium_amount = Column(DECIMAL(10, 2))
    
    # Dates
    effective_date = Column(DateTime(timezone=True))
    expiration_date = Column(DateTime(timezone=True))
    
    # Status and Risk
    status = Column(
        SQLEnum(
            PolicyStatus,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        default=PolicyStatus.DRAFT,
    )
    risk_score = Column(Float)
    risk_factors = Column(JSONB, default=list)
    
    # Metadata
    policy_data = Column(JSONB, default=dict)
    underwriting_notes = Column(Text)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    organization = relationship("Organization", back_populates="policies")
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_policies")
    claims = relationship("Claim", back_populates="policy", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="policy", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_policy_number', 'policy_number'),
        Index('idx_policy_status', 'status'),
        Index('idx_policy_type', 'policy_type'),
        Index('idx_policy_dates', 'effective_date', 'expiration_date'),
        CheckConstraint('coverage_amount > 0', name='check_positive_coverage'),
        CheckConstraint('premium_amount > 0', name='check_positive_premium'),
    )

class Claim(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = 'claims'
    
    claim_number = Column(String(50), unique=True, nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    policy_id = Column(UUID(as_uuid=True), ForeignKey('policies.id', ondelete='CASCADE'), nullable=False)
    
    # Claim Details
    incident_date = Column(DateTime(timezone=True), nullable=False)
    reported_date = Column(DateTime(timezone=True), nullable=False)
    incident_description = Column(Text, nullable=False)
    incident_location = Column(String(500))
    
    # Financial
    claimed_amount = Column(DECIMAL(15, 2))
    estimated_amount = Column(DECIMAL(15, 2))
    approved_amount = Column(DECIMAL(15, 2))
    paid_amount = Column(DECIMAL(15, 2), default=0)
    
    # Status and Assignment
    status = Column(
        SQLEnum(
            ClaimStatus,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        default=ClaimStatus.SUBMITTED,
    )
    priority = Column(Integer, default=3)  # 1=High, 3=Normal, 5=Low
    assigned_to = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Investigation
    investigation_notes = Column(Text)
    fraud_score = Column(Float)
    fraud_indicators = Column(JSONB, default=list)
    
    # Metadata
    claim_data = Column(JSONB, default=dict)
    settlement_details = Column(JSONB, default=dict)
    
    # Relationships
    organization = relationship("Organization", back_populates="claims")
    policy = relationship("Policy", back_populates="claims")
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_claims")
    assignee = relationship("User", foreign_keys=[assigned_to], back_populates="assigned_claims")
    documents = relationship("Document", back_populates="claim", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_claim_number', 'claim_number'),
        Index('idx_claim_status', 'status'),
        Index('idx_claim_dates', 'incident_date', 'reported_date'),
        Index('idx_claim_amounts', 'claimed_amount', 'approved_amount'),
        CheckConstraint('claimed_amount > 0', name='check_positive_claimed'),
        CheckConstraint('priority BETWEEN 1 AND 5', name='check_priority_range'),
    )

class Document(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = 'documents'
    
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_hash = Column(String(64))  # SHA-256
    
    # Classification
    document_type = Column(
        SQLEnum(
            DocumentType,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )
    category = Column(String(100))
    tags = Column(ARRAY(String), default=list)
    
    # Relationships
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'))
    policy_id = Column(UUID(as_uuid=True), ForeignKey('policies.id', ondelete='SET NULL'))
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id', ondelete='SET NULL'))
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Analysis
    ocr_text = Column(Text)
    extracted_data = Column(JSONB, default=dict)
    confidence_score = Column(Float)
    
    # Security
    is_encrypted = Column(Boolean, default=False)
    access_level = Column(String(50), default='internal')
    
    # Relationships
    policy = relationship("Policy", back_populates="documents")
    claim = relationship("Claim", back_populates="documents")
    uploader = relationship("User")
    analysis_results = relationship("DocumentAnalysis", back_populates="document", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_document_type', 'document_type'),
        Index('idx_document_hash', 'file_hash'),
        Index('idx_document_relationships', 'policy_id', 'claim_id'),
    )

class DocumentAnalysis(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'document_analyses'
    
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    agent_type = Column(
        SQLEnum(
            AgentType,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )
    
    # Analysis Results
    extracted_text = Column(Text)
    extracted_entities = Column(JSONB, default=dict)
    key_value_pairs = Column(JSONB, default=dict)
    confidence_scores = Column(JSONB, default=dict)
    
    # Processing
    processing_time = Column(Float)  # seconds
    model_version = Column(String(50))
    
    # Status
    status = Column(String(50), default='completed')
    error_message = Column(Text)
    
    # Relationships
    document = relationship("Document", back_populates="analysis_results")
    
    __table_args__ = (
        Index('idx_analysis_document', 'document_id'),
        Index('idx_analysis_agent', 'agent_type'),
    )

class EvidenceAnalysis(Base, UUIDMixin, TimestampMixin):
    """Detailed analysis results for claim evidence"""
    __tablename__ = 'evidence_analyses'

    evidence_id = Column(UUID(as_uuid=True), ForeignKey('evidences.id', ondelete='CASCADE'), nullable=False)
    analysis_type = Column(String(100), nullable=False)
    analysis_data = Column(JSONB, default=dict)
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

class Workflow(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'workflows'
    
    workflow_type = Column(String(50), nullable=False)  # underwriting, claims
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    
    # Configuration
    name = Column(String(255), nullable=False)
    description = Column(Text)
    priority = Column(Integer, default=3)
    
    # Input/Output
    input_data = Column(JSONB, default=dict)
    output_data = Column(JSONB, default=dict)
    
    # Execution
    status = Column(
        SQLEnum(
            WorkflowStatus,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        default=WorkflowStatus.PENDING,
    )
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Progress
    current_step = Column(String(100))
    total_steps = Column(Integer)
    completed_steps = Column(Integer, default=0)
    
    # Error Handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Metadata
    metadata_json = Column(JSONB, default=dict)
    
    # Relationships
    organization = relationship("Organization", back_populates="workflows")
    executions = relationship("AgentExecution", back_populates="workflow", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_workflow_type', 'workflow_type'),
        Index('idx_workflow_status', 'status'),
        Index('idx_workflow_priority', 'priority'),
    )

class AgentExecution(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'agent_executions'
    
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id', ondelete='CASCADE'), nullable=False)
    agent_type = Column(
        SQLEnum(
            AgentType,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )
    
    # Execution Details
    agent_name = Column(String(100), nullable=False)
    step_number = Column(Integer, nullable=False)
    
    # Input/Output
    input_data = Column(JSONB, default=dict)
    output_data = Column(JSONB, default=dict)
    
    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    execution_time = Column(Float)  # seconds
    
    # Status
    status = Column(String(50), default='pending')
    success = Column(Boolean)
    error_message = Column(Text)
    
    # Performance
    confidence_score = Column(Float)
    quality_score = Column(Float)
    
    # Metadata
    metadata_json = Column(JSONB, default=dict)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    
    __table_args__ = (
        Index('idx_execution_workflow', 'workflow_id'),
        Index('idx_execution_agent', 'agent_type'),
        Index('idx_execution_status', 'status'),
    )

class AuditLog(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'audit_logs'
    
    # Actor
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'))
    
    # Action
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(UUID(as_uuid=True))
    
    # Details
    old_values = Column(JSONB, default=dict)
    new_values = Column(JSONB, default=dict)
    
    # Context
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    session_id = Column(String(100))
    
    # Metadata
    metadata_json = Column(JSONB, default=dict)
    
    __table_args__ = (
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_timestamp', 'created_at'),
    )

class SystemMetric(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'system_metrics'
    
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    
    # Context
    component = Column(String(50))  # api, agent, database, etc.
    instance_id = Column(String(100))
    
    # Metadata
    labels = Column(JSONB, default=dict)
    
    __table_args__ = (
        Index('idx_metric_name', 'metric_name'),
        Index('idx_metric_component', 'component'),
        Index('idx_metric_timestamp', 'created_at'),
    )

# Import additional models that are defined in the single module version of this
# package. These models are referenced throughout the codebase but were missing
# from this package's namespace which caused ``ImportError`` when other modules
# attempted ``from backend.shared.models import UnderwritingSubmission``.  By
# importing them here we expose a consistent API regardless of whether callers
# import from ``backend.shared.models`` (this package) or from
# ``backend.shared.models`` as a module.
try:  # pragma: no cover - import guard for environments without dependencies
    from .models import (
        APIKey,
        AgentMetric,
        ClaimActivity,
        Communication,
        CommunicationTemplate,
        Evidence,
        EvidenceAnalysis,
        FeatureFlag,
        HealthCheck,
        KnowledgeBase,
        ModelPerformance,
        Prompt,
        SecurityEvent,
        SystemConfig,
        SystemEvent,
        UnderwritingSubmission,
        WebSocketConnection,
    )
except Exception:  # pragma: no cover - fail silently if dependencies missing
    try:
        import importlib.util
        from pathlib import Path

        _single_path = Path(__file__).with_name("models.py")
        spec = importlib.util.spec_from_file_location("backend.shared.models_single", _single_path)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)  # type: ignore[arg-type]

        APIKey = getattr(_mod, "APIKey", None)
        AgentMetric = getattr(_mod, "AgentMetric", None)
        ClaimActivity = getattr(_mod, "ClaimActivity", None)
        Communication = getattr(_mod, "Communication", None)
        CommunicationTemplate = getattr(_mod, "CommunicationTemplate", None)
        Evidence = getattr(_mod, "Evidence", None)
        EvidenceAnalysis = getattr(_mod, "EvidenceAnalysis", None)
        FeatureFlag = getattr(_mod, "FeatureFlag", None)
        HealthCheck = getattr(_mod, "HealthCheck", None)
        KnowledgeBase = getattr(_mod, "KnowledgeBase", None)
        ModelPerformance = getattr(_mod, "ModelPerformance", None)
        Prompt = getattr(_mod, "Prompt", None)
        SecurityEvent = getattr(_mod, "SecurityEvent", None)
        SystemConfig = getattr(_mod, "SystemConfig", None)
        SystemEvent = getattr(_mod, "SystemEvent", None)
        UnderwritingSubmission = getattr(_mod, "UnderwritingSubmission", None)
        WebSocketConnection = getattr(_mod, "WebSocketConnection", None)
    except Exception:
        pass

