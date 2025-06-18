-- Insurance AI Agent System - Database Schema
-- PostgreSQL 15+ with pgvector extension
-- Production-ready schema with all tables, indexes, and constraints

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Organizations table for multi-tenancy
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Users table with role-based access
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'underwriter', 'claims_adjuster', 'broker', 'viewer')),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    status VARCHAR(50) DEFAULT 'pending_verification',
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API Keys for external integrations
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    permissions JSONB DEFAULT '[]',
    rate_limit INTEGER DEFAULT 1000,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflows for orchestrating agent processes
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(50) NOT NULL CHECK (type IN ('underwriting', 'claims', 'validation', 'analysis')),
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    input_data JSONB NOT NULL,
    output_data JSONB,
    metadata JSONB DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    timeout_at TIMESTAMP WITH TIME ZONE,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_by UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent executions for tracking individual agent runs
CREATE TABLE agent_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows(id) ON DELETE CASCADE,
    agent_name VARCHAR(100) NOT NULL,
    agent_version VARCHAR(20) DEFAULT '1.0.0',
    input_data JSONB NOT NULL,
    output_data JSONB,
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    error_message TEXT,
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    cpu_usage_percent DECIMAL(5,2),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- UNDERWRITING TABLES
-- =============================================================================

-- Underwriting submissions
CREATE TABLE underwriting_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows(id) ON DELETE CASCADE,
    submission_number VARCHAR(100) UNIQUE NOT NULL,
    broker_id VARCHAR(255),
    broker_name VARCHAR(255),
    insured_name VARCHAR(255) NOT NULL,
    insured_industry VARCHAR(100),
    policy_type VARCHAR(100) NOT NULL,
    coverage_amount DECIMAL(15,2),
    premium_amount DECIMAL(12,2),
    submission_data JSONB NOT NULL,
    risk_score DECIMAL(5,2) CHECK (risk_score BETWEEN 0 AND 100),
    decision VARCHAR(50) CHECK (decision IN ('accept', 'decline', 'refer', 'pending')),
    decision_reasons JSONB,
    decision_confidence DECIMAL(5,2) CHECK (decision_confidence BETWEEN 0 AND 100),
    underwriter_notes TEXT,
    effective_date DATE,
    expiry_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Policy documents
CREATE TABLE policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID REFERENCES underwriting_submissions(id) ON DELETE CASCADE,
    policy_number VARCHAR(100) UNIQUE NOT NULL,
    policy_data JSONB NOT NULL,
    terms_and_conditions JSONB,
    coverage_details JSONB,
    exclusions JSONB,
    limits JSONB,
    deductibles JSONB,
    coverage_limits JSONB,

    customer_id UUID,
    policy_type_id INTEGER,
    premium_amount DECIMAL(12,2),
    deductible DECIMAL(12,2),
    policy_terms JSONB,
    underwriting_data JSONB,
    risk_score DECIMAL(5,2) CHECK (risk_score BETWEEN 0 AND 100),
    agent_id UUID REFERENCES users(id),
    underwriter_id UUID REFERENCES users(id),
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN ('draft', 'quoted', 'bound', 'active', 'cancelled', 'expired', 'suspended')),

    effective_date DATE NOT NULL,
    expiration_date DATE NOT NULL,
    renewal_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CHECK (effective_date < expiration_date)
);

-- =============================================================================
-- CLAIMS TABLES
-- =============================================================================

-- Claims processing
CREATE TABLE claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows(id) ON DELETE CASCADE,
    claim_number VARCHAR(100) UNIQUE NOT NULL,
    policy_id UUID REFERENCES policies(id),
    policy_number VARCHAR(100),
    claim_type VARCHAR(100) NOT NULL,
    incident_date DATE,
    reported_date DATE DEFAULT CURRENT_DATE,
    description TEXT,
    location JSONB,
    claim_amount DECIMAL(15,2),
    reserve_amount DECIMAL(15,2),
    paid_amount DECIMAL(15,2) DEFAULT 0,
    claim_data JSONB NOT NULL,
    liability_assessment JSONB,
    settlement_amount DECIMAL(12,2),
    settlement_date DATE,
    status VARCHAR(50) DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'settled', 'closed', 'denied')),
    priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    adjuster_id UUID REFERENCES users(id),
    adjuster_notes TEXT,
    fraud_indicators JSONB,
    fraud_score DECIMAL(5,2) CHECK (fraud_score BETWEEN 0 AND 100),
    stp_eligible BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Evidence storage metadata
CREATE TABLE evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
    file_key VARCHAR(500) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INTEGER,
    mime_type VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    analysis_results JSONB,
    analysis_confidence DECIMAL(5,2),
    tags TEXT[],
    is_sensitive BOOLEAN DEFAULT false,
    uploaded_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Claim activities and timeline
CREATE TABLE claim_activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
    activity_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    performed_by UUID REFERENCES users(id),
    performed_by_agent VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- KNOWLEDGE BASE AND AI TABLES
-- =============================================================================

-- Knowledge base for RAG
CREATE TABLE knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category VARCHAR(100) NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    tags TEXT[],
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    source_url VARCHAR(500),
    source_type VARCHAR(50),
    confidence_score DECIMAL(5,2),
    created_by UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Prompts management for AI agents
CREATE TABLE prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    prompt_text TEXT NOT NULL,
    variables JSONB DEFAULT '[]',
    agent_name VARCHAR(100),
    model_name VARCHAR(100),
    temperature DECIMAL(3,2) DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 4000,
    is_active BOOLEAN DEFAULT true,
    performance_metrics JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

-- Model performance tracking
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    metrics JSONB NOT NULL,
    evaluation_date DATE DEFAULT CURRENT_DATE,
    dataset_size INTEGER,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- COMMUNICATION TABLES
-- =============================================================================

-- Communication logs
CREATE TABLE communications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows(id),
    claim_id UUID REFERENCES claims(id),
    submission_id UUID REFERENCES underwriting_submissions(id),
    communication_type VARCHAR(50) NOT NULL CHECK (communication_type IN ('email', 'sms', 'phone', 'letter', 'portal')),
    direction VARCHAR(20) NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    recipient_email VARCHAR(255),
    recipient_phone VARCHAR(20),
    sender_email VARCHAR(255),
    subject VARCHAR(500),
    content TEXT,
    template_id UUID,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'delivered', 'failed', 'bounced')),
    external_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    sent_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Communication templates
CREATE TABLE communication_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    template_type VARCHAR(50) NOT NULL,
    subject_template VARCHAR(500),
    content_template TEXT NOT NULL,
    variables JSONB DEFAULT '[]',
    language VARCHAR(10) DEFAULT 'en',
    is_active BOOLEAN DEFAULT true,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- AUDIT AND SECURITY TABLES
-- =============================================================================

-- Comprehensive audit trail
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    request_id VARCHAR(255),
    severity VARCHAR(20) DEFAULT 'info' CHECK (severity IN ('debug', 'info', 'warning', 'error', 'critical')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Security events
CREATE TABLE security_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    user_id UUID REFERENCES users(id),
    ip_address INET,
    details JSONB NOT NULL,
    resolved BOOLEAN DEFAULT false,
    resolved_by UUID REFERENCES users(id),
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- CONFIGURATION TABLES
-- =============================================================================

-- System configuration
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    description TEXT,
    is_sensitive BOOLEAN DEFAULT false,
    organization_id UUID REFERENCES organizations(id),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Feature flags
CREATE TABLE feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    is_enabled BOOLEAN DEFAULT false,
    conditions JSONB DEFAULT '{}',
    rollout_percentage INTEGER DEFAULT 0 CHECK (rollout_percentage BETWEEN 0 AND 100),
    organization_id UUID REFERENCES organizations(id),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- PERFORMANCE AND MONITORING TABLES
-- =============================================================================

-- Agent performance metrics
CREATE TABLE agent_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50),
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    organization_id UUID REFERENCES organizations(id)
);

-- System health checks
CREATE TABLE health_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('healthy', 'degraded', 'unhealthy')),
    response_time_ms INTEGER,
    details JSONB DEFAULT '{}',
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Core table indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_organization ON users(organization_id);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active);

-- Workflow indexes
CREATE INDEX idx_workflows_status ON workflows(status);
CREATE INDEX idx_workflows_type ON workflows(type);
CREATE INDEX idx_workflows_created_at ON workflows(created_at DESC);
CREATE INDEX idx_workflows_organization ON workflows(organization_id);
CREATE INDEX idx_workflows_priority ON workflows(priority DESC);

-- Agent execution indexes
CREATE INDEX idx_agent_executions_workflow ON agent_executions(workflow_id);
CREATE INDEX idx_agent_executions_agent ON agent_executions(agent_name);
CREATE INDEX idx_agent_executions_status ON agent_executions(status);
CREATE INDEX idx_agent_executions_created_at ON agent_executions(created_at DESC);

-- Underwriting indexes
CREATE INDEX idx_underwriting_submissions_number ON underwriting_submissions(submission_number);
CREATE INDEX idx_underwriting_submissions_broker ON underwriting_submissions(broker_id);
CREATE INDEX idx_underwriting_submissions_decision ON underwriting_submissions(decision);
CREATE INDEX idx_underwriting_submissions_created_at ON underwriting_submissions(created_at DESC);

-- Policy indexes
CREATE INDEX idx_policies_number ON policies(policy_number);
CREATE INDEX idx_policies_status ON policies(status);
CREATE INDEX idx_policies_effective_date ON policies(effective_date);
CREATE INDEX idx_policies_expiration_date ON policies(expiration_date);

-- Claims indexes
CREATE INDEX idx_claims_number ON claims(claim_number);
CREATE INDEX idx_claims_policy ON claims(policy_id);
CREATE INDEX idx_claims_status ON claims(status);
CREATE INDEX idx_claims_priority ON claims(priority);
CREATE INDEX idx_claims_created_at ON claims(created_at DESC);
CREATE INDEX idx_claims_incident_date ON claims(incident_date);
CREATE INDEX idx_claims_adjuster ON claims(adjuster_id);

-- Evidence indexes
CREATE INDEX idx_evidence_claim ON evidence(claim_id);
CREATE INDEX idx_evidence_file_type ON evidence(file_type);
CREATE INDEX idx_evidence_created_at ON evidence(created_at DESC);

-- Knowledge base indexes
CREATE INDEX idx_knowledge_base_category ON knowledge_base(category);
CREATE INDEX idx_knowledge_base_active ON knowledge_base(is_active);
CREATE INDEX idx_knowledge_base_embedding ON knowledge_base USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_knowledge_base_tags ON knowledge_base USING GIN(tags);

-- Audit log indexes
CREATE INDEX idx_audit_logs_user_action ON audit_logs(user_id, action, created_at DESC);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_organization ON audit_logs(organization_id);

-- Communication indexes
CREATE INDEX idx_communications_workflow ON communications(workflow_id);
CREATE INDEX idx_communications_claim ON communications(claim_id);
CREATE INDEX idx_communications_type ON communications(communication_type);
CREATE INDEX idx_communications_status ON communications(status);

-- Performance indexes
CREATE INDEX idx_agent_metrics_agent_timestamp ON agent_metrics(agent_name, timestamp DESC);
CREATE INDEX idx_health_checks_service_timestamp ON health_checks(service_name, checked_at DESC);

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers to relevant tables
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workflows_updated_at BEFORE UPDATE ON workflows FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_underwriting_submissions_updated_at BEFORE UPDATE ON underwriting_submissions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_policies_updated_at BEFORE UPDATE ON policies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_claims_updated_at BEFORE UPDATE ON claims FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_knowledge_base_updated_at BEFORE UPDATE ON knowledge_base FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_communication_templates_updated_at BEFORE UPDATE ON communication_templates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_feature_flags_updated_at BEFORE UPDATE ON feature_flags FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Active workflows view
CREATE VIEW active_workflows AS
SELECT 
    w.*,
    u.email as created_by_email,
    o.name as organization_name
FROM workflows w
LEFT JOIN users u ON w.created_by = u.id
LEFT JOIN organizations o ON w.organization_id = o.id
WHERE w.status IN ('pending', 'running');

-- Claims summary view
CREATE VIEW claims_summary AS
SELECT
    c.*,
    p.policy_number as policy_number_ref,
    u.email as adjuster_email,
    COUNT(e.id) as evidence_count,
    COUNT(ca.id) as activity_count
FROM claims c
LEFT JOIN policies p ON c.policy_id = p.id
LEFT JOIN users u ON c.adjuster_id = u.id
LEFT JOIN evidence e ON c.id = e.claim_id
LEFT JOIN claim_activities ca ON c.id = ca.claim_id
GROUP BY c.id, p.policy_number, u.email;

-- Agent performance view
CREATE VIEW agent_performance_summary AS
SELECT 
    agent_name,
    DATE(created_at) as date,
    COUNT(*) as total_executions,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_executions,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_executions,
    AVG(execution_time_ms) as avg_execution_time_ms,
    MAX(execution_time_ms) as max_execution_time_ms,
    MIN(execution_time_ms) as min_execution_time_ms
FROM agent_executions
GROUP BY agent_name, DATE(created_at);

-- =============================================================================
-- INITIAL DATA AND CONFIGURATION
-- =============================================================================

-- Insert default organization
INSERT INTO organizations (id, name, settings) VALUES 
('00000000-0000-0000-0000-000000000001', 'Zurich Insurance', '{"timezone": "UTC", "currency": "USD"}');

-- Insert default admin user (password: admin123)
INSERT INTO users (id, email, password_hash, role, organization_id, first_name, last_name) VALUES 
('00000000-0000-0000-0000-000000000001', 'admin@zurich.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.PmvlG.', 'admin', '00000000-0000-0000-0000-000000000001', 'System', 'Administrator');

-- Insert default system configuration
INSERT INTO system_config (key, value, description, organization_id) VALUES
('max_file_size_mb', to_jsonb(50), 'Maximum file upload size in MB', '00000000-0000-0000-0000-000000000001'),
('session_timeout_minutes', to_jsonb(1440), 'Session timeout in minutes', '00000000-0000-0000-0000-000000000001'),
('rate_limit_per_minute', to_jsonb(1000), 'API rate limit per minute', '00000000-0000-0000-0000-000000000001'),
('workflow_timeout_seconds', to_jsonb(300), 'Default workflow timeout in seconds', '00000000-0000-0000-0000-000000000001'),
('max_retries', to_jsonb(3), 'Maximum retry attempts for failed operations', '00000000-0000-0000-0000-000000000001');

-- Insert default feature flags
INSERT INTO feature_flags (name, description, is_enabled, organization_id) VALUES 
('document_ocr', 'Enable OCR processing for documents', true, '00000000-0000-0000-0000-000000000001'),
('evidence_analysis', 'Enable automated evidence analysis', true, '00000000-0000-0000-0000-000000000001'),
('automated_decisions', 'Enable automated decision making', true, '00000000-0000-0000-0000-000000000001'),
('real_time_notifications', 'Enable real-time notifications', true, '00000000-0000-0000-0000-000000000001'),
('audit_logging', 'Enable comprehensive audit logging', true, '00000000-0000-0000-0000-000000000001'),
('performance_monitoring', 'Enable performance monitoring', true, '00000000-0000-0000-0000-000000000001');

-- =============================================================================
-- SECURITY POLICIES
-- =============================================================================

-- Row Level Security (RLS) policies can be added here for multi-tenant security
-- ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
-- etc.

COMMIT;

