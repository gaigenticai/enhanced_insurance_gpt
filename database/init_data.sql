-- Insurance AI Agent System Database Initialization
-- Production-ready database schema and initial setup
-- This script creates the complete database structure and populates reference data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Set timezone
SET timezone = 'UTC';

-- Create custom types
CREATE TYPE policy_status AS ENUM ('draft', 'quoted', 'bound', 'active', 'cancelled', 'expired', 'suspended');
CREATE TYPE claim_status AS ENUM ('reported', 'assigned', 'investigating', 'pending_approval', 'approved', 'denied', 'closed', 'reopened');
CREATE TYPE document_status AS ENUM ('uploaded', 'processing', 'processed', 'approved', 'rejected', 'archived');
CREATE TYPE evidence_status AS ENUM ('collected', 'processing', 'analyzed', 'verified', 'disputed', 'archived');
CREATE TYPE workflow_status AS ENUM ('pending', 'running', 'paused', 'completed', 'failed', 'cancelled');
CREATE TYPE notification_status AS ENUM ('pending', 'sent', 'delivered', 'read', 'failed', 'cancelled');
CREATE TYPE ai_operation_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');
CREATE TYPE priority_level AS ENUM ('low', 'normal', 'high', 'urgent', 'critical');
CREATE TYPE risk_level AS ENUM ('very_low', 'low', 'medium', 'high', 'very_high');

-- Core system configuration table
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    organization_id UUID REFERENCES organizations(id),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Application settings
CREATE TABLE IF NOT EXISTS app_settings (
    id SERIAL PRIMARY KEY,
    category VARCHAR(100) NOT NULL,
    setting_key VARCHAR(255) NOT NULL,
    setting_value JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category, setting_key)
);

-- Feature flags for gradual rollout
CREATE TABLE IF NOT EXISTS feature_flags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    is_enabled BOOLEAN DEFAULT FALSE,
    rollout_percentage INTEGER DEFAULT 0 CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100),
    target_users JSONB,
    conditions JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User roles and permissions
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    permissions JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Users table with comprehensive security features
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    password_changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(255),
    backup_codes TEXT[],
    preferences JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- User role assignments
CREATE TABLE IF NOT EXISTS user_roles (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assigned_by UUID REFERENCES users(id),
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(user_id, role_id)
);

-- User sessions for authentication tracking
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    refresh_token VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Password reset functionality
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Email verification
CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    verified_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Policy types and configurations
CREATE TABLE IF NOT EXISTS policy_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50),
    base_premium DECIMAL(12,2),
    risk_factors JSONB,
    coverage_options JSONB,
    underwriting_rules JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Customer information
CREATE TABLE IF NOT EXISTS customers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_number VARCHAR(50) NOT NULL UNIQUE,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    date_of_birth DATE,
    ssn_encrypted VARCHAR(255),
    address JSONB,
    credit_score INTEGER CHECK (credit_score >= 300 AND credit_score <= 850),
    risk_profile risk_level DEFAULT 'medium',
    customer_since DATE DEFAULT CURRENT_DATE,
    lifetime_value DECIMAL(12,2) DEFAULT 0,
    preferences JSONB DEFAULT '{}',
    kyc_status VARCHAR(50) DEFAULT 'pending',
    kyc_verified_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insurance policies
CREATE TABLE IF NOT EXISTS policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_number VARCHAR(50) NOT NULL UNIQUE,
    customer_id UUID NOT NULL REFERENCES customers(id),
    policy_type_id INTEGER NOT NULL REFERENCES policy_types(id),
    status policy_status DEFAULT 'draft',
    effective_date DATE NOT NULL,
    expiration_date DATE NOT NULL,
    premium_amount DECIMAL(12,2) NOT NULL,
    deductible DECIMAL(12,2),
    coverage_limits JSONB,
    policy_terms JSONB,
    underwriting_data JSONB,
    risk_score DECIMAL(5,2) CHECK (risk_score >= 0 AND risk_score <= 100),
    agent_id UUID REFERENCES users(id),
    underwriter_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CHECK (effective_date < expiration_date),
    CHECK (premium_amount > 0)
);

-- Policy endorsements and changes
CREATE TABLE IF NOT EXISTS policy_endorsements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    endorsement_number VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    effective_date DATE NOT NULL,
    premium_change DECIMAL(12,2) DEFAULT 0,
    coverage_changes JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_by UUID REFERENCES users(id),
    approved_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    approved_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(policy_id, endorsement_number)
);

-- Claim types
CREATE TABLE IF NOT EXISTS claim_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50),
    typical_settlement_days INTEGER,
    requires_investigation BOOLEAN DEFAULT FALSE,
    auto_approval_limit DECIMAL(12,2),
    workflow_template JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insurance claims
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_number VARCHAR(50) NOT NULL UNIQUE,
    policy_id UUID NOT NULL REFERENCES policies(id),
    claim_type VARCHAR(100) NOT NULL,
    status claim_status DEFAULT 'reported',
    priority priority_level DEFAULT 'normal',
    incident_date DATE NOT NULL,
    reported_date DATE NOT NULL DEFAULT CURRENT_DATE,
    description TEXT NOT NULL,
    location JSONB,
    estimated_amount DECIMAL(12,2),
    reserved_amount DECIMAL(12,2),
    paid_amount DECIMAL(12,2) DEFAULT 0,
    deductible_amount DECIMAL(12,2),
    fault_determination VARCHAR(100),
    investigation_required BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(5,2) CHECK (fraud_score >= 0 AND fraud_score <= 100),
    complexity_score DECIMAL(5,2) CHECK (complexity_score >= 0 AND complexity_score <= 100),
    adjuster_id UUID REFERENCES users(id),
    supervisor_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP WITH TIME ZONE,
    CHECK (incident_date <= reported_date),
    CHECK (estimated_amount >= 0),
    CHECK (paid_amount >= 0)
);

-- Claim participants (claimants, witnesses, etc.)
CREATE TABLE IF NOT EXISTS claim_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    contact_info JSONB,
    role_description TEXT,
    statement TEXT,
    statement_date DATE,
    credibility_score DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Document types and configurations
CREATE TABLE IF NOT EXISTS document_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50),
    required_fields JSONB,
    validation_rules JSONB,
    retention_period_days INTEGER,
    auto_processing_enabled BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Document storage and management
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_type_id INTEGER NOT NULL REFERENCES document_types(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size > 0),
    mime_type VARCHAR(100),
    checksum VARCHAR(64),
    version INTEGER DEFAULT 1 CHECK (version > 0),
    status document_status DEFAULT 'uploaded',
    metadata JSONB DEFAULT '{}',
    tags TEXT[],
    uploaded_by UUID REFERENCES users(id),
    processed_by VARCHAR(100), -- AI agent name
    processing_results JSONB,
    confidence_score DECIMAL(5,2) CHECK (confidence_score >= 0 AND confidence_score <= 100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Document relationships to entities
CREATE TABLE IF NOT EXISTS document_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    related_entity_type VARCHAR(50) NOT NULL,
    related_entity_id UUID NOT NULL,
    relationship_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

-- Evidence types
CREATE TABLE IF NOT EXISTS evidence_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50),
    analysis_methods JSONB,
    quality_requirements JSONB,
    chain_of_custody_required BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Evidence items
CREATE TABLE IF NOT EXISTS evidence_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL REFERENCES claims(id),
    evidence_type_id INTEGER NOT NULL REFERENCES evidence_types(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    source VARCHAR(100),
    collection_date DATE,
    location JSONB,
    chain_of_custody JSONB,
    integrity_verified BOOLEAN DEFAULT FALSE,
    status evidence_status DEFAULT 'collected',
    analysis_results JSONB,
    quality_score DECIMAL(5,2) CHECK (quality_score >= 0 AND quality_score <= 100),
    metadata JSONB DEFAULT '{}',
    collected_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- AI agents configuration
CREATE TABLE IF NOT EXISTS ai_agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    version VARCHAR(20),
    capabilities JSONB,
    configuration JSONB,
    model_info JSONB,
    status VARCHAR(50) DEFAULT 'active',
    performance_metrics JSONB,
    last_health_check TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- AI agent operations tracking
CREATE TABLE IF NOT EXISTS ai_agent_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES ai_agents(id),
    operation_type VARCHAR(50) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    status ai_operation_status DEFAULT 'pending',
    confidence_score DECIMAL(5,2) CHECK (confidence_score >= 0 AND confidence_score <= 100),
    processing_time_ms INTEGER,
    error_message TEXT,
    context JSONB,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Workflow definitions
CREATE TABLE IF NOT EXISTS workflow_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50),
    version VARCHAR(20) DEFAULT '1.0',
    workflow_schema JSONB NOT NULL,
    trigger_conditions JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflow instances
CREATE TABLE IF NOT EXISTS workflow_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_definition_id UUID NOT NULL REFERENCES workflow_definitions(id),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    status workflow_status DEFAULT 'pending',
    current_step VARCHAR(100),
    input_data JSONB,
    output_data JSONB,
    context JSONB,
    priority priority_level DEFAULT 'normal',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- Notification templates
CREATE TABLE IF NOT EXISTS notification_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,
    subject_template TEXT,
    body_template TEXT NOT NULL,
    variables JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Notifications
CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id UUID REFERENCES notification_templates(id),
    recipient_type VARCHAR(50) NOT NULL,
    recipient_id UUID,
    recipient_contact VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    subject TEXT,
    content TEXT NOT NULL,
    status notification_status DEFAULT 'pending',
    priority priority_level DEFAULT 'normal',
    scheduled_for TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sent_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    read_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Audit log for compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    old_values JSONB,
    new_values JSONB,
    changed_fields TEXT[],
    user_id UUID REFERENCES users(id),
    ip_address INET,
    user_agent TEXT,
    session_id UUID,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reason TEXT
);

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active ON users(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customers_customer_number ON customers(customer_number);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_policies_policy_number ON policies(policy_number);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_policies_customer_id ON policies(customer_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_policies_status ON policies(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_claims_claim_number ON claims(claim_number);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_claims_policy_id ON claims(policy_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_claims_incident_date ON claims(incident_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_file_path ON documents(file_path);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_relationships_entity ON document_relationships(related_entity_type, related_entity_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evidence_items_claim_id ON evidence_items(claim_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_operations_agent_id ON ai_agent_operations(agent_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_operations_status ON ai_agent_operations(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_workflow_instances_entity ON workflow_instances(entity_type, entity_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_notifications_recipient ON notifications(recipient_type, recipient_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_notifications_status ON notifications(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_entity ON audit_log(entity_type, entity_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);

-- Create GIN indexes for JSONB columns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_policies_coverage_limits_gin ON policies USING GIN(coverage_limits);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_claims_location_gin ON claims USING GIN(location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN(metadata);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evidence_analysis_gin ON evidence_items USING GIN(analysis_results);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_operations_context_gin ON ai_agent_operations USING GIN(context);

-- Create text search indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customers_name_trgm ON customers USING GIN((first_name || ' ' || last_name) gin_trgm_ops);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_claims_description_trgm ON claims USING GIN(description gin_trgm_ops);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_title_trgm ON documents USING GIN(title gin_trgm_ops);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Ensure triggers do not already exist before creation
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP TRIGGER IF EXISTS update_customers_updated_at ON customers;
DROP TRIGGER IF EXISTS update_policies_updated_at ON policies;
DROP TRIGGER IF EXISTS update_claims_updated_at ON claims;
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
DROP TRIGGER IF EXISTS update_evidence_items_updated_at ON evidence_items;
DROP TRIGGER IF EXISTS update_ai_agents_updated_at ON ai_agents;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_customers_updated_at BEFORE UPDATE ON customers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_policies_updated_at BEFORE UPDATE ON policies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_claims_updated_at BEFORE UPDATE ON claims FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_evidence_items_updated_at BEFORE UPDATE ON evidence_items FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ai_agents_updated_at BEFORE UPDATE ON ai_agents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
DECLARE
    old_data JSONB;
    new_data JSONB;
    changed_fields TEXT[];
BEGIN
    IF TG_OP = 'DELETE' THEN
        old_data = to_jsonb(OLD);
        INSERT INTO audit_log (entity_type, entity_id, action, old_values, user_id)
        VALUES (TG_TABLE_NAME, OLD.id, 'DELETE', old_data, current_setting('app.current_user_id', true)::UUID);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        old_data = to_jsonb(OLD);
        new_data = to_jsonb(NEW);
        
        -- Find changed fields
        SELECT array_agg(key) INTO changed_fields
        FROM jsonb_each(old_data) o
        WHERE o.value IS DISTINCT FROM new_data->o.key;
        
        IF array_length(changed_fields, 1) > 0 THEN
            INSERT INTO audit_log (entity_type, entity_id, action, old_values, new_values, changed_fields, user_id)
            VALUES (TG_TABLE_NAME, NEW.id, 'UPDATE', old_data, new_data, changed_fields, current_setting('app.current_user_id', true)::UUID);
        END IF;
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        new_data = to_jsonb(NEW);
        INSERT INTO audit_log (entity_type, entity_id, action, new_values, user_id)
        VALUES (TG_TABLE_NAME, NEW.id, 'INSERT', new_data, current_setting('app.current_user_id', true)::UUID);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers for key tables
CREATE TRIGGER audit_users_trigger AFTER INSERT OR UPDATE OR DELETE ON users FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
CREATE TRIGGER audit_customers_trigger AFTER INSERT OR UPDATE OR DELETE ON customers FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
CREATE TRIGGER audit_policies_trigger AFTER INSERT OR UPDATE OR DELETE ON policies FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
CREATE TRIGGER audit_claims_trigger AFTER INSERT OR UPDATE OR DELETE ON claims FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Create views for common queries
CREATE OR REPLACE VIEW active_policies AS
SELECT 
    p.*,
    c.first_name || ' ' || c.last_name AS customer_name,
    c.email AS customer_email,
    pt.name AS policy_type_name,
    pt.category AS policy_category
FROM policies p
JOIN customers c ON p.customer_id = c.id
JOIN policy_types pt ON p.policy_type_id = pt.id
WHERE p.status = 'active'
AND p.expiration_date > CURRENT_DATE;

CREATE OR REPLACE VIEW open_claims AS
SELECT
    cl.*,
    p.policy_number AS policy_number_ref,
    c.first_name || ' ' || c.last_name AS customer_name,
    cl.claim_type AS claim_type_ref,
    u.first_name || ' ' || u.last_name AS adjuster_name
FROM claims cl
JOIN policies p ON cl.policy_id = p.id
JOIN customers c ON p.customer_id = c.id
LEFT JOIN users u ON cl.adjuster_id = u.id
WHERE cl.status NOT IN ('closed', 'denied');

CREATE OR REPLACE VIEW user_permissions AS
SELECT 
    u.id AS user_id,
    u.email,
    u.first_name || ' ' || u.last_name AS full_name,
    r.name AS role_name,
    r.permissions
FROM users u
JOIN user_roles ur ON u.id = ur.user_id
JOIN roles r ON ur.role_id = r.id
WHERE u.is_active = TRUE
AND r.is_active = TRUE
AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP);

-- Insert initial system configuration
INSERT INTO system_config (key, value, description) VALUES
('app_name', to_jsonb('Insurance AI Agent System'::text), 'Application name'),
('app_version', to_jsonb('1.0.0'::text), 'Application version'),
('max_file_size_mb', to_jsonb(100), 'Maximum file upload size in MB'),
('session_timeout_minutes', to_jsonb(60), 'User session timeout in minutes'),
('auto_approval_limit', to_jsonb(50000), 'Auto approval limit for claims in USD'),
('high_risk_threshold', to_jsonb(0.8), 'High risk threshold score'),
('fraud_detection_enabled', to_jsonb(true), 'Enable fraud detection'),
('ai_agents_enabled', to_jsonb(true), 'Enable AI agents')
ON CONFLICT (key) DO NOTHING;

-- Insert default roles
INSERT INTO roles (name, description, permissions) VALUES
('super_admin', 'Super Administrator', '{"users": ["create", "read", "update", "delete"], "policies": ["create", "read", "update", "delete"], "claims": ["create", "read", "update", "delete", "approve"], "system": ["configure", "monitor"]}'),
('admin', 'Administrator', '{"users": ["create", "read", "update"], "policies": ["create", "read", "update", "delete"], "claims": ["create", "read", "update", "delete"]}'),
('underwriter', 'Underwriter', '{"policies": ["create", "read", "update"], "customers": ["create", "read", "update"], "risk_assessment": ["read", "update"]}'),
('claims_adjuster', 'Claims Adjuster', '{"claims": ["create", "read", "update"], "policies": ["read"], "customers": ["read", "update"], "evidence": ["create", "read", "update"]}'),
('agent', 'Insurance Agent', '{"policies": ["create", "read", "update"], "customers": ["create", "read", "update"], "quotes": ["create", "read", "update"]}'),
('viewer', 'Read-Only Viewer', '{"policies": ["read"], "claims": ["read"], "customers": ["read"]}')
ON CONFLICT (name) DO NOTHING;

-- Insert default admin user (password: admin123)
INSERT INTO users (id, email, password_hash, role, first_name, last_name, is_verified) VALUES
('00000000-0000-0000-0000-000000000001', 'admin@zurich.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.PmvlIW', 'admin', 'System', 'Administrator', true)
ON CONFLICT (email) DO NOTHING;

-- Assign super_admin role to default admin
INSERT INTO user_roles (user_id, role_id) 
SELECT '00000000-0000-0000-0000-000000000001', id FROM roles WHERE name = 'super_admin'
ON CONFLICT (user_id, role_id) DO NOTHING;

-- Grant necessary permissions
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO postgres;

-- Create database statistics
ANALYZE;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Insurance AI Agent System database initialization completed successfully at %', CURRENT_TIMESTAMP;
END $$;

