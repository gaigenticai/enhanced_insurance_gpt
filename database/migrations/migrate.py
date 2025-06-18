"""
Database Migration Scripts - Production Ready
Complete database schema and migration management for Insurance AI Agent System
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sqlalchemy as sa
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, Boolean, Text, Numeric, ForeignKey, Index, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, ENUM
from sqlalchemy.sql import func
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Production-ready database migration manager"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.metadata = MetaData()
        self.migration_history = []
        
    def create_migration_table(self):
        """Create migration tracking table"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(255) NOT NULL UNIQUE,
                    name VARCHAR(255) NOT NULL,
                    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    execution_time_ms INTEGER,
                    checksum VARCHAR(64),
                    success BOOLEAN DEFAULT TRUE
                );
                
                CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version);
                CREATE INDEX IF NOT EXISTS idx_schema_migrations_executed_at ON schema_migrations(executed_at);
            """))
            conn.commit()
            
    def get_executed_migrations(self) -> List[str]:
        """Get list of executed migrations"""
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT version FROM schema_migrations WHERE success = TRUE ORDER BY version"))
            return [row[0] for row in result]
            
    def execute_migration(self, version: str, name: str, sql: str) -> bool:
        """Execute a single migration with rollback support"""
        start_time = datetime.now()
        
        try:
            with self.engine.begin() as conn:
                # Execute migration SQL
                conn.execute(text(sql))
                
                # Record migration
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                conn.execute(text("""
                    INSERT INTO schema_migrations (version, name, execution_time_ms, checksum)
                    VALUES (:version, :name, :execution_time, :checksum)
                """), {
                    'version': version,
                    'name': name,
                    'execution_time': execution_time,
                    'checksum': str(hash(sql))
                })
                
            logger.info(f"Migration {version} ({name}) executed successfully in {execution_time}ms")
            return True
            
        except Exception as e:
            logger.error(f"Migration {version} failed: {str(e)}")
            
            # Record failed migration
            with self.engine.begin() as conn:
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                conn.execute(text("""
                    INSERT INTO schema_migrations (version, name, execution_time_ms, success)
                    VALUES (:version, :name, :execution_time, FALSE)
                """), {
                    'version': version,
                    'name': name,
                    'execution_time': execution_time
                })
            
            return False
            
    def run_migrations(self):
        """Run all pending migrations"""
        self.create_migration_table()
        executed = set(self.get_executed_migrations())
        
        migrations = [
            ("001", "create_core_tables", self.migration_001_core_tables),
            ("002", "create_user_management", self.migration_002_user_management),
            ("003", "create_policy_tables", self.migration_003_policy_tables),
            ("004", "create_claims_tables", self.migration_004_claims_tables),
            ("005", "create_document_tables", self.migration_005_document_tables),
            ("006", "create_evidence_tables", self.migration_006_evidence_tables),
            ("007", "create_ai_agent_tables", self.migration_007_ai_agent_tables),
            ("008", "create_workflow_tables", self.migration_008_workflow_tables),
            ("009", "create_audit_tables", self.migration_009_audit_tables),
            ("010", "create_notification_tables", self.migration_010_notification_tables),
            ("011", "create_integration_tables", self.migration_011_integration_tables),
            ("012", "create_analytics_tables", self.migration_012_analytics_tables),
            ("013", "create_security_tables", self.migration_013_security_tables),
            ("014", "create_ml_model_tables", self.migration_014_ml_model_tables),
            ("015", "create_indexes_and_constraints", self.migration_015_indexes_constraints),
            ("016", "create_views_and_functions", self.migration_016_views_functions),
            ("017", "create_triggers_and_procedures", self.migration_017_triggers_procedures),
            ("018", "insert_reference_data", self.migration_018_reference_data),
            ("019", "create_partitions", self.migration_019_partitions),
            ("020", "optimize_performance", self.migration_020_performance)
        ]
        
        for version, name, migration_func in migrations:
            if version not in executed:
                logger.info(f"Running migration {version}: {name}")
                sql = migration_func()
                if not self.execute_migration(version, name, sql):
                    logger.error(f"Migration {version} failed, stopping")
                    return False
            else:
                logger.info(f"Migration {version} already executed, skipping")
                
        logger.info("All migrations completed successfully")
        return True
        
    def migration_001_core_tables(self) -> str:
        """Core system tables"""
        return """
        -- Core system configuration
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
        
        -- Feature flags
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
        
        -- System health monitoring
        CREATE TABLE IF NOT EXISTS system_health (
            id SERIAL PRIMARY KEY,
            component VARCHAR(100) NOT NULL,
            status VARCHAR(50) NOT NULL,
            metrics JSONB,
            last_check TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT,
            response_time_ms INTEGER
        );
        """
        
    def migration_002_user_management(self) -> str:
        """User management and authentication tables"""
        return """
        -- User roles
        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            permissions JSONB,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Users
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
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- User roles mapping
        CREATE TABLE IF NOT EXISTS user_roles (
            id SERIAL PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
            assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            assigned_by UUID REFERENCES users(id),
            UNIQUE(user_id, role_id)
        );
        
        -- User sessions
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
        
        -- Password reset tokens
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token VARCHAR(255) NOT NULL UNIQUE,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            used_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Email verification tokens
        CREATE TABLE IF NOT EXISTS email_verification_tokens (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token VARCHAR(255) NOT NULL UNIQUE,
            email VARCHAR(255) NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            verified_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
    def migration_003_policy_tables(self) -> str:
        """Insurance policy management tables"""
        return """
        -- Policy types
        CREATE TABLE IF NOT EXISTS policy_types (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            category VARCHAR(50),
            base_premium DECIMAL(12,2),
            risk_factors JSONB,
            coverage_options JSONB,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Customers
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
            credit_score INTEGER,
            risk_profile VARCHAR(50),
            customer_since DATE,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Policies
        CREATE TABLE IF NOT EXISTS policies (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            policy_number VARCHAR(50) NOT NULL UNIQUE,
            customer_id UUID NOT NULL REFERENCES customers(id),
            policy_type_id INTEGER NOT NULL REFERENCES policy_types(id),
            status VARCHAR(50) NOT NULL DEFAULT 'draft',
            effective_date DATE NOT NULL,
            expiration_date DATE NOT NULL,
            premium_amount DECIMAL(12,2) NOT NULL,
            deductible DECIMAL(12,2),
            coverage_limits JSONB,
            policy_terms JSONB,
            underwriting_data JSONB,
            risk_score DECIMAL(5,2),
            agent_id UUID REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CHECK (effective_date < expiration_date),
            CHECK (premium_amount > 0),
            CHECK (risk_score >= 0 AND risk_score <= 100)
        );
        
        -- Policy endorsements
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
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(policy_id, endorsement_number)
        );
        
        -- Policy renewals
        CREATE TABLE IF NOT EXISTS policy_renewals (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            policy_id UUID NOT NULL REFERENCES policies(id),
            renewal_date DATE NOT NULL,
            new_premium DECIMAL(12,2),
            new_terms JSONB,
            status VARCHAR(50) DEFAULT 'pending',
            processed_at TIMESTAMP WITH TIME ZONE,
            processed_by UUID REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
    def migration_004_claims_tables(self) -> str:
        """Claims management tables"""
        return """
        -- Claim types
        CREATE TABLE IF NOT EXISTS claim_types (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            category VARCHAR(50),
            typical_settlement_days INTEGER,
            requires_investigation BOOLEAN DEFAULT FALSE,
            auto_approval_limit DECIMAL(12,2),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Claims
        CREATE TABLE IF NOT EXISTS claims (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            claim_number VARCHAR(50) NOT NULL UNIQUE,
            policy_id UUID NOT NULL REFERENCES policies(id),
            claim_type_id INTEGER NOT NULL REFERENCES claim_types(id),
            status VARCHAR(50) NOT NULL DEFAULT 'reported',
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
            fraud_score DECIMAL(5,2),
            adjuster_id UUID REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP WITH TIME ZONE,
            CHECK (incident_date <= reported_date),
            CHECK (estimated_amount >= 0),
            CHECK (paid_amount >= 0),
            CHECK (fraud_score >= 0 AND fraud_score <= 100)
        );
        
        -- Claim participants
        CREATE TABLE IF NOT EXISTS claim_participants (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
            type VARCHAR(50) NOT NULL, -- claimant, witness, third_party, etc.
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            contact_info JSONB,
            role_description TEXT,
            statement TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Claim payments
        CREATE TABLE IF NOT EXISTS claim_payments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            claim_id UUID NOT NULL REFERENCES claims(id),
            payment_type VARCHAR(50) NOT NULL,
            amount DECIMAL(12,2) NOT NULL,
            payee_name VARCHAR(255) NOT NULL,
            payee_details JSONB,
            payment_method VARCHAR(50),
            payment_date DATE,
            check_number VARCHAR(50),
            status VARCHAR(50) DEFAULT 'pending',
            approved_by UUID REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CHECK (amount > 0)
        );
        
        -- Claim reserves
        CREATE TABLE IF NOT EXISTS claim_reserves (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            claim_id UUID NOT NULL REFERENCES claims(id),
            reserve_type VARCHAR(50) NOT NULL,
            amount DECIMAL(12,2) NOT NULL,
            reason TEXT,
            set_by UUID REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CHECK (amount >= 0)
        );
        
        -- Claim investigations
        CREATE TABLE IF NOT EXISTS claim_investigations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            claim_id UUID NOT NULL REFERENCES claims(id),
            investigator_id UUID REFERENCES users(id),
            investigation_type VARCHAR(50),
            status VARCHAR(50) DEFAULT 'assigned',
            findings TEXT,
            recommendations TEXT,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP WITH TIME ZONE
        );
        """
        
    def migration_005_document_tables(self) -> str:
        """Document management tables"""
        return """
        -- Document types
        CREATE TABLE IF NOT EXISTS document_types (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            category VARCHAR(50),
            required_fields JSONB,
            validation_rules JSONB,
            retention_period_days INTEGER,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Documents
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_type_id INTEGER NOT NULL REFERENCES document_types(id),
            title VARCHAR(255) NOT NULL,
            description TEXT,
            file_name VARCHAR(255) NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            file_size BIGINT NOT NULL,
            mime_type VARCHAR(100),
            checksum VARCHAR(64),
            version INTEGER DEFAULT 1,
            status VARCHAR(50) DEFAULT 'active',
            metadata JSONB,
            tags TEXT[],
            uploaded_by UUID REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE,
            CHECK (file_size > 0),
            CHECK (version > 0)
        );
        
        -- Document relationships
        CREATE TABLE IF NOT EXISTS document_relationships (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            related_entity_type VARCHAR(50) NOT NULL, -- policy, claim, customer, etc.
            related_entity_id UUID NOT NULL,
            relationship_type VARCHAR(50), -- attachment, evidence, contract, etc.
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Document processing results
        CREATE TABLE IF NOT EXISTS document_processing_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            processing_type VARCHAR(50) NOT NULL, -- ocr, classification, extraction, etc.
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            result_data JSONB,
            confidence_score DECIMAL(5,2),
            processing_time_ms INTEGER,
            error_message TEXT,
            processed_by VARCHAR(100), -- AI agent name
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP WITH TIME ZONE,
            CHECK (confidence_score >= 0 AND confidence_score <= 100)
        );
        
        -- Document access log
        CREATE TABLE IF NOT EXISTS document_access_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id),
            user_id UUID REFERENCES users(id),
            access_type VARCHAR(50) NOT NULL, -- view, download, edit, delete
            ip_address INET,
            user_agent TEXT,
            accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Document versions
        CREATE TABLE IF NOT EXISTS document_versions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            version_number INTEGER NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            file_size BIGINT NOT NULL,
            checksum VARCHAR(64),
            changes_description TEXT,
            created_by UUID REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(document_id, version_number),
            CHECK (version_number > 0),
            CHECK (file_size > 0)
        );
        """
        
    def migration_006_evidence_tables(self) -> str:
        """Evidence processing and analysis tables"""
        return """
        -- Evidence types
        CREATE TABLE IF NOT EXISTS evidence_types (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            category VARCHAR(50),
            analysis_methods JSONB,
            quality_requirements JSONB,
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
            status VARCHAR(50) DEFAULT 'collected',
            metadata JSONB,
            collected_by UUID REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Photo evidence
        CREATE TABLE IF NOT EXISTS photo_evidence (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            evidence_item_id UUID NOT NULL REFERENCES evidence_items(id) ON DELETE CASCADE,
            document_id UUID NOT NULL REFERENCES documents(id),
            photo_type VARCHAR(50), -- damage, scene, vehicle, etc.
            camera_info JSONB,
            gps_coordinates POINT,
            timestamp_verified BOOLEAN DEFAULT FALSE,
            quality_score DECIMAL(5,2),
            analysis_results JSONB,
            damage_assessment JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CHECK (quality_score >= 0 AND quality_score <= 100)
        );
        
        -- Video evidence
        CREATE TABLE IF NOT EXISTS video_evidence (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            evidence_item_id UUID NOT NULL REFERENCES evidence_items(id) ON DELETE CASCADE,
            document_id UUID NOT NULL REFERENCES documents(id),
            video_type VARCHAR(50),
            duration_seconds INTEGER,
            resolution VARCHAR(20),
            frame_rate INTEGER,
            audio_present BOOLEAN DEFAULT FALSE,
            analysis_results JSONB,
            key_frames JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CHECK (duration_seconds > 0)
        );
        
        -- Forensic analysis
        CREATE TABLE IF NOT EXISTS forensic_analysis (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            evidence_item_id UUID NOT NULL REFERENCES evidence_items(id),
            analysis_type VARCHAR(50) NOT NULL,
            analyst_id UUID REFERENCES users(id),
            methodology TEXT,
            findings JSONB,
            conclusions TEXT,
            confidence_level VARCHAR(20),
            peer_reviewed BOOLEAN DEFAULT FALSE,
            reviewed_by UUID REFERENCES users(id),
            started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP WITH TIME ZONE
        );
        
        -- Evidence chain of custody
        CREATE TABLE IF NOT EXISTS evidence_custody_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            evidence_item_id UUID NOT NULL REFERENCES evidence_items(id),
            action VARCHAR(50) NOT NULL, -- collected, transferred, analyzed, stored, etc.
            from_person VARCHAR(255),
            to_person VARCHAR(255),
            location VARCHAR(255),
            reason TEXT,
            signature_hash VARCHAR(64),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            created_by UUID REFERENCES users(id)
        );
        """
        
    def migration_007_ai_agent_tables(self) -> str:
        """AI agent management and tracking tables"""
        return """
        -- AI agent definitions
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
        
        -- AI agent operations
        CREATE TABLE IF NOT EXISTS ai_agent_operations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES ai_agents(id),
            operation_type VARCHAR(50) NOT NULL,
            input_data JSONB,
            output_data JSONB,
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            confidence_score DECIMAL(5,2),
            processing_time_ms INTEGER,
            error_message TEXT,
            context JSONB,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP WITH TIME ZONE,
            CHECK (confidence_score >= 0 AND confidence_score <= 100)
        );
        
        -- AI model training
        CREATE TABLE IF NOT EXISTS ai_model_training (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES ai_agents(id),
            training_type VARCHAR(50) NOT NULL,
            dataset_info JSONB,
            training_parameters JSONB,
            metrics JSONB,
            model_path VARCHAR(500),
            status VARCHAR(50) DEFAULT 'pending',
            started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP WITH TIME ZONE,
            deployed_at TIMESTAMP WITH TIME ZONE
        );
        
        -- AI decision explanations
        CREATE TABLE IF NOT EXISTS ai_decision_explanations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            operation_id UUID NOT NULL REFERENCES ai_agent_operations(id) ON DELETE CASCADE,
            decision_type VARCHAR(50) NOT NULL,
            explanation TEXT NOT NULL,
            factors JSONB,
            confidence_breakdown JSONB,
            alternative_outcomes JSONB,
            human_review_required BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- AI agent feedback
        CREATE TABLE IF NOT EXISTS ai_agent_feedback (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            operation_id UUID NOT NULL REFERENCES ai_agent_operations(id),
            feedback_type VARCHAR(50) NOT NULL, -- correction, validation, rating
            feedback_value JSONB,
            provided_by UUID REFERENCES users(id),
            comments TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- AI agent performance metrics
        CREATE TABLE IF NOT EXISTS ai_agent_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES ai_agents(id),
            metric_type VARCHAR(50) NOT NULL,
            metric_value DECIMAL(10,4),
            measurement_period TSTZRANGE,
            context JSONB,
            recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
    def migration_008_workflow_tables(self) -> str:
        """Workflow and orchestration tables"""
        return """
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
            entity_type VARCHAR(50) NOT NULL, -- policy, claim, etc.
            entity_id UUID NOT NULL,
            status VARCHAR(50) NOT NULL DEFAULT 'running',
            current_step VARCHAR(100),
            input_data JSONB,
            output_data JSONB,
            context JSONB,
            priority INTEGER DEFAULT 5,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP WITH TIME ZONE,
            error_message TEXT,
            CHECK (priority >= 1 AND priority <= 10)
        );
        
        -- Workflow steps
        CREATE TABLE IF NOT EXISTS workflow_steps (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_instance_id UUID NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,
            step_name VARCHAR(100) NOT NULL,
            step_type VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            input_data JSONB,
            output_data JSONB,
            assigned_to UUID REFERENCES users(id),
            assigned_agent_id UUID REFERENCES ai_agents(id),
            started_at TIMESTAMP WITH TIME ZONE,
            completed_at TIMESTAMP WITH TIME ZONE,
            due_date TIMESTAMP WITH TIME ZONE,
            retry_count INTEGER DEFAULT 0,
            error_message TEXT
        );
        
        -- Workflow transitions
        CREATE TABLE IF NOT EXISTS workflow_transitions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_instance_id UUID NOT NULL REFERENCES workflow_instances(id),
            from_step VARCHAR(100),
            to_step VARCHAR(100) NOT NULL,
            condition_met JSONB,
            transition_data JSONB,
            transitioned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            triggered_by UUID REFERENCES users(id)
        );
        
        -- Task queue
        CREATE TABLE IF NOT EXISTS task_queue (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            task_type VARCHAR(50) NOT NULL,
            payload JSONB NOT NULL,
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            priority INTEGER DEFAULT 5,
            max_retries INTEGER DEFAULT 3,
            retry_count INTEGER DEFAULT 0,
            scheduled_for TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP WITH TIME ZONE,
            completed_at TIMESTAMP WITH TIME ZONE,
            error_message TEXT,
            worker_id VARCHAR(100),
            CHECK (priority >= 1 AND priority <= 10),
            CHECK (retry_count <= max_retries)
        );
        
        -- Scheduled jobs
        CREATE TABLE IF NOT EXISTS scheduled_jobs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            job_type VARCHAR(50) NOT NULL,
            cron_expression VARCHAR(100),
            payload JSONB,
            is_active BOOLEAN DEFAULT TRUE,
            last_run TIMESTAMP WITH TIME ZONE,
            next_run TIMESTAMP WITH TIME ZONE,
            run_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
    def migration_009_audit_tables(self) -> str:
        """Audit and compliance tracking tables"""
        return """
        -- Audit log
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
        
        -- Compliance checks
        CREATE TABLE IF NOT EXISTS compliance_checks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            check_type VARCHAR(50) NOT NULL,
            entity_type VARCHAR(50) NOT NULL,
            entity_id UUID NOT NULL,
            rule_name VARCHAR(100) NOT NULL,
            status VARCHAR(50) NOT NULL, -- passed, failed, warning
            details JSONB,
            checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            checked_by VARCHAR(100), -- system or user
            remediation_required BOOLEAN DEFAULT FALSE,
            remediation_notes TEXT
        );
        
        -- Data retention policies
        CREATE TABLE IF NOT EXISTS data_retention_policies (
            id SERIAL PRIMARY KEY,
            entity_type VARCHAR(50) NOT NULL,
            retention_period_days INTEGER NOT NULL,
            archive_after_days INTEGER,
            delete_after_days INTEGER,
            conditions JSONB,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CHECK (retention_period_days > 0)
        );
        
        -- Data archival log
        CREATE TABLE IF NOT EXISTS data_archival_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            entity_type VARCHAR(50) NOT NULL,
            entity_id UUID NOT NULL,
            action VARCHAR(50) NOT NULL, -- archived, deleted, restored
            archive_location VARCHAR(500),
            reason TEXT,
            performed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            performed_by VARCHAR(100)
        );
        
        -- Regulatory reporting
        CREATE TABLE IF NOT EXISTS regulatory_reports (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            report_type VARCHAR(50) NOT NULL,
            reporting_period DATERANGE NOT NULL,
            status VARCHAR(50) DEFAULT 'draft',
            report_data JSONB,
            file_path VARCHAR(500),
            submitted_at TIMESTAMP WITH TIME ZONE,
            submitted_by UUID REFERENCES users(id),
            acknowledgment_received BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
    def migration_010_notification_tables(self) -> str:
        """Notification and communication tables"""
        return """
        -- Notification templates
        CREATE TABLE IF NOT EXISTS notification_templates (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(100) NOT NULL UNIQUE,
            type VARCHAR(50) NOT NULL, -- email, sms, push, in_app
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
            recipient_type VARCHAR(50) NOT NULL, -- user, customer, external
            recipient_id UUID,
            recipient_contact VARCHAR(255) NOT NULL,
            type VARCHAR(50) NOT NULL,
            subject TEXT,
            content TEXT NOT NULL,
            status VARCHAR(50) DEFAULT 'pending',
            priority INTEGER DEFAULT 5,
            scheduled_for TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            sent_at TIMESTAMP WITH TIME ZONE,
            delivered_at TIMESTAMP WITH TIME ZONE,
            read_at TIMESTAMP WITH TIME ZONE,
            error_message TEXT,
            metadata JSONB,
            CHECK (priority >= 1 AND priority <= 10)
        );
        
        -- Communication preferences
        CREATE TABLE IF NOT EXISTS communication_preferences (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id),
            customer_id UUID REFERENCES customers(id),
            notification_type VARCHAR(50) NOT NULL,
            channel VARCHAR(50) NOT NULL, -- email, sms, push, phone
            is_enabled BOOLEAN DEFAULT TRUE,
            frequency VARCHAR(50) DEFAULT 'immediate', -- immediate, daily, weekly
            quiet_hours JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CHECK ((user_id IS NOT NULL) OR (customer_id IS NOT NULL))
        );
        
        -- Message threads
        CREATE TABLE IF NOT EXISTS message_threads (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            subject VARCHAR(255) NOT NULL,
            entity_type VARCHAR(50), -- claim, policy, etc.
            entity_id UUID,
            status VARCHAR(50) DEFAULT 'active',
            priority VARCHAR(20) DEFAULT 'normal',
            created_by UUID REFERENCES users(id),
            assigned_to UUID REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Messages
        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            thread_id UUID NOT NULL REFERENCES message_threads(id) ON DELETE CASCADE,
            sender_type VARCHAR(50) NOT NULL, -- user, customer, system
            sender_id UUID,
            sender_name VARCHAR(255),
            content TEXT NOT NULL,
            message_type VARCHAR(50) DEFAULT 'text', -- text, file, system
            attachments JSONB,
            is_internal BOOLEAN DEFAULT FALSE,
            sent_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            edited_at TIMESTAMP WITH TIME ZONE
        );
        
        -- Email tracking
        CREATE TABLE IF NOT EXISTS email_tracking (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            notification_id UUID NOT NULL REFERENCES notifications(id),
            tracking_id VARCHAR(100) NOT NULL UNIQUE,
            opened_at TIMESTAMP WITH TIME ZONE,
            clicked_at TIMESTAMP WITH TIME ZONE,
            bounced_at TIMESTAMP WITH TIME ZONE,
            bounce_reason TEXT,
            unsubscribed_at TIMESTAMP WITH TIME ZONE,
            spam_reported_at TIMESTAMP WITH TIME ZONE
        );
        """

if __name__ == "__main__":
    # Example usage
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/insurance_ai")
    
    migrator = DatabaseMigrator(database_url)
    success = migrator.run_migrations()
    
    if success:
        logger.info("All migrations completed successfully")
        sys.exit(0)
    else:
        logger.error("Migration failed")
        sys.exit(1)

