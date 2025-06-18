"""
Database Seeding System - Production Ready
Initial data seeding for Insurance AI Agent System
"""

import os
import sys
import logging
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from werkzeug.security import generate_password_hash
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseSeeder:
    """Production-ready database seeding system"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
    def seed_all(self):
        """Seed all reference data and initial records"""
        try:
            logger.info("Starting database seeding...")
            
            # Seed in dependency order
            self.seed_system_config()
            self.seed_feature_flags()
            self.seed_roles()
            self.seed_users()
            self.seed_policy_types()
            self.seed_claim_types()
            self.seed_document_types()
            self.seed_evidence_types()
            self.seed_ai_agents()
            self.seed_workflow_definitions()
            self.seed_notification_templates()
            self.seed_data_retention_policies()
            self.seed_sample_customers()
            self.seed_sample_policies()
            self.seed_sample_claims()
            
            logger.info("Database seeding completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database seeding failed: {str(e)}")
            return False
            
    def seed_system_config(self):
        """Seed system configuration"""
        configs = [
            ('app_name', 'Insurance AI Agent System', 'Application name'),
            ('app_version', '1.0.0', 'Application version'),
            ('max_file_size_mb', '100', 'Maximum file upload size in MB'),
            ('session_timeout_minutes', '60', 'User session timeout in minutes'),
            ('password_min_length', '8', 'Minimum password length'),
            ('password_require_special', 'true', 'Require special characters in password'),
            ('max_login_attempts', '5', 'Maximum failed login attempts before lockout'),
            ('lockout_duration_minutes', '30', 'Account lockout duration in minutes'),
            ('backup_retention_days', '90', 'Backup retention period in days'),
            ('audit_log_retention_days', '2555', 'Audit log retention period in days (7 years)'),
            ('auto_approval_limit', '50000', 'Auto approval limit for claims in USD'),
            ('high_risk_threshold', '0.8', 'High risk threshold score'),
            ('fraud_detection_enabled', 'true', 'Enable fraud detection'),
            ('ai_agents_enabled', 'true', 'Enable AI agents'),
            ('document_ocr_enabled', 'true', 'Enable document OCR processing'),
            ('email_notifications_enabled', 'true', 'Enable email notifications'),
            ('sms_notifications_enabled', 'true', 'Enable SMS notifications'),
            ('push_notifications_enabled', 'true', 'Enable push notifications'),
            ('maintenance_mode', 'false', 'System maintenance mode'),
            ('api_rate_limit_per_minute', '60', 'API rate limit per minute per user'),
            ('api_rate_limit_per_hour', '1000', 'API rate limit per hour per user'),
            ('default_timezone', 'UTC', 'Default system timezone'),
            ('currency_code', 'USD', 'Default currency code'),
            ('date_format', 'YYYY-MM-DD', 'Default date format'),
            ('time_format', 'HH:mm:ss', 'Default time format')
        ]
        
        with self.engine.begin() as conn:
            for key, value, description in configs:
                conn.execute(text("""
                    INSERT INTO system_config (key, value, description)
                    VALUES (:key, :value, :description)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        description = EXCLUDED.description,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'key': key,
                    'value': value,
                    'description': description
                })
                
        logger.info("System configuration seeded")
        
    def seed_feature_flags(self):
        """Seed feature flags"""
        flags = [
            ('ai_underwriting', 'Enable AI-powered underwriting', True, 100),
            ('ai_claims_processing', 'Enable AI-powered claims processing', True, 100),
            ('automated_fraud_detection', 'Enable automated fraud detection', True, 100),
            ('document_auto_classification', 'Enable automatic document classification', True, 100),
            ('real_time_risk_assessment', 'Enable real-time risk assessment', True, 100),
            ('predictive_analytics', 'Enable predictive analytics', True, 80),
            ('advanced_reporting', 'Enable advanced reporting features', True, 100),
            ('mobile_app_support', 'Enable mobile application support', True, 100),
            ('third_party_integrations', 'Enable third-party system integrations', True, 90),
            ('blockchain_verification', 'Enable blockchain-based verification', False, 0),
            ('voice_assistant', 'Enable voice assistant features', False, 10),
            ('augmented_reality_claims', 'Enable AR for claims assessment', False, 5),
            ('iot_device_integration', 'Enable IoT device data integration', True, 50),
            ('social_media_monitoring', 'Enable social media monitoring for fraud', True, 30),
            ('telematics_integration', 'Enable vehicle telematics integration', True, 70)
        ]
        
        with self.engine.begin() as conn:
            for name, description, is_enabled, rollout_percentage in flags:
                conn.execute(text("""
                    INSERT INTO feature_flags (name, description, is_enabled, rollout_percentage)
                    VALUES (:name, :description, :is_enabled, :rollout_percentage)
                    ON CONFLICT (name) DO UPDATE SET
                        description = EXCLUDED.description,
                        is_enabled = EXCLUDED.is_enabled,
                        rollout_percentage = EXCLUDED.rollout_percentage,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'name': name,
                    'description': description,
                    'is_enabled': is_enabled,
                    'rollout_percentage': rollout_percentage
                })
                
        logger.info("Feature flags seeded")
        
    def seed_roles(self):
        """Seed user roles and permissions"""
        roles = [
            ('super_admin', 'Super Administrator', {
                'users': ['create', 'read', 'update', 'delete'],
                'policies': ['create', 'read', 'update', 'delete'],
                'claims': ['create', 'read', 'update', 'delete', 'approve'],
                'documents': ['create', 'read', 'update', 'delete'],
                'reports': ['create', 'read', 'export'],
                'system': ['configure', 'monitor', 'backup'],
                'ai_agents': ['configure', 'monitor', 'train']
            }),
            ('admin', 'Administrator', {
                'users': ['create', 'read', 'update'],
                'policies': ['create', 'read', 'update', 'delete'],
                'claims': ['create', 'read', 'update', 'delete'],
                'documents': ['create', 'read', 'update', 'delete'],
                'reports': ['create', 'read', 'export'],
                'system': ['monitor']
            }),
            ('underwriter', 'Underwriter', {
                'policies': ['create', 'read', 'update'],
                'customers': ['create', 'read', 'update'],
                'documents': ['create', 'read', 'update'],
                'reports': ['read', 'export'],
                'risk_assessment': ['read', 'update']
            }),
            ('claims_adjuster', 'Claims Adjuster', {
                'claims': ['create', 'read', 'update'],
                'policies': ['read'],
                'customers': ['read', 'update'],
                'documents': ['create', 'read', 'update'],
                'evidence': ['create', 'read', 'update'],
                'reports': ['read', 'export']
            }),
            ('claims_manager', 'Claims Manager', {
                'claims': ['create', 'read', 'update', 'approve'],
                'policies': ['read'],
                'customers': ['read', 'update'],
                'documents': ['create', 'read', 'update', 'delete'],
                'evidence': ['create', 'read', 'update', 'delete'],
                'reports': ['create', 'read', 'export'],
                'team_management': ['read', 'assign']
            }),
            ('agent', 'Insurance Agent', {
                'policies': ['create', 'read', 'update'],
                'customers': ['create', 'read', 'update'],
                'quotes': ['create', 'read', 'update'],
                'documents': ['create', 'read'],
                'reports': ['read']
            }),
            ('customer_service', 'Customer Service Representative', {
                'customers': ['read', 'update'],
                'policies': ['read'],
                'claims': ['read', 'update'],
                'documents': ['read'],
                'communications': ['create', 'read', 'update']
            }),
            ('auditor', 'Auditor', {
                'policies': ['read'],
                'claims': ['read'],
                'documents': ['read'],
                'reports': ['read', 'export'],
                'audit_logs': ['read'],
                'compliance': ['read']
            }),
            ('viewer', 'Read-Only Viewer', {
                'policies': ['read'],
                'claims': ['read'],
                'customers': ['read'],
                'documents': ['read'],
                'reports': ['read']
            })
        ]
        
        with self.engine.begin() as conn:
            for name, description, permissions in roles:
                conn.execute(text("""
                    INSERT INTO roles (name, description, permissions)
                    VALUES (:name, :description, :permissions)
                    ON CONFLICT (name) DO UPDATE SET
                        description = EXCLUDED.description,
                        permissions = EXCLUDED.permissions,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'name': name,
                    'description': description,
                    'permissions': json.dumps(permissions)
                })
                
        logger.info("Roles seeded")
        
    def seed_users(self):
        """Seed initial users"""
        users = [
            ('admin@zurich.com', 'admin123', 'System', 'Administrator', '+1-555-0001', 'super_admin'),
            ('underwriter@zurich.com', 'underwriter123', 'John', 'Smith', '+1-555-0002', 'underwriter'),
            ('adjuster@zurich.com', 'adjuster123', 'Jane', 'Doe', '+1-555-0003', 'claims_adjuster'),
            ('manager@zurich.com', 'manager123', 'Mike', 'Johnson', '+1-555-0004', 'claims_manager'),
            ('agent@zurich.com', 'agent123', 'Sarah', 'Wilson', '+1-555-0005', 'agent'),
            ('support@zurich.com', 'support123', 'David', 'Brown', '+1-555-0006', 'customer_service'),
            ('auditor@zurich.com', 'auditor123', 'Lisa', 'Davis', '+1-555-0007', 'auditor')
        ]
        
        with self.engine.begin() as conn:
            for email, password, first_name, last_name, phone, role_name in users:
                # Get role ID
                role_result = conn.execute(text("SELECT id FROM roles WHERE name = :role_name"), {'role_name': role_name})
                role_id = role_result.fetchone()[0]
                
                # Create user
                user_id = str(uuid.uuid4())
                password_hash = generate_password_hash(password)
                
                conn.execute(text("""
                    INSERT INTO users (id, email, password_hash, first_name, last_name, phone, is_active, is_verified)
                    VALUES (:id, :email, :password_hash, :first_name, :last_name, :phone, TRUE, TRUE)
                    ON CONFLICT (email) DO UPDATE SET
                        password_hash = EXCLUDED.password_hash,
                        first_name = EXCLUDED.first_name,
                        last_name = EXCLUDED.last_name,
                        phone = EXCLUDED.phone,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'id': user_id,
                    'email': email,
                    'password_hash': password_hash,
                    'first_name': first_name,
                    'last_name': last_name,
                    'phone': phone
                })
                
                # Assign role
                conn.execute(text("""
                    INSERT INTO user_roles (user_id, role_id)
                    VALUES (:user_id, :role_id)
                    ON CONFLICT (user_id, role_id) DO NOTHING
                """), {
                    'user_id': user_id,
                    'role_id': role_id
                })
                
        logger.info("Initial users seeded")
        
    def seed_policy_types(self):
        """Seed policy types"""
        policy_types = [
            ('auto_liability', 'Auto Liability Insurance', 'auto', 1200.00, {
                'age': {'min': 16, 'max': 85},
                'driving_record': ['clean', 'minor_violations', 'major_violations'],
                'vehicle_age': {'max': 25},
                'credit_score': {'min': 300, 'max': 850}
            }, {
                'bodily_injury': [25000, 50000, 100000, 250000, 500000],
                'property_damage': [25000, 50000, 100000, 250000],
                'uninsured_motorist': [25000, 50000, 100000, 250000],
                'medical_payments': [1000, 2500, 5000, 10000]
            }),
            ('auto_comprehensive', 'Auto Comprehensive Insurance', 'auto', 800.00, {
                'vehicle_value': {'min': 1000, 'max': 200000},
                'deductible': [250, 500, 1000, 2500]
            }, {
                'comprehensive': 'actual_cash_value',
                'collision': 'actual_cash_value',
                'rental_reimbursement': [30, 50, 75, 100],
                'roadside_assistance': True
            }),
            ('homeowners', 'Homeowners Insurance', 'property', 1500.00, {
                'home_age': {'max': 100},
                'home_value': {'min': 50000, 'max': 5000000},
                'location_risk': ['low', 'medium', 'high'],
                'security_features': ['alarm', 'cameras', 'monitoring']
            }, {
                'dwelling': [100000, 250000, 500000, 1000000, 2000000],
                'personal_property': [50000, 100000, 250000, 500000],
                'liability': [100000, 300000, 500000, 1000000],
                'medical_payments': [1000, 5000, 10000]
            }),
            ('renters', 'Renters Insurance', 'property', 200.00, {
                'rental_value': {'min': 500, 'max': 10000},
                'location_risk': ['low', 'medium', 'high']
            }, {
                'personal_property': [15000, 25000, 50000, 100000],
                'liability': [100000, 300000, 500000],
                'additional_living_expenses': [5000, 10000, 20000]
            }),
            ('life_term', 'Term Life Insurance', 'life', 500.00, {
                'age': {'min': 18, 'max': 75},
                'health_status': ['excellent', 'good', 'fair', 'poor'],
                'smoking_status': ['never', 'former', 'current'],
                'occupation_risk': ['low', 'medium', 'high']
            }, {
                'death_benefit': [50000, 100000, 250000, 500000, 1000000, 2000000],
                'term_length': [10, 15, 20, 25, 30],
                'conversion_option': True
            }),
            ('life_whole', 'Whole Life Insurance', 'life', 2000.00, {
                'age': {'min': 18, 'max': 65},
                'health_status': ['excellent', 'good', 'fair'],
                'investment_risk': ['conservative', 'moderate', 'aggressive']
            }, {
                'death_benefit': [50000, 100000, 250000, 500000, 1000000],
                'cash_value': True,
                'dividend_option': ['cash', 'reduce_premium', 'buy_additions']
            }),
            ('health_individual', 'Individual Health Insurance', 'health', 3000.00, {
                'age': {'min': 18, 'max': 65},
                'pre_existing_conditions': True,
                'family_size': {'max': 10},
                'income_level': ['low', 'medium', 'high']
            }, {
                'deductible': [1000, 2500, 5000, 10000],
                'out_of_pocket_max': [5000, 10000, 15000, 20000],
                'copay': [20, 30, 40, 50],
                'prescription_coverage': True
            }),
            ('disability_short', 'Short-term Disability Insurance', 'disability', 300.00, {
                'occupation': ['office', 'manual', 'hazardous'],
                'income': {'min': 20000, 'max': 500000},
                'health_status': ['excellent', 'good', 'fair']
            }, {
                'benefit_period': [3, 6, 12, 24],
                'benefit_amount': [0.6, 0.7, 0.8],
                'elimination_period': [0, 7, 14, 30]
            }),
            ('disability_long', 'Long-term Disability Insurance', 'disability', 800.00, {
                'occupation': ['office', 'manual', 'hazardous'],
                'income': {'min': 30000, 'max': 1000000},
                'age': {'min': 18, 'max': 60}
            }, {
                'benefit_period': ['5_years', '10_years', 'to_age_65', 'lifetime'],
                'benefit_amount': [0.6, 0.7, 0.8],
                'elimination_period': [90, 180, 365, 730]
            }),
            ('commercial_general', 'Commercial General Liability', 'commercial', 2500.00, {
                'business_type': ['retail', 'manufacturing', 'service', 'professional'],
                'revenue': {'min': 50000, 'max': 100000000},
                'employee_count': {'max': 10000},
                'location_count': {'max': 100}
            }, {
                'general_aggregate': [1000000, 2000000, 5000000, 10000000],
                'products_aggregate': [1000000, 2000000, 5000000],
                'personal_injury': [1000000, 2000000, 5000000],
                'medical_expenses': [5000, 10000, 25000]
            })
        ]
        
        with self.engine.begin() as conn:
            for name, description, category, base_premium, risk_factors, coverage_options in policy_types:
                conn.execute(text("""
                    INSERT INTO policy_types (name, description, category, base_premium, risk_factors, coverage_options)
                    VALUES (:name, :description, :category, :base_premium, :risk_factors, :coverage_options)
                    ON CONFLICT (name) DO UPDATE SET
                        description = EXCLUDED.description,
                        category = EXCLUDED.category,
                        base_premium = EXCLUDED.base_premium,
                        risk_factors = EXCLUDED.risk_factors,
                        coverage_options = EXCLUDED.coverage_options,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'name': name,
                    'description': description,
                    'category': category,
                    'base_premium': base_premium,
                    'risk_factors': json.dumps(risk_factors),
                    'coverage_options': json.dumps(coverage_options)
                })
                
        logger.info("Policy types seeded")
        
    def seed_claim_types(self):
        """Seed claim types"""
        claim_types = [
            ('auto_collision', 'Auto Collision', 'auto', 14, True, 25000.00),
            ('auto_comprehensive', 'Auto Comprehensive', 'auto', 10, False, 15000.00),
            ('auto_liability', 'Auto Liability', 'auto', 21, True, 50000.00),
            ('auto_pip', 'Auto Personal Injury Protection', 'auto', 7, False, 10000.00),
            ('property_fire', 'Property Fire Damage', 'property', 30, True, 100000.00),
            ('property_water', 'Property Water Damage', 'property', 14, True, 50000.00),
            ('property_theft', 'Property Theft', 'property', 21, True, 25000.00),
            ('property_vandalism', 'Property Vandalism', 'property', 10, False, 10000.00),
            ('property_wind', 'Property Wind Damage', 'property', 21, True, 75000.00),
            ('property_hail', 'Property Hail Damage', 'property', 14, False, 30000.00),
            ('life_death', 'Life Insurance Death Benefit', 'life', 45, True, 0.00),
            ('life_accidental', 'Accidental Death Benefit', 'life', 30, True, 0.00),
            ('health_medical', 'Health Medical Claim', 'health', 3, False, 5000.00),
            ('health_prescription', 'Health Prescription Claim', 'health', 1, False, 500.00),
            ('health_emergency', 'Health Emergency Claim', 'health', 1, False, 10000.00),
            ('disability_short', 'Short-term Disability Claim', 'disability', 14, False, 0.00),
            ('disability_long', 'Long-term Disability Claim', 'disability', 60, True, 0.00),
            ('workers_comp', 'Workers Compensation', 'workers_comp', 30, True, 100000.00),
            ('commercial_liability', 'Commercial Liability', 'commercial', 45, True, 500000.00),
            ('commercial_property', 'Commercial Property', 'commercial', 30, True, 250000.00)
        ]
        
        with self.engine.begin() as conn:
            for name, description, category, settlement_days, requires_investigation, auto_approval_limit in claim_types:
                conn.execute(text("""
                    INSERT INTO claim_types (name, description, category, typical_settlement_days, requires_investigation, auto_approval_limit)
                    VALUES (:name, :description, :category, :settlement_days, :requires_investigation, :auto_approval_limit)
                    ON CONFLICT (name) DO UPDATE SET
                        description = EXCLUDED.description,
                        category = EXCLUDED.category,
                        typical_settlement_days = EXCLUDED.typical_settlement_days,
                        requires_investigation = EXCLUDED.requires_investigation,
                        auto_approval_limit = EXCLUDED.auto_approval_limit
                """), {
                    'name': name,
                    'description': description,
                    'category': category,
                    'settlement_days': settlement_days,
                    'requires_investigation': requires_investigation,
                    'auto_approval_limit': auto_approval_limit
                })
                
        logger.info("Claim types seeded")
        
    def seed_document_types(self):
        """Seed document types"""
        document_types = [
            ('policy_application', 'Policy Application', 'policy', {
                'applicant_name': 'required',
                'application_date': 'required',
                'coverage_type': 'required',
                'signature': 'required'
            }, {
                'required_fields': ['applicant_name', 'application_date'],
                'max_file_size_mb': 10,
                'allowed_formats': ['pdf', 'jpg', 'png']
            }, 2555),
            ('policy_contract', 'Policy Contract', 'policy', {
                'policy_number': 'required',
                'effective_date': 'required',
                'premium_amount': 'required'
            }, {
                'digital_signature': True,
                'version_control': True
            }, 2555),
            ('claim_form', 'Claim Form', 'claim', {
                'claim_number': 'required',
                'incident_date': 'required',
                'description': 'required',
                'claimant_signature': 'required'
            }, {
                'auto_populate': ['policy_number', 'customer_info'],
                'validation': 'strict'
            }, 2555),
            ('police_report', 'Police Report', 'claim', {
                'report_number': 'required',
                'incident_date': 'required',
                'officer_name': 'required'
            }, {
                'official_document': True,
                'verification_required': True
            }, 2555),
            ('medical_report', 'Medical Report', 'claim', {
                'patient_name': 'required',
                'diagnosis': 'required',
                'treatment_date': 'required',
                'physician_signature': 'required'
            }, {
                'hipaa_compliance': True,
                'encryption_required': True
            }, 2555),
            ('repair_estimate', 'Repair Estimate', 'claim', {
                'estimate_date': 'required',
                'total_amount': 'required',
                'itemized_costs': 'required'
            }, {
                'vendor_verification': True,
                'cost_validation': True
            }, 1825),
            ('photo_evidence', 'Photo Evidence', 'evidence', {
                'timestamp': 'required',
                'location': 'optional',
                'description': 'required'
            }, {
                'metadata_preservation': True,
                'chain_of_custody': True
            }, 2555),
            ('video_evidence', 'Video Evidence', 'evidence', {
                'timestamp': 'required',
                'duration': 'required',
                'description': 'required'
            }, {
                'compression_allowed': False,
                'chain_of_custody': True
            }, 2555),
            ('witness_statement', 'Witness Statement', 'claim', {
                'witness_name': 'required',
                'contact_info': 'required',
                'statement_date': 'required',
                'signature': 'required'
            }, {
                'notarization': 'optional',
                'verification': True
            }, 2555),
            ('financial_statement', 'Financial Statement', 'underwriting', {
                'statement_date': 'required',
                'total_assets': 'required',
                'total_liabilities': 'required'
            }, {
                'cpa_verification': 'optional',
                'confidentiality': 'high'
            }, 2555),
            ('driving_record', 'Driving Record', 'underwriting', {
                'license_number': 'required',
                'issue_date': 'required',
                'violations': 'optional'
            }, {
                'dmv_verification': True,
                'auto_refresh': True
            }, 1095),
            ('credit_report', 'Credit Report', 'underwriting', {
                'report_date': 'required',
                'credit_score': 'required',
                'bureau_name': 'required'
            }, {
                'consent_required': True,
                'auto_refresh': True
            }, 1095),
            ('inspection_report', 'Inspection Report', 'underwriting', {
                'inspection_date': 'required',
                'inspector_name': 'required',
                'findings': 'required'
            }, {
                'photo_required': True,
                'certification': True
            }, 1825),
            ('correspondence', 'Correspondence', 'communication', {
                'date': 'required',
                'sender': 'required',
                'recipient': 'required'
            }, {
                'thread_tracking': True,
                'response_tracking': True
            }, 1095),
            ('legal_document', 'Legal Document', 'legal', {
                'document_type': 'required',
                'jurisdiction': 'required',
                'effective_date': 'required'
            }, {
                'legal_review': True,
                'version_control': True
            }, 2555)
        ]
        
        with self.engine.begin() as conn:
            for name, description, category, required_fields, validation_rules, retention_days in document_types:
                conn.execute(text("""
                    INSERT INTO document_types (name, description, category, required_fields, validation_rules, retention_period_days)
                    VALUES (:name, :description, :category, :required_fields, :validation_rules, :retention_days)
                    ON CONFLICT (name) DO UPDATE SET
                        description = EXCLUDED.description,
                        category = EXCLUDED.category,
                        required_fields = EXCLUDED.required_fields,
                        validation_rules = EXCLUDED.validation_rules,
                        retention_period_days = EXCLUDED.retention_period_days
                """), {
                    'name': name,
                    'description': description,
                    'category': category,
                    'required_fields': json.dumps(required_fields),
                    'validation_rules': json.dumps(validation_rules),
                    'retention_days': retention_days
                })
                
        logger.info("Document types seeded")
        
    def seed_evidence_types(self):
        """Seed evidence types"""
        evidence_types = [
            ('accident_photos', 'Accident Scene Photos', 'visual', {
                'photo_analysis': True,
                'damage_detection': True,
                'object_recognition': True
            }, {
                'min_resolution': '1920x1080',
                'max_file_size_mb': 50,
                'metadata_required': True
            }),
            ('vehicle_damage_photos', 'Vehicle Damage Photos', 'visual', {
                'damage_assessment': True,
                'cost_estimation': True,
                'repair_analysis': True
            }, {
                'multiple_angles': True,
                'close_up_required': True,
                'lighting_standards': True
            }),
            ('property_damage_photos', 'Property Damage Photos', 'visual', {
                'structural_analysis': True,
                'material_identification': True,
                'extent_assessment': True
            }, {
                'before_after': 'preferred',
                'scale_reference': True,
                'timestamp_verification': True
            }),
            ('surveillance_video', 'Surveillance Video', 'video', {
                'motion_detection': True,
                'facial_recognition': False,
                'timeline_analysis': True
            }, {
                'min_quality': '720p',
                'frame_rate': '15fps',
                'audio_optional': True
            }),
            ('dashcam_footage', 'Dashcam Footage', 'video', {
                'speed_analysis': True,
                'trajectory_tracking': True,
                'impact_analysis': True
            }, {
                'gps_data': 'preferred',
                'timestamp_sync': True,
                'multiple_cameras': 'optional'
            }),
            ('medical_records', 'Medical Records', 'document', {
                'injury_classification': True,
                'treatment_validation': True,
                'cost_verification': True
            }, {
                'hipaa_compliance': True,
                'physician_verification': True,
                'diagnostic_codes': True
            }),
            ('repair_invoices', 'Repair Invoices', 'document', {
                'cost_analysis': True,
                'vendor_verification': True,
                'parts_validation': True
            }, {
                'itemized_breakdown': True,
                'labor_rates': True,
                'warranty_info': True
            }),
            ('witness_statements', 'Witness Statements', 'testimony', {
                'credibility_assessment': True,
                'consistency_check': True,
                'bias_detection': True
            }, {
                'contact_verification': True,
                'statement_recording': 'optional',
                'follow_up_required': True
            }),
            ('expert_opinions', 'Expert Opinions', 'professional', {
                'qualification_verification': True,
                'methodology_review': True,
                'peer_validation': True
            }, {
                'credentials_required': True,
                'methodology_documentation': True,
                'supporting_data': True
            }),
            ('financial_records', 'Financial Records', 'document', {
                'income_verification': True,
                'loss_calculation': True,
                'fraud_indicators': True
            }, {
                'bank_verification': True,
                'tax_document_matching': True,
                'third_party_validation': True
            }),
            ('communication_logs', 'Communication Logs', 'digital', {
                'timeline_reconstruction': True,
                'sentiment_analysis': True,
                'pattern_detection': True
            }, {
                'metadata_preservation': True,
                'chain_of_custody': True,
                'privacy_compliance': True
            }),
            ('sensor_data', 'Sensor Data', 'digital', {
                'data_validation': True,
                'pattern_analysis': True,
                'anomaly_detection': True
            }, {
                'calibration_records': True,
                'sampling_rate': 'high',
                'data_integrity': True
            }),
            ('social_media', 'Social Media Evidence', 'digital', {
                'authenticity_verification': True,
                'timeline_analysis': True,
                'content_analysis': True
            }, {
                'screenshot_preservation': True,
                'metadata_capture': True,
                'privacy_considerations': True
            }),
            ('gps_data', 'GPS Location Data', 'digital', {
                'route_reconstruction': True,
                'speed_analysis': True,
                'location_verification': True
            }, {
                'accuracy_validation': True,
                'timestamp_sync': True,
                'privacy_compliance': True
            }),
            ('weather_data', 'Weather Data', 'environmental', {
                'condition_analysis': True,
                'visibility_assessment': True,
                'impact_correlation': True
            }, {
                'official_sources': True,
                'location_specific': True,
                'time_accuracy': True
            })
        ]
        
        with self.engine.begin() as conn:
            for name, description, category, analysis_methods, quality_requirements in evidence_types:
                conn.execute(text("""
                    INSERT INTO evidence_types (name, description, category, analysis_methods, quality_requirements)
                    VALUES (:name, :description, :category, :analysis_methods, :quality_requirements)
                    ON CONFLICT (name) DO UPDATE SET
                        description = EXCLUDED.description,
                        category = EXCLUDED.category,
                        analysis_methods = EXCLUDED.analysis_methods,
                        quality_requirements = EXCLUDED.quality_requirements
                """), {
                    'name': name,
                    'description': description,
                    'category': category,
                    'analysis_methods': json.dumps(analysis_methods),
                    'quality_requirements': json.dumps(quality_requirements)
                })
                
        logger.info("Evidence types seeded")

if __name__ == "__main__":
    # Example usage
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/insurance_ai")
    
    seeder = DatabaseSeeder(database_url)
    success = seeder.seed_all()
    
    if success:
        logger.info("Database seeding completed successfully")
        sys.exit(0)
    else:
        logger.error("Database seeding failed")
        sys.exit(1)

