"""
Validation Agent - Production Ready Implementation
Comprehensive data validation, quality assurance, and business rule enforcement
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import numpy as np

# Validation libraries
import cerberus
import jsonschema
from marshmallow import Schema, fields, validate, ValidationError as MarshmallowValidationError
import validators
from email_validator import validate_email, EmailNotValidError

# Machine Learning for anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

validation_checks_total = Counter('validation_checks_total', 'Total validation checks', ['validation_type', 'status'])
validation_duration = Histogram('validation_duration_seconds', 'Validation duration')
validation_errors_gauge = Gauge('validation_errors_current', 'Current validation errors count')

Base = declarative_base()

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationType(Enum):
    DATA_TYPE = "data_type"
    FORMAT = "format"
    RANGE = "range"
    BUSINESS_RULE = "business_rule"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    ANOMALY = "anomaly"

class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class ValidationRule:
    rule_id: str
    rule_name: str
    rule_description: str
    validation_type: ValidationType
    severity: ValidationSeverity
    field_names: List[str]
    rule_logic: Dict[str, Any]
    error_message: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class ValidationError:
    error_id: str
    rule_id: str
    field_name: str
    error_message: str
    severity: ValidationSeverity
    actual_value: Any
    expected_value: Any
    record_id: Optional[str]
    context: Dict[str, Any]
    detected_at: datetime

@dataclass
class ValidationResult:
    validation_id: str
    data_source: str
    total_records: int
    records_validated: int
    validation_rules_applied: List[str]
    passed_validations: int
    failed_validations: int
    warning_validations: int
    errors: List[ValidationError]
    summary: Dict[str, Any]
    validation_duration: float
    validated_at: datetime

class ValidationRuleRecord(Base):
    __tablename__ = 'validation_rules'
    
    rule_id = Column(String, primary_key=True)
    rule_name = Column(String, nullable=False)
    rule_description = Column(Text)
    validation_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    field_names = Column(JSON)
    rule_logic = Column(JSON)
    error_message = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class ValidationResultRecord(Base):
    __tablename__ = 'validation_results'
    
    validation_id = Column(String, primary_key=True)
    data_source = Column(String, nullable=False)
    total_records = Column(Integer)
    records_validated = Column(Integer)
    validation_rules_applied = Column(JSON)
    passed_validations = Column(Integer)
    failed_validations = Column(Integer)
    warning_validations = Column(Integer)
    errors = Column(JSON)
    summary = Column(JSON)
    validation_duration = Column(Float)
    validated_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False)

class ValidationAgent:
    """Production-ready Validation Agent for comprehensive data validation"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # Validation rules
        self.validation_rules = {}
        
        # Schema validators
        self.schema_validators = {}
        
        # Business rule validators
        self.business_rules = {}
        
        # Anomaly detection models
        self.anomaly_models = {}
        
        # Reference data for validation
        self.reference_data = {}
        
        self._initialize_validation_rules()
        self._initialize_schema_validators()
        self._initialize_business_rules()
        self._initialize_anomaly_models()
        self._load_reference_data()
        
        logger.info("ValidationAgent initialized successfully")

    def _initialize_validation_rules(self):
        """Initialize standard validation rules"""
        
        # Policy validation rules
        self.validation_rules['policy'] = [
            ValidationRule(
                rule_id="POL001",
                rule_name="Policy Number Format",
                rule_description="Policy number must follow standard format",
                validation_type=ValidationType.FORMAT,
                severity=ValidationSeverity.ERROR,
                field_names=["policy_number"],
                rule_logic={"pattern": r"^[A-Z]{2}\d{8}$"},
                error_message="Policy number must be 2 letters followed by 8 digits",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            ValidationRule(
                rule_id="POL002",
                rule_name="Policy Effective Date",
                rule_description="Policy effective date must be valid and not in the past",
                validation_type=ValidationType.BUSINESS_RULE,
                severity=ValidationSeverity.ERROR,
                field_names=["effective_date"],
                rule_logic={"min_date": "today", "max_date": "today+1year"},
                error_message="Policy effective date must be between today and one year from today",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            ValidationRule(
                rule_id="POL003",
                rule_name="Premium Amount Range",
                rule_description="Premium amount must be within reasonable range",
                validation_type=ValidationType.RANGE,
                severity=ValidationSeverity.WARNING,
                field_names=["premium_amount"],
                rule_logic={"min_value": 100, "max_value": 50000},
                error_message="Premium amount should be between $100 and $50,000",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        ]
        
        # Claim validation rules
        self.validation_rules['claim'] = [
            ValidationRule(
                rule_id="CLM001",
                rule_name="Claim Number Format",
                rule_description="Claim number must follow standard format",
                validation_type=ValidationType.FORMAT,
                severity=ValidationSeverity.ERROR,
                field_names=["claim_number"],
                rule_logic={"pattern": r"^CLM\d{10}$"},
                error_message="Claim number must be 'CLM' followed by 10 digits",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            ValidationRule(
                rule_id="CLM002",
                rule_name="Loss Date Validation",
                rule_description="Loss date must be before or equal to report date",
                validation_type=ValidationType.BUSINESS_RULE,
                severity=ValidationSeverity.ERROR,
                field_names=["loss_date", "report_date"],
                rule_logic={"comparison": "loss_date <= report_date"},
                error_message="Loss date cannot be after report date",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            ValidationRule(
                rule_id="CLM003",
                rule_name="Claim Amount Reasonableness",
                rule_description="Claim amount must be reasonable for coverage type",
                validation_type=ValidationType.BUSINESS_RULE,
                severity=ValidationSeverity.WARNING,
                field_names=["claim_amount", "coverage_type"],
                rule_logic={"coverage_limits": {"liability": 1000000, "collision": 100000, "comprehensive": 75000}},
                error_message="Claim amount exceeds typical limits for coverage type",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        ]
        
        # Customer validation rules
        self.validation_rules['customer'] = [
            ValidationRule(
                rule_id="CUST001",
                rule_name="Email Format",
                rule_description="Email address must be valid format",
                validation_type=ValidationType.FORMAT,
                severity=ValidationSeverity.ERROR,
                field_names=["email"],
                rule_logic={"validation_type": "email"},
                error_message="Invalid email address format",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            ValidationRule(
                rule_id="CUST002",
                rule_name="Phone Number Format",
                rule_description="Phone number must be valid US format",
                validation_type=ValidationType.FORMAT,
                severity=ValidationSeverity.ERROR,
                field_names=["phone"],
                rule_logic={"pattern": r"^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$"},
                error_message="Invalid phone number format",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            ValidationRule(
                rule_id="CUST003",
                rule_name="Date of Birth Range",
                rule_description="Date of birth must be reasonable",
                validation_type=ValidationType.RANGE,
                severity=ValidationSeverity.ERROR,
                field_names=["date_of_birth"],
                rule_logic={"min_age": 16, "max_age": 120},
                error_message="Date of birth indicates age outside reasonable range (16-120 years)",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        ]

    def _initialize_schema_validators(self):
        """Initialize JSON schema validators"""
        
        # Policy schema
        self.schema_validators['policy'] = {
            "type": "object",
            "properties": {
                "policy_number": {"type": "string", "pattern": "^[A-Z]{2}\\d{8}$"},
                "policy_holder_name": {"type": "string", "minLength": 1, "maxLength": 100},
                "effective_date": {"type": "string", "format": "date"},
                "expiration_date": {"type": "string", "format": "date"},
                "premium_amount": {"type": "number", "minimum": 0, "maximum": 1000000},
                "coverage_type": {"type": "string", "enum": ["liability", "collision", "comprehensive", "uninsured"]},
                "deductible": {"type": "number", "minimum": 0, "maximum": 10000},
                "limits": {
                    "type": "object",
                    "properties": {
                        "bodily_injury_per_person": {"type": "number", "minimum": 0},
                        "bodily_injury_per_accident": {"type": "number", "minimum": 0},
                        "property_damage": {"type": "number", "minimum": 0}
                    }
                }
            },
            "required": ["policy_number", "policy_holder_name", "effective_date", "premium_amount"]
        }
        
        # Claim schema
        self.schema_validators['claim'] = {
            "type": "object",
            "properties": {
                "claim_number": {"type": "string", "pattern": "^CLM\\d{10}$"},
                "policy_number": {"type": "string", "pattern": "^[A-Z]{2}\\d{8}$"},
                "loss_date": {"type": "string", "format": "date"},
                "report_date": {"type": "string", "format": "date"},
                "claim_amount": {"type": "number", "minimum": 0, "maximum": 10000000},
                "coverage_type": {"type": "string", "enum": ["liability", "collision", "comprehensive", "uninsured"]},
                "loss_description": {"type": "string", "minLength": 10, "maxLength": 1000},
                "claimant_name": {"type": "string", "minLength": 1, "maxLength": 100},
                "status": {"type": "string", "enum": ["open", "closed", "pending", "denied"]},
                "adjuster_id": {"type": "string", "pattern": "^ADJ\\d{6}$"}
            },
            "required": ["claim_number", "policy_number", "loss_date", "report_date", "claim_amount"]
        }
        
        # Customer schema
        self.schema_validators['customer'] = {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "pattern": "^CUST\\d{8}$"},
                "first_name": {"type": "string", "minLength": 1, "maxLength": 50},
                "last_name": {"type": "string", "minLength": 1, "maxLength": 50},
                "email": {"type": "string", "format": "email"},
                "phone": {"type": "string", "pattern": "^\\+?1?[-\\.\\s]?\\(?[0-9]{3}\\)?[-\\.\\s]?[0-9]{3}[-\\.\\s]?[0-9]{4}$"},
                "date_of_birth": {"type": "string", "format": "date"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string", "minLength": 1, "maxLength": 100},
                        "city": {"type": "string", "minLength": 1, "maxLength": 50},
                        "state": {"type": "string", "pattern": "^[A-Z]{2}$"},
                        "zip_code": {"type": "string", "pattern": "^\\d{5}(-\\d{4})?$"}
                    },
                    "required": ["street", "city", "state", "zip_code"]
                },
                "driver_license": {
                    "type": "object",
                    "properties": {
                        "license_number": {"type": "string", "minLength": 5, "maxLength": 20},
                        "state": {"type": "string", "pattern": "^[A-Z]{2}$"},
                        "expiration_date": {"type": "string", "format": "date"}
                    },
                    "required": ["license_number", "state"]
                }
            },
            "required": ["customer_id", "first_name", "last_name", "email", "phone"]
        }

    def _initialize_business_rules(self):
        """Initialize business rule validators"""
        
        self.business_rules = {
            'policy_consistency': {
                'rule': 'effective_date < expiration_date',
                'message': 'Policy effective date must be before expiration date'
            },
            'claim_timing': {
                'rule': 'loss_date <= report_date <= today',
                'message': 'Loss date must be before or equal to report date, and report date cannot be in the future'
            },
            'coverage_limits': {
                'rule': 'claim_amount <= policy_limits[coverage_type]',
                'message': 'Claim amount cannot exceed policy coverage limits'
            },
            'age_driving': {
                'rule': 'age >= 16',
                'message': 'Driver must be at least 16 years old'
            },
            'premium_calculation': {
                'rule': 'premium_amount >= base_premium * risk_factor',
                'message': 'Premium amount appears too low for risk profile'
            }
        }

    def _initialize_anomaly_models(self):
        """Initialize anomaly detection models"""
        
        # Isolation Forest for numerical anomalies
        self.anomaly_models['numerical'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # DBSCAN for clustering-based anomalies
        self.anomaly_models['clustering'] = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        
        # Standard scaler for preprocessing
        self.anomaly_models['scaler'] = StandardScaler()

    def _load_reference_data(self):
        """Load reference data for validation"""
        
        # State codes
        self.reference_data['state_codes'] = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        }
        
        # Coverage types
        self.reference_data['coverage_types'] = {
            'liability', 'collision', 'comprehensive', 'uninsured_motorist',
            'underinsured_motorist', 'personal_injury_protection', 'medical_payments'
        }
        
        # Claim statuses
        self.reference_data['claim_statuses'] = {
            'open', 'closed', 'pending', 'denied', 'settled', 'litigated'
        }
        
        # Vehicle makes (sample)
        self.reference_data['vehicle_makes'] = {
            'TOYOTA', 'HONDA', 'FORD', 'CHEVROLET', 'NISSAN', 'BMW', 'MERCEDES',
            'AUDI', 'VOLKSWAGEN', 'HYUNDAI', 'KIA', 'SUBARU', 'MAZDA', 'ACURA',
            'LEXUS', 'INFINITI', 'CADILLAC', 'LINCOLN', 'BUICK', 'GMC'
        }

    async def validate_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                          data_type: str, 
                          validation_level: str = "comprehensive") -> ValidationResult:
        """Validate data using comprehensive validation rules"""
        
        validation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        validation_checks_total.labels(validation_type=data_type, status='started').inc()
        
        try:
            with validation_duration.time():
                # Normalize data to list format
                if isinstance(data, dict):
                    data_list = [data]
                else:
                    data_list = data
                
                total_records = len(data_list)
                errors = []
                passed_count = 0
                failed_count = 0
                warning_count = 0
                
                # Get validation rules for data type
                rules = self.validation_rules.get(data_type, [])
                rule_ids = [rule.rule_id for rule in rules]
                
                # Validate each record
                for i, record in enumerate(data_list):
                    record_id = record.get('id', f"record_{i}")
                    
                    # Schema validation
                    schema_errors = await self._validate_schema(record, data_type, record_id)
                    errors.extend(schema_errors)
                    
                    # Field validation
                    field_errors = await self._validate_fields(record, rules, record_id)
                    errors.extend(field_errors)
                    
                    # Business rule validation
                    business_errors = await self._validate_business_rules(record, data_type, record_id)
                    errors.extend(business_errors)
                    
                    # Completeness validation
                    completeness_errors = await self._validate_completeness(record, data_type, record_id)
                    errors.extend(completeness_errors)
                    
                    # Consistency validation
                    consistency_errors = await self._validate_consistency(record, data_type, record_id)
                    errors.extend(consistency_errors)
                    
                    if validation_level == "comprehensive":
                        # Anomaly detection
                        anomaly_errors = await self._detect_anomalies(record, data_type, record_id)
                        errors.extend(anomaly_errors)
                        
                        # Cross-reference validation
                        reference_errors = await self._validate_references(record, data_type, record_id)
                        errors.extend(reference_errors)
                
                # Count validation results
                for error in errors:
                    if error.severity == ValidationSeverity.ERROR or error.severity == ValidationSeverity.CRITICAL:
                        failed_count += 1
                    elif error.severity == ValidationSeverity.WARNING:
                        warning_count += 1
                    else:
                        passed_count += 1
                
                # Calculate summary statistics
                validation_duration_seconds = (datetime.utcnow() - start_time).total_seconds()
                
                summary = {
                    'validation_rate': (total_records - failed_count) / total_records if total_records > 0 else 0,
                    'error_rate': failed_count / total_records if total_records > 0 else 0,
                    'warning_rate': warning_count / total_records if total_records > 0 else 0,
                    'most_common_errors': self._get_most_common_errors(errors),
                    'validation_coverage': len(rule_ids),
                    'data_quality_score': self._calculate_data_quality_score(errors, total_records)
                }
                
                # Create validation result
                result = ValidationResult(
                    validation_id=validation_id,
                    data_source=data_type,
                    total_records=total_records,
                    records_validated=total_records,
                    validation_rules_applied=rule_ids,
                    passed_validations=passed_count,
                    failed_validations=failed_count,
                    warning_validations=warning_count,
                    errors=errors,
                    summary=summary,
                    validation_duration=validation_duration_seconds,
                    validated_at=start_time
                )
                
                # Store validation result
                await self._store_validation_result(result)
                
                # Update metrics
                validation_checks_total.labels(validation_type=data_type, status='completed').inc()
                validation_errors_gauge.set(failed_count)
                
                return result
                
        except Exception as e:
            validation_checks_total.labels(validation_type=data_type, status='failed').inc()
            logger.error(f"Data validation failed: {e}")
            raise

    async def _validate_schema(self, record: Dict[str, Any], data_type: str, record_id: str) -> List[ValidationError]:
        """Validate record against JSON schema"""
        
        errors = []
        
        try:
            schema = self.schema_validators.get(data_type)
            if schema:
                jsonschema.validate(record, schema)
        except jsonschema.ValidationError as e:
            error = ValidationError(
                error_id=str(uuid.uuid4()),
                rule_id="SCHEMA001",
                field_name=e.path[-1] if e.path else "unknown",
                error_message=f"Schema validation failed: {e.message}",
                severity=ValidationSeverity.ERROR,
                actual_value=e.instance,
                expected_value=None,
                record_id=record_id,
                context={"schema_path": list(e.path)},
                detected_at=datetime.utcnow()
            )
            errors.append(error)
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
        
        return errors

    async def _validate_fields(self, record: Dict[str, Any], rules: List[ValidationRule], record_id: str) -> List[ValidationError]:
        """Validate individual fields against validation rules"""
        
        errors = []
        
        for rule in rules:
            if not rule.is_active:
                continue
            
            try:
                # Check if all required fields are present
                missing_fields = [field for field in rule.field_names if field not in record]
                if missing_fields:
                    continue  # Skip rule if required fields are missing
                
                # Apply validation based on type
                if rule.validation_type == ValidationType.FORMAT:
                    field_errors = await self._validate_format(record, rule, record_id)
                elif rule.validation_type == ValidationType.RANGE:
                    field_errors = await self._validate_range(record, rule, record_id)
                elif rule.validation_type == ValidationType.DATA_TYPE:
                    field_errors = await self._validate_data_type(record, rule, record_id)
                else:
                    continue
                
                errors.extend(field_errors)
                
            except Exception as e:
                logger.error(f"Field validation error for rule {rule.rule_id}: {e}")
        
        return errors

    async def _validate_format(self, record: Dict[str, Any], rule: ValidationRule, record_id: str) -> List[ValidationError]:
        """Validate field format"""
        
        errors = []
        
        for field_name in rule.field_names:
            value = record.get(field_name)
            if value is None:
                continue
            
            try:
                if rule.rule_logic.get('validation_type') == 'email':
                    # Email validation
                    validate_email(str(value))
                elif 'pattern' in rule.rule_logic:
                    # Regex pattern validation
                    pattern = rule.rule_logic['pattern']
                    if not re.match(pattern, str(value)):
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id=rule.rule_id,
                            field_name=field_name,
                            error_message=rule.error_message,
                            severity=rule.severity,
                            actual_value=value,
                            expected_value=pattern,
                            record_id=record_id,
                            context={"rule_logic": rule.rule_logic},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
                        
            except EmailNotValidError:
                error = ValidationError(
                    error_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    field_name=field_name,
                    error_message=rule.error_message,
                    severity=rule.severity,
                    actual_value=value,
                    expected_value="valid email format",
                    record_id=record_id,
                    context={"validation_type": "email"},
                    detected_at=datetime.utcnow()
                )
                errors.append(error)
            except Exception as e:
                logger.error(f"Format validation error: {e}")
        
        return errors

    async def _validate_range(self, record: Dict[str, Any], rule: ValidationRule, record_id: str) -> List[ValidationError]:
        """Validate field range"""
        
        errors = []
        
        for field_name in rule.field_names:
            value = record.get(field_name)
            if value is None:
                continue
            
            try:
                # Numeric range validation
                if 'min_value' in rule.rule_logic or 'max_value' in rule.rule_logic:
                    numeric_value = float(value)
                    
                    min_val = rule.rule_logic.get('min_value')
                    max_val = rule.rule_logic.get('max_value')
                    
                    if min_val is not None and numeric_value < min_val:
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id=rule.rule_id,
                            field_name=field_name,
                            error_message=f"{rule.error_message} (minimum: {min_val})",
                            severity=rule.severity,
                            actual_value=value,
                            expected_value=f">= {min_val}",
                            record_id=record_id,
                            context={"min_value": min_val, "max_value": max_val},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
                    
                    if max_val is not None and numeric_value > max_val:
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id=rule.rule_id,
                            field_name=field_name,
                            error_message=f"{rule.error_message} (maximum: {max_val})",
                            severity=rule.severity,
                            actual_value=value,
                            expected_value=f"<= {max_val}",
                            record_id=record_id,
                            context={"min_value": min_val, "max_value": max_val},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
                
                # Date range validation
                elif 'min_age' in rule.rule_logic or 'max_age' in rule.rule_logic:
                    date_value = datetime.strptime(str(value), '%Y-%m-%d').date()
                    today = datetime.utcnow().date()
                    age = (today - date_value).days / 365.25
                    
                    min_age = rule.rule_logic.get('min_age')
                    max_age = rule.rule_logic.get('max_age')
                    
                    if min_age is not None and age < min_age:
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id=rule.rule_id,
                            field_name=field_name,
                            error_message=f"{rule.error_message} (minimum age: {min_age})",
                            severity=rule.severity,
                            actual_value=value,
                            expected_value=f"age >= {min_age}",
                            record_id=record_id,
                            context={"calculated_age": age, "min_age": min_age, "max_age": max_age},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
                    
                    if max_age is not None and age > max_age:
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id=rule.rule_id,
                            field_name=field_name,
                            error_message=f"{rule.error_message} (maximum age: {max_age})",
                            severity=rule.severity,
                            actual_value=value,
                            expected_value=f"age <= {max_age}",
                            record_id=record_id,
                            context={"calculated_age": age, "min_age": min_age, "max_age": max_age},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
                        
            except (ValueError, TypeError) as e:
                error = ValidationError(
                    error_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    field_name=field_name,
                    error_message=f"Invalid data type for range validation: {e}",
                    severity=ValidationSeverity.ERROR,
                    actual_value=value,
                    expected_value="numeric or date value",
                    record_id=record_id,
                    context={"error": str(e)},
                    detected_at=datetime.utcnow()
                )
                errors.append(error)
            except Exception as e:
                logger.error(f"Range validation error: {e}")
        
        return errors

    async def _validate_data_type(self, record: Dict[str, Any], rule: ValidationRule, record_id: str) -> List[ValidationError]:
        """Validate field data types"""
        
        errors = []
        
        for field_name in rule.field_names:
            value = record.get(field_name)
            if value is None:
                continue
            
            expected_type = rule.rule_logic.get('expected_type')
            if not expected_type:
                continue
            
            try:
                if expected_type == 'string' and not isinstance(value, str):
                    error = ValidationError(
                        error_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        field_name=field_name,
                        error_message=f"Expected string, got {type(value).__name__}",
                        severity=rule.severity,
                        actual_value=value,
                        expected_value="string",
                        record_id=record_id,
                        context={"actual_type": type(value).__name__},
                        detected_at=datetime.utcnow()
                    )
                    errors.append(error)
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    error = ValidationError(
                        error_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        field_name=field_name,
                        error_message=f"Expected number, got {type(value).__name__}",
                        severity=rule.severity,
                        actual_value=value,
                        expected_value="number",
                        record_id=record_id,
                        context={"actual_type": type(value).__name__},
                        detected_at=datetime.utcnow()
                    )
                    errors.append(error)
                elif expected_type == 'boolean' and not isinstance(value, bool):
                    error = ValidationError(
                        error_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        field_name=field_name,
                        error_message=f"Expected boolean, got {type(value).__name__}",
                        severity=rule.severity,
                        actual_value=value,
                        expected_value="boolean",
                        record_id=record_id,
                        context={"actual_type": type(value).__name__},
                        detected_at=datetime.utcnow()
                    )
                    errors.append(error)
                    
            except Exception as e:
                logger.error(f"Data type validation error: {e}")
        
        return errors

    async def _validate_business_rules(self, record: Dict[str, Any], data_type: str, record_id: str) -> List[ValidationError]:
        """Validate business rules"""
        
        errors = []
        
        try:
            # Policy-specific business rules
            if data_type == 'policy':
                # Effective date must be before expiration date
                if 'effective_date' in record and 'expiration_date' in record:
                    effective = datetime.strptime(record['effective_date'], '%Y-%m-%d').date()
                    expiration = datetime.strptime(record['expiration_date'], '%Y-%m-%d').date()
                    
                    if effective >= expiration:
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id="BUS001",
                            field_name="effective_date",
                            error_message="Policy effective date must be before expiration date",
                            severity=ValidationSeverity.ERROR,
                            actual_value=record['effective_date'],
                            expected_value=f"< {record['expiration_date']}",
                            record_id=record_id,
                            context={"effective_date": record['effective_date'], "expiration_date": record['expiration_date']},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
            
            # Claim-specific business rules
            elif data_type == 'claim':
                # Loss date must be before or equal to report date
                if 'loss_date' in record and 'report_date' in record:
                    loss_date = datetime.strptime(record['loss_date'], '%Y-%m-%d').date()
                    report_date = datetime.strptime(record['report_date'], '%Y-%m-%d').date()
                    
                    if loss_date > report_date:
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id="BUS002",
                            field_name="loss_date",
                            error_message="Loss date cannot be after report date",
                            severity=ValidationSeverity.ERROR,
                            actual_value=record['loss_date'],
                            expected_value=f"<= {record['report_date']}",
                            record_id=record_id,
                            context={"loss_date": record['loss_date'], "report_date": record['report_date']},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
                
                # Report date cannot be in the future
                if 'report_date' in record:
                    report_date = datetime.strptime(record['report_date'], '%Y-%m-%d').date()
                    today = datetime.utcnow().date()
                    
                    if report_date > today:
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id="BUS003",
                            field_name="report_date",
                            error_message="Report date cannot be in the future",
                            severity=ValidationSeverity.ERROR,
                            actual_value=record['report_date'],
                            expected_value=f"<= {today}",
                            record_id=record_id,
                            context={"report_date": record['report_date'], "today": str(today)},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
            
            # Customer-specific business rules
            elif data_type == 'customer':
                # Driver must be at least 16 years old
                if 'date_of_birth' in record:
                    birth_date = datetime.strptime(record['date_of_birth'], '%Y-%m-%d').date()
                    today = datetime.utcnow().date()
                    age = (today - birth_date).days / 365.25
                    
                    if age < 16:
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id="BUS004",
                            field_name="date_of_birth",
                            error_message="Driver must be at least 16 years old",
                            severity=ValidationSeverity.ERROR,
                            actual_value=record['date_of_birth'],
                            expected_value="age >= 16",
                            record_id=record_id,
                            context={"calculated_age": age, "minimum_age": 16},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
                        
        except Exception as e:
            logger.error(f"Business rule validation error: {e}")
        
        return errors

    async def _validate_completeness(self, record: Dict[str, Any], data_type: str, record_id: str) -> List[ValidationError]:
        """Validate data completeness"""
        
        errors = []
        
        # Define required fields by data type
        required_fields = {
            'policy': ['policy_number', 'policy_holder_name', 'effective_date', 'premium_amount'],
            'claim': ['claim_number', 'policy_number', 'loss_date', 'report_date', 'claim_amount'],
            'customer': ['customer_id', 'first_name', 'last_name', 'email', 'phone']
        }
        
        required = required_fields.get(data_type, [])
        
        for field in required:
            if field not in record or record[field] is None or record[field] == '':
                error = ValidationError(
                    error_id=str(uuid.uuid4()),
                    rule_id="COMP001",
                    field_name=field,
                    error_message=f"Required field '{field}' is missing or empty",
                    severity=ValidationSeverity.ERROR,
                    actual_value=record.get(field),
                    expected_value="non-empty value",
                    record_id=record_id,
                    context={"required_fields": required},
                    detected_at=datetime.utcnow()
                )
                errors.append(error)
        
        return errors

    async def _validate_consistency(self, record: Dict[str, Any], data_type: str, record_id: str) -> List[ValidationError]:
        """Validate internal data consistency"""
        
        errors = []
        
        try:
            # Check for duplicate values in fields that should be unique
            if data_type == 'policy' and 'policy_number' in record:
                # This would typically check against database for uniqueness
                # For now, just validate format consistency
                pass
            
            # Check for logical consistency in related fields
            if data_type == 'claim':
                # Claim amount should be reasonable for coverage type
                if 'claim_amount' in record and 'coverage_type' in record:
                    claim_amount = float(record['claim_amount'])
                    coverage_type = record['coverage_type']
                    
                    # Define reasonable limits by coverage type
                    coverage_limits = {
                        'liability': 1000000,
                        'collision': 100000,
                        'comprehensive': 75000,
                        'uninsured': 500000
                    }
                    
                    limit = coverage_limits.get(coverage_type, 100000)
                    
                    if claim_amount > limit:
                        error = ValidationError(
                            error_id=str(uuid.uuid4()),
                            rule_id="CONS001",
                            field_name="claim_amount",
                            error_message=f"Claim amount ${claim_amount:,.2f} is unusually high for {coverage_type} coverage",
                            severity=ValidationSeverity.WARNING,
                            actual_value=claim_amount,
                            expected_value=f"<= {limit}",
                            record_id=record_id,
                            context={"coverage_type": coverage_type, "typical_limit": limit},
                            detected_at=datetime.utcnow()
                        )
                        errors.append(error)
                        
        except Exception as e:
            logger.error(f"Consistency validation error: {e}")
        
        return errors

    async def _detect_anomalies(self, record: Dict[str, Any], data_type: str, record_id: str) -> List[ValidationError]:
        """Detect anomalies using machine learning"""
        
        errors = []
        
        try:
            # Extract numerical features for anomaly detection
            numerical_features = []
            feature_names = []
            
            if data_type == 'policy':
                if 'premium_amount' in record:
                    numerical_features.append(float(record['premium_amount']))
                    feature_names.append('premium_amount')
                if 'deductible' in record:
                    numerical_features.append(float(record['deductible']))
                    feature_names.append('deductible')
            elif data_type == 'claim':
                if 'claim_amount' in record:
                    numerical_features.append(float(record['claim_amount']))
                    feature_names.append('claim_amount')
            
            if len(numerical_features) >= 2:
                # Reshape for sklearn
                features = np.array(numerical_features).reshape(1, -1)
                
                # Scale features
                scaled_features = self.anomaly_models['scaler'].fit_transform(features)
                
                # Detect anomalies
                anomaly_score = self.anomaly_models['numerical'].decision_function(scaled_features)[0]
                is_anomaly = self.anomaly_models['numerical'].predict(scaled_features)[0] == -1
                
                if is_anomaly:
                    error = ValidationError(
                        error_id=str(uuid.uuid4()),
                        rule_id="ANOM001",
                        field_name="multiple_fields",
                        error_message=f"Anomalous data pattern detected (score: {anomaly_score:.3f})",
                        severity=ValidationSeverity.WARNING,
                        actual_value=dict(zip(feature_names, numerical_features)),
                        expected_value="typical data pattern",
                        record_id=record_id,
                        context={"anomaly_score": anomaly_score, "features": feature_names},
                        detected_at=datetime.utcnow()
                    )
                    errors.append(error)
                    
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
        
        return errors

    async def _validate_references(self, record: Dict[str, Any], data_type: str, record_id: str) -> List[ValidationError]:
        """Validate against reference data"""
        
        errors = []
        
        try:
            # Validate state codes
            if 'state' in record:
                state = record['state'].upper()
                if state not in self.reference_data['state_codes']:
                    error = ValidationError(
                        error_id=str(uuid.uuid4()),
                        rule_id="REF001",
                        field_name="state",
                        error_message=f"Invalid state code: {state}",
                        severity=ValidationSeverity.ERROR,
                        actual_value=record['state'],
                        expected_value="valid US state code",
                        record_id=record_id,
                        context={"valid_states": list(self.reference_data['state_codes'])},
                        detected_at=datetime.utcnow()
                    )
                    errors.append(error)
            
            # Validate coverage types
            if 'coverage_type' in record:
                coverage = record['coverage_type'].lower()
                if coverage not in self.reference_data['coverage_types']:
                    error = ValidationError(
                        error_id=str(uuid.uuid4()),
                        rule_id="REF002",
                        field_name="coverage_type",
                        error_message=f"Invalid coverage type: {coverage}",
                        severity=ValidationSeverity.ERROR,
                        actual_value=record['coverage_type'],
                        expected_value="valid coverage type",
                        record_id=record_id,
                        context={"valid_coverage_types": list(self.reference_data['coverage_types'])},
                        detected_at=datetime.utcnow()
                    )
                    errors.append(error)
            
            # Validate claim statuses
            if 'status' in record and data_type == 'claim':
                status = record['status'].lower()
                if status not in self.reference_data['claim_statuses']:
                    error = ValidationError(
                        error_id=str(uuid.uuid4()),
                        rule_id="REF003",
                        field_name="status",
                        error_message=f"Invalid claim status: {status}",
                        severity=ValidationSeverity.ERROR,
                        actual_value=record['status'],
                        expected_value="valid claim status",
                        record_id=record_id,
                        context={"valid_statuses": list(self.reference_data['claim_statuses'])},
                        detected_at=datetime.utcnow()
                    )
                    errors.append(error)
                    
        except Exception as e:
            logger.error(f"Reference validation error: {e}")
        
        return errors

    def _get_most_common_errors(self, errors: List[ValidationError]) -> List[Dict[str, Any]]:
        """Get most common validation errors"""
        
        error_counts = {}
        
        for error in errors:
            key = f"{error.rule_id}:{error.field_name}"
            if key not in error_counts:
                error_counts[key] = {
                    'rule_id': error.rule_id,
                    'field_name': error.field_name,
                    'error_message': error.error_message,
                    'severity': error.severity.value,
                    'count': 0
                }
            error_counts[key]['count'] += 1
        
        # Sort by count and return top 10
        sorted_errors = sorted(error_counts.values(), key=lambda x: x['count'], reverse=True)
        return sorted_errors[:10]

    def _calculate_data_quality_score(self, errors: List[ValidationError], total_records: int) -> float:
        """Calculate overall data quality score"""
        
        if total_records == 0:
            return 0.0
        
        # Weight errors by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 4,
            ValidationSeverity.ERROR: 3,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.INFO: 0.5
        }
        
        total_weighted_errors = sum(severity_weights.get(error.severity, 1) for error in errors)
        max_possible_errors = total_records * 4  # Assuming worst case all critical errors
        
        # Calculate score (0-100)
        if max_possible_errors > 0:
            quality_score = max(0, 100 - (total_weighted_errors / max_possible_errors * 100))
        else:
            quality_score = 100
        
        return round(quality_score, 2)

    async def _store_validation_result(self, result: ValidationResult):
        """Store validation result in database"""
        
        try:
            with self.Session() as session:
                record = ValidationResultRecord(
                    validation_id=result.validation_id,
                    data_source=result.data_source,
                    total_records=result.total_records,
                    records_validated=result.records_validated,
                    validation_rules_applied=result.validation_rules_applied,
                    passed_validations=result.passed_validations,
                    failed_validations=result.failed_validations,
                    warning_validations=result.warning_validations,
                    errors=[asdict(error) for error in result.errors],
                    summary=result.summary,
                    validation_duration=result.validation_duration,
                    validated_at=result.validated_at,
                    created_at=datetime.utcnow()
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing validation result: {e}")

def create_validation_agent(db_url: str = None, redis_url: str = None) -> ValidationAgent:
    """Create and configure ValidationAgent instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return ValidationAgent(db_url=db_url, redis_url=redis_url)

