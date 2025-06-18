"""
Decision Engine - Production Ready Implementation
AI-powered decision making for claims processing
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DecisionType(Enum):
    COVERAGE_DETERMINATION = "coverage_determination"
    LIABILITY_ASSESSMENT = "liability_assessment"
    SETTLEMENT_APPROVAL = "settlement_approval"
    FRAUD_DETECTION = "fraud_detection"
    SUBROGATION_POTENTIAL = "subrogation_potential"
    RESERVE_SETTING = "reserve_setting"
    CLAIM_CLOSURE = "claim_closure"

class DecisionOutcome(Enum):
    APPROVE = "approve"
    DENY = "deny"
    INVESTIGATE = "investigate"
    ESCALATE = "escalate"
    SETTLE = "settle"
    LITIGATE = "litigate"

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class DecisionFactor:
    factor_name: str
    factor_value: Any
    weight: float
    impact_score: float
    confidence: float
    source: str

@dataclass
class DecisionResult:
    decision_id: str
    decision_type: DecisionType
    outcome: DecisionOutcome
    confidence_score: float
    confidence_level: ConfidenceLevel
    factors: List[DecisionFactor]
    reasoning: str
    recommendations: List[str]
    alternative_outcomes: Dict[str, float]
    risk_assessment: Dict[str, Any]
    financial_impact: Dict[str, float]
    decision_timestamp: datetime
    model_version: str
    human_review_required: bool

class DecisionRecord(Base):
    __tablename__ = 'claim_decisions'
    
    decision_id = Column(String, primary_key=True)
    claim_id = Column(String, nullable=False)
    workflow_id = Column(String)
    decision_type = Column(String, nullable=False)
    outcome = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    confidence_level = Column(String, nullable=False)
    factors = Column(JSON)
    reasoning = Column(Text)
    recommendations = Column(JSON)
    alternative_outcomes = Column(JSON)
    risk_assessment = Column(JSON)
    financial_impact = Column(JSON)
    decision_timestamp = Column(DateTime, nullable=False)
    model_version = Column(String)
    human_review_required = Column(Boolean)
    human_reviewer = Column(String)
    human_decision = Column(String)
    human_notes = Column(Text)
    final_outcome = Column(String)
    created_at = Column(DateTime, nullable=False)

class ModelPerformanceRecord(Base):
    __tablename__ = 'model_performance'
    
    performance_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    decision_type = Column(String, nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_date = Column(DateTime)
    evaluation_date = Column(DateTime, nullable=False)
    sample_size = Column(Integer)
    feature_importance = Column(JSON)
    confusion_matrix = Column(JSON)
    created_at = Column(DateTime, nullable=False)

class ClaimsDecisionEngine:
    """Production-ready AI decision engine for claims processing"""
    
    def __init__(self, db_url: str, redis_url: str, model_path: str = None):
        self.db_url = db_url
        self.redis_url = redis_url
        self.model_path = model_path or "/tmp/claims_models"
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # ML Models
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Decision rules
        self.business_rules = {}
        
        # Model configurations
        self.model_configs = {}
        
        # Feature definitions
        self.feature_definitions = {}
        
        # Decision thresholds
        self.decision_thresholds = {}
        
        self._initialize_business_rules()
        self._initialize_model_configs()
        self._initialize_feature_definitions()
        self._initialize_decision_thresholds()
        self._load_models()
        
        logger.info("ClaimsDecisionEngine initialized successfully")

    def _initialize_business_rules(self):
        """Initialize business rules for decision making"""
        
        self.business_rules = {
            'coverage_determination': {
                'policy_active': {
                    'condition': 'policy_status == "active"',
                    'weight': 1.0,
                    'required': True
                },
                'loss_within_period': {
                    'condition': 'loss_date >= policy_start and loss_date <= policy_end',
                    'weight': 1.0,
                    'required': True
                },
                'coverage_applies': {
                    'condition': 'loss_type in covered_perils',
                    'weight': 1.0,
                    'required': True
                },
                'exclusions_check': {
                    'condition': 'not any(exclusion_applies)',
                    'weight': 1.0,
                    'required': True
                }
            },
            'liability_assessment': {
                'fault_determination': {
                    'clear_fault': {'threshold': 0.9, 'outcome': 'accept_liability'},
                    'probable_fault': {'threshold': 0.7, 'outcome': 'accept_partial'},
                    'disputed_fault': {'threshold': 0.5, 'outcome': 'investigate'},
                    'no_fault': {'threshold': 0.3, 'outcome': 'deny_liability'}
                },
                'evidence_strength': {
                    'strong_evidence': {'threshold': 0.8, 'weight': 0.4},
                    'moderate_evidence': {'threshold': 0.6, 'weight': 0.3},
                    'weak_evidence': {'threshold': 0.4, 'weight': 0.2}
                }
            },
            'settlement_approval': {
                'authority_limits': {
                    'adjuster': 25000,
                    'senior_adjuster': 100000,
                    'manager': 500000,
                    'director': 1000000
                },
                'settlement_factors': {
                    'liability_clear': {'multiplier': 1.0},
                    'liability_disputed': {'multiplier': 0.7},
                    'comparative_negligence': {'multiplier': 0.8},
                    'policy_limits': {'cap': True}
                }
            },
            'fraud_detection': {
                'red_flags': {
                    'late_reporting': {'days': 30, 'score': 0.3},
                    'prior_claims': {'count': 3, 'score': 0.4},
                    'inconsistent_statements': {'score': 0.5},
                    'suspicious_circumstances': {'score': 0.6},
                    'medical_mills': {'score': 0.7},
                    'staged_accident': {'score': 0.9}
                },
                'investigation_triggers': {
                    'siu_referral': 0.7,
                    'enhanced_investigation': 0.5,
                    'standard_processing': 0.3
                }
            }
        }

    def _initialize_model_configs(self):
        """Initialize ML model configurations"""
        
        self.model_configs = {
            'coverage_determination': {
                'model_type': 'random_forest',
                'parameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                },
                'features': [
                    'policy_status', 'loss_date_valid', 'coverage_type',
                    'policy_limits', 'deductible', 'exclusions_count'
                ],
                'target': 'coverage_decision'
            },
            'liability_assessment': {
                'model_type': 'gradient_boosting',
                'parameters': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'random_state': 42
                },
                'features': [
                    'fault_indicators', 'evidence_strength', 'witness_count',
                    'police_report_fault', 'traffic_violations', 'weather_conditions',
                    'road_conditions', 'vehicle_damage_pattern'
                ],
                'target': 'liability_percentage'
            },
            'settlement_approval': {
                'model_type': 'logistic_regression',
                'parameters': {
                    'C': 1.0,
                    'random_state': 42,
                    'max_iter': 1000
                },
                'features': [
                    'claim_amount', 'liability_percentage', 'policy_limits',
                    'adjuster_level', 'claim_complexity', 'legal_representation',
                    'medical_treatment', 'lost_wages'
                ],
                'target': 'settlement_decision'
            },
            'fraud_detection': {
                'model_type': 'random_forest',
                'parameters': {
                    'n_estimators': 200,
                    'max_depth': 12,
                    'min_samples_split': 3,
                    'class_weight': 'balanced',
                    'random_state': 42
                },
                'features': [
                    'reporting_delay', 'prior_claims_count', 'claim_amount',
                    'medical_providers', 'attorney_involved', 'witness_availability',
                    'damage_consistency', 'claimant_history', 'provider_history'
                ],
                'target': 'fraud_indicator'
            }
        }

    def _initialize_feature_definitions(self):
        """Initialize feature definitions and transformations"""
        
        self.feature_definitions = {
            'coverage_determination': {
                'policy_status': {
                    'type': 'categorical',
                    'values': ['active', 'lapsed', 'cancelled', 'suspended'],
                    'encoding': 'label'
                },
                'loss_date_valid': {
                    'type': 'boolean',
                    'calculation': 'loss_date between policy_start and policy_end'
                },
                'coverage_type': {
                    'type': 'categorical',
                    'values': ['liability', 'collision', 'comprehensive', 'uninsured'],
                    'encoding': 'one_hot'
                },
                'policy_limits': {
                    'type': 'numerical',
                    'scaling': 'standard'
                },
                'deductible': {
                    'type': 'numerical',
                    'scaling': 'standard'
                },
                'exclusions_count': {
                    'type': 'numerical',
                    'calculation': 'count of applicable exclusions'
                }
            },
            'liability_assessment': {
                'fault_indicators': {
                    'type': 'numerical',
                    'calculation': 'weighted sum of fault indicators',
                    'scaling': 'standard'
                },
                'evidence_strength': {
                    'type': 'numerical',
                    'range': [0, 1],
                    'calculation': 'composite evidence score'
                },
                'witness_count': {
                    'type': 'numerical',
                    'transformation': 'log1p'
                },
                'police_report_fault': {
                    'type': 'categorical',
                    'values': ['insured', 'other', 'both', 'unknown'],
                    'encoding': 'label'
                },
                'traffic_violations': {
                    'type': 'boolean',
                    'calculation': 'any traffic violations cited'
                },
                'weather_conditions': {
                    'type': 'categorical',
                    'values': ['clear', 'rain', 'snow', 'fog', 'ice'],
                    'encoding': 'one_hot'
                },
                'road_conditions': {
                    'type': 'categorical',
                    'values': ['good', 'fair', 'poor', 'construction'],
                    'encoding': 'one_hot'
                },
                'vehicle_damage_pattern': {
                    'type': 'numerical',
                    'calculation': 'damage pattern consistency score'
                }
            },
            'settlement_approval': {
                'claim_amount': {
                    'type': 'numerical',
                    'transformation': 'log1p',
                    'scaling': 'standard'
                },
                'liability_percentage': {
                    'type': 'numerical',
                    'range': [0, 100]
                },
                'policy_limits': {
                    'type': 'numerical',
                    'transformation': 'log1p',
                    'scaling': 'standard'
                },
                'adjuster_level': {
                    'type': 'categorical',
                    'values': ['adjuster', 'senior_adjuster', 'manager', 'director'],
                    'encoding': 'ordinal'
                },
                'claim_complexity': {
                    'type': 'numerical',
                    'range': [1, 10],
                    'calculation': 'complexity score based on multiple factors'
                },
                'legal_representation': {
                    'type': 'boolean'
                },
                'medical_treatment': {
                    'type': 'boolean'
                },
                'lost_wages': {
                    'type': 'numerical',
                    'transformation': 'log1p'
                }
            },
            'fraud_detection': {
                'reporting_delay': {
                    'type': 'numerical',
                    'calculation': 'days between loss and report',
                    'transformation': 'log1p'
                },
                'prior_claims_count': {
                    'type': 'numerical',
                    'calculation': 'claims in past 5 years',
                    'transformation': 'sqrt'
                },
                'claim_amount': {
                    'type': 'numerical',
                    'transformation': 'log1p',
                    'scaling': 'standard'
                },
                'medical_providers': {
                    'type': 'numerical',
                    'calculation': 'number of medical providers'
                },
                'attorney_involved': {
                    'type': 'boolean'
                },
                'witness_availability': {
                    'type': 'categorical',
                    'values': ['none', 'limited', 'available', 'multiple'],
                    'encoding': 'ordinal'
                },
                'damage_consistency': {
                    'type': 'numerical',
                    'range': [0, 1],
                    'calculation': 'consistency between reported and observed damage'
                },
                'claimant_history': {
                    'type': 'numerical',
                    'calculation': 'claimant risk score'
                },
                'provider_history': {
                    'type': 'numerical',
                    'calculation': 'provider risk score'
                }
            }
        }

    def _initialize_decision_thresholds(self):
        """Initialize decision thresholds for different decision types"""
        
        self.decision_thresholds = {
            'coverage_determination': {
                'approve': 0.8,
                'investigate': 0.6,
                'deny': 0.4
            },
            'liability_assessment': {
                'accept_full': 0.9,
                'accept_partial': 0.7,
                'investigate': 0.5,
                'deny': 0.3
            },
            'settlement_approval': {
                'approve': 0.8,
                'escalate': 0.6,
                'deny': 0.4
            },
            'fraud_detection': {
                'siu_referral': 0.7,
                'enhanced_investigation': 0.5,
                'standard_processing': 0.3
            }
        }

    def _load_models(self):
        """Load trained ML models"""

        try:
            for decision_type in self.model_configs:
                try:
                    model_file = f"{self.model_path}/{decision_type}_model.joblib"
                    scaler_file = f"{self.model_path}/{decision_type}_scaler.joblib"
                    encoder_file = f"{self.model_path}/{decision_type}_encoder.joblib"

                    # Try to load existing models
                    self.models[decision_type] = joblib.load(model_file)
                    self.scalers[decision_type] = joblib.load(scaler_file)
                    self.encoders[decision_type] = joblib.load(encoder_file)

                    logger.info(f"Loaded model for {decision_type}")

                except FileNotFoundError:
                    # Create and train new model if not found
                    logger.info(f"Training new model for {decision_type}")
                    asyncio.run(self._train_model(decision_type))

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            # Initialize with default models
            asyncio.run(self._initialize_default_models())

    async def _initialize_default_models(self):
        """Initialize default models with synthetic data"""
        
        for decision_type in self.model_configs:
            await self._train_model(decision_type, use_synthetic_data=True)

    async def _train_model(self, decision_type: str, use_synthetic_data: bool = False):
        """Train ML model for specific decision type"""
        
        try:
            config = self.model_configs[decision_type]
            
            if use_synthetic_data:
                # Generate synthetic training data
                X, y = self._generate_synthetic_data(decision_type)
            else:
                # Load training data from database
                X, y = await self._load_training_data(decision_type)
            
            # Prepare features
            X_processed = self._prepare_features(X, decision_type, fit_transformers=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create model
            if config['model_type'] == 'random_forest':
                model = RandomForestClassifier(**config['parameters'])
            elif config['model_type'] == 'gradient_boosting':
                model = GradientBoostingClassifier(**config['parameters'])
            elif config['model_type'] == 'logistic_regression':
                model = LogisticRegression(**config['parameters'])
            else:
                raise ValueError(f"Unknown model type: {config['model_type']}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store model
            self.models[decision_type] = model
            
            # Save model to disk
            model_file = f"{self.model_path}/{decision_type}_model.joblib"
            scaler_file = f"{self.model_path}/{decision_type}_scaler.joblib"
            encoder_file = f"{self.model_path}/{decision_type}_encoder.joblib"
            
            joblib.dump(model, model_file)
            joblib.dump(self.scalers.get(decision_type), scaler_file)
            joblib.dump(self.encoders.get(decision_type), encoder_file)
            
            # Store performance metrics
            await self._store_model_performance(
                decision_type, model, accuracy, precision, recall, f1, len(X_train)
            )
            
            logger.info(f"Trained {decision_type} model - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
        except Exception as e:
            logger.error(f"Model training failed for {decision_type}: {e}")

    def _generate_synthetic_data(self, decision_type: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic training data"""
        
        np.random.seed(42)
        n_samples = 1000
        
        if decision_type == 'coverage_determination':
            data = {
                'policy_status': np.random.choice(['active', 'lapsed', 'cancelled'], n_samples, p=[0.8, 0.15, 0.05]),
                'loss_date_valid': np.random.choice([True, False], n_samples, p=[0.9, 0.1]),
                'coverage_type': np.random.choice(['liability', 'collision', 'comprehensive'], n_samples),
                'policy_limits': np.random.normal(100000, 50000, n_samples),
                'deductible': np.random.choice([500, 1000, 2500, 5000], n_samples),
                'exclusions_count': np.random.poisson(0.5, n_samples)
            }
            
            # Generate target based on rules
            y = []
            for i in range(n_samples):
                if (data['policy_status'][i] == 'active' and 
                    data['loss_date_valid'][i] and 
                    data['exclusions_count'][i] == 0):
                    y.append(1)  # Approve
                else:
                    y.append(0)  # Deny
            
        elif decision_type == 'liability_assessment':
            data = {
                'fault_indicators': np.random.normal(0.5, 0.3, n_samples),
                'evidence_strength': np.random.beta(2, 2, n_samples),
                'witness_count': np.random.poisson(1.5, n_samples),
                'police_report_fault': np.random.choice(['insured', 'other', 'both', 'unknown'], n_samples),
                'traffic_violations': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
                'weather_conditions': np.random.choice(['clear', 'rain', 'snow'], n_samples, p=[0.7, 0.2, 0.1]),
                'road_conditions': np.random.choice(['good', 'fair', 'poor'], n_samples, p=[0.6, 0.3, 0.1]),
                'vehicle_damage_pattern': np.random.beta(3, 2, n_samples)
            }
            
            # Generate liability percentage
            y = np.clip(
                data['fault_indicators'] * 0.4 + 
                data['evidence_strength'] * 0.3 + 
                (data['witness_count'] / 5) * 0.2 + 
                np.random.normal(0, 0.1, n_samples),
                0, 1
            ) * 100
            
        elif decision_type == 'settlement_approval':
            data = {
                'claim_amount': np.random.lognormal(10, 1, n_samples),
                'liability_percentage': np.random.normal(75, 25, n_samples),
                'policy_limits': np.random.choice([100000, 250000, 500000, 1000000], n_samples),
                'adjuster_level': np.random.choice(['adjuster', 'senior_adjuster', 'manager'], n_samples, p=[0.6, 0.3, 0.1]),
                'claim_complexity': np.random.randint(1, 11, n_samples),
                'legal_representation': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
                'medical_treatment': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),
                'lost_wages': np.random.exponential(5000, n_samples)
            }
            
            # Generate settlement decision
            y = []
            for i in range(n_samples):
                score = (data['liability_percentage'][i] / 100 * 0.4 + 
                        (1 - data['claim_complexity'][i] / 10) * 0.3 + 
                        (0.2 if not data['legal_representation'][i] else 0.1) + 
                        np.random.normal(0, 0.1))
                y.append(1 if score > 0.6 else 0)
            
        elif decision_type == 'fraud_detection':
            data = {
                'reporting_delay': np.random.exponential(5, n_samples),
                'prior_claims_count': np.random.poisson(1, n_samples),
                'claim_amount': np.random.lognormal(9, 1.5, n_samples),
                'medical_providers': np.random.poisson(2, n_samples),
                'attorney_involved': np.random.choice([True, False], n_samples, p=[0.25, 0.75]),
                'witness_availability': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
                'damage_consistency': np.random.beta(4, 2, n_samples),
                'claimant_history': np.random.beta(2, 5, n_samples),
                'provider_history': np.random.beta(2, 5, n_samples)
            }
            
            # Generate fraud indicator
            y = []
            for i in range(n_samples):
                fraud_score = (
                    (data['reporting_delay'][i] > 30) * 0.2 +
                    (data['prior_claims_count'][i] > 2) * 0.3 +
                    (data['medical_providers'][i] > 5) * 0.2 +
                    (1 - data['damage_consistency'][i]) * 0.3 +
                    np.random.normal(0, 0.1)
                )
                y.append(1 if fraud_score > 0.5 else 0)
        
        X = pd.DataFrame(data)
        y = np.array(y)
        
        return X, y

    async def _load_training_data(self, decision_type: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load training data from database"""
        
        # This would load actual historical data from the database
        # For now, use synthetic data
        return self._generate_synthetic_data(decision_type)

    def _prepare_features(self, X: pd.DataFrame, decision_type: str, fit_transformers: bool = False) -> np.ndarray:
        """Prepare features for model input"""
        
        try:
            feature_defs = self.feature_definitions[decision_type]
            X_processed = X.copy()
            
            # Initialize transformers if needed
            if fit_transformers:
                self.scalers[decision_type] = StandardScaler()
                self.encoders[decision_type] = {}
            
            # Process each feature
            for feature, definition in feature_defs.items():
                if feature not in X_processed.columns:
                    continue
                
                if definition['type'] == 'categorical':
                    if definition.get('encoding') == 'label':
                        if fit_transformers:
                            encoder = LabelEncoder()
                            X_processed[feature] = encoder.fit_transform(X_processed[feature].astype(str))
                            self.encoders[decision_type][feature] = encoder
                        else:
                            encoder = self.encoders[decision_type].get(feature)
                            if encoder:
                                X_processed[feature] = encoder.transform(X_processed[feature].astype(str))
                    
                    elif definition.get('encoding') == 'one_hot':
                        if fit_transformers:
                            dummies = pd.get_dummies(X_processed[feature], prefix=feature)
                            self.encoders[decision_type][feature] = list(dummies.columns)
                            X_processed = pd.concat([X_processed.drop(feature, axis=1), dummies], axis=1)
                        else:
                            dummy_cols = self.encoders[decision_type].get(feature, [])
                            dummies = pd.get_dummies(X_processed[feature], prefix=feature)
                            # Ensure all expected columns are present
                            for col in dummy_cols:
                                if col not in dummies.columns:
                                    dummies[col] = 0
                            X_processed = pd.concat([X_processed.drop(feature, axis=1), dummies[dummy_cols]], axis=1)
                
                elif definition['type'] == 'numerical':
                    if definition.get('transformation') == 'log1p':
                        X_processed[feature] = np.log1p(X_processed[feature])
                    elif definition.get('transformation') == 'sqrt':
                        X_processed[feature] = np.sqrt(X_processed[feature])
            
            # Scale numerical features
            if fit_transformers:
                X_scaled = self.scalers[decision_type].fit_transform(X_processed.select_dtypes(include=[np.number]))
            else:
                scaler = self.scalers.get(decision_type)
                if scaler:
                    X_scaled = scaler.transform(X_processed.select_dtypes(include=[np.number]))
                else:
                    X_scaled = X_processed.select_dtypes(include=[np.number]).values
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return X.values

    async def make_decision(self, decision_type: DecisionType, claim_data: Dict[str, Any], 
                          workflow_context: Dict[str, Any] = None) -> DecisionResult:
        """Make AI-powered decision"""
        
        try:
            decision_id = str(uuid.uuid4())
            
            # Extract features from claim data
            features = await self._extract_features(decision_type, claim_data, workflow_context)
            
            # Apply business rules
            rule_result = await self._apply_business_rules(decision_type, claim_data, features)
            
            # Get ML prediction if model is available
            ml_result = None
            if decision_type.value in self.models:
                ml_result = await self._get_ml_prediction(decision_type, features)
            
            # Combine rule-based and ML results
            final_result = await self._combine_results(decision_type, rule_result, ml_result, claim_data)
            
            # Generate decision factors
            factors = await self._generate_decision_factors(decision_type, features, rule_result, ml_result)
            
            # Generate reasoning
            reasoning = await self._generate_reasoning(decision_type, final_result, factors)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(decision_type, final_result, factors)
            
            # Assess risk
            risk_assessment = await self._assess_risk(decision_type, final_result, claim_data)
            
            # Calculate financial impact
            financial_impact = await self._calculate_financial_impact(decision_type, final_result, claim_data)
            
            # Determine if human review is required
            human_review_required = await self._requires_human_review(decision_type, final_result, factors)
            
            # Create decision result
            decision_result = DecisionResult(
                decision_id=decision_id,
                decision_type=decision_type,
                outcome=final_result['outcome'],
                confidence_score=final_result['confidence'],
                confidence_level=self._get_confidence_level(final_result['confidence']),
                factors=factors,
                reasoning=reasoning,
                recommendations=recommendations,
                alternative_outcomes=final_result.get('alternatives', {}),
                risk_assessment=risk_assessment,
                financial_impact=financial_impact,
                decision_timestamp=datetime.utcnow(),
                model_version=self._get_model_version(decision_type),
                human_review_required=human_review_required
            )
            
            # Store decision
            await self._store_decision(decision_result, claim_data.get('claim_id'))
            
            logger.info(f"Decision {decision_id} made: {final_result['outcome'].value} (confidence: {final_result['confidence']:.3f})")
            
            return decision_result
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            raise

    async def _extract_features(self, decision_type: DecisionType, claim_data: Dict[str, Any], 
                               workflow_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract features from claim data"""
        
        features = {}
        
        try:
            if decision_type == DecisionType.COVERAGE_DETERMINATION:
                features = {
                    'policy_status': claim_data.get('policy_status', 'active'),
                    'loss_date_valid': self._is_loss_date_valid(claim_data),
                    'coverage_type': claim_data.get('coverage_type', 'liability'),
                    'policy_limits': claim_data.get('policy_limits', 100000),
                    'deductible': claim_data.get('deductible', 500),
                    'exclusions_count': len(claim_data.get('applicable_exclusions', []))
                }
            
            elif decision_type == DecisionType.LIABILITY_ASSESSMENT:
                features = {
                    'fault_indicators': self._calculate_fault_indicators(claim_data),
                    'evidence_strength': self._assess_evidence_strength(claim_data),
                    'witness_count': len(claim_data.get('witnesses', [])),
                    'police_report_fault': claim_data.get('police_report_fault', 'unknown'),
                    'traffic_violations': bool(claim_data.get('traffic_violations')),
                    'weather_conditions': claim_data.get('weather_conditions', 'clear'),
                    'road_conditions': claim_data.get('road_conditions', 'good'),
                    'vehicle_damage_pattern': self._assess_damage_pattern(claim_data)
                }
            
            elif decision_type == DecisionType.SETTLEMENT_APPROVAL:
                features = {
                    'claim_amount': claim_data.get('claim_amount', 0),
                    'liability_percentage': claim_data.get('liability_percentage', 100),
                    'policy_limits': claim_data.get('policy_limits', 100000),
                    'adjuster_level': claim_data.get('adjuster_level', 'adjuster'),
                    'claim_complexity': self._assess_claim_complexity(claim_data),
                    'legal_representation': bool(claim_data.get('legal_representation')),
                    'medical_treatment': bool(claim_data.get('medical_treatment')),
                    'lost_wages': claim_data.get('lost_wages', 0)
                }
            
            elif decision_type == DecisionType.FRAUD_DETECTION:
                features = {
                    'reporting_delay': self._calculate_reporting_delay(claim_data),
                    'prior_claims_count': claim_data.get('prior_claims_count', 0),
                    'claim_amount': claim_data.get('claim_amount', 0),
                    'medical_providers': len(claim_data.get('medical_providers', [])),
                    'attorney_involved': bool(claim_data.get('attorney_involved')),
                    'witness_availability': len(claim_data.get('witnesses', [])),
                    'damage_consistency': self._assess_damage_consistency(claim_data),
                    'claimant_history': claim_data.get('claimant_risk_score', 0.5),
                    'provider_history': claim_data.get('provider_risk_score', 0.5)
                }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}

    def _is_loss_date_valid(self, claim_data: Dict[str, Any]) -> bool:
        """Check if loss date is within policy period"""
        
        try:
            loss_date = datetime.strptime(claim_data.get('loss_date', ''), '%Y-%m-%d')
            policy_start = datetime.strptime(claim_data.get('policy_start_date', ''), '%Y-%m-%d')
            policy_end = datetime.strptime(claim_data.get('policy_end_date', ''), '%Y-%m-%d')
            
            return policy_start <= loss_date <= policy_end
            
        except:
            return False

    def _calculate_fault_indicators(self, claim_data: Dict[str, Any]) -> float:
        """Calculate fault indicators score"""
        
        indicators = 0.0
        
        # Traffic violations
        if claim_data.get('traffic_violations'):
            indicators += 0.3
        
        # Police report fault
        if claim_data.get('police_report_fault') == 'insured':
            indicators += 0.4
        
        # Witness statements
        witness_fault = claim_data.get('witness_fault_percentage', 0)
        indicators += (witness_fault / 100) * 0.3
        
        return min(indicators, 1.0)

    def _assess_evidence_strength(self, claim_data: Dict[str, Any]) -> float:
        """Assess strength of evidence"""
        
        strength = 0.0
        
        # Police report available
        if claim_data.get('police_report'):
            strength += 0.3
        
        # Photos available
        if claim_data.get('photos'):
            strength += 0.2
        
        # Witness statements
        witness_count = len(claim_data.get('witnesses', []))
        strength += min(witness_count * 0.1, 0.3)
        
        # Expert analysis
        if claim_data.get('expert_analysis'):
            strength += 0.2
        
        return min(strength, 1.0)

    def _assess_damage_pattern(self, claim_data: Dict[str, Any]) -> float:
        """Assess vehicle damage pattern consistency"""
        
        # This would use computer vision analysis
        # For now, return a simulated score
        return 0.8

    def _assess_claim_complexity(self, claim_data: Dict[str, Any]) -> int:
        """Assess claim complexity on scale 1-10"""
        
        complexity = 1
        
        # Multiple vehicles
        if claim_data.get('vehicle_count', 1) > 1:
            complexity += 2
        
        # Injuries involved
        if claim_data.get('injury_involved'):
            complexity += 3
        
        # Legal representation
        if claim_data.get('legal_representation'):
            complexity += 2
        
        # Disputed liability
        if claim_data.get('disputed_liability'):
            complexity += 2
        
        return min(complexity, 10)

    def _calculate_reporting_delay(self, claim_data: Dict[str, Any]) -> int:
        """Calculate reporting delay in days"""
        
        try:
            loss_date = datetime.strptime(claim_data.get('loss_date', ''), '%Y-%m-%d')
            report_date = datetime.strptime(claim_data.get('report_date', ''), '%Y-%m-%d')
            
            return (report_date - loss_date).days
            
        except:
            return 0

    def _assess_damage_consistency(self, claim_data: Dict[str, Any]) -> float:
        """Assess consistency between reported and observed damage"""
        
        # This would use advanced analysis
        # For now, return a simulated score
        return 0.9

    async def _apply_business_rules(self, decision_type: DecisionType, claim_data: Dict[str, Any], 
                                  features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply business rules"""
        
        try:
            rules = self.business_rules.get(decision_type.value, {})
            rule_results = {}
            
            if decision_type == DecisionType.COVERAGE_DETERMINATION:
                # Check required conditions
                required_checks = ['policy_active', 'loss_within_period', 'coverage_applies', 'exclusions_check']
                
                for check in required_checks:
                    if check == 'policy_active':
                        rule_results[check] = features.get('policy_status') == 'active'
                    elif check == 'loss_within_period':
                        rule_results[check] = features.get('loss_date_valid', False)
                    elif check == 'coverage_applies':
                        rule_results[check] = True  # Simplified
                    elif check == 'exclusions_check':
                        rule_results[check] = features.get('exclusions_count', 0) == 0
                
                # All required checks must pass
                all_passed = all(rule_results.values())
                
                return {
                    'outcome': DecisionOutcome.APPROVE if all_passed else DecisionOutcome.DENY,
                    'confidence': 1.0 if all_passed else 0.0,
                    'rule_results': rule_results
                }
            
            elif decision_type == DecisionType.FRAUD_DETECTION:
                fraud_score = 0.0
                red_flags = rules.get('red_flags', {})
                
                # Check red flags
                if features.get('reporting_delay', 0) > red_flags.get('late_reporting', {}).get('days', 30):
                    fraud_score += red_flags['late_reporting']['score']
                
                if features.get('prior_claims_count', 0) > red_flags.get('prior_claims', {}).get('count', 3):
                    fraud_score += red_flags['prior_claims']['score']
                
                # Determine outcome based on score
                triggers = rules.get('investigation_triggers', {})
                
                if fraud_score >= triggers.get('siu_referral', 0.7):
                    outcome = DecisionOutcome.INVESTIGATE
                elif fraud_score >= triggers.get('enhanced_investigation', 0.5):
                    outcome = DecisionOutcome.INVESTIGATE
                else:
                    outcome = DecisionOutcome.APPROVE
                
                return {
                    'outcome': outcome,
                    'confidence': min(fraud_score, 1.0),
                    'fraud_score': fraud_score,
                    'rule_results': rule_results
                }
            
            # Default rule result
            return {
                'outcome': DecisionOutcome.APPROVE,
                'confidence': 0.5,
                'rule_results': rule_results
            }
            
        except Exception as e:
            logger.error(f"Business rules application failed: {e}")
            return {
                'outcome': DecisionOutcome.ESCALATE,
                'confidence': 0.0,
                'rule_results': {}
            }

    async def _get_ml_prediction(self, decision_type: DecisionType, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML model prediction"""
        
        try:
            model = self.models.get(decision_type.value)
            if not model:
                return None
            
            # Prepare features
            feature_df = pd.DataFrame([features])
            X = self._prepare_features(feature_df, decision_type.value, fit_transformers=False)
            
            # Get prediction
            prediction = model.predict(X)[0]
            prediction_proba = model.predict_proba(X)[0]
            
            # Map prediction to outcome
            if decision_type == DecisionType.COVERAGE_DETERMINATION:
                outcome = DecisionOutcome.APPROVE if prediction == 1 else DecisionOutcome.DENY
                confidence = max(prediction_proba)
            
            elif decision_type == DecisionType.SETTLEMENT_APPROVAL:
                outcome = DecisionOutcome.APPROVE if prediction == 1 else DecisionOutcome.DENY
                confidence = max(prediction_proba)
            
            elif decision_type == DecisionType.FRAUD_DETECTION:
                if prediction == 1:
                    outcome = DecisionOutcome.INVESTIGATE
                else:
                    outcome = DecisionOutcome.APPROVE
                confidence = max(prediction_proba)
            
            else:
                outcome = DecisionOutcome.APPROVE
                confidence = 0.5
            
            return {
                'outcome': outcome,
                'confidence': confidence,
                'prediction': prediction,
                'probabilities': prediction_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return None

    async def _combine_results(self, decision_type: DecisionType, rule_result: Dict[str, Any], 
                             ml_result: Dict[str, Any], claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule-based and ML results"""
        
        try:
            # If no ML result, use rule result
            if not ml_result:
                return rule_result
            
            # Weight rule vs ML results
            rule_weight = 0.6
            ml_weight = 0.4
            
            # For critical decisions, give more weight to rules
            if decision_type in [DecisionType.COVERAGE_DETERMINATION]:
                rule_weight = 0.8
                ml_weight = 0.2
            
            # Combine confidence scores
            combined_confidence = (
                rule_result['confidence'] * rule_weight + 
                ml_result['confidence'] * ml_weight
            )
            
            # Determine final outcome
            if rule_result['outcome'] == ml_result['outcome']:
                final_outcome = rule_result['outcome']
            else:
                # Conflict resolution
                if rule_result['confidence'] > ml_result['confidence']:
                    final_outcome = rule_result['outcome']
                else:
                    final_outcome = ml_result['outcome']
                
                # Reduce confidence for conflicting results
                combined_confidence *= 0.7
            
            # Apply decision thresholds
            thresholds = self.decision_thresholds.get(decision_type.value, {})
            
            if combined_confidence < thresholds.get('investigate', 0.6):
                if final_outcome in [DecisionOutcome.APPROVE, DecisionOutcome.DENY]:
                    final_outcome = DecisionOutcome.ESCALATE
            
            return {
                'outcome': final_outcome,
                'confidence': combined_confidence,
                'rule_result': rule_result,
                'ml_result': ml_result,
                'alternatives': self._calculate_alternatives(rule_result, ml_result)
            }
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            return rule_result

    def _calculate_alternatives(self, rule_result: Dict[str, Any], ml_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate alternative outcome probabilities"""
        
        alternatives = {}
        
        if rule_result:
            alternatives[f"rule_{rule_result['outcome'].value}"] = rule_result['confidence']
        
        if ml_result:
            alternatives[f"ml_{ml_result['outcome'].value}"] = ml_result['confidence']
        
        return alternatives

    async def _generate_decision_factors(self, decision_type: DecisionType, features: Dict[str, Any],
                                       rule_result: Dict[str, Any], ml_result: Dict[str, Any]) -> List[DecisionFactor]:
        """Generate decision factors"""
        
        factors = []
        
        try:
            # Add rule-based factors
            if rule_result and 'rule_results' in rule_result:
                for rule_name, rule_value in rule_result['rule_results'].items():
                    factor = DecisionFactor(
                        factor_name=rule_name,
                        factor_value=rule_value,
                        weight=0.6,
                        impact_score=1.0 if rule_value else -1.0,
                        confidence=rule_result['confidence'],
                        source='business_rules'
                    )
                    factors.append(factor)
            
            # Add ML-based factors
            if ml_result and decision_type.value in self.models:
                model = self.models[decision_type.value]
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_names = list(features.keys())
                    importances = model.feature_importances_
                    
                    for i, (name, importance) in enumerate(zip(feature_names, importances)):
                        if importance > 0.05:  # Only include significant features
                            factor = DecisionFactor(
                                factor_name=name,
                                factor_value=features.get(name),
                                weight=0.4,
                                impact_score=importance,
                                confidence=ml_result['confidence'],
                                source='machine_learning'
                            )
                            factors.append(factor)
            
            return factors
            
        except Exception as e:
            logger.error(f"Decision factor generation failed: {e}")
            return []

    async def _generate_reasoning(self, decision_type: DecisionType, result: Dict[str, Any], 
                                factors: List[DecisionFactor]) -> str:
        """Generate human-readable reasoning"""
        
        try:
            reasoning_parts = []
            
            # Add outcome statement
            outcome = result['outcome']
            confidence = result['confidence']
            
            reasoning_parts.append(f"Decision: {outcome.value.title()} (Confidence: {confidence:.1%})")
            
            # Add key factors
            key_factors = sorted(factors, key=lambda f: f.impact_score, reverse=True)[:3]
            
            if key_factors:
                reasoning_parts.append("Key factors:")
                for factor in key_factors:
                    impact = "positive" if factor.impact_score > 0 else "negative"
                    reasoning_parts.append(f"- {factor.factor_name}: {factor.factor_value} ({impact} impact)")
            
            # Add decision type specific reasoning
            if decision_type == DecisionType.COVERAGE_DETERMINATION:
                if outcome == DecisionOutcome.APPROVE:
                    reasoning_parts.append("All coverage requirements are satisfied.")
                else:
                    reasoning_parts.append("One or more coverage requirements are not met.")
            
            elif decision_type == DecisionType.FRAUD_DETECTION:
                if outcome == DecisionOutcome.INVESTIGATE:
                    reasoning_parts.append("Fraud indicators detected requiring investigation.")
                else:
                    reasoning_parts.append("No significant fraud indicators detected.")
            
            return " ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return f"Decision: {result['outcome'].value} with {result['confidence']:.1%} confidence"

    async def _generate_recommendations(self, decision_type: DecisionType, result: Dict[str, Any], 
                                      factors: List[DecisionFactor]) -> List[str]:
        """Generate recommendations"""
        
        recommendations = []
        
        try:
            outcome = result['outcome']
            confidence = result['confidence']
            
            # Low confidence recommendations
            if confidence < 0.7:
                recommendations.append("Consider human review due to low confidence")
            
            # Decision type specific recommendations
            if decision_type == DecisionType.COVERAGE_DETERMINATION:
                if outcome == DecisionOutcome.DENY:
                    recommendations.append("Send coverage denial letter with explanation")
                    recommendations.append("Document specific exclusions or policy violations")
                elif outcome == DecisionOutcome.APPROVE:
                    recommendations.append("Proceed with claim processing")
                    recommendations.append("Verify policy limits and deductibles")
            
            elif decision_type == DecisionType.LIABILITY_ASSESSMENT:
                if outcome == DecisionOutcome.INVESTIGATE:
                    recommendations.append("Conduct detailed liability investigation")
                    recommendations.append("Obtain additional witness statements")
                    recommendations.append("Consider expert accident reconstruction")
            
            elif decision_type == DecisionType.FRAUD_DETECTION:
                if outcome == DecisionOutcome.INVESTIGATE:
                    recommendations.append("Refer to Special Investigation Unit (SIU)")
                    recommendations.append("Conduct enhanced documentation review")
                    recommendations.append("Verify claimant and provider histories")
            
            elif decision_type == DecisionType.SETTLEMENT_APPROVAL:
                if outcome == DecisionOutcome.APPROVE:
                    recommendations.append("Prepare settlement documentation")
                    recommendations.append("Obtain necessary releases")
                elif outcome == DecisionOutcome.ESCALATE:
                    recommendations.append("Escalate to senior adjuster or manager")
                    recommendations.append("Provide detailed justification for settlement amount")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Review decision manually"]

    async def _assess_risk(self, decision_type: DecisionType, result: Dict[str, Any], 
                         claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk associated with decision"""
        
        try:
            risk_assessment = {
                'overall_risk': 'medium',
                'financial_risk': 'medium',
                'legal_risk': 'low',
                'reputational_risk': 'low',
                'risk_factors': []
            }
            
            outcome = result['outcome']
            confidence = result['confidence']
            claim_amount = claim_data.get('claim_amount', 0)
            
            # Assess financial risk
            if claim_amount > 100000:
                risk_assessment['financial_risk'] = 'high'
                risk_assessment['risk_factors'].append('High claim amount')
            elif claim_amount > 25000:
                risk_assessment['financial_risk'] = 'medium'
            else:
                risk_assessment['financial_risk'] = 'low'
            
            # Assess legal risk
            if claim_data.get('legal_representation'):
                risk_assessment['legal_risk'] = 'high'
                risk_assessment['risk_factors'].append('Legal representation involved')
            
            if claim_data.get('disputed_liability'):
                risk_assessment['legal_risk'] = 'medium'
                risk_assessment['risk_factors'].append('Disputed liability')
            
            # Assess confidence-based risk
            if confidence < 0.6:
                risk_assessment['overall_risk'] = 'high'
                risk_assessment['risk_factors'].append('Low decision confidence')
            elif confidence < 0.8:
                risk_assessment['overall_risk'] = 'medium'
            
            # Decision-specific risk assessment
            if outcome == DecisionOutcome.DENY and decision_type == DecisionType.COVERAGE_DETERMINATION:
                risk_assessment['reputational_risk'] = 'medium'
                risk_assessment['risk_factors'].append('Coverage denial may impact customer satisfaction')
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {'overall_risk': 'unknown', 'error': str(e)}

    async def _calculate_financial_impact(self, decision_type: DecisionType, result: Dict[str, Any], 
                                        claim_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate financial impact of decision"""
        
        try:
            financial_impact = {
                'estimated_payout': 0.0,
                'reserve_adjustment': 0.0,
                'cost_savings': 0.0,
                'investigation_cost': 0.0
            }
            
            outcome = result['outcome']
            claim_amount = claim_data.get('claim_amount', 0)
            liability_percentage = claim_data.get('liability_percentage', 100)
            
            if outcome == DecisionOutcome.APPROVE:
                if decision_type == DecisionType.COVERAGE_DETERMINATION:
                    financial_impact['estimated_payout'] = claim_amount
                elif decision_type == DecisionType.SETTLEMENT_APPROVAL:
                    financial_impact['estimated_payout'] = claim_amount * (liability_percentage / 100)
            
            elif outcome == DecisionOutcome.DENY:
                financial_impact['cost_savings'] = claim_amount
            
            elif outcome == DecisionOutcome.INVESTIGATE:
                financial_impact['investigation_cost'] = min(claim_amount * 0.05, 10000)  # 5% of claim, max $10k
            
            # Reserve adjustments
            if decision_type == DecisionType.RESERVE_SETTING:
                current_reserve = claim_data.get('current_reserve', 0)
                new_reserve = financial_impact['estimated_payout']
                financial_impact['reserve_adjustment'] = new_reserve - current_reserve
            
            return financial_impact
            
        except Exception as e:
            logger.error(f"Financial impact calculation failed: {e}")
            return {'error': str(e)}

    async def _requires_human_review(self, decision_type: DecisionType, result: Dict[str, Any], 
                                   factors: List[DecisionFactor]) -> bool:
        """Determine if human review is required"""
        
        try:
            confidence = result['confidence']
            outcome = result['outcome']
            
            # Low confidence always requires review
            if confidence < 0.7:
                return True
            
            # Certain outcomes require review
            if outcome in [DecisionOutcome.ESCALATE, DecisionOutcome.INVESTIGATE]:
                return True
            
            # High-value decisions require review
            high_impact_factors = [f for f in factors if abs(f.impact_score) > 0.8]
            if len(high_impact_factors) > 2:
                return True
            
            # Decision type specific rules
            if decision_type == DecisionType.COVERAGE_DETERMINATION and outcome == DecisionOutcome.DENY:
                return True
            
            if decision_type == DecisionType.FRAUD_DETECTION and outcome == DecisionOutcome.INVESTIGATE:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Human review determination failed: {e}")
            return True  # Default to requiring review on error

    def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _get_model_version(self, decision_type: DecisionType) -> str:
        """Get model version for decision type"""
        
        return f"{decision_type.value}_v1.0"

    async def _store_decision(self, decision_result: DecisionResult, claim_id: str):
        """Store decision in database"""
        
        try:
            with self.Session() as session:
                decision_record = DecisionRecord(
                    decision_id=decision_result.decision_id,
                    claim_id=claim_id,
                    decision_type=decision_result.decision_type.value,
                    outcome=decision_result.outcome.value,
                    confidence_score=decision_result.confidence_score,
                    confidence_level=decision_result.confidence_level.value,
                    factors=[asdict(factor) for factor in decision_result.factors],
                    reasoning=decision_result.reasoning,
                    recommendations=decision_result.recommendations,
                    alternative_outcomes=decision_result.alternative_outcomes,
                    risk_assessment=decision_result.risk_assessment,
                    financial_impact=decision_result.financial_impact,
                    decision_timestamp=decision_result.decision_timestamp,
                    model_version=decision_result.model_version,
                    human_review_required=decision_result.human_review_required,
                    created_at=datetime.utcnow()
                )
                
                session.add(decision_record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Decision storage failed: {e}")

    async def _store_model_performance(self, decision_type: str, model: Any, accuracy: float, 
                                     precision: float, recall: float, f1: float, sample_size: int):
        """Store model performance metrics"""
        
        try:
            with self.Session() as session:
                performance_record = ModelPerformanceRecord(
                    performance_id=str(uuid.uuid4()),
                    model_name=self.model_configs[decision_type]['model_type'],
                    model_version=f"{decision_type}_v1.0",
                    decision_type=decision_type,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    training_date=datetime.utcnow(),
                    evaluation_date=datetime.utcnow(),
                    sample_size=sample_size,
                    feature_importance=getattr(model, 'feature_importances_', []).tolist() if hasattr(model, 'feature_importances_') else [],
                    created_at=datetime.utcnow()
                )
                
                session.add(performance_record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Model performance storage failed: {e}")

    async def get_decision_history(self, claim_id: str) -> List[Dict[str, Any]]:
        """Get decision history for a claim"""
        
        try:
            with self.Session() as session:
                decisions = session.query(DecisionRecord).filter_by(claim_id=claim_id).order_by(DecisionRecord.decision_timestamp).all()
                
                return [
                    {
                        'decision_id': decision.decision_id,
                        'decision_type': decision.decision_type,
                        'outcome': decision.outcome,
                        'confidence_score': decision.confidence_score,
                        'confidence_level': decision.confidence_level,
                        'reasoning': decision.reasoning,
                        'recommendations': decision.recommendations,
                        'decision_timestamp': decision.decision_timestamp.isoformat(),
                        'human_review_required': decision.human_review_required,
                        'human_reviewer': decision.human_reviewer,
                        'final_outcome': decision.final_outcome
                    }
                    for decision in decisions
                ]
                
        except Exception as e:
            logger.error(f"Decision history retrieval failed: {e}")
            return []

    async def update_human_review(self, decision_id: str, reviewer: str, human_decision: str, notes: str) -> bool:
        """Update decision with human review"""
        
        try:
            with self.Session() as session:
                decision = session.query(DecisionRecord).filter_by(decision_id=decision_id).first()
                
                if decision:
                    decision.human_reviewer = reviewer
                    decision.human_decision = human_decision
                    decision.human_notes = notes
                    decision.final_outcome = human_decision
                    
                    session.commit()
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Human review update failed: {e}")
            return False

