"""
Decision Engine - Production Ready Implementation
AI-powered decision making system for insurance operations
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
decisions_made_total = Counter('decisions_made_total', 'Total decisions made', ['decision_type', 'outcome'])
decision_processing_duration = Histogram('decision_processing_duration_seconds', 'Time to process decisions')
active_decision_sessions = Gauge('active_decision_sessions', 'Number of active decision sessions')
model_accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy', ['model_type'])

Base = declarative_base()

class DecisionType(Enum):
    UNDERWRITING = "underwriting"
    CLAIMS = "claims"
    FRAUD_DETECTION = "fraud_detection"
    PRICING = "pricing"
    RISK_ASSESSMENT = "risk_assessment"
    POLICY_RENEWAL = "policy_renewal"
    COVERAGE_RECOMMENDATION = "coverage_recommendation"

class DecisionOutcome(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REFER = "refer"
    INVESTIGATE = "investigate"
    ESCALATE = "escalate"
    PENDING = "pending"

class ConfidenceLevel(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

@dataclass
class DecisionRequest:
    """Represents a decision request"""
    request_id: str
    decision_type: DecisionType
    input_data: Dict[str, Any]
    context: Dict[str, Any]
    requester_id: str
    priority: int = 5  # 1-10 scale
    timeout_seconds: int = 300
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class DecisionResult:
    """Represents a decision result"""
    request_id: str
    decision_type: DecisionType
    outcome: DecisionOutcome
    confidence: float
    reasoning: List[str]
    supporting_data: Dict[str, Any]
    model_used: str
    processing_time: float
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DecisionModel(Base):
    """SQLAlchemy model for persisting decisions"""
    __tablename__ = 'decisions'
    
    request_id = Column(String, primary_key=True)
    decision_type = Column(String, nullable=False)
    outcome = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(JSON)
    supporting_data = Column(JSON)
    model_used = Column(String, nullable=False)
    processing_time = Column(Float, nullable=False)
    input_data = Column(JSON)
    context = Column(JSON)
    requester_id = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime)
    metadata = Column(JSON)

class DecisionEngine:
    """
    Production-ready Decision Engine for Insurance AI System
    Provides AI-powered decision making with ML models, business rules, and explainable AI
    """
    
    def __init__(self, db_url: str, redis_url: str, model_path: str = "/tmp/models"):
        self.db_url = db_url
        self.redis_url = redis_url
        self.model_path = model_path
        
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis setup
        self.redis_client = redis.from_url(redis_url)
        
        # Model management
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, Dict[str, LabelEncoder]] = {}
        
        # Decision cache
        self.decision_cache: Dict[str, DecisionResult] = {}
        
        # Business rules
        self.business_rules: Dict[DecisionType, List[Dict[str, Any]]] = {}
        
        # Processing queue
        self.processing_queue = asyncio.Queue()
        self.processing_active = False
        
        # Initialize components
        self._initialize_models()
        self._load_business_rules()
        
        logger.info("DecisionEngine initialized successfully")

    def _initialize_models(self):
        """Initialize ML models for different decision types"""
        
        # Underwriting model
        self.models['underwriting'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Claims model
        self.models['claims'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Fraud detection model
        self.models['fraud_detection'] = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42
        )
        
        # Risk assessment model
        self.models['risk_assessment'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=3,
            random_state=42
        )
        
        # Pricing model
        self.models['pricing'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        )
        
        # Initialize scalers and encoders
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
            self.encoders[model_name] = {}
        
        # Load pre-trained models if available
        self._load_pretrained_models()
        
        logger.info(f"Initialized {len(self.models)} ML models")

    def _load_pretrained_models(self):
        """Load pre-trained models from disk"""
        
        import os
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            return
        
        for model_name in self.models.keys():
            model_file = f"{self.model_path}/{model_name}_model.joblib"
            scaler_file = f"{self.model_path}/{model_name}_scaler.joblib"
            encoder_file = f"{self.model_path}/{model_name}_encoders.joblib"
            
            try:
                if os.path.exists(model_file):
                    self.models[model_name] = joblib.load(model_file)
                    logger.info(f"Loaded pre-trained model: {model_name}")
                
                if os.path.exists(scaler_file):
                    self.scalers[model_name] = joblib.load(scaler_file)
                    
                if os.path.exists(encoder_file):
                    self.encoders[model_name] = joblib.load(encoder_file)
                    
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model {model_name}: {e}")

    def _load_business_rules(self):
        """Load business rules for decision making"""
        
        self.business_rules = {
            DecisionType.UNDERWRITING: [
                {
                    "rule_id": "age_limit",
                    "condition": lambda data: data.get("age", 0) >= 18 and data.get("age", 0) <= 75,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.3,
                    "reason": "Age within acceptable range"
                },
                {
                    "rule_id": "income_requirement",
                    "condition": lambda data: data.get("annual_income", 0) >= 25000,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.4,
                    "reason": "Meets minimum income requirement"
                },
                {
                    "rule_id": "credit_score",
                    "condition": lambda data: data.get("credit_score", 0) >= 650,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.3,
                    "reason": "Good credit score"
                },
                {
                    "rule_id": "high_risk_occupation",
                    "condition": lambda data: data.get("occupation", "").lower() not in ["pilot", "miner", "stuntman"],
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.2,
                    "reason": "Low-risk occupation"
                }
            ],
            DecisionType.CLAIMS: [
                {
                    "rule_id": "policy_active",
                    "condition": lambda data: data.get("policy_status", "").lower() == "active",
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.5,
                    "reason": "Policy is active"
                },
                {
                    "rule_id": "within_coverage",
                    "condition": lambda data: data.get("claim_amount", 0) <= data.get("coverage_limit", 0),
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.4,
                    "reason": "Claim within coverage limits"
                },
                {
                    "rule_id": "timely_reporting",
                    "condition": lambda data: (datetime.utcnow() - datetime.fromisoformat(data.get("incident_date", datetime.utcnow().isoformat()))).days <= 30,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.3,
                    "reason": "Claim reported within time limit"
                }
            ],
            DecisionType.FRAUD_DETECTION: [
                {
                    "rule_id": "multiple_claims",
                    "condition": lambda data: data.get("claims_last_year", 0) <= 2,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.4,
                    "reason": "Normal claim frequency"
                },
                {
                    "rule_id": "claim_amount_reasonable",
                    "condition": lambda data: data.get("claim_amount", 0) <= data.get("policy_value", 0) * 0.8,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.3,
                    "reason": "Reasonable claim amount"
                },
                {
                    "rule_id": "documentation_complete",
                    "condition": lambda data: len(data.get("supporting_documents", [])) >= 2,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.3,
                    "reason": "Adequate documentation provided"
                }
            ],
            DecisionType.RISK_ASSESSMENT: [
                {
                    "rule_id": "location_risk",
                    "condition": lambda data: data.get("location_risk_score", 5) <= 7,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.3,
                    "reason": "Acceptable location risk"
                },
                {
                    "rule_id": "property_age",
                    "condition": lambda data: data.get("property_age", 0) <= 50,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.2,
                    "reason": "Property age acceptable"
                },
                {
                    "rule_id": "safety_features",
                    "condition": lambda data: len(data.get("safety_features", [])) >= 3,
                    "outcome": DecisionOutcome.APPROVE,
                    "weight": 0.3,
                    "reason": "Adequate safety features"
                }
            ]
        }
        
        logger.info(f"Loaded business rules for {len(self.business_rules)} decision types")

    async def make_decision(self, request: DecisionRequest) -> DecisionResult:
        """Make a decision based on the request"""
        
        start_time = datetime.utcnow()
        
        with decision_processing_duration.time():
            try:
                active_decision_sessions.inc()
                
                # Check cache first
                cached_result = await self._get_cached_decision(request)
                if cached_result:
                    logger.info(f"Returning cached decision for {request.request_id}")
                    return cached_result
                
                # Apply business rules
                rule_result = await self._apply_business_rules(request)
                
                # Get ML model prediction
                ml_result = await self._get_ml_prediction(request)
                
                # Combine results
                final_result = await self._combine_results(request, rule_result, ml_result)
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                final_result.processing_time = processing_time
                
                # Store result
                await self._store_decision(final_result)
                
                # Cache result
                await self._cache_decision(final_result)
                
                # Update metrics
                decisions_made_total.labels(
                    decision_type=request.decision_type.value,
                    outcome=final_result.outcome.value
                ).inc()
                
                logger.info(f"Decision made for {request.request_id}: {final_result.outcome.value} (confidence: {final_result.confidence:.2f})")
                
                return final_result
                
            except Exception as e:
                logger.error(f"Error making decision for {request.request_id}: {e}")
                
                # Return error result
                error_result = DecisionResult(
                    request_id=request.request_id,
                    decision_type=request.decision_type,
                    outcome=DecisionOutcome.ESCALATE,
                    confidence=0.0,
                    reasoning=[f"Error in decision processing: {str(e)}"],
                    supporting_data={"error": str(e)},
                    model_used="error_handler",
                    processing_time=(datetime.utcnow() - start_time).total_seconds(),
                    created_at=datetime.utcnow()
                )
                
                return error_result
                
            finally:
                active_decision_sessions.dec()

    async def _apply_business_rules(self, request: DecisionRequest) -> Dict[str, Any]:
        """Apply business rules to the decision request"""
        
        decision_type = request.decision_type
        input_data = request.input_data
        
        if decision_type not in self.business_rules:
            return {
                "outcome": DecisionOutcome.REFER,
                "confidence": 0.5,
                "reasoning": ["No business rules defined for this decision type"],
                "rule_scores": {}
            }
        
        rules = self.business_rules[decision_type]
        rule_scores = {}
        total_weight = 0
        weighted_score = 0
        reasoning = []
        
        for rule in rules:
            try:
                rule_id = rule["rule_id"]
                condition = rule["condition"]
                weight = rule["weight"]
                reason = rule["reason"]
                
                # Evaluate condition
                if condition(input_data):
                    score = 1.0
                    reasoning.append(f"✓ {reason}")
                else:
                    score = 0.0
                    reasoning.append(f"✗ Failed: {reason}")
                
                rule_scores[rule_id] = {
                    "score": score,
                    "weight": weight,
                    "reason": reason
                }
                
                weighted_score += score * weight
                total_weight += weight
                
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule.get('rule_id', 'unknown')}: {e}")
                reasoning.append(f"⚠ Rule evaluation error: {rule.get('rule_id', 'unknown')}")
        
        # Calculate overall confidence
        confidence = weighted_score / total_weight if total_weight > 0 else 0.5
        
        # Determine outcome based on confidence
        if confidence >= 0.8:
            outcome = DecisionOutcome.APPROVE
        elif confidence >= 0.6:
            outcome = DecisionOutcome.REFER
        elif confidence >= 0.3:
            outcome = DecisionOutcome.INVESTIGATE
        else:
            outcome = DecisionOutcome.REJECT
        
        return {
            "outcome": outcome,
            "confidence": confidence,
            "reasoning": reasoning,
            "rule_scores": rule_scores,
            "total_weight": total_weight,
            "weighted_score": weighted_score
        }

    async def _get_ml_prediction(self, request: DecisionRequest) -> Dict[str, Any]:
        """Get ML model prediction"""
        
        decision_type = request.decision_type.value
        input_data = request.input_data
        
        if decision_type not in self.models:
            return {
                "outcome": DecisionOutcome.REFER,
                "confidence": 0.5,
                "reasoning": ["No ML model available for this decision type"],
                "model_used": "none"
            }
        
        try:
            model = self.models[decision_type]
            scaler = self.scalers[decision_type]
            encoders = self.encoders[decision_type]
            
            # Prepare features
            features = await self._prepare_features(input_data, decision_type)
            
            if len(features) == 0:
                return {
                    "outcome": DecisionOutcome.REFER,
                    "confidence": 0.5,
                    "reasoning": ["Insufficient data for ML prediction"],
                    "model_used": decision_type
                }
            
            # Make prediction
            feature_array = np.array(features).reshape(1, -1)
            
            # Check if model is trained
            if not hasattr(model, 'classes_'):
                # Model not trained, use default logic
                return await self._get_default_prediction(request)
            
            # Scale features if scaler is fitted
            if hasattr(scaler, 'scale_'):
                feature_array = scaler.transform(feature_array)
            
            # Get prediction and probability
            prediction = model.predict(feature_array)[0]
            probabilities = model.predict_proba(feature_array)[0]
            
            # Map prediction to outcome
            outcome_mapping = {
                0: DecisionOutcome.REJECT,
                1: DecisionOutcome.APPROVE,
                2: DecisionOutcome.REFER,
                3: DecisionOutcome.INVESTIGATE
            }
            
            outcome = outcome_mapping.get(prediction, DecisionOutcome.REFER)
            confidence = max(probabilities)
            
            # Generate reasoning based on feature importance
            reasoning = await self._generate_ml_reasoning(model, features, decision_type)
            
            return {
                "outcome": outcome,
                "confidence": confidence,
                "reasoning": reasoning,
                "model_used": decision_type,
                "prediction": int(prediction),
                "probabilities": probabilities.tolist(),
                "feature_count": len(features)
            }
            
        except Exception as e:
            logger.error(f"ML prediction error for {decision_type}: {e}")
            return await self._get_default_prediction(request)

    async def _prepare_features(self, input_data: Dict[str, Any], decision_type: str) -> List[float]:
        """Prepare features for ML model"""
        
        features = []
        
        # Feature mappings for different decision types
        feature_mappings = {
            "underwriting": [
                "age", "annual_income", "credit_score", "coverage_amount",
                "property_value", "location_risk_score", "claims_history"
            ],
            "claims": [
                "claim_amount", "policy_value", "days_since_incident",
                "claimant_age", "claim_type_code", "location_risk_score"
            ],
            "fraud_detection": [
                "claim_amount", "policy_value", "claims_last_year",
                "days_since_policy_start", "documentation_score",
                "claimant_history_score"
            ],
            "risk_assessment": [
                "property_age", "property_value", "location_risk_score",
                "safety_features_count", "previous_claims", "construction_type_code"
            ],
            "pricing": [
                "coverage_amount", "deductible", "risk_score",
                "location_factor", "age_factor", "claims_history"
            ]
        }
        
        feature_names = feature_mappings.get(decision_type, [])
        
        for feature_name in feature_names:
            value = input_data.get(feature_name, 0)
            
            # Handle different data types
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Encode string values
                if feature_name not in self.encoders[decision_type]:
                    self.encoders[decision_type][feature_name] = LabelEncoder()
                
                encoder = self.encoders[decision_type][feature_name]
                
                # Fit encoder if not already fitted
                if not hasattr(encoder, 'classes_'):
                    # Use common values for initialization
                    common_values = self._get_common_values(feature_name)
                    encoder.fit(common_values)
                
                try:
                    encoded_value = encoder.transform([value])[0]
                    features.append(float(encoded_value))
                except ValueError:
                    # Unknown value, use default
                    features.append(0.0)
            elif isinstance(value, list):
                features.append(float(len(value)))
            else:
                features.append(0.0)
        
        return features

    def _get_common_values(self, feature_name: str) -> List[str]:
        """Get common values for categorical features"""
        
        common_values = {
            "occupation": ["engineer", "teacher", "doctor", "manager", "clerk", "other"],
            "claim_type": ["auto", "home", "life", "health", "other"],
            "construction_type": ["wood", "brick", "concrete", "steel", "other"],
            "location_type": ["urban", "suburban", "rural"],
            "policy_type": ["basic", "standard", "premium"]
        }
        
        return common_values.get(feature_name, ["unknown", "other"])

    async def _generate_ml_reasoning(self, model, features: List[float], decision_type: str) -> List[str]:
        """Generate reasoning based on ML model decision"""
        
        reasoning = []
        
        try:
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Get top 3 most important features
                top_indices = np.argsort(importances)[-3:][::-1]
                
                feature_names = {
                    "underwriting": ["age", "income", "credit_score", "coverage", "property_value", "location_risk", "claims_history"],
                    "claims": ["claim_amount", "policy_value", "days_since_incident", "claimant_age", "claim_type", "location_risk"],
                    "fraud_detection": ["claim_amount", "policy_value", "claims_frequency", "policy_age", "documentation", "history"],
                    "risk_assessment": ["property_age", "property_value", "location_risk", "safety_features", "claims", "construction"],
                    "pricing": ["coverage", "deductible", "risk_score", "location", "age", "history"]
                }.get(decision_type, [f"feature_{i}" for i in range(len(features))])
                
                for idx in top_indices:
                    if idx < len(feature_names) and idx < len(features):
                        feature_name = feature_names[idx]
                        feature_value = features[idx]
                        importance = importances[idx]
                        
                        reasoning.append(f"Key factor: {feature_name} (value: {feature_value:.2f}, importance: {importance:.3f})")
            
            # Add model-specific reasoning
            reasoning.append(f"ML model ({type(model).__name__}) prediction based on {len(features)} features")
            
        except Exception as e:
            logger.warning(f"Error generating ML reasoning: {e}")
            reasoning.append("ML model prediction (detailed reasoning unavailable)")
        
        return reasoning

    async def _get_default_prediction(self, request: DecisionRequest) -> Dict[str, Any]:
        """Get default prediction when ML model is unavailable"""
        
        # Simple heuristic-based prediction
        input_data = request.input_data
        decision_type = request.decision_type
        
        if decision_type == DecisionType.UNDERWRITING:
            # Simple underwriting logic
            age = input_data.get("age", 0)
            income = input_data.get("annual_income", 0)
            credit_score = input_data.get("credit_score", 0)
            
            if age >= 18 and age <= 75 and income >= 25000 and credit_score >= 650:
                outcome = DecisionOutcome.APPROVE
                confidence = 0.7
            else:
                outcome = DecisionOutcome.REFER
                confidence = 0.6
                
        elif decision_type == DecisionType.CLAIMS:
            # Simple claims logic
            claim_amount = input_data.get("claim_amount", 0)
            coverage_limit = input_data.get("coverage_limit", 0)
            
            if claim_amount <= coverage_limit and claim_amount > 0:
                outcome = DecisionOutcome.APPROVE
                confidence = 0.7
            else:
                outcome = DecisionOutcome.INVESTIGATE
                confidence = 0.6
                
        else:
            outcome = DecisionOutcome.REFER
            confidence = 0.5
        
        return {
            "outcome": outcome,
            "confidence": confidence,
            "reasoning": ["Default heuristic-based prediction"],
            "model_used": "default_heuristic"
        }

    async def _combine_results(self, request: DecisionRequest, rule_result: Dict[str, Any], ml_result: Dict[str, Any]) -> DecisionResult:
        """Combine business rules and ML results"""
        
        # Weight the results (rules: 60%, ML: 40%)
        rule_weight = 0.6
        ml_weight = 0.4
        
        rule_confidence = rule_result["confidence"]
        ml_confidence = ml_result["confidence"]
        
        # Combine confidences
        combined_confidence = (rule_confidence * rule_weight) + (ml_confidence * ml_weight)
        
        # Determine final outcome
        rule_outcome = rule_result["outcome"]
        ml_outcome = ml_result["outcome"]
        
        # Priority logic for combining outcomes
        if rule_outcome == DecisionOutcome.REJECT or ml_outcome == DecisionOutcome.REJECT:
            final_outcome = DecisionOutcome.REJECT
        elif rule_outcome == DecisionOutcome.INVESTIGATE or ml_outcome == DecisionOutcome.INVESTIGATE:
            final_outcome = DecisionOutcome.INVESTIGATE
        elif rule_outcome == DecisionOutcome.APPROVE and ml_outcome == DecisionOutcome.APPROVE:
            final_outcome = DecisionOutcome.APPROVE
        else:
            final_outcome = DecisionOutcome.REFER
        
        # Combine reasoning
        combined_reasoning = []
        combined_reasoning.extend(rule_result["reasoning"])
        combined_reasoning.extend(ml_result["reasoning"])
        combined_reasoning.append(f"Combined decision (Rules: {rule_weight*100}%, ML: {ml_weight*100}%)")
        
        # Supporting data
        supporting_data = {
            "rule_analysis": rule_result,
            "ml_analysis": ml_result,
            "combination_weights": {"rules": rule_weight, "ml": ml_weight},
            "input_data": request.input_data,
            "context": request.context
        }
        
        return DecisionResult(
            request_id=request.request_id,
            decision_type=request.decision_type,
            outcome=final_outcome,
            confidence=combined_confidence,
            reasoning=combined_reasoning,
            supporting_data=supporting_data,
            model_used=f"combined_{ml_result['model_used']}",
            processing_time=0.0,  # Will be set by caller
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24)  # Cache for 24 hours
        )

    async def _get_cached_decision(self, request: DecisionRequest) -> Optional[DecisionResult]:
        """Get cached decision if available"""
        
        # Create cache key
        cache_key = self._create_cache_key(request)
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                
                # Check expiration
                expires_at = datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
                if expires_at and expires_at > datetime.utcnow():
                    # Convert back to DecisionResult
                    result = DecisionResult(
                        request_id=data["request_id"],
                        decision_type=DecisionType(data["decision_type"]),
                        outcome=DecisionOutcome(data["outcome"]),
                        confidence=data["confidence"],
                        reasoning=data["reasoning"],
                        supporting_data=data["supporting_data"],
                        model_used=data["model_used"],
                        processing_time=data["processing_time"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        expires_at=expires_at,
                        metadata=data.get("metadata", {})
                    )
                    
                    return result
                else:
                    # Expired, remove from cache
                    self.redis_client.delete(cache_key)
                    
        except Exception as e:
            logger.warning(f"Error retrieving cached decision: {e}")
        
        return None

    def _create_cache_key(self, request: DecisionRequest) -> str:
        """Create cache key for decision request"""
        
        # Create hash of input data for caching
        import hashlib
        
        cache_data = {
            "decision_type": request.decision_type.value,
            "input_data": request.input_data,
            "context": request.context
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        
        return f"decision_cache:{cache_hash}"

    async def _cache_decision(self, result: DecisionResult):
        """Cache decision result"""
        
        try:
            # Convert to dict for JSON serialization
            cache_data = {
                "request_id": result.request_id,
                "decision_type": result.decision_type.value,
                "outcome": result.outcome.value,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "supporting_data": result.supporting_data,
                "model_used": result.model_used,
                "processing_time": result.processing_time,
                "created_at": result.created_at.isoformat(),
                "expires_at": result.expires_at.isoformat() if result.expires_at else None,
                "metadata": result.metadata
            }
            
            # Create cache key
            cache_key = f"decision_result:{result.request_id}"
            
            # Cache for 24 hours
            self.redis_client.setex(
                cache_key,
                24 * 3600,
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.warning(f"Error caching decision: {e}")

    async def _store_decision(self, result: DecisionResult):
        """Store decision in database"""
        
        try:
            with self.Session() as session:
                model = DecisionModel(
                    request_id=result.request_id,
                    decision_type=result.decision_type.value,
                    outcome=result.outcome.value,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    supporting_data=result.supporting_data,
                    model_used=result.model_used,
                    processing_time=result.processing_time,
                    input_data=result.supporting_data.get("input_data", {}),
                    context=result.supporting_data.get("context", {}),
                    requester_id=result.supporting_data.get("requester_id", "unknown"),
                    created_at=result.created_at,
                    expires_at=result.expires_at,
                    metadata=result.metadata
                )
                
                session.merge(model)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing decision: {e}")

    async def train_model(self, decision_type: DecisionType, training_data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Train ML model with new data"""
        
        try:
            model_name = decision_type.value
            
            if model_name not in self.models:
                raise ValueError(f"Unknown decision type: {model_name}")
            
            # Prepare features and target
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns]
            y = training_data[target_column]
            
            # Handle categorical features
            for col in X.select_dtypes(include=['object']).columns:
                if col not in self.encoders[model_name]:
                    self.encoders[model_name][col] = LabelEncoder()
                
                X[col] = self.encoders[model_name][col].fit_transform(X[col])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = self.scalers[model_name]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Update metrics
            model_accuracy_gauge.labels(model_type=model_name).set(accuracy)
            
            # Save model
            await self._save_model(model_name, model, scaler, self.encoders[model_name])
            
            training_results = {
                "model_type": model_name,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "feature_count": len(feature_columns),
                "training_time": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Model {model_name} trained successfully. Accuracy: {accuracy:.3f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error training model {decision_type.value}: {e}")
            raise

    async def _save_model(self, model_name: str, model, scaler, encoders):
        """Save trained model to disk"""
        
        try:
            import os
            
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            
            # Save model
            model_file = f"{self.model_path}/{model_name}_model.joblib"
            joblib.dump(model, model_file)
            
            # Save scaler
            scaler_file = f"{self.model_path}/{model_name}_scaler.joblib"
            joblib.dump(scaler, scaler_file)
            
            # Save encoders
            encoder_file = f"{self.model_path}/{model_name}_encoders.joblib"
            joblib.dump(encoders, encoder_file)
            
            logger.info(f"Saved model {model_name} to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")

    async def get_decision_history(self, requester_id: str = None, decision_type: DecisionType = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get decision history"""
        
        try:
            with self.Session() as session:
                query = session.query(DecisionModel)
                
                if requester_id:
                    query = query.filter(DecisionModel.requester_id == requester_id)
                
                if decision_type:
                    query = query.filter(DecisionModel.decision_type == decision_type.value)
                
                decisions = query.order_by(DecisionModel.created_at.desc()).limit(limit).all()
                
                history = []
                for decision in decisions:
                    history.append({
                        "request_id": decision.request_id,
                        "decision_type": decision.decision_type,
                        "outcome": decision.outcome,
                        "confidence": decision.confidence,
                        "model_used": decision.model_used,
                        "processing_time": decision.processing_time,
                        "created_at": decision.created_at.isoformat(),
                        "reasoning": decision.reasoning[:3] if decision.reasoning else []  # First 3 reasons
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error retrieving decision history: {e}")
            return []

    async def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision statistics"""
        
        try:
            with self.Session() as session:
                # Total decisions
                total_decisions = session.query(DecisionModel).count()
                
                # Decisions by type
                type_stats = {}
                for decision_type in DecisionType:
                    count = session.query(DecisionModel).filter(
                        DecisionModel.decision_type == decision_type.value
                    ).count()
                    type_stats[decision_type.value] = count
                
                # Decisions by outcome
                outcome_stats = {}
                for outcome in DecisionOutcome:
                    count = session.query(DecisionModel).filter(
                        DecisionModel.outcome == outcome.value
                    ).count()
                    outcome_stats[outcome.value] = count
                
                # Average processing time
                avg_processing_time = session.query(DecisionModel.processing_time).all()
                avg_time = sum(t[0] for t in avg_processing_time) / len(avg_processing_time) if avg_processing_time else 0
                
                # Recent decisions (last 24 hours)
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                recent_count = session.query(DecisionModel).filter(
                    DecisionModel.created_at >= recent_cutoff
                ).count()
                
                return {
                    "total_decisions": total_decisions,
                    "decisions_by_type": type_stats,
                    "decisions_by_outcome": outcome_stats,
                    "average_processing_time": round(avg_time, 3),
                    "recent_24h": recent_count,
                    "active_models": len(self.models),
                    "cache_size": len(self.decision_cache)
                }
                
        except Exception as e:
            logger.error(f"Error getting decision statistics: {e}")
            return {}

    async def shutdown(self):
        """Graceful shutdown of the decision engine"""
        
        logger.info("Shutting down DecisionEngine...")
        
        # Stop processing
        self.processing_active = False
        
        # Clear cache
        self.decision_cache.clear()
        
        logger.info("DecisionEngine shutdown complete")

# Factory function
def create_decision_engine(db_url: str = None, redis_url: str = None, model_path: str = None) -> DecisionEngine:
    """Create and configure a DecisionEngine instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    if not model_path:
        model_path = "/tmp/insurance_ai_models"
    
    return DecisionEngine(db_url=db_url, redis_url=redis_url, model_path=model_path)

# Example usage
if __name__ == "__main__":
    async def test_decision_engine():
        """Test the decision engine functionality"""
        
        engine = create_decision_engine()
        
        # Test underwriting decision
        request = DecisionRequest(
            request_id=str(uuid.uuid4()),
            decision_type=DecisionType.UNDERWRITING,
            input_data={
                "age": 35,
                "annual_income": 75000,
                "credit_score": 720,
                "coverage_amount": 500000,
                "property_value": 300000,
                "location_risk_score": 4,
                "claims_history": 0
            },
            context={"application_id": "APP123456"},
            requester_id="underwriter_001"
        )
        
        result = await engine.make_decision(request)
        
        print(f"Decision: {result.outcome.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        
        # Get statistics
        stats = await engine.get_decision_statistics()
        print(f"Statistics: {stats}")
    
    # Run test
    # asyncio.run(test_decision_engine())

