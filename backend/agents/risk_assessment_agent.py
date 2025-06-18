"""
Insurance AI Agent System - Risk Assessment Agent
Production-ready agent for comprehensive risk analysis and scoring
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from decimal import Decimal
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Machine learning libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Statistical analysis
from scipy import stats
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

# Database and utilities
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
import structlog
import redis.asyncio as redis

from backend.shared.models import (
    Policy, Claim, AgentExecution, User, Organization,
    Evidence, EvidenceAnalysis
)
from backend.shared.schemas import (
    AgentExecutionStatus,
)
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

logger = structlog.get_logger(__name__)

@dataclass
class RiskFactor:
    """Risk factor data structure"""
    name: str
    value: float
    weight: float
    category: str
    description: str
    confidence: float

class RiskCategory(Enum):
    """Risk categories"""
    FRAUD = "fraud"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    REPUTATION = "reputation"

class RiskLevel(Enum):
    """Risk levels used for assessment output"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class RiskAssessmentAgent:
    """
    Advanced risk assessment agent for comprehensive risk analysis
    Provides multi-dimensional risk scoring and predictive analytics
    """
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
        self.agent_name = "risk_assessment_agent"
        self.agent_version = "1.0.0"
        self.logger = structlog.get_logger(self.agent_name)
        
        # Initialize ML models
        self._initialize_models()
        
        # Risk assessment configuration
        self.risk_config = {
            "fraud_threshold": 70.0,
            "high_risk_threshold": 80.0,
            "medium_risk_threshold": 50.0,
            "low_risk_threshold": 30.0,
            "confidence_threshold": 0.7,
            "model_retrain_interval_days": 30
        }
        
        # Risk factor weights
        self.risk_weights = {
            RiskCategory.FRAUD: {
                "evidence_authenticity": 0.25,
                "claim_patterns": 0.20,
                "behavioral_indicators": 0.15,
                "financial_anomalies": 0.15,
                "historical_claims": 0.15,
                "external_data": 0.10
            },
            RiskCategory.FINANCIAL: {
                "claim_amount": 0.30,
                "policy_limits": 0.20,
                "payment_history": 0.20,
                "credit_score": 0.15,
                "financial_stability": 0.15
            },
            RiskCategory.OPERATIONAL: {
                "processing_complexity": 0.25,
                "resource_requirements": 0.25,
                "timeline_constraints": 0.20,
                "regulatory_requirements": 0.15,
                "stakeholder_involvement": 0.15
            }
        }
        
        # Feature engineering configuration
        self.feature_config = {
            "temporal_features": ["hour_of_day", "day_of_week", "month", "season"],
            "categorical_features": ["claim_type", "policy_type", "location", "cause"],
            "numerical_features": ["amount", "age", "tenure", "previous_claims"],
            "derived_features": ["amount_to_limit_ratio", "claims_frequency", "severity_index"]
        }
        
        # Model cache
        self.model_cache = {}
        self.scaler_cache = {}
        
        # Risk assessment cache TTL (seconds)
        self.cache_ttl = 3600  # 1 hour
    
    def _initialize_models(self):
        """Initialize machine learning models"""
        
        try:
            # Fraud detection model
            self.fraud_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            # Risk scoring model
            self.risk_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Feature scalers
            self.fraud_scaler = StandardScaler()
            self.risk_scaler = StandardScaler()
            
            # Label encoders for categorical features
            self.label_encoders = {}
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize ML models", error=str(e))
            # Continue without ML models
            self.fraud_model = None
            self.risk_model = None
    
    async def assess_risk(
        self,
        entity_type: str,
        entity_id: uuid.UUID,
        assessment_type: str = "comprehensive",
        risk_categories: Optional[List[RiskCategory]] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment
        
        Args:
            entity_type: Type of entity (claim, policy, user, organization)
            entity_id: UUID of the entity to assess
            assessment_type: Type of assessment (comprehensive, fraud, financial, operational)
            risk_categories: Specific risk categories to assess
            context_data: Additional context for assessment
            
        Returns:
            Dictionary containing risk assessment results
        """
        
        async with performance_monitor.monitor_operation("risk_assessment"):
            try:
                # Create agent execution record
                execution = AgentExecution(
                    agent_name=self.agent_name,
                    agent_version=self.agent_version,
                    input_data={
                        "entity_type": entity_type,
                        "entity_id": str(entity_id),
                        "assessment_type": assessment_type,
                        "risk_categories": [cat.value for cat in risk_categories] if risk_categories else None,
                        "context_data": context_data or {}
                    },
                    status=AgentExecutionStatus.RUNNING
                )
                
                self.db_session.add(execution)
                await self.db_session.commit()
                await self.db_session.refresh(execution)
                
                start_time = datetime.utcnow()
                
                # Check cache first
                cache_key = f"risk_assessment:{entity_type}:{entity_id}:{assessment_type}"
                cached_result = await self.redis_client.get(cache_key)
                
                if cached_result:
                    cached_data = json.loads(cached_result)
                    self.logger.info("Risk assessment retrieved from cache", 
                                   entity_type=entity_type, entity_id=str(entity_id))
                    return cached_data
                
                # Gather entity data
                entity_data = await self._gather_entity_data(entity_type, entity_id)
                
                # Determine risk categories to assess
                if not risk_categories:
                    risk_categories = self._determine_risk_categories(assessment_type)
                
                # Perform risk assessments by category
                risk_assessments = {}
                overall_risk_factors = []
                
                for category in risk_categories:
                    category_assessment = await self._assess_category_risk(
                        category, entity_data, context_data or {}
                    )
                    risk_assessments[category.value] = category_assessment
                    overall_risk_factors.extend(category_assessment.get("risk_factors", []))
                
                # Calculate overall risk score
                overall_score = self._calculate_overall_risk_score(risk_assessments)
                
                # Determine risk level
                risk_level = self._determine_risk_level(overall_score)
                
                # Generate recommendations
                recommendations = await self._generate_recommendations(
                    risk_assessments, overall_score, entity_type, entity_data
                )
                
                # Prepare final result
                assessment_result = {
                    "entity_type": entity_type,
                    "entity_id": str(entity_id),
                    "assessment_type": assessment_type,
                    "assessment_timestamp": datetime.utcnow().isoformat(),
                    "overall_risk_score": overall_score,
                    "risk_level": risk_level.value,
                    "risk_assessments": risk_assessments,
                    "overall_risk_factors": overall_risk_factors,
                    "recommendations": recommendations,
                    "confidence_score": self._calculate_confidence_score(risk_assessments),
                    "assessment_metadata": {
                        "agent_name": self.agent_name,
                        "agent_version": self.agent_version,
                        "processing_time_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
                        "data_sources": list(entity_data.keys()),
                        "risk_categories_assessed": [cat.value for cat in risk_categories]
                    }
                }
                
                # Save assessment results
                await self._save_assessment_results(entity_type, entity_id, assessment_result)
                
                # Cache results
                await self.redis_client.setex(
                    cache_key, 
                    self.cache_ttl, 
                    json.dumps(assessment_result, default=str)
                )
                
                # Update execution record
                execution.status = AgentExecutionStatus.COMPLETED
                execution.output_data = assessment_result
                execution.execution_time_ms = assessment_result["assessment_metadata"]["processing_time_ms"]
                execution.completed_at = datetime.utcnow()
                
                await self.db_session.commit()
                
                # Record metrics
                metrics.record_agent_execution(
                    self.agent_name, 
                    execution.execution_time_ms / 1000, 
                    success=True
                )
                
                # Log assessment
                audit_logger.log_user_action(
                    user_id="system",
                    action="risk_assessment_completed",
                    resource_type=entity_type,
                    resource_id=str(entity_id),
                    details={
                        "assessment_type": assessment_type,
                        "overall_risk_score": overall_score,
                        "risk_level": risk_level.value,
                        "processing_time_ms": execution.execution_time_ms
                    }
                )
                
                self.logger.info(
                    "Risk assessment completed",
                    entity_type=entity_type,
                    entity_id=str(entity_id),
                    overall_risk_score=overall_score,
                    risk_level=risk_level.value,
                    processing_time_ms=execution.execution_time_ms
                )
                
                return assessment_result
                
            except Exception as e:
                # Update execution record with error
                execution.status = AgentExecutionStatus.FAILED
                execution.error_message = str(e)
                execution.completed_at = datetime.utcnow()
                
                await self.db_session.commit()
                
                # Record metrics
                metrics.record_agent_execution(self.agent_name, 0, success=False)
                
                self.logger.error(
                    "Risk assessment failed",
                    entity_type=entity_type,
                    entity_id=str(entity_id),
                    error=str(e)
                )
                raise ServiceException(f"Risk assessment failed: {str(e)}")
    
    async def _gather_entity_data(self, entity_type: str, entity_id: uuid.UUID) -> Dict[str, Any]:
        """Gather comprehensive data for entity"""
        
        try:
            entity_data = {}
            
            if entity_type == "claim":
                entity_data = await self._gather_claim_data(entity_id)
            elif entity_type == "policy":
                entity_data = await self._gather_policy_data(entity_id)
            elif entity_type == "user":
                entity_data = await self._gather_user_data(entity_id)
            elif entity_type == "organization":
                entity_data = await self._gather_organization_data(entity_id)
            else:
                raise ServiceException(f"Unsupported entity type: {entity_type}")
            
            return entity_data
            
        except Exception as e:
            self.logger.error("Failed to gather entity data", error=str(e))
            raise
    
    async def _gather_claim_data(self, claim_id: uuid.UUID) -> Dict[str, Any]:
        """Gather comprehensive claim data"""
        
        try:
            # Get claim details
            claim_service = BaseService(Claim, self.db_session)
            claim = await claim_service.get(claim_id)
            
            if not claim:
                raise ServiceException(f"Claim not found: {claim_id}")
            
            # Get related policy
            policy_service = BaseService(Policy, self.db_session)
            policy = await policy_service.get(claim.policy_id) if claim.policy_id else None
            
            # Get evidence and analysis
            evidence_query = select(Evidence).where(Evidence.claim_id == claim_id)
            evidence_result = await self.db_session.execute(evidence_query)
            evidence_list = evidence_result.scalars().all()
            
            # Get evidence analyses
            evidence_analyses = []
            for evidence in evidence_list:
                analysis_query = select(EvidenceAnalysis).where(
                    EvidenceAnalysis.evidence_id == evidence.id
                )
                analysis_result = await self.db_session.execute(analysis_query)
                analyses = analysis_result.scalars().all()
                evidence_analyses.extend(analyses)
            
            # Get historical claims for the same policy/user
            historical_claims = []
            if policy:
                hist_query = select(Claim).where(
                    and_(
                        Claim.policy_id == policy.id,
                        Claim.id != claim_id
                    )
                )
                hist_result = await self.db_session.execute(hist_query)
                historical_claims = hist_result.scalars().all()
            
            # Compile claim data
            claim_data = {
                "claim": {
                    "id": str(claim.id),
                    "claim_number": claim.claim_number,
                    "status": claim.status.value,
                    "claim_type": claim.claim_type,
                    "incident_date": claim.incident_date.isoformat() if claim.incident_date else None,
                    "reported_date": claim.reported_date.isoformat() if claim.reported_date else None,
                    "amount_claimed": float(claim.amount_claimed) if claim.amount_claimed else 0.0,
                    "amount_approved": float(claim.amount_approved) if claim.amount_approved else 0.0,
                    "description": claim.description,
                    "location": claim.incident_location,
                    "cause": claim.cause_of_loss,
                    "created_at": claim.created_at.isoformat()
                },
                "policy": {
                    "id": str(policy.id) if policy else None,
                    "policy_number": policy.policy_number if policy else None,
                    "policy_type": policy.policy_type if policy else None,
                    "coverage_amount": float(policy.coverage_amount) if policy and policy.coverage_amount else 0.0,
                    "premium_amount": float(policy.premium_amount) if policy and policy.premium_amount else 0.0,
                    "effective_date": policy.effective_date.isoformat() if policy and policy.effective_date else None,
                    "expiry_date": policy.expiry_date.isoformat() if policy and policy.expiry_date else None,
                    "status": policy.status.value if policy else None
                },
                "evidence": [
                    {
                        "id": str(ev.id),
                        "file_type": ev.file_type,
                        "file_size": ev.file_size,
                        "upload_date": ev.created_at.isoformat(),
                        "description": ev.description
                    }
                    for ev in evidence_list
                ],
                "evidence_analyses": [
                    {
                        "id": str(analysis.id),
                        "evidence_id": str(analysis.evidence_id),
                        "analysis_type": analysis.analysis_type,
                        "quality_score": float(analysis.quality_score) if analysis.quality_score else 0.0,
                        "authenticity_score": float(analysis.authenticity_score) if analysis.authenticity_score else 0.0,
                        "fraud_score": float(analysis.fraud_score) if analysis.fraud_score else 0.0,
                        "analysis_data": analysis.analysis_data
                    }
                    for analysis in evidence_analyses
                ],
                "historical_claims": [
                    {
                        "id": str(hc.id),
                        "claim_number": hc.claim_number,
                        "status": hc.status.value,
                        "claim_type": hc.claim_type,
                        "amount_claimed": float(hc.amount_claimed) if hc.amount_claimed else 0.0,
                        "amount_approved": float(hc.amount_approved) if hc.amount_approved else 0.0,
                        "incident_date": hc.incident_date.isoformat() if hc.incident_date else None,
                        "reported_date": hc.reported_date.isoformat() if hc.reported_date else None
                    }
                    for hc in historical_claims
                ]
            }
            
            return claim_data
            
        except Exception as e:
            self.logger.error("Failed to gather claim data", error=str(e))
            raise
    
    async def _gather_policy_data(self, policy_id: uuid.UUID) -> Dict[str, Any]:
        """Gather comprehensive policy data"""
        
        try:
            # Get policy details
            policy_service = BaseService(Policy, self.db_session)
            policy = await policy_service.get(policy_id)
            
            if not policy:
                raise ServiceException(f"Policy not found: {policy_id}")
            
            # Get claims for this policy
            claims_query = select(Claim).where(Claim.policy_id == policy_id)
            claims_result = await self.db_session.execute(claims_query)
            claims = claims_result.scalars().all()
            
            # Get user/organization data
            user_data = {}
            if policy.user_id:
                user_service = BaseService(User, self.db_session)
                user = await user_service.get(policy.user_id)
                if user:
                    user_data = {
                        "id": str(user.id),
                        "email": user.email,
                        "created_at": user.created_at.isoformat(),
                        "is_active": user.is_active
                    }
            
            policy_data = {
                "policy": {
                    "id": str(policy.id),
                    "policy_number": policy.policy_number,
                    "policy_type": policy.policy_type,
                    "status": policy.status.value,
                    "coverage_amount": float(policy.coverage_amount) if policy.coverage_amount else 0.0,
                    "premium_amount": float(policy.premium_amount) if policy.premium_amount else 0.0,
                    "deductible_amount": float(policy.deductible_amount) if policy.deductible_amount else 0.0,
                    "effective_date": policy.effective_date.isoformat() if policy.effective_date else None,
                    "expiry_date": policy.expiry_date.isoformat() if policy.expiry_date else None,
                    "created_at": policy.created_at.isoformat()
                },
                "user": user_data,
                "claims": [
                    {
                        "id": str(claim.id),
                        "claim_number": claim.claim_number,
                        "status": claim.status.value,
                        "claim_type": claim.claim_type,
                        "amount_claimed": float(claim.amount_claimed) if claim.amount_claimed else 0.0,
                        "amount_approved": float(claim.amount_approved) if claim.amount_approved else 0.0,
                        "incident_date": claim.incident_date.isoformat() if claim.incident_date else None,
                        "reported_date": claim.reported_date.isoformat() if claim.reported_date else None
                    }
                    for claim in claims
                ]
            }
            
            return policy_data
            
        except Exception as e:
            self.logger.error("Failed to gather policy data", error=str(e))
            raise
    
    async def _gather_user_data(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """Gather comprehensive user data"""
        
        try:
            # Get user details
            user_service = BaseService(User, self.db_session)
            user = await user_service.get(user_id)
            
            if not user:
                raise ServiceException(f"User not found: {user_id}")
            
            # Get user's policies
            policies_query = select(Policy).where(Policy.user_id == user_id)
            policies_result = await self.db_session.execute(policies_query)
            policies = policies_result.scalars().all()
            
            # Get user's claims
            claims_query = select(Claim).join(Policy).where(Policy.user_id == user_id)
            claims_result = await self.db_session.execute(claims_query)
            claims = claims_result.scalars().all()
            
            user_data = {
                "user": {
                    "id": str(user.id),
                    "email": user.email,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None
                },
                "policies": [
                    {
                        "id": str(policy.id),
                        "policy_number": policy.policy_number,
                        "policy_type": policy.policy_type,
                        "status": policy.status.value,
                        "coverage_amount": float(policy.coverage_amount) if policy.coverage_amount else 0.0,
                        "premium_amount": float(policy.premium_amount) if policy.premium_amount else 0.0,
                        "effective_date": policy.effective_date.isoformat() if policy.effective_date else None,
                        "expiry_date": policy.expiry_date.isoformat() if policy.expiry_date else None
                    }
                    for policy in policies
                ],
                "claims": [
                    {
                        "id": str(claim.id),
                        "claim_number": claim.claim_number,
                        "status": claim.status.value,
                        "claim_type": claim.claim_type,
                        "amount_claimed": float(claim.amount_claimed) if claim.amount_claimed else 0.0,
                        "amount_approved": float(claim.amount_approved) if claim.amount_approved else 0.0,
                        "incident_date": claim.incident_date.isoformat() if claim.incident_date else None,
                        "reported_date": claim.reported_date.isoformat() if claim.reported_date else None
                    }
                    for claim in claims
                ]
            }
            
            return user_data
            
        except Exception as e:
            self.logger.error("Failed to gather user data", error=str(e))
            raise
    
    async def _gather_organization_data(self, org_id: uuid.UUID) -> Dict[str, Any]:
        """Gather comprehensive organization data"""
        
        try:
            # Get organization details
            org_service = BaseService(Organization, self.db_session)
            organization = await org_service.get(org_id)
            
            if not organization:
                raise ServiceException(f"Organization not found: {org_id}")
            
            # Get organization's users
            users_query = select(User).where(User.organization_id == org_id)
            users_result = await self.db_session.execute(users_query)
            users = users_result.scalars().all()
            
            # Get organization's policies
            policies_query = select(Policy).where(Policy.organization_id == org_id)
            policies_result = await self.db_session.execute(policies_query)
            policies = policies_result.scalars().all()
            
            org_data = {
                "organization": {
                    "id": str(organization.id),
                    "name": organization.name,
                    "organization_type": organization.organization_type,
                    "is_active": organization.is_active,
                    "created_at": organization.created_at.isoformat()
                },
                "users": [
                    {
                        "id": str(user.id),
                        "email": user.email,
                        "is_active": user.is_active,
                        "created_at": user.created_at.isoformat()
                    }
                    for user in users
                ],
                "policies": [
                    {
                        "id": str(policy.id),
                        "policy_number": policy.policy_number,
                        "policy_type": policy.policy_type,
                        "status": policy.status.value,
                        "coverage_amount": float(policy.coverage_amount) if policy.coverage_amount else 0.0,
                        "premium_amount": float(policy.premium_amount) if policy.premium_amount else 0.0
                    }
                    for policy in policies
                ]
            }
            
            return org_data
            
        except Exception as e:
            self.logger.error("Failed to gather organization data", error=str(e))
            raise
    
    def _determine_risk_categories(self, assessment_type: str) -> List[RiskCategory]:
        """Determine risk categories based on assessment type"""
        
        if assessment_type == "fraud":
            return [RiskCategory.FRAUD]
        elif assessment_type == "financial":
            return [RiskCategory.FINANCIAL]
        elif assessment_type == "operational":
            return [RiskCategory.OPERATIONAL]
        else:  # comprehensive
            return [RiskCategory.FRAUD, RiskCategory.FINANCIAL, RiskCategory.OPERATIONAL]
    
    async def _assess_category_risk(
        self, 
        category: RiskCategory, 
        entity_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risk for specific category"""
        
        try:
            if category == RiskCategory.FRAUD:
                return await self._assess_fraud_risk(entity_data, context_data)
            elif category == RiskCategory.FINANCIAL:
                return await self._assess_financial_risk(entity_data, context_data)
            elif category == RiskCategory.OPERATIONAL:
                return await self._assess_operational_risk(entity_data, context_data)
            else:
                return {"risk_score": 0.0, "risk_factors": [], "confidence": 0.0}
                
        except Exception as e:
            self.logger.error(f"Failed to assess {category.value} risk", error=str(e))
            return {"risk_score": 0.0, "risk_factors": [], "confidence": 0.0}
    
    async def _assess_fraud_risk(self, entity_data: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess fraud risk"""
        
        try:
            risk_factors = []
            fraud_score = 0.0
            
            # Evidence authenticity analysis
            if "evidence_analyses" in entity_data:
                evidence_risk = self._analyze_evidence_fraud_risk(entity_data["evidence_analyses"])
                risk_factors.extend(evidence_risk["factors"])
                fraud_score += evidence_risk["score"] * self.risk_weights[RiskCategory.FRAUD]["evidence_authenticity"]
            
            # Claim pattern analysis
            if "claim" in entity_data:
                pattern_risk = self._analyze_claim_patterns(entity_data["claim"], entity_data.get("historical_claims", []))
                risk_factors.extend(pattern_risk["factors"])
                fraud_score += pattern_risk["score"] * self.risk_weights[RiskCategory.FRAUD]["claim_patterns"]
            
            # Behavioral indicators
            behavioral_risk = self._analyze_behavioral_indicators(entity_data)
            risk_factors.extend(behavioral_risk["factors"])
            fraud_score += behavioral_risk["score"] * self.risk_weights[RiskCategory.FRAUD]["behavioral_indicators"]
            
            # Financial anomalies
            financial_risk = self._analyze_financial_anomalies(entity_data)
            risk_factors.extend(financial_risk["factors"])
            fraud_score += financial_risk["score"] * self.risk_weights[RiskCategory.FRAUD]["financial_anomalies"]
            
            # Historical claims analysis
            historical_risk = self._analyze_historical_claims_risk(entity_data.get("historical_claims", []))
            risk_factors.extend(historical_risk["factors"])
            fraud_score += historical_risk["score"] * self.risk_weights[RiskCategory.FRAUD]["historical_claims"]
            
            # Normalize score to 0-100
            fraud_score = min(100.0, max(0.0, fraud_score * 100))
            
            # Calculate confidence based on data availability
            confidence = self._calculate_fraud_confidence(entity_data)
            
            return {
                "risk_score": fraud_score,
                "risk_factors": risk_factors,
                "confidence": confidence,
                "category": RiskCategory.FRAUD.value,
                "assessment_details": {
                    "evidence_authenticity_score": evidence_risk.get("score", 0.0) if "evidence_analyses" in entity_data else 0.0,
                    "claim_pattern_score": pattern_risk.get("score", 0.0) if "claim" in entity_data else 0.0,
                    "behavioral_score": behavioral_risk.get("score", 0.0),
                    "financial_anomaly_score": financial_risk.get("score", 0.0),
                    "historical_claims_score": historical_risk.get("score", 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error("Fraud risk assessment failed", error=str(e))
            return {"risk_score": 0.0, "risk_factors": [], "confidence": 0.0}
    
    def _analyze_evidence_fraud_risk(self, evidence_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fraud risk from evidence analyses"""
        
        try:
            if not evidence_analyses:
                return {"score": 0.0, "factors": []}
            
            fraud_scores = []
            authenticity_scores = []
            quality_scores = []
            factors = []
            
            for analysis in evidence_analyses:
                fraud_score = analysis.get("fraud_score", 0.0)
                authenticity_score = analysis.get("authenticity_score", 1.0)
                quality_score = analysis.get("quality_score", 1.0)
                
                fraud_scores.append(fraud_score)
                authenticity_scores.append(authenticity_score)
                quality_scores.append(quality_score)
                
                # Add specific risk factors
                if fraud_score > 70:
                    factors.append(RiskFactor(
                        name="high_fraud_score_evidence",
                        value=fraud_score,
                        weight=0.8,
                        category="evidence",
                        description=f"Evidence shows high fraud score: {fraud_score:.1f}",
                        confidence=0.8
                    ))
                
                if authenticity_score < 0.3:
                    factors.append(RiskFactor(
                        name="low_authenticity_evidence",
                        value=authenticity_score,
                        weight=0.7,
                        category="evidence",
                        description=f"Evidence shows low authenticity: {authenticity_score:.2f}",
                        confidence=0.7
                    ))
                
                if quality_score < 0.3:
                    factors.append(RiskFactor(
                        name="poor_quality_evidence",
                        value=quality_score,
                        weight=0.4,
                        category="evidence",
                        description=f"Evidence quality is poor: {quality_score:.2f}",
                        confidence=0.6
                    ))
            
            # Calculate overall evidence risk score
            avg_fraud_score = np.mean(fraud_scores)
            avg_authenticity_score = np.mean(authenticity_scores)
            avg_quality_score = np.mean(quality_scores)
            
            # Combine scores (higher fraud score and lower authenticity = higher risk)
            evidence_risk_score = (avg_fraud_score / 100.0 + (1.0 - avg_authenticity_score)) / 2.0
            
            return {
                "score": min(1.0, evidence_risk_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Evidence fraud risk analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _analyze_claim_patterns(self, claim_data: Dict[str, Any], historical_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze claim patterns for fraud indicators"""
        
        try:
            factors = []
            pattern_score = 0.0
            
            # Analyze claim timing
            if claim_data.get("incident_date") and claim_data.get("reported_date"):
                incident_date = datetime.fromisoformat(claim_data["incident_date"])
                reported_date = datetime.fromisoformat(claim_data["reported_date"])
                
                reporting_delay = (reported_date - incident_date).days
                
                if reporting_delay > 30:
                    factors.append(RiskFactor(
                        name="delayed_reporting",
                        value=reporting_delay,
                        weight=0.6,
                        category="timing",
                        description=f"Claim reported {reporting_delay} days after incident",
                        confidence=0.8
                    ))
                    pattern_score += 0.3
                
                # Weekend/holiday reporting (simplified check)
                if reported_date.weekday() >= 5:  # Saturday or Sunday
                    factors.append(RiskFactor(
                        name="weekend_reporting",
                        value=1.0,
                        weight=0.3,
                        category="timing",
                        description="Claim reported on weekend",
                        confidence=0.5
                    ))
                    pattern_score += 0.1
            
            # Analyze claim amount patterns
            claim_amount = claim_data.get("amount_claimed", 0.0)
            
            if historical_claims:
                historical_amounts = [hc.get("amount_claimed", 0.0) for hc in historical_claims]
                
                if historical_amounts:
                    avg_historical = np.mean(historical_amounts)
                    
                    # Check for unusually high claim amount
                    if claim_amount > avg_historical * 3:
                        factors.append(RiskFactor(
                            name="unusually_high_amount",
                            value=claim_amount / avg_historical,
                            weight=0.7,
                            category="amount",
                            description=f"Claim amount is {claim_amount/avg_historical:.1f}x higher than historical average",
                            confidence=0.7
                        ))
                        pattern_score += 0.4
                    
                    # Check for frequent claims
                    recent_claims = [
                        hc for hc in historical_claims 
                        if hc.get("reported_date") and 
                        (datetime.utcnow() - datetime.fromisoformat(hc["reported_date"])).days <= 365
                    ]
                    
                    if len(recent_claims) >= 3:
                        factors.append(RiskFactor(
                            name="frequent_claims",
                            value=len(recent_claims),
                            weight=0.6,
                            category="frequency",
                            description=f"{len(recent_claims)} claims in the past year",
                            confidence=0.8
                        ))
                        pattern_score += 0.3
            
            # Analyze claim type patterns
            claim_type = claim_data.get("claim_type", "")
            if claim_type in ["theft", "fire", "water_damage"]:
                # These types have higher fraud rates
                factors.append(RiskFactor(
                    name="high_risk_claim_type",
                    value=1.0,
                    weight=0.4,
                    category="type",
                    description=f"Claim type '{claim_type}' has higher fraud risk",
                    confidence=0.6
                ))
                pattern_score += 0.2
            
            return {
                "score": min(1.0, pattern_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Claim pattern analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _analyze_behavioral_indicators(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral indicators for fraud risk"""
        
        try:
            factors = []
            behavioral_score = 0.0
            
            # Analyze user behavior if available
            if "user" in entity_data:
                user_data = entity_data["user"]
                
                # Check account age
                if user_data.get("created_at"):
                    account_age = (datetime.utcnow() - datetime.fromisoformat(user_data["created_at"])).days
                    
                    if account_age < 30:  # Very new account
                        factors.append(RiskFactor(
                            name="new_account",
                            value=account_age,
                            weight=0.5,
                            category="behavior",
                            description=f"Account created {account_age} days ago",
                            confidence=0.7
                        ))
                        behavioral_score += 0.3
                
                # Check activity patterns
                if not user_data.get("is_active", True):
                    factors.append(RiskFactor(
                        name="inactive_account",
                        value=1.0,
                        weight=0.3,
                        category="behavior",
                        description="Account is inactive",
                        confidence=0.6
                    ))
                    behavioral_score += 0.2
            
            # Analyze policy behavior
            if "policies" in entity_data:
                policies = entity_data["policies"]
                
                # Check for multiple policies
                if len(policies) > 5:
                    factors.append(RiskFactor(
                        name="multiple_policies",
                        value=len(policies),
                        weight=0.4,
                        category="behavior",
                        description=f"User has {len(policies)} policies",
                        confidence=0.6
                    ))
                    behavioral_score += 0.2
                
                # Check for recent policy changes
                recent_policies = [
                    p for p in policies 
                    if p.get("effective_date") and 
                    (datetime.utcnow() - datetime.fromisoformat(p["effective_date"])).days <= 90
                ]
                
                if len(recent_policies) > 1:
                    factors.append(RiskFactor(
                        name="recent_policy_changes",
                        value=len(recent_policies),
                        weight=0.5,
                        category="behavior",
                        description=f"{len(recent_policies)} policies started in last 90 days",
                        confidence=0.7
                    ))
                    behavioral_score += 0.3
            
            return {
                "score": min(1.0, behavioral_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Behavioral analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _analyze_financial_anomalies(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial anomalies for fraud risk"""
        
        try:
            factors = []
            financial_score = 0.0
            
            # Analyze claim to policy ratio
            if "claim" in entity_data and "policy" in entity_data:
                claim_amount = entity_data["claim"].get("amount_claimed", 0.0)
                coverage_amount = entity_data["policy"].get("coverage_amount", 0.0)
                
                if coverage_amount > 0:
                    claim_ratio = claim_amount / coverage_amount
                    
                    if claim_ratio > 0.8:  # Claiming more than 80% of coverage
                        factors.append(RiskFactor(
                            name="high_claim_to_coverage_ratio",
                            value=claim_ratio,
                            weight=0.7,
                            category="financial",
                            description=f"Claim amount is {claim_ratio:.1%} of coverage",
                            confidence=0.8
                        ))
                        financial_score += 0.4
                    
                    # Check for round numbers (potential indicator of estimation)
                    if claim_amount % 1000 == 0 and claim_amount > 5000:
                        factors.append(RiskFactor(
                            name="round_claim_amount",
                            value=claim_amount,
                            weight=0.3,
                            category="financial",
                            description=f"Claim amount is a round number: ${claim_amount:,.0f}",
                            confidence=0.5
                        ))
                        financial_score += 0.1
            
            # Analyze premium to claim ratio
            if "policy" in entity_data and "claims" in entity_data:
                premium_amount = entity_data["policy"].get("premium_amount", 0.0)
                total_claims = sum(claim.get("amount_claimed", 0.0) for claim in entity_data["claims"])
                
                if premium_amount > 0 and total_claims > 0:
                    claims_to_premium_ratio = total_claims / premium_amount
                    
                    if claims_to_premium_ratio > 5:  # Claims exceed 5x annual premium
                        factors.append(RiskFactor(
                            name="high_claims_to_premium_ratio",
                            value=claims_to_premium_ratio,
                            weight=0.6,
                            category="financial",
                            description=f"Total claims are {claims_to_premium_ratio:.1f}x annual premium",
                            confidence=0.7
                        ))
                        financial_score += 0.3
            
            return {
                "score": min(1.0, financial_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Financial anomaly analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _analyze_historical_claims_risk(self, historical_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical claims for risk patterns"""
        
        try:
            factors = []
            historical_score = 0.0
            
            if not historical_claims:
                return {"score": 0.0, "factors": []}
            
            # Analyze claim frequency
            claim_count = len(historical_claims)
            if claim_count >= 5:
                factors.append(RiskFactor(
                    name="high_claim_frequency",
                    value=claim_count,
                    weight=0.6,
                    category="history",
                    description=f"User has {claim_count} historical claims",
                    confidence=0.8
                ))
                historical_score += 0.4
            
            # Analyze claim success rate
            approved_claims = [c for c in historical_claims if c.get("amount_approved", 0) > 0]
            if claim_count > 0:
                approval_rate = len(approved_claims) / claim_count
                
                if approval_rate < 0.5:  # Less than 50% approval rate
                    factors.append(RiskFactor(
                        name="low_approval_rate",
                        value=approval_rate,
                        weight=0.5,
                        category="history",
                        description=f"Historical approval rate: {approval_rate:.1%}",
                        confidence=0.7
                    ))
                    historical_score += 0.3
            
            # Analyze claim timing patterns
            claim_dates = []
            for claim in historical_claims:
                if claim.get("incident_date"):
                    claim_dates.append(datetime.fromisoformat(claim["incident_date"]))
            
            if len(claim_dates) >= 3:
                # Check for seasonal patterns (simplified)
                months = [date.month for date in claim_dates]
                month_counts = {}
                for month in months:
                    month_counts[month] = month_counts.get(month, 0) + 1
                
                max_month_count = max(month_counts.values())
                if max_month_count >= len(claim_dates) * 0.6:  # 60% of claims in same month
                    factors.append(RiskFactor(
                        name="seasonal_claim_pattern",
                        value=max_month_count / len(claim_dates),
                        weight=0.4,
                        category="history",
                        description=f"{max_month_count} out of {len(claim_dates)} claims in same month",
                        confidence=0.6
                    ))
                    historical_score += 0.2
            
            return {
                "score": min(1.0, historical_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Historical claims analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _calculate_fraud_confidence(self, entity_data: Dict[str, Any]) -> float:
        """Calculate confidence score for fraud assessment"""
        
        try:
            confidence_factors = []
            
            # Data availability factors
            if "evidence_analyses" in entity_data and entity_data["evidence_analyses"]:
                confidence_factors.append(0.3)  # Evidence available
            
            if "historical_claims" in entity_data and entity_data["historical_claims"]:
                confidence_factors.append(0.2)  # Historical data available
            
            if "policy" in entity_data and entity_data["policy"]:
                confidence_factors.append(0.2)  # Policy data available
            
            if "user" in entity_data and entity_data["user"]:
                confidence_factors.append(0.2)  # User data available
            
            # Always have basic claim data
            confidence_factors.append(0.1)
            
            return min(1.0, sum(confidence_factors))
            
        except Exception:
            return 0.5
    
    async def _assess_financial_risk(self, entity_data: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess financial risk"""
        
        try:
            risk_factors = []
            financial_score = 0.0
            
            # Claim amount analysis
            if "claim" in entity_data:
                amount_risk = self._analyze_claim_amount_risk(entity_data["claim"], entity_data.get("policy"))
                risk_factors.extend(amount_risk["factors"])
                financial_score += amount_risk["score"] * self.risk_weights[RiskCategory.FINANCIAL]["claim_amount"]
            
            # Policy limits analysis
            if "policy" in entity_data:
                limits_risk = self._analyze_policy_limits_risk(entity_data["policy"])
                risk_factors.extend(limits_risk["factors"])
                financial_score += limits_risk["score"] * self.risk_weights[RiskCategory.FINANCIAL]["policy_limits"]
            
            # Payment history analysis
            payment_risk = self._analyze_payment_history_risk(entity_data)
            risk_factors.extend(payment_risk["factors"])
            financial_score += payment_risk["score"] * self.risk_weights[RiskCategory.FINANCIAL]["payment_history"]
            
            # Normalize score to 0-100
            financial_score = min(100.0, max(0.0, financial_score * 100))
            
            # Calculate confidence
            confidence = self._calculate_financial_confidence(entity_data)
            
            return {
                "risk_score": financial_score,
                "risk_factors": risk_factors,
                "confidence": confidence,
                "category": RiskCategory.FINANCIAL.value,
                "assessment_details": {
                    "claim_amount_score": amount_risk.get("score", 0.0) if "claim" in entity_data else 0.0,
                    "policy_limits_score": limits_risk.get("score", 0.0) if "policy" in entity_data else 0.0,
                    "payment_history_score": payment_risk.get("score", 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error("Financial risk assessment failed", error=str(e))
            return {"risk_score": 0.0, "risk_factors": [], "confidence": 0.0}
    
    def _analyze_claim_amount_risk(self, claim_data: Dict[str, Any], policy_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze claim amount risk"""
        
        try:
            factors = []
            amount_score = 0.0
            
            claim_amount = claim_data.get("amount_claimed", 0.0)
            
            # High claim amount risk
            if claim_amount > 100000:  # $100k+
                factors.append(RiskFactor(
                    name="high_claim_amount",
                    value=claim_amount,
                    weight=0.8,
                    category="financial",
                    description=f"High claim amount: ${claim_amount:,.0f}",
                    confidence=0.9
                ))
                amount_score += 0.6
            elif claim_amount > 50000:  # $50k+
                factors.append(RiskFactor(
                    name="medium_claim_amount",
                    value=claim_amount,
                    weight=0.5,
                    category="financial",
                    description=f"Medium-high claim amount: ${claim_amount:,.0f}",
                    confidence=0.8
                ))
                amount_score += 0.3
            
            # Policy coverage analysis
            if policy_data:
                coverage_amount = policy_data.get("coverage_amount", 0.0)
                if coverage_amount > 0:
                    coverage_ratio = claim_amount / coverage_amount
                    
                    if coverage_ratio > 0.9:  # Claiming >90% of coverage
                        factors.append(RiskFactor(
                            name="near_policy_limit",
                            value=coverage_ratio,
                            weight=0.7,
                            category="financial",
                            description=f"Claim is {coverage_ratio:.1%} of policy limit",
                            confidence=0.8
                        ))
                        amount_score += 0.4
            
            return {
                "score": min(1.0, amount_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Claim amount risk analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _analyze_policy_limits_risk(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze policy limits risk"""
        
        try:
            factors = []
            limits_score = 0.0
            
            coverage_amount = policy_data.get("coverage_amount", 0.0)
            premium_amount = policy_data.get("premium_amount", 0.0)
            
            # High coverage risk
            if coverage_amount > 1000000:  # $1M+ coverage
                factors.append(RiskFactor(
                    name="high_coverage_policy",
                    value=coverage_amount,
                    weight=0.6,
                    category="financial",
                    description=f"High coverage amount: ${coverage_amount:,.0f}",
                    confidence=0.7
                ))
                limits_score += 0.3
            
            # Premium to coverage ratio
            if premium_amount > 0 and coverage_amount > 0:
                premium_ratio = premium_amount / coverage_amount
                
                if premium_ratio < 0.01:  # Very low premium for coverage
                    factors.append(RiskFactor(
                        name="low_premium_ratio",
                        value=premium_ratio,
                        weight=0.5,
                        category="financial",
                        description=f"Low premium ratio: {premium_ratio:.3%}",
                        confidence=0.6
                    ))
                    limits_score += 0.2
            
            return {
                "score": min(1.0, limits_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Policy limits risk analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _analyze_payment_history_risk(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze payment history risk"""
        
        try:
            factors = []
            payment_score = 0.0
            
            # Simplified payment history analysis
            # In a real implementation, this would analyze actual payment records
            
            if "claims" in entity_data:
                claims = entity_data["claims"]
                
                # Analyze claim payment patterns
                total_claimed = sum(claim.get("amount_claimed", 0.0) for claim in claims)
                total_approved = sum(claim.get("amount_approved", 0.0) for claim in claims)
                
                if total_claimed > 0:
                    approval_ratio = total_approved / total_claimed
                    
                    if approval_ratio < 0.5:  # Low approval ratio
                        factors.append(RiskFactor(
                            name="low_payment_approval_ratio",
                            value=approval_ratio,
                            weight=0.4,
                            category="financial",
                            description=f"Low historical approval ratio: {approval_ratio:.1%}",
                            confidence=0.6
                        ))
                        payment_score += 0.2
            
            return {
                "score": min(1.0, payment_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Payment history risk analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _calculate_financial_confidence(self, entity_data: Dict[str, Any]) -> float:
        """Calculate confidence score for financial assessment"""
        
        try:
            confidence_factors = []
            
            if "claim" in entity_data and entity_data["claim"].get("amount_claimed"):
                confidence_factors.append(0.3)
            
            if "policy" in entity_data and entity_data["policy"].get("coverage_amount"):
                confidence_factors.append(0.3)
            
            if "claims" in entity_data and entity_data["claims"]:
                confidence_factors.append(0.2)
            
            # Base confidence
            confidence_factors.append(0.2)
            
            return min(1.0, sum(confidence_factors))
            
        except Exception:
            return 0.5
    
    async def _assess_operational_risk(self, entity_data: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess operational risk"""
        
        try:
            risk_factors = []
            operational_score = 0.0
            
            # Processing complexity analysis
            complexity_risk = self._analyze_processing_complexity(entity_data)
            risk_factors.extend(complexity_risk["factors"])
            operational_score += complexity_risk["score"] * self.risk_weights[RiskCategory.OPERATIONAL]["processing_complexity"]
            
            # Resource requirements analysis
            resource_risk = self._analyze_resource_requirements(entity_data)
            risk_factors.extend(resource_risk["factors"])
            operational_score += resource_risk["score"] * self.risk_weights[RiskCategory.OPERATIONAL]["resource_requirements"]
            
            # Timeline constraints analysis
            timeline_risk = self._analyze_timeline_constraints(entity_data)
            risk_factors.extend(timeline_risk["factors"])
            operational_score += timeline_risk["score"] * self.risk_weights[RiskCategory.OPERATIONAL]["timeline_constraints"]
            
            # Normalize score to 0-100
            operational_score = min(100.0, max(0.0, operational_score * 100))
            
            # Calculate confidence
            confidence = self._calculate_operational_confidence(entity_data)
            
            return {
                "risk_score": operational_score,
                "risk_factors": risk_factors,
                "confidence": confidence,
                "category": RiskCategory.OPERATIONAL.value,
                "assessment_details": {
                    "processing_complexity_score": complexity_risk.get("score", 0.0),
                    "resource_requirements_score": resource_risk.get("score", 0.0),
                    "timeline_constraints_score": timeline_risk.get("score", 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error("Operational risk assessment failed", error=str(e))
            return {"risk_score": 0.0, "risk_factors": [], "confidence": 0.0}
    
    def _analyze_processing_complexity(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze processing complexity risk"""
        
        try:
            factors = []
            complexity_score = 0.0
            
            # Evidence complexity
            if "evidence" in entity_data:
                evidence_count = len(entity_data["evidence"])
                
                if evidence_count > 10:
                    factors.append(RiskFactor(
                        name="high_evidence_volume",
                        value=evidence_count,
                        weight=0.6,
                        category="operational",
                        description=f"High number of evidence items: {evidence_count}",
                        confidence=0.8
                    ))
                    complexity_score += 0.4
                
                # Check for complex evidence types
                evidence_types = [ev.get("file_type", "") for ev in entity_data["evidence"]]
                complex_types = ["video", "audio", "document"]
                
                complex_evidence_count = sum(1 for et in evidence_types if any(ct in et.lower() for ct in complex_types))
                
                if complex_evidence_count > 0:
                    factors.append(RiskFactor(
                        name="complex_evidence_types",
                        value=complex_evidence_count,
                        weight=0.5,
                        category="operational",
                        description=f"Complex evidence types requiring specialized processing: {complex_evidence_count}",
                        confidence=0.7
                    ))
                    complexity_score += 0.3
            
            # Claim complexity
            if "claim" in entity_data:
                claim_type = entity_data["claim"].get("claim_type", "")
                
                complex_claim_types = ["liability", "property_damage", "personal_injury"]
                if claim_type.lower() in complex_claim_types:
                    factors.append(RiskFactor(
                        name="complex_claim_type",
                        value=1.0,
                        weight=0.4,
                        category="operational",
                        description=f"Complex claim type: {claim_type}",
                        confidence=0.6
                    ))
                    complexity_score += 0.2
            
            return {
                "score": min(1.0, complexity_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Processing complexity analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _analyze_resource_requirements(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource requirements risk"""
        
        try:
            factors = []
            resource_score = 0.0
            
            # High-value claims require more resources
            if "claim" in entity_data:
                claim_amount = entity_data["claim"].get("amount_claimed", 0.0)
                
                if claim_amount > 500000:  # $500k+
                    factors.append(RiskFactor(
                        name="high_value_claim_resources",
                        value=claim_amount,
                        weight=0.7,
                        category="operational",
                        description=f"High-value claim requiring specialized resources: ${claim_amount:,.0f}",
                        confidence=0.8
                    ))
                    resource_score += 0.5
            
            # Multiple stakeholders increase resource needs
            stakeholder_count = 0
            if "user" in entity_data:
                stakeholder_count += 1
            if "organization" in entity_data:
                stakeholder_count += 1
            
            # Estimate additional stakeholders based on claim type
            if "claim" in entity_data:
                claim_type = entity_data["claim"].get("claim_type", "")
                if claim_type.lower() in ["liability", "auto", "property"]:
                    stakeholder_count += 2  # Likely third parties involved
            
            if stakeholder_count > 3:
                factors.append(RiskFactor(
                    name="multiple_stakeholders",
                    value=stakeholder_count,
                    weight=0.5,
                    category="operational",
                    description=f"Multiple stakeholders requiring coordination: {stakeholder_count}",
                    confidence=0.6
                ))
                resource_score += 0.3
            
            return {
                "score": min(1.0, resource_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Resource requirements analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _analyze_timeline_constraints(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze timeline constraints risk"""
        
        try:
            factors = []
            timeline_score = 0.0
            
            # Analyze claim age
            if "claim" in entity_data:
                if entity_data["claim"].get("reported_date"):
                    reported_date = datetime.fromisoformat(entity_data["claim"]["reported_date"])
                    claim_age = (datetime.utcnow() - reported_date).days
                    
                    if claim_age > 90:  # Claim older than 90 days
                        factors.append(RiskFactor(
                            name="aged_claim",
                            value=claim_age,
                            weight=0.6,
                            category="operational",
                            description=f"Claim is {claim_age} days old, approaching time limits",
                            confidence=0.8
                        ))
                        timeline_score += 0.4
                    
                    # Check for urgent processing needs
                    if claim_age > 180:  # Very old claim
                        factors.append(RiskFactor(
                            name="urgent_processing_needed",
                            value=claim_age,
                            weight=0.8,
                            category="operational",
                            description=f"Urgent processing needed for {claim_age}-day-old claim",
                            confidence=0.9
                        ))
                        timeline_score += 0.6
            
            # Policy expiry constraints
            if "policy" in entity_data:
                if entity_data["policy"].get("expiry_date"):
                    expiry_date = datetime.fromisoformat(entity_data["policy"]["expiry_date"])
                    days_to_expiry = (expiry_date - datetime.utcnow()).days
                    
                    if days_to_expiry < 30:  # Policy expires soon
                        factors.append(RiskFactor(
                            name="policy_expiry_constraint",
                            value=days_to_expiry,
                            weight=0.5,
                            category="operational",
                            description=f"Policy expires in {days_to_expiry} days",
                            confidence=0.7
                        ))
                        timeline_score += 0.3
            
            return {
                "score": min(1.0, timeline_score),
                "factors": [factor.__dict__ for factor in factors]
            }
            
        except Exception as e:
            self.logger.warning("Timeline constraints analysis failed", error=str(e))
            return {"score": 0.0, "factors": []}
    
    def _calculate_operational_confidence(self, entity_data: Dict[str, Any]) -> float:
        """Calculate confidence score for operational assessment"""
        
        try:
            confidence_factors = []
            
            if "claim" in entity_data:
                confidence_factors.append(0.4)
            
            if "evidence" in entity_data:
                confidence_factors.append(0.3)
            
            if "policy" in entity_data:
                confidence_factors.append(0.3)
            
            return min(1.0, sum(confidence_factors))
            
        except Exception:
            return 0.5
    
    def _calculate_overall_risk_score(self, risk_assessments: Dict[str, Any]) -> float:
        """Calculate overall risk score from category assessments"""
        
        try:
            total_score = 0.0
            total_weight = 0.0
            
            # Weight categories based on their importance
            category_weights = {
                RiskCategory.FRAUD.value: 0.5,
                RiskCategory.FINANCIAL.value: 0.3,
                RiskCategory.OPERATIONAL.value: 0.2
            }
            
            for category, assessment in risk_assessments.items():
                risk_score = assessment.get("risk_score", 0.0)
                confidence = assessment.get("confidence", 0.0)
                weight = category_weights.get(category, 0.1) * confidence
                
                total_score += risk_score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error("Overall risk score calculation failed", error=str(e))
            return 0.0
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        
        if risk_score >= self.risk_config["high_risk_threshold"]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_config["medium_risk_threshold"]:
            return RiskLevel.MEDIUM
        elif risk_score >= self.risk_config["low_risk_threshold"]:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _calculate_confidence_score(self, risk_assessments: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        
        try:
            confidences = [
                assessment.get("confidence", 0.0) 
                for assessment in risk_assessments.values()
            ]
            
            return np.mean(confidences) if confidences else 0.0
            
        except Exception:
            return 0.5
    
    async def _generate_recommendations(
        self, 
        risk_assessments: Dict[str, Any], 
        overall_score: float, 
        entity_type: str, 
        entity_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate risk mitigation recommendations"""
        
        try:
            recommendations = []
            
            # High-level recommendations based on overall score
            if overall_score >= self.risk_config["high_risk_threshold"]:
                recommendations.append({
                    "priority": "high",
                    "category": "general",
                    "title": "Manual Review Required",
                    "description": "High risk score requires immediate manual review and approval",
                    "action": "escalate_to_senior_adjuster",
                    "estimated_impact": "high"
                })
            
            # Category-specific recommendations
            for category, assessment in risk_assessments.items():
                category_score = assessment.get("risk_score", 0.0)
                risk_factors = assessment.get("risk_factors", [])
                
                if category == RiskCategory.FRAUD.value and category_score > 70:
                    recommendations.append({
                        "priority": "high",
                        "category": "fraud",
                        "title": "Fraud Investigation Required",
                        "description": "High fraud indicators detected, initiate special investigation unit review",
                        "action": "initiate_siu_investigation",
                        "estimated_impact": "high"
                    })
                
                if category == RiskCategory.FINANCIAL.value and category_score > 60:
                    recommendations.append({
                        "priority": "medium",
                        "category": "financial",
                        "title": "Financial Verification",
                        "description": "Verify claim amounts and financial documentation",
                        "action": "request_additional_documentation",
                        "estimated_impact": "medium"
                    })
                
                if category == RiskCategory.OPERATIONAL.value and category_score > 50:
                    recommendations.append({
                        "priority": "medium",
                        "category": "operational",
                        "title": "Resource Allocation",
                        "description": "Allocate additional resources for complex processing requirements",
                        "action": "assign_senior_adjuster",
                        "estimated_impact": "medium"
                    })
                
                # Factor-specific recommendations
                for factor in risk_factors:
                    if factor.get("weight", 0) > 0.7:  # High-weight factors
                        recommendations.append({
                            "priority": "medium",
                            "category": factor.get("category", "general"),
                            "title": f"Address {factor.get('name', 'Risk Factor')}",
                            "description": factor.get("description", "High-impact risk factor identified"),
                            "action": "investigate_specific_factor",
                            "estimated_impact": "medium"
                        })
            
            # Evidence-specific recommendations
            if "evidence_analyses" in entity_data:
                for analysis in entity_data["evidence_analyses"]:
                    fraud_score = analysis.get("fraud_score", 0.0)
                    if fraud_score > 80:
                        recommendations.append({
                            "priority": "high",
                            "category": "evidence",
                            "title": "Evidence Authentication Required",
                            "description": f"Evidence shows high fraud score ({fraud_score:.1f}), requires expert authentication",
                            "action": "forensic_evidence_analysis",
                            "estimated_impact": "high"
                        })
            
            # Remove duplicates and sort by priority
            unique_recommendations = []
            seen_titles = set()
            
            for rec in recommendations:
                if rec["title"] not in seen_titles:
                    unique_recommendations.append(rec)
                    seen_titles.add(rec["title"])
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            unique_recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
            
            return unique_recommendations[:10]  # Limit to top 10 recommendations
            
        except Exception as e:
            self.logger.error("Recommendation generation failed", error=str(e))
            return []
    
    async def _save_assessment_results(
        self,
        entity_type: str,
        entity_id: uuid.UUID,
        assessment_result: Dict[str, Any]
    ):
        """Persist or log assessment results."""

        try:
            self.logger.info(
                "Assessment results ready",
                entity_type=entity_type,
                entity_id=str(entity_id),
                results=assessment_result,
            )
            # TODO: persist results via a proper service
        except Exception as e:
            self.logger.error("Failed to save assessment results", error=str(e))
            # Don't raise exception as this is not critical for the assessment itself

# Agent factory function
async def create_risk_assessment_agent(db_session: AsyncSession, redis_client: redis.Redis) -> RiskAssessmentAgent:
    """Create risk assessment agent instance"""
    return RiskAssessmentAgent(db_session, redis_client)

