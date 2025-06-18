"""
Liability Assessor - Production Ready Implementation
Comprehensive liability assessment and risk calculation for insurance operations
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from decimal import Decimal
import math
import statistics

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

liability_assessments_total = Counter('liability_assessments_total', 'Total liability assessments', ['assessment_type'])
assessment_duration = Histogram('assessment_duration_seconds', 'Assessment duration')
liability_amount_gauge = Gauge('liability_amount_current', 'Current liability amount being assessed')

Base = declarative_base()

class LiabilityType(Enum):
    BODILY_INJURY = "bodily_injury"
    PROPERTY_DAMAGE = "property_damage"
    MEDICAL_PAYMENTS = "medical_payments"
    COLLISION = "collision"
    COMPREHENSIVE = "comprehensive"
    UNINSURED_MOTORIST = "uninsured_motorist"
    PERSONAL_INJURY_PROTECTION = "personal_injury_protection"
    GENERAL_LIABILITY = "general_liability"
    PROFESSIONAL_LIABILITY = "professional_liability"
    PRODUCT_LIABILITY = "product_liability"

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class AssessmentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REQUIRES_REVIEW = "requires_review"
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass
class LiabilityFactor:
    factor_name: str
    factor_value: Any
    weight: float
    impact_score: float
    confidence: float
    source: str

@dataclass
class RiskAssessment:
    assessment_id: str
    liability_type: LiabilityType
    risk_level: RiskLevel
    risk_score: float
    confidence_level: float
    contributing_factors: List[LiabilityFactor]
    recommended_premium: Decimal
    coverage_limits: Dict[str, Decimal]
    exclusions: List[str]
    conditions: List[str]
    assessment_date: datetime
    valid_until: datetime

@dataclass
class LiabilityCalculation:
    calculation_id: str
    policy_data: Dict[str, Any]
    claim_data: Optional[Dict[str, Any]]
    assessment_results: List[RiskAssessment]
    total_liability_amount: Decimal
    reserve_amount: Decimal
    settlement_recommendation: Optional[Decimal]
    legal_exposure: Decimal
    confidence_score: float
    calculation_method: str
    supporting_documentation: List[str]
    created_at: datetime
    updated_at: datetime

class LiabilityAssessmentRecord(Base):
    __tablename__ = 'liability_assessments'
    
    assessment_id = Column(String, primary_key=True)
    policy_number = Column(String, nullable=False, index=True)
    claim_number = Column(String, index=True)
    liability_type = Column(String, nullable=False)
    risk_level = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False)
    confidence_level = Column(Float, nullable=False)
    contributing_factors = Column(JSON)
    recommended_premium = Column(Numeric(12, 2))
    coverage_limits = Column(JSON)
    exclusions = Column(JSON)
    conditions = Column(JSON)
    total_liability_amount = Column(Numeric(15, 2))
    reserve_amount = Column(Numeric(15, 2))
    settlement_recommendation = Column(Numeric(15, 2))
    legal_exposure = Column(Numeric(15, 2))
    confidence_score = Column(Float)
    calculation_method = Column(String)
    supporting_documentation = Column(JSON)
    status = Column(String, nullable=False, default='pending')
    assessment_date = Column(DateTime, nullable=False)
    valid_until = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class RiskFactorRecord(Base):
    __tablename__ = 'risk_factors'
    
    factor_id = Column(String, primary_key=True)
    factor_name = Column(String, nullable=False, index=True)
    factor_category = Column(String, nullable=False)
    base_weight = Column(Float, nullable=False)
    adjustment_rules = Column(JSON)
    historical_impact = Column(JSON)
    industry_benchmarks = Column(JSON)
    last_calibrated = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class LiabilityAssessor:
    """Production-ready Liability Assessor for comprehensive risk assessment"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # Risk assessment models
        self.ml_models = {}
        self.scalers = {}
        
        # Risk factors and weights
        self.risk_factors = {}
        self.industry_benchmarks = {}
        
        # Actuarial tables
        self.actuarial_tables = {}
        
        self._initialize_models()
        self._load_risk_factors()
        self._load_actuarial_tables()
        
        logger.info("LiabilityAssessor initialized successfully")

    def _initialize_models(self):
        """Initialize machine learning models for risk assessment"""
        
        # Bodily Injury Liability Model
        self.ml_models['bodily_injury'] = {
            'severity_model': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'frequency_model': RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42
            )
        }
        
        # Property Damage Liability Model
        self.ml_models['property_damage'] = {
            'severity_model': RandomForestRegressor(
                n_estimators=180,
                max_depth=12,
                random_state=42
            ),
            'frequency_model': GradientBoostingRegressor(
                n_estimators=160,
                max_depth=6,
                learning_rate=0.12,
                random_state=42
            )
        }
        
        # Comprehensive Risk Model
        self.ml_models['comprehensive'] = {
            'risk_classifier': LogisticRegression(
                solver='liblinear',
                random_state=42
            ),
            'cost_predictor': GradientBoostingRegressor(
                n_estimators=220,
                max_depth=10,
                learning_rate=0.08,
                random_state=42
            )
        }
        
        # Initialize scalers
        for model_type in self.ml_models:
            self.scalers[model_type] = StandardScaler()

    def _load_risk_factors(self):
        """Load risk factors and their weights from database"""
        
        try:
            with self.Session() as session:
                factors = session.query(RiskFactorRecord).all()
                
                for factor in factors:
                    self.risk_factors[factor.factor_name] = {
                        'category': factor.factor_category,
                        'base_weight': factor.base_weight,
                        'adjustment_rules': factor.adjustment_rules or {},
                        'historical_impact': factor.historical_impact or {},
                        'industry_benchmarks': factor.industry_benchmarks or {}
                    }
                
                # If no factors in database, initialize default factors
                if not self.risk_factors:
                    self._initialize_default_risk_factors()
                
                logger.info(f"Loaded {len(self.risk_factors)} risk factors")
                
        except Exception as e:
            logger.error(f"Error loading risk factors: {e}")
            self._initialize_default_risk_factors()

    def _initialize_default_risk_factors(self):
        """Initialize default risk factors"""
        
        default_factors = {
            # Driver factors
            'driver_age': {'category': 'driver', 'base_weight': 0.15, 'adjustment_rules': {'young_driver_penalty': 1.5, 'senior_discount': 0.9}},
            'driving_experience': {'category': 'driver', 'base_weight': 0.12, 'adjustment_rules': {'new_driver_penalty': 1.3}},
            'driving_record': {'category': 'driver', 'base_weight': 0.20, 'adjustment_rules': {'clean_record_discount': 0.8, 'violation_penalty': 1.4}},
            'credit_score': {'category': 'financial', 'base_weight': 0.10, 'adjustment_rules': {'excellent_discount': 0.85, 'poor_penalty': 1.25}},
            
            # Vehicle factors
            'vehicle_age': {'category': 'vehicle', 'base_weight': 0.08, 'adjustment_rules': {'new_car_premium': 1.1, 'old_car_discount': 0.9}},
            'vehicle_value': {'category': 'vehicle', 'base_weight': 0.12, 'adjustment_rules': {'luxury_penalty': 1.3, 'economy_discount': 0.9}},
            'safety_rating': {'category': 'vehicle', 'base_weight': 0.10, 'adjustment_rules': {'top_safety_discount': 0.85, 'poor_safety_penalty': 1.2}},
            'theft_rating': {'category': 'vehicle', 'base_weight': 0.06, 'adjustment_rules': {'high_theft_penalty': 1.15}},
            
            # Location factors
            'location_risk': {'category': 'location', 'base_weight': 0.14, 'adjustment_rules': {'urban_penalty': 1.2, 'rural_discount': 0.9}},
            'weather_risk': {'category': 'location', 'base_weight': 0.05, 'adjustment_rules': {'severe_weather_penalty': 1.1}},
            
            # Usage factors
            'annual_mileage': {'category': 'usage', 'base_weight': 0.08, 'adjustment_rules': {'high_mileage_penalty': 1.15, 'low_mileage_discount': 0.9}},
            'usage_type': {'category': 'usage', 'base_weight': 0.06, 'adjustment_rules': {'business_use_penalty': 1.2}}
        }
        
        self.risk_factors = default_factors
        
        # Store in database
        try:
            with self.Session() as session:
                for factor_name, factor_data in default_factors.items():
                    record = RiskFactorRecord(
                        factor_id=str(uuid.uuid4()),
                        factor_name=factor_name,
                        factor_category=factor_data['category'],
                        base_weight=factor_data['base_weight'],
                        adjustment_rules=factor_data['adjustment_rules'],
                        historical_impact={},
                        industry_benchmarks={},
                        last_calibrated=datetime.utcnow(),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    session.add(record)
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing default risk factors: {e}")

    def _load_actuarial_tables(self):
        """Load actuarial tables for liability calculations"""
        
        # Bodily Injury Severity Distribution
        self.actuarial_tables['bodily_injury_severity'] = {
            'minor': {'probability': 0.70, 'avg_cost': 5000, 'std_dev': 2000, 'max_cost': 15000},
            'moderate': {'probability': 0.20, 'avg_cost': 25000, 'std_dev': 10000, 'max_cost': 75000},
            'severe': {'probability': 0.08, 'avg_cost': 150000, 'std_dev': 75000, 'max_cost': 500000},
            'catastrophic': {'probability': 0.02, 'avg_cost': 750000, 'std_dev': 500000, 'max_cost': 5000000}
        }
        
        # Property Damage Severity Distribution
        self.actuarial_tables['property_damage_severity'] = {
            'minor': {'probability': 0.60, 'avg_cost': 3000, 'std_dev': 1500, 'max_cost': 8000},
            'moderate': {'probability': 0.25, 'avg_cost': 15000, 'std_dev': 8000, 'max_cost': 40000},
            'major': {'probability': 0.12, 'avg_cost': 75000, 'std_dev': 40000, 'max_cost': 200000},
            'catastrophic': {'probability': 0.03, 'avg_cost': 400000, 'std_dev': 300000, 'max_cost': 2000000}
        }
        
        # Frequency multipliers by risk factors
        self.actuarial_tables['frequency_multipliers'] = {
            'age_16_25': 2.1,
            'age_26_35': 1.3,
            'age_36_50': 1.0,
            'age_51_65': 0.8,
            'age_65_plus': 0.9,
            'urban_area': 1.4,
            'suburban_area': 1.0,
            'rural_area': 0.7,
            'high_mileage': 1.3,
            'average_mileage': 1.0,
            'low_mileage': 0.8
        }

    async def assess_liability(self, policy_data: Dict[str, Any], 
                            claim_data: Optional[Dict[str, Any]] = None) -> LiabilityCalculation:
        """Perform comprehensive liability assessment"""
        
        calculation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        liability_assessments_total.labels(assessment_type='comprehensive').inc()
        
        try:
            with assessment_duration.time():
                # Extract key information
                policy_number = policy_data.get('policy_number')
                claim_number = claim_data.get('claim_number') if claim_data else None
                
                # Perform risk assessments for each liability type
                assessment_results = []
                
                # Bodily Injury Assessment
                if policy_data.get('bodily_injury_coverage'):
                    bi_assessment = await self._assess_bodily_injury_liability(policy_data, claim_data)
                    assessment_results.append(bi_assessment)
                
                # Property Damage Assessment
                if policy_data.get('property_damage_coverage'):
                    pd_assessment = await self._assess_property_damage_liability(policy_data, claim_data)
                    assessment_results.append(pd_assessment)
                
                # Comprehensive Assessment
                if policy_data.get('comprehensive_coverage'):
                    comp_assessment = await self._assess_comprehensive_liability(policy_data, claim_data)
                    assessment_results.append(comp_assessment)
                
                # Calculate total liability amounts
                total_liability = sum(Decimal(str(assessment.recommended_premium)) for assessment in assessment_results)
                
                # Calculate reserves
                reserve_amount = await self._calculate_reserve_amount(assessment_results, claim_data)
                
                # Settlement recommendation
                settlement_recommendation = None
                if claim_data:
                    settlement_recommendation = await self._calculate_settlement_recommendation(
                        assessment_results, claim_data
                    )
                
                # Legal exposure assessment
                legal_exposure = await self._assess_legal_exposure(policy_data, claim_data, assessment_results)
                
                # Overall confidence score
                confidence_score = self._calculate_overall_confidence(assessment_results)
                
                # Create liability calculation result
                calculation = LiabilityCalculation(
                    calculation_id=calculation_id,
                    policy_data=policy_data,
                    claim_data=claim_data,
                    assessment_results=assessment_results,
                    total_liability_amount=total_liability,
                    reserve_amount=reserve_amount,
                    settlement_recommendation=settlement_recommendation,
                    legal_exposure=legal_exposure,
                    confidence_score=confidence_score,
                    calculation_method="ml_actuarial_hybrid",
                    supporting_documentation=[],
                    created_at=start_time,
                    updated_at=datetime.utcnow()
                )
                
                # Store assessment results
                await self._store_liability_assessment(calculation)
                
                liability_amount_gauge.set(float(total_liability))
                
                return calculation
                
        except Exception as e:
            logger.error(f"Liability assessment failed: {e}")
            raise

    async def _assess_bodily_injury_liability(self, policy_data: Dict[str, Any], 
                                           claim_data: Optional[Dict[str, Any]]) -> RiskAssessment:
        """Assess bodily injury liability"""
        
        assessment_id = str(uuid.uuid4())
        
        # Extract relevant features
        features = self._extract_bodily_injury_features(policy_data, claim_data)
        
        # Calculate risk factors
        contributing_factors = []
        
        # Driver age factor
        driver_age = policy_data.get('driver_age', 35)
        age_factor = self._calculate_age_risk_factor(driver_age)
        contributing_factors.append(LiabilityFactor(
            factor_name="driver_age",
            factor_value=driver_age,
            weight=self.risk_factors['driver_age']['base_weight'],
            impact_score=age_factor,
            confidence=0.9,
            source="actuarial_table"
        ))
        
        # Driving record factor
        violations = policy_data.get('violations', [])
        record_factor = self._calculate_driving_record_factor(violations)
        contributing_factors.append(LiabilityFactor(
            factor_name="driving_record",
            factor_value=len(violations),
            weight=self.risk_factors['driving_record']['base_weight'],
            impact_score=record_factor,
            confidence=0.95,
            source="dmv_records"
        ))
        
        # Location risk factor
        location = policy_data.get('location', {})
        location_factor = self._calculate_location_risk_factor(location)
        contributing_factors.append(LiabilityFactor(
            factor_name="location_risk",
            factor_value=location.get('risk_score', 1.0),
            weight=self.risk_factors['location_risk']['base_weight'],
            impact_score=location_factor,
            confidence=0.85,
            source="geographic_data"
        ))
        
        # Vehicle safety factor
        vehicle_data = policy_data.get('vehicle', {})
        safety_factor = self._calculate_vehicle_safety_factor(vehicle_data)
        contributing_factors.append(LiabilityFactor(
            factor_name="safety_rating",
            factor_value=vehicle_data.get('safety_rating', 3),
            weight=self.risk_factors['safety_rating']['base_weight'],
            impact_score=safety_factor,
            confidence=0.8,
            source="nhtsa_data"
        ))
        
        # Calculate overall risk score
        risk_score = self._calculate_composite_risk_score(contributing_factors)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Calculate recommended premium
        base_premium = Decimal('1200.00')  # Base bodily injury premium
        risk_multiplier = Decimal(str(risk_score))
        recommended_premium = base_premium * risk_multiplier
        
        # Set coverage limits
        coverage_limits = {
            'per_person': Decimal(str(policy_data.get('bi_per_person_limit', 100000))),
            'per_accident': Decimal(str(policy_data.get('bi_per_accident_limit', 300000)))
        }
        
        # Define exclusions and conditions
        exclusions = [
            "Intentional acts",
            "Racing or speed contests",
            "Using vehicle for business purposes without commercial coverage",
            "Driving under the influence"
        ]
        
        conditions = [
            "Driver must have valid license",
            "Vehicle must be properly maintained",
            "Accidents must be reported within 24 hours",
            "Cooperation with investigation required"
        ]
        
        return RiskAssessment(
            assessment_id=assessment_id,
            liability_type=LiabilityType.BODILY_INJURY,
            risk_level=risk_level,
            risk_score=risk_score,
            confidence_level=self._calculate_assessment_confidence(contributing_factors),
            contributing_factors=contributing_factors,
            recommended_premium=recommended_premium,
            coverage_limits=coverage_limits,
            exclusions=exclusions,
            conditions=conditions,
            assessment_date=datetime.utcnow(),
            valid_until=datetime.utcnow() + timedelta(days=365)
        )

    async def _assess_property_damage_liability(self, policy_data: Dict[str, Any], 
                                             claim_data: Optional[Dict[str, Any]]) -> RiskAssessment:
        """Assess property damage liability"""
        
        assessment_id = str(uuid.uuid4())
        
        # Extract features for property damage assessment
        features = self._extract_property_damage_features(policy_data, claim_data)
        
        # Calculate risk factors specific to property damage
        contributing_factors = []
        
        # Vehicle value factor
        vehicle_value = policy_data.get('vehicle', {}).get('value', 25000)
        value_factor = self._calculate_vehicle_value_factor(vehicle_value)
        contributing_factors.append(LiabilityFactor(
            factor_name="vehicle_value",
            factor_value=vehicle_value,
            weight=self.risk_factors['vehicle_value']['base_weight'],
            impact_score=value_factor,
            confidence=0.9,
            source="kbb_valuation"
        ))
        
        # Annual mileage factor
        annual_mileage = policy_data.get('annual_mileage', 12000)
        mileage_factor = self._calculate_mileage_factor(annual_mileage)
        contributing_factors.append(LiabilityFactor(
            factor_name="annual_mileage",
            factor_value=annual_mileage,
            weight=self.risk_factors['annual_mileage']['base_weight'],
            impact_score=mileage_factor,
            confidence=0.85,
            source="odometer_reading"
        ))
        
        # Usage type factor
        usage_type = policy_data.get('usage_type', 'personal')
        usage_factor = self._calculate_usage_factor(usage_type)
        contributing_factors.append(LiabilityFactor(
            factor_name="usage_type",
            factor_value=usage_type,
            weight=self.risk_factors['usage_type']['base_weight'],
            impact_score=usage_factor,
            confidence=0.95,
            source="policy_declaration"
        ))
        
        # Calculate risk score
        risk_score = self._calculate_composite_risk_score(contributing_factors)
        risk_level = self._determine_risk_level(risk_score)
        
        # Calculate recommended premium
        base_premium = Decimal('800.00')  # Base property damage premium
        recommended_premium = base_premium * Decimal(str(risk_score))
        
        # Coverage limits
        coverage_limits = {
            'per_accident': Decimal(str(policy_data.get('pd_per_accident_limit', 100000)))
        }
        
        exclusions = [
            "Damage to insured's own property",
            "Intentional damage",
            "Wear and tear",
            "Racing activities"
        ]
        
        conditions = [
            "Prompt notification of claims",
            "Cooperation with investigation",
            "Right to inspect damaged property",
            "Subrogation rights reserved"
        ]
        
        return RiskAssessment(
            assessment_id=assessment_id,
            liability_type=LiabilityType.PROPERTY_DAMAGE,
            risk_level=risk_level,
            risk_score=risk_score,
            confidence_level=self._calculate_assessment_confidence(contributing_factors),
            contributing_factors=contributing_factors,
            recommended_premium=recommended_premium,
            coverage_limits=coverage_limits,
            exclusions=exclusions,
            conditions=conditions,
            assessment_date=datetime.utcnow(),
            valid_until=datetime.utcnow() + timedelta(days=365)
        )

    async def _assess_comprehensive_liability(self, policy_data: Dict[str, Any], 
                                           claim_data: Optional[Dict[str, Any]]) -> RiskAssessment:
        """Assess comprehensive liability coverage"""
        
        assessment_id = str(uuid.uuid4())
        
        # Extract comprehensive coverage features
        features = self._extract_comprehensive_features(policy_data, claim_data)
        
        contributing_factors = []
        
        # Theft rating factor
        vehicle_data = policy_data.get('vehicle', {})
        theft_rating = vehicle_data.get('theft_rating', 3)
        theft_factor = self._calculate_theft_risk_factor(theft_rating)
        contributing_factors.append(LiabilityFactor(
            factor_name="theft_rating",
            factor_value=theft_rating,
            weight=self.risk_factors['theft_rating']['base_weight'],
            impact_score=theft_factor,
            confidence=0.8,
            source="nicb_data"
        ))
        
        # Weather risk factor
        location = policy_data.get('location', {})
        weather_risk = location.get('weather_risk_score', 1.0)
        weather_factor = self._calculate_weather_risk_factor(weather_risk)
        contributing_factors.append(LiabilityFactor(
            factor_name="weather_risk",
            factor_value=weather_risk,
            weight=self.risk_factors['weather_risk']['base_weight'],
            impact_score=weather_factor,
            confidence=0.75,
            source="noaa_data"
        ))
        
        # Vehicle age factor
        vehicle_age = vehicle_data.get('age', 5)
        age_factor = self._calculate_vehicle_age_factor(vehicle_age)
        contributing_factors.append(LiabilityFactor(
            factor_name="vehicle_age",
            factor_value=vehicle_age,
            weight=self.risk_factors['vehicle_age']['base_weight'],
            impact_score=age_factor,
            confidence=0.9,
            source="vehicle_registration"
        ))
        
        # Calculate risk metrics
        risk_score = self._calculate_composite_risk_score(contributing_factors)
        risk_level = self._determine_risk_level(risk_score)
        
        # Premium calculation
        base_premium = Decimal('600.00')  # Base comprehensive premium
        recommended_premium = base_premium * Decimal(str(risk_score))
        
        # Coverage details
        coverage_limits = {
            'actual_cash_value': Decimal(str(vehicle_data.get('value', 25000))),
            'deductible': Decimal(str(policy_data.get('comprehensive_deductible', 500)))
        }
        
        exclusions = [
            "Mechanical breakdown",
            "Wear and tear",
            "Freezing",
            "Road damage to tires",
            "Racing activities"
        ]
        
        conditions = [
            "Deductible applies to each claim",
            "Coverage based on actual cash value",
            "Right to repair or replace",
            "Salvage rights reserved"
        ]
        
        return RiskAssessment(
            assessment_id=assessment_id,
            liability_type=LiabilityType.COMPREHENSIVE,
            risk_level=risk_level,
            risk_score=risk_score,
            confidence_level=self._calculate_assessment_confidence(contributing_factors),
            contributing_factors=contributing_factors,
            recommended_premium=recommended_premium,
            coverage_limits=coverage_limits,
            exclusions=exclusions,
            conditions=conditions,
            assessment_date=datetime.utcnow(),
            valid_until=datetime.utcnow() + timedelta(days=365)
        )

    # Risk calculation helper methods
    
    def _calculate_age_risk_factor(self, age: int) -> float:
        """Calculate risk factor based on driver age"""
        
        if age < 25:
            return 1.8 - (age - 16) * 0.08  # Higher risk for young drivers
        elif age < 65:
            return 0.8 + (65 - age) * 0.005  # Lower risk for middle-aged drivers
        else:
            return 1.0 + (age - 65) * 0.02  # Slightly higher risk for seniors

    def _calculate_driving_record_factor(self, violations: List[Dict[str, Any]]) -> float:
        """Calculate risk factor based on driving record"""
        
        if not violations:
            return 0.8  # Clean record discount
        
        base_factor = 1.0
        
        for violation in violations:
            violation_type = violation.get('type', '').lower()
            violation_date = violation.get('date')
            
            # Calculate recency factor
            if violation_date:
                try:
                    viol_date = datetime.fromisoformat(violation_date.replace('Z', '+00:00'))
                    years_ago = (datetime.utcnow() - viol_date).days / 365.25
                    recency_factor = max(0.1, 1.0 - years_ago * 0.2)  # Decay over time
                except:
                    recency_factor = 1.0
            else:
                recency_factor = 1.0
            
            # Violation severity multipliers
            if 'dui' in violation_type or 'dwi' in violation_type:
                base_factor *= (1.5 * recency_factor)
            elif 'reckless' in violation_type:
                base_factor *= (1.3 * recency_factor)
            elif 'speeding' in violation_type:
                base_factor *= (1.15 * recency_factor)
            else:
                base_factor *= (1.1 * recency_factor)
        
        return min(base_factor, 2.5)  # Cap at 2.5x

    def _calculate_location_risk_factor(self, location: Dict[str, Any]) -> float:
        """Calculate risk factor based on location"""
        
        base_factor = 1.0
        
        # Urban/rural factor
        area_type = location.get('area_type', 'suburban').lower()
        if area_type == 'urban':
            base_factor *= 1.2
        elif area_type == 'rural':
            base_factor *= 0.9
        
        # Crime rate factor
        crime_rate = location.get('crime_rate', 1.0)
        base_factor *= (0.8 + crime_rate * 0.4)
        
        # Traffic density factor
        traffic_density = location.get('traffic_density', 1.0)
        base_factor *= (0.9 + traffic_density * 0.2)
        
        return base_factor

    def _calculate_vehicle_safety_factor(self, vehicle_data: Dict[str, Any]) -> float:
        """Calculate risk factor based on vehicle safety features"""
        
        safety_rating = vehicle_data.get('safety_rating', 3)  # 1-5 scale
        base_factor = 1.3 - (safety_rating * 0.1)  # Better rating = lower factor
        
        # Safety features adjustments
        safety_features = vehicle_data.get('safety_features', [])
        
        feature_discounts = {
            'abs': 0.05,
            'airbags': 0.08,
            'electronic_stability_control': 0.06,
            'backup_camera': 0.03,
            'blind_spot_monitoring': 0.04,
            'automatic_emergency_braking': 0.07,
            'lane_departure_warning': 0.03
        }
        
        total_discount = sum(feature_discounts.get(feature.lower(), 0) for feature in safety_features)
        base_factor *= (1.0 - min(total_discount, 0.25))  # Cap discount at 25%
        
        return max(base_factor, 0.6)  # Minimum factor of 0.6

    def _calculate_vehicle_value_factor(self, value: float) -> float:
        """Calculate risk factor based on vehicle value"""
        
        if value < 15000:
            return 0.9  # Lower value vehicles
        elif value < 30000:
            return 1.0  # Average value vehicles
        elif value < 50000:
            return 1.1  # Higher value vehicles
        elif value < 100000:
            return 1.25  # Luxury vehicles
        else:
            return 1.4  # Super luxury vehicles

    def _calculate_mileage_factor(self, annual_mileage: int) -> float:
        """Calculate risk factor based on annual mileage"""
        
        if annual_mileage < 7500:
            return 0.9  # Low mileage discount
        elif annual_mileage < 15000:
            return 1.0  # Average mileage
        elif annual_mileage < 20000:
            return 1.1  # High mileage penalty
        else:
            return 1.2  # Very high mileage penalty

    def _calculate_usage_factor(self, usage_type: str) -> float:
        """Calculate risk factor based on vehicle usage"""
        
        usage_factors = {
            'personal': 1.0,
            'commuting': 1.05,
            'business': 1.15,
            'commercial': 1.3,
            'rideshare': 1.25,
            'delivery': 1.35
        }
        
        return usage_factors.get(usage_type.lower(), 1.0)

    def _calculate_theft_risk_factor(self, theft_rating: int) -> float:
        """Calculate risk factor based on vehicle theft rating"""
        
        # Theft rating: 1 (most stolen) to 5 (least stolen)
        return 1.5 - (theft_rating * 0.1)

    def _calculate_weather_risk_factor(self, weather_risk_score: float) -> float:
        """Calculate risk factor based on weather conditions"""
        
        return 0.8 + (weather_risk_score * 0.4)

    def _calculate_vehicle_age_factor(self, vehicle_age: int) -> float:
        """Calculate risk factor based on vehicle age"""
        
        if vehicle_age < 3:
            return 1.05  # New cars slightly higher risk
        elif vehicle_age < 8:
            return 1.0  # Prime age vehicles
        elif vehicle_age < 15:
            return 0.95  # Older but reliable
        else:
            return 1.1  # Very old vehicles higher risk

    def _calculate_composite_risk_score(self, factors: List[LiabilityFactor]) -> float:
        """Calculate composite risk score from individual factors"""
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor in factors:
            weighted_sum += factor.impact_score * factor.weight * factor.confidence
            total_weight += factor.weight * factor.confidence
        
        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 1.0
        
        # Ensure score is within reasonable bounds
        return max(0.5, min(composite_score, 3.0))

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on risk score"""
        
        if risk_score < 0.7:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.9:
            return RiskLevel.LOW
        elif risk_score < 1.2:
            return RiskLevel.MODERATE
        elif risk_score < 1.5:
            return RiskLevel.HIGH
        elif risk_score < 2.0:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME

    def _calculate_assessment_confidence(self, factors: List[LiabilityFactor]) -> float:
        """Calculate overall confidence level for assessment"""
        
        if not factors:
            return 0.5
        
        confidence_scores = [factor.confidence for factor in factors]
        return statistics.mean(confidence_scores)

    def _calculate_overall_confidence(self, assessments: List[RiskAssessment]) -> float:
        """Calculate overall confidence across all assessments"""
        
        if not assessments:
            return 0.5
        
        confidence_scores = [assessment.confidence_level for assessment in assessments]
        return statistics.mean(confidence_scores)

    # Feature extraction methods
    
    def _extract_bodily_injury_features(self, policy_data: Dict[str, Any], 
                                      claim_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features for bodily injury assessment"""
        
        features = {
            'driver_age': policy_data.get('driver_age', 35),
            'driving_experience': policy_data.get('driving_experience', 10),
            'violation_count': len(policy_data.get('violations', [])),
            'accident_count': len(policy_data.get('accidents', [])),
            'vehicle_safety_rating': policy_data.get('vehicle', {}).get('safety_rating', 3),
            'location_risk_score': policy_data.get('location', {}).get('risk_score', 1.0),
            'annual_mileage': policy_data.get('annual_mileage', 12000),
            'coverage_limit': policy_data.get('bi_per_person_limit', 100000)
        }
        
        if claim_data:
            features.update({
                'claim_severity': claim_data.get('severity_score', 1.0),
                'injury_type': claim_data.get('injury_type', 'minor'),
                'medical_costs': claim_data.get('medical_costs', 0),
                'lost_wages': claim_data.get('lost_wages', 0)
            })
        
        return features

    def _extract_property_damage_features(self, policy_data: Dict[str, Any], 
                                        claim_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features for property damage assessment"""
        
        features = {
            'vehicle_value': policy_data.get('vehicle', {}).get('value', 25000),
            'vehicle_age': policy_data.get('vehicle', {}).get('age', 5),
            'driver_age': policy_data.get('driver_age', 35),
            'annual_mileage': policy_data.get('annual_mileage', 12000),
            'usage_type': policy_data.get('usage_type', 'personal'),
            'location_density': policy_data.get('location', {}).get('traffic_density', 1.0),
            'coverage_limit': policy_data.get('pd_per_accident_limit', 100000)
        }
        
        if claim_data:
            features.update({
                'damage_amount': claim_data.get('damage_amount', 0),
                'fault_percentage': claim_data.get('fault_percentage', 100),
                'property_type': claim_data.get('property_type', 'vehicle')
            })
        
        return features

    def _extract_comprehensive_features(self, policy_data: Dict[str, Any], 
                                      claim_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features for comprehensive assessment"""
        
        features = {
            'vehicle_value': policy_data.get('vehicle', {}).get('value', 25000),
            'vehicle_age': policy_data.get('vehicle', {}).get('age', 5),
            'theft_rating': policy_data.get('vehicle', {}).get('theft_rating', 3),
            'location_crime_rate': policy_data.get('location', {}).get('crime_rate', 1.0),
            'weather_risk': policy_data.get('location', {}).get('weather_risk_score', 1.0),
            'deductible': policy_data.get('comprehensive_deductible', 500)
        }
        
        if claim_data:
            features.update({
                'loss_type': claim_data.get('loss_type', 'theft'),
                'loss_amount': claim_data.get('loss_amount', 0),
                'recovery_potential': claim_data.get('recovery_potential', 0.0)
            })
        
        return features

    # Additional calculation methods
    
    async def _calculate_reserve_amount(self, assessments: List[RiskAssessment], 
                                      claim_data: Optional[Dict[str, Any]]) -> Decimal:
        """Calculate reserve amount for potential claims"""
        
        if not claim_data:
            # No active claim - calculate based on risk assessments
            total_exposure = sum(
                max(assessment.coverage_limits.values()) for assessment in assessments
            )
            
            # Reserve 15-25% of total exposure based on risk level
            avg_risk_score = statistics.mean(assessment.risk_score for assessment in assessments)
            reserve_percentage = 0.15 + (avg_risk_score - 1.0) * 0.1
            reserve_percentage = max(0.10, min(reserve_percentage, 0.30))
            
            return total_exposure * Decimal(str(reserve_percentage))
        else:
            # Active claim - calculate specific reserves
            estimated_total = Decimal(str(claim_data.get('estimated_total_cost', 0)))
            
            # Add uncertainty buffer based on claim complexity
            complexity_factor = claim_data.get('complexity_score', 1.0)
            buffer_percentage = 0.20 + (complexity_factor - 1.0) * 0.15
            buffer_percentage = max(0.10, min(buffer_percentage, 0.50))
            
            return estimated_total * Decimal(str(1.0 + buffer_percentage))

    async def _calculate_settlement_recommendation(self, assessments: List[RiskAssessment], 
                                                 claim_data: Dict[str, Any]) -> Decimal:
        """Calculate recommended settlement amount"""
        
        # Base settlement on actual damages and liability assessment
        actual_damages = Decimal(str(claim_data.get('actual_damages', 0)))
        fault_percentage = claim_data.get('fault_percentage', 100) / 100.0
        
        # Adjust for liability
        liable_amount = actual_damages * Decimal(str(fault_percentage))
        
        # Consider legal costs and settlement incentives
        legal_cost_estimate = Decimal(str(claim_data.get('estimated_legal_costs', 0)))
        
        # Settlement discount for avoiding litigation
        if claim_data.get('litigation_risk', 'low') == 'high':
            settlement_factor = Decimal('0.85')  # 15% discount for high litigation risk
        elif claim_data.get('litigation_risk', 'low') == 'medium':
            settlement_factor = Decimal('0.90')  # 10% discount for medium litigation risk
        else:
            settlement_factor = Decimal('0.95')  # 5% discount for low litigation risk
        
        recommended_settlement = (liable_amount + legal_cost_estimate) * settlement_factor
        
        return recommended_settlement

    async def _assess_legal_exposure(self, policy_data: Dict[str, Any], 
                                   claim_data: Optional[Dict[str, Any]], 
                                   assessments: List[RiskAssessment]) -> Decimal:
        """Assess potential legal exposure"""
        
        if not claim_data:
            return Decimal('0.00')
        
        # Base legal exposure on coverage limits and claim severity
        max_coverage = max(
            max(assessment.coverage_limits.values()) for assessment in assessments
        )
        
        claim_amount = Decimal(str(claim_data.get('claim_amount', 0)))
        
        # Excess exposure (amount above coverage limits)
        excess_exposure = max(Decimal('0.00'), claim_amount - max_coverage)
        
        # Bad faith exposure
        bad_faith_factors = claim_data.get('bad_faith_factors', [])
        if bad_faith_factors:
            bad_faith_multiplier = 1.0 + len(bad_faith_factors) * 0.5
            excess_exposure *= Decimal(str(bad_faith_multiplier))
        
        # Punitive damages potential
        if claim_data.get('punitive_damages_risk', False):
            punitive_estimate = claim_amount * Decimal('0.5')  # 50% of compensatory
            excess_exposure += punitive_estimate
        
        return excess_exposure

    async def _store_liability_assessment(self, calculation: LiabilityCalculation):
        """Store liability assessment in database"""
        
        try:
            with self.Session() as session:
                for assessment in calculation.assessment_results:
                    record = LiabilityAssessmentRecord(
                        assessment_id=assessment.assessment_id,
                        policy_number=calculation.policy_data.get('policy_number'),
                        claim_number=calculation.claim_data.get('claim_number') if calculation.claim_data else None,
                        liability_type=assessment.liability_type.value,
                        risk_level=assessment.risk_level.value,
                        risk_score=assessment.risk_score,
                        confidence_level=assessment.confidence_level,
                        contributing_factors=[asdict(factor) for factor in assessment.contributing_factors],
                        recommended_premium=assessment.recommended_premium,
                        coverage_limits=assessment.coverage_limits,
                        exclusions=assessment.exclusions,
                        conditions=assessment.conditions,
                        total_liability_amount=calculation.total_liability_amount,
                        reserve_amount=calculation.reserve_amount,
                        settlement_recommendation=calculation.settlement_recommendation,
                        legal_exposure=calculation.legal_exposure,
                        confidence_score=calculation.confidence_score,
                        calculation_method=calculation.calculation_method,
                        supporting_documentation=calculation.supporting_documentation,
                        status=AssessmentStatus.COMPLETED.value,
                        assessment_date=assessment.assessment_date,
                        valid_until=assessment.valid_until,
                        created_at=calculation.created_at,
                        updated_at=calculation.updated_at
                    )
                    
                    session.add(record)
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing liability assessment: {e}")

def create_liability_assessor(db_url: str = None, redis_url: str = None) -> LiabilityAssessor:
    """Create and configure LiabilityAssessor instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return LiabilityAssessor(db_url=db_url, redis_url=redis_url)

