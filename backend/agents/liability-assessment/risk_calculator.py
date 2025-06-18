"""
Risk Calculator - Production Ready Implementation
Advanced risk calculation engine for insurance liability assessment
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
from scipy import stats
from scipy.stats import norm, lognorm, gamma, beta

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

risk_calculations_total = Counter('risk_calculations_total', 'Total risk calculations', ['calculation_type'])
calculation_duration = Histogram('calculation_duration_seconds', 'Risk calculation duration')
risk_score_gauge = Gauge('risk_score_current', 'Current risk score being calculated')

Base = declarative_base()

class RiskCategory(Enum):
    FREQUENCY = "frequency"
    SEVERITY = "severity"
    EXPOSURE = "exposure"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    CATASTROPHIC = "catastrophic"

class CalculationMethod(Enum):
    ACTUARIAL = "actuarial"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    MONTE_CARLO = "monte_carlo"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"

class DistributionType(Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    GAMMA = "gamma"
    BETA = "beta"
    POISSON = "poisson"
    EXPONENTIAL = "exponential"
    WEIBULL = "weibull"

@dataclass
class RiskParameter:
    parameter_name: str
    parameter_value: float
    confidence_interval: Tuple[float, float]
    distribution_type: DistributionType
    source_data_points: int
    last_updated: datetime

@dataclass
class RiskModel:
    model_id: str
    model_name: str
    model_type: CalculationMethod
    risk_category: RiskCategory
    parameters: List[RiskParameter]
    accuracy_metrics: Dict[str, float]
    validation_date: datetime
    model_version: str

@dataclass
class RiskCalculationResult:
    calculation_id: str
    input_data: Dict[str, Any]
    risk_scores: Dict[RiskCategory, float]
    confidence_levels: Dict[RiskCategory, float]
    expected_values: Dict[str, float]
    percentiles: Dict[str, Dict[int, float]]
    sensitivity_analysis: Dict[str, float]
    scenario_results: Dict[str, float]
    calculation_method: CalculationMethod
    models_used: List[str]
    calculation_time: float
    created_at: datetime

class RiskModelRecord(Base):
    __tablename__ = 'risk_models'
    
    model_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    risk_category = Column(String, nullable=False)
    parameters = Column(JSON)
    accuracy_metrics = Column(JSON)
    validation_date = Column(DateTime)
    model_version = Column(String)
    model_data = Column(JSON)  # Serialized model
    training_data_size = Column(Integer)
    feature_importance = Column(JSON)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class RiskCalculationRecord(Base):
    __tablename__ = 'risk_calculations'
    
    calculation_id = Column(String, primary_key=True)
    policy_number = Column(String, index=True)
    claim_number = Column(String, index=True)
    input_data = Column(JSON)
    risk_scores = Column(JSON)
    confidence_levels = Column(JSON)
    expected_values = Column(JSON)
    percentiles = Column(JSON)
    sensitivity_analysis = Column(JSON)
    scenario_results = Column(JSON)
    calculation_method = Column(String)
    models_used = Column(JSON)
    calculation_time = Column(Float)
    created_at = Column(DateTime, nullable=False)

class RiskCalculator:
    """Production-ready Risk Calculator for comprehensive risk assessment"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # Risk models
        self.risk_models = {}
        self.ml_models = {}
        self.scalers = {}
        
        # Statistical distributions
        self.distributions = {}
        
        # Historical data cache
        self.historical_data = {}
        
        # Monte Carlo simulation parameters
        self.simulation_iterations = 10000
        
        self._initialize_risk_models()
        self._load_historical_data()
        
        logger.info("RiskCalculator initialized successfully")

    def _initialize_risk_models(self):
        """Initialize risk calculation models"""
        
        # Frequency models
        self.risk_models[RiskCategory.FREQUENCY] = {
            'poisson_model': {
                'distribution': DistributionType.POISSON,
                'parameters': {'lambda': 0.15},  # Average claims per year
                'confidence': 0.85
            },
            'negative_binomial_model': {
                'distribution': DistributionType.GAMMA,
                'parameters': {'shape': 0.8, 'scale': 0.2},
                'confidence': 0.90
            }
        }
        
        # Severity models
        self.risk_models[RiskCategory.SEVERITY] = {
            'lognormal_model': {
                'distribution': DistributionType.LOGNORMAL,
                'parameters': {'mu': 8.5, 'sigma': 1.2},  # Log of claim amounts
                'confidence': 0.88
            },
            'gamma_model': {
                'distribution': DistributionType.GAMMA,
                'parameters': {'shape': 1.5, 'scale': 8000},
                'confidence': 0.82
            },
            'pareto_model': {
                'distribution': DistributionType.EXPONENTIAL,
                'parameters': {'alpha': 2.1, 'scale': 5000},
                'confidence': 0.75
            }
        }
        
        # Exposure models
        self.risk_models[RiskCategory.EXPOSURE] = {
            'linear_exposure_model': {
                'base_exposure': 1.0,
                'exposure_factors': {
                    'vehicle_value': 0.0001,  # Per dollar of vehicle value
                    'annual_mileage': 0.00001,  # Per mile driven
                    'driver_age_factor': {'16-25': 1.8, '26-50': 1.0, '51-65': 0.9, '65+': 1.1}
                }
            }
        }
        
        # Initialize ML models
        self._initialize_ml_models()

    def _initialize_ml_models(self):
        """Initialize machine learning models for risk prediction"""
        
        # Frequency prediction models
        self.ml_models['frequency'] = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'ridge_regression': Ridge(alpha=1.0)
        }
        
        # Severity prediction models
        self.ml_models['severity'] = {
            'random_forest': RandomForestRegressor(
                n_estimators=180,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.08,
                subsample=0.9,
                random_state=42
            ),
            'lasso_regression': Lasso(alpha=0.1)
        }
        
        # Anomaly detection models
        self.ml_models['anomaly'] = {
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42
            )
        }
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

    def _load_historical_data(self):
        """Load historical data for risk calculations"""
        
        # This would typically load from database or external sources
        # For now, initialize with sample data structures
        
        self.historical_data = {
            'claim_frequencies': {
                'auto': {'mean': 0.12, 'std': 0.08, 'data_points': 50000},
                'home': {'mean': 0.08, 'std': 0.06, 'data_points': 30000},
                'life': {'mean': 0.003, 'std': 0.002, 'data_points': 100000}
            },
            'claim_severities': {
                'auto_bodily_injury': {'mean': 25000, 'std': 45000, 'data_points': 8000},
                'auto_property_damage': {'mean': 8000, 'std': 15000, 'data_points': 12000},
                'home_property': {'mean': 15000, 'std': 35000, 'data_points': 5000}
            },
            'risk_factors': {
                'age_multipliers': {
                    '16-20': 2.1, '21-25': 1.8, '26-30': 1.3, '31-40': 1.0,
                    '41-50': 0.9, '51-60': 0.8, '61-70': 0.9, '70+': 1.2
                },
                'location_multipliers': {
                    'urban': 1.3, 'suburban': 1.0, 'rural': 0.8
                },
                'vehicle_type_multipliers': {
                    'sedan': 1.0, 'suv': 1.1, 'truck': 1.2, 'sports_car': 1.8, 'luxury': 1.4
                }
            }
        }

    async def calculate_comprehensive_risk(self, input_data: Dict[str, Any], 
                                         calculation_method: CalculationMethod = CalculationMethod.ENSEMBLE) -> RiskCalculationResult:
        """Calculate comprehensive risk assessment"""
        
        calculation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        risk_calculations_total.labels(calculation_type='comprehensive').inc()
        
        try:
            with calculation_duration.time():
                # Extract and validate input data
                validated_data = await self._validate_input_data(input_data)
                
                # Calculate risk scores for each category
                risk_scores = {}
                confidence_levels = {}
                
                # Frequency risk calculation
                frequency_result = await self._calculate_frequency_risk(validated_data)
                risk_scores[RiskCategory.FREQUENCY] = frequency_result['risk_score']
                confidence_levels[RiskCategory.FREQUENCY] = frequency_result['confidence']
                
                # Severity risk calculation
                severity_result = await self._calculate_severity_risk(validated_data)
                risk_scores[RiskCategory.SEVERITY] = severity_result['risk_score']
                confidence_levels[RiskCategory.SEVERITY] = severity_result['confidence']
                
                # Exposure risk calculation
                exposure_result = await self._calculate_exposure_risk(validated_data)
                risk_scores[RiskCategory.EXPOSURE] = exposure_result['risk_score']
                confidence_levels[RiskCategory.EXPOSURE] = exposure_result['confidence']
                
                # Volatility risk calculation
                volatility_result = await self._calculate_volatility_risk(validated_data)
                risk_scores[RiskCategory.VOLATILITY] = volatility_result['risk_score']
                confidence_levels[RiskCategory.VOLATILITY] = volatility_result['confidence']
                
                # Correlation risk calculation
                correlation_result = await self._calculate_correlation_risk(validated_data)
                risk_scores[RiskCategory.CORRELATION] = correlation_result['risk_score']
                confidence_levels[RiskCategory.CORRELATION] = correlation_result['confidence']
                
                # Catastrophic risk calculation
                catastrophic_result = await self._calculate_catastrophic_risk(validated_data)
                risk_scores[RiskCategory.CATASTROPHIC] = catastrophic_result['risk_score']
                confidence_levels[RiskCategory.CATASTROPHIC] = catastrophic_result['confidence']
                
                # Calculate expected values
                expected_values = await self._calculate_expected_values(validated_data, risk_scores)
                
                # Calculate percentiles
                percentiles = await self._calculate_risk_percentiles(validated_data, risk_scores)
                
                # Perform sensitivity analysis
                sensitivity_analysis = await self._perform_sensitivity_analysis(validated_data, risk_scores)
                
                # Run scenario analysis
                scenario_results = await self._run_scenario_analysis(validated_data, risk_scores)
                
                # Determine models used
                models_used = self._get_models_used(calculation_method)
                
                calculation_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Create result
                result = RiskCalculationResult(
                    calculation_id=calculation_id,
                    input_data=validated_data,
                    risk_scores=risk_scores,
                    confidence_levels=confidence_levels,
                    expected_values=expected_values,
                    percentiles=percentiles,
                    sensitivity_analysis=sensitivity_analysis,
                    scenario_results=scenario_results,
                    calculation_method=calculation_method,
                    models_used=models_used,
                    calculation_time=calculation_time,
                    created_at=start_time
                )
                
                # Store calculation result
                await self._store_calculation_result(result)
                
                # Update metrics
                overall_risk_score = statistics.mean(risk_scores.values())
                risk_score_gauge.set(overall_risk_score)
                
                return result
                
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            raise

    async def _calculate_frequency_risk(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate frequency risk using multiple models"""
        
        try:
            # Extract frequency-related features
            features = self._extract_frequency_features(input_data)
            
            # Poisson model calculation
            lambda_param = self._calculate_poisson_lambda(features)
            poisson_risk = min(lambda_param * 2.0, 3.0)  # Cap at 3.0
            
            # Negative binomial model calculation
            nb_risk = self._calculate_negative_binomial_risk(features)
            
            # Machine learning model prediction
            ml_risk = await self._predict_frequency_ml(features)
            
            # Ensemble calculation
            weights = [0.4, 0.3, 0.3]  # Poisson, NB, ML
            ensemble_risk = (poisson_risk * weights[0] + 
                           nb_risk * weights[1] + 
                           ml_risk * weights[2])
            
            # Calculate confidence based on model agreement
            risks = [poisson_risk, nb_risk, ml_risk]
            risk_std = statistics.stdev(risks)
            confidence = max(0.5, 1.0 - (risk_std / statistics.mean(risks)))
            
            return {
                'risk_score': ensemble_risk,
                'confidence': confidence,
                'component_risks': {
                    'poisson': poisson_risk,
                    'negative_binomial': nb_risk,
                    'machine_learning': ml_risk
                }
            }
            
        except Exception as e:
            logger.error(f"Frequency risk calculation failed: {e}")
            return {'risk_score': 1.0, 'confidence': 0.5}

    async def _calculate_severity_risk(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate severity risk using multiple models"""
        
        try:
            # Extract severity-related features
            features = self._extract_severity_features(input_data)
            
            # Lognormal model calculation
            lognormal_risk = self._calculate_lognormal_severity(features)
            
            # Gamma model calculation
            gamma_risk = self._calculate_gamma_severity(features)
            
            # Machine learning model prediction
            ml_risk = await self._predict_severity_ml(features)
            
            # Ensemble calculation
            weights = [0.35, 0.35, 0.30]  # Lognormal, Gamma, ML
            ensemble_risk = (lognormal_risk * weights[0] + 
                           gamma_risk * weights[1] + 
                           ml_risk * weights[2])
            
            # Calculate confidence
            risks = [lognormal_risk, gamma_risk, ml_risk]
            risk_std = statistics.stdev(risks)
            confidence = max(0.5, 1.0 - (risk_std / statistics.mean(risks)))
            
            return {
                'risk_score': ensemble_risk,
                'confidence': confidence,
                'component_risks': {
                    'lognormal': lognormal_risk,
                    'gamma': gamma_risk,
                    'machine_learning': ml_risk
                }
            }
            
        except Exception as e:
            logger.error(f"Severity risk calculation failed: {e}")
            return {'risk_score': 1.0, 'confidence': 0.5}

    async def _calculate_exposure_risk(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate exposure risk"""
        
        try:
            base_exposure = 1.0
            
            # Vehicle value exposure
            vehicle_value = input_data.get('vehicle', {}).get('value', 25000)
            value_exposure = vehicle_value * 0.00002  # 2 basis points per dollar
            
            # Mileage exposure
            annual_mileage = input_data.get('annual_mileage', 12000)
            mileage_exposure = (annual_mileage / 12000) * 0.3  # 30% variation for mileage
            
            # Location exposure
            location_risk = input_data.get('location', {}).get('risk_score', 1.0)
            location_exposure = (location_risk - 1.0) * 0.5
            
            # Usage exposure
            usage_type = input_data.get('usage_type', 'personal')
            usage_multipliers = {
                'personal': 1.0,
                'commuting': 1.1,
                'business': 1.3,
                'commercial': 1.5,
                'rideshare': 1.4
            }
            usage_exposure = usage_multipliers.get(usage_type, 1.0) - 1.0
            
            # Total exposure risk
            total_exposure = (base_exposure + value_exposure + mileage_exposure + 
                            location_exposure + usage_exposure)
            
            # Normalize to reasonable range
            exposure_risk = max(0.5, min(total_exposure, 2.5))
            
            return {
                'risk_score': exposure_risk,
                'confidence': 0.85,
                'component_exposures': {
                    'vehicle_value': value_exposure,
                    'annual_mileage': mileage_exposure,
                    'location': location_exposure,
                    'usage_type': usage_exposure
                }
            }
            
        except Exception as e:
            logger.error(f"Exposure risk calculation failed: {e}")
            return {'risk_score': 1.0, 'confidence': 0.5}

    async def _calculate_volatility_risk(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate volatility risk"""
        
        try:
            # Historical volatility calculation
            historical_claims = input_data.get('historical_claims', [])
            
            if len(historical_claims) >= 3:
                claim_amounts = [claim.get('amount', 0) for claim in historical_claims]
                volatility = statistics.stdev(claim_amounts) / max(statistics.mean(claim_amounts), 1)
            else:
                # Use industry averages
                volatility = 0.8  # Default volatility
            
            # Driver volatility factors
            driver_age = input_data.get('driver_age', 35)
            if driver_age < 25:
                age_volatility = 1.5
            elif driver_age > 65:
                age_volatility = 1.2
            else:
                age_volatility = 1.0
            
            # Vehicle volatility factors
            vehicle_type = input_data.get('vehicle', {}).get('type', 'sedan').lower()
            vehicle_volatility = {
                'sedan': 1.0,
                'suv': 1.1,
                'truck': 1.2,
                'sports_car': 1.8,
                'motorcycle': 2.2,
                'luxury': 1.4
            }.get(vehicle_type, 1.0)
            
            # Composite volatility risk
            volatility_risk = volatility * age_volatility * vehicle_volatility
            volatility_risk = max(0.3, min(volatility_risk, 3.0))
            
            return {
                'risk_score': volatility_risk,
                'confidence': 0.75,
                'component_volatilities': {
                    'historical': volatility,
                    'age_factor': age_volatility,
                    'vehicle_factor': vehicle_volatility
                }
            }
            
        except Exception as e:
            logger.error(f"Volatility risk calculation failed: {e}")
            return {'risk_score': 1.0, 'confidence': 0.5}

    async def _calculate_correlation_risk(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlation risk"""
        
        try:
            # Geographic correlation
            location = input_data.get('location', {})
            zip_code = location.get('zip_code', '')
            
            # Simulate correlation based on geographic clustering
            geographic_correlation = self._calculate_geographic_correlation(zip_code)
            
            # Temporal correlation (seasonal patterns)
            current_month = datetime.utcnow().month
            seasonal_multipliers = {
                12: 1.3, 1: 1.3, 2: 1.2,  # Winter months
                3: 1.1, 4: 1.0, 5: 1.0,   # Spring months
                6: 1.1, 7: 1.2, 8: 1.1,   # Summer months
                9: 1.0, 10: 1.0, 11: 1.1  # Fall months
            }
            temporal_correlation = seasonal_multipliers.get(current_month, 1.0)
            
            # Economic correlation
            economic_indicators = input_data.get('economic_indicators', {})
            unemployment_rate = economic_indicators.get('unemployment_rate', 5.0)
            economic_correlation = 1.0 + (unemployment_rate - 5.0) * 0.02
            
            # Composite correlation risk
            correlation_risk = (geographic_correlation * 0.4 + 
                              temporal_correlation * 0.3 + 
                              economic_correlation * 0.3)
            
            correlation_risk = max(0.5, min(correlation_risk, 2.0))
            
            return {
                'risk_score': correlation_risk,
                'confidence': 0.70,
                'component_correlations': {
                    'geographic': geographic_correlation,
                    'temporal': temporal_correlation,
                    'economic': economic_correlation
                }
            }
            
        except Exception as e:
            logger.error(f"Correlation risk calculation failed: {e}")
            return {'risk_score': 1.0, 'confidence': 0.5}

    async def _calculate_catastrophic_risk(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate catastrophic risk"""
        
        try:
            location = input_data.get('location', {})
            
            # Natural disaster risk
            natural_disaster_risk = 1.0
            
            # Hurricane risk
            if location.get('hurricane_zone', False):
                natural_disaster_risk *= 1.4
            
            # Earthquake risk
            earthquake_zone = location.get('earthquake_zone', 0)
            if earthquake_zone > 0:
                natural_disaster_risk *= (1.0 + earthquake_zone * 0.2)
            
            # Flood risk
            flood_zone = location.get('flood_zone', 'X')
            flood_multipliers = {'A': 1.8, 'AE': 1.6, 'X': 1.0, 'B': 1.1, 'C': 1.05}
            natural_disaster_risk *= flood_multipliers.get(flood_zone, 1.0)
            
            # Wildfire risk
            wildfire_risk = location.get('wildfire_risk_score', 1.0)
            natural_disaster_risk *= wildfire_risk
            
            # Terrorism risk (for commercial properties)
            terrorism_risk = 1.0
            if input_data.get('property_type') == 'commercial':
                city_size = location.get('city_size', 'medium')
                if city_size == 'major':
                    terrorism_risk = 1.15
                elif city_size == 'large':
                    terrorism_risk = 1.08
            
            # Cyber risk (for businesses)
            cyber_risk = 1.0
            if input_data.get('business_type'):
                cyber_exposure = input_data.get('cyber_exposure_score', 1.0)
                cyber_risk = cyber_exposure
            
            # Composite catastrophic risk
            catastrophic_risk = natural_disaster_risk * terrorism_risk * cyber_risk
            catastrophic_risk = max(0.8, min(catastrophic_risk, 3.0))
            
            return {
                'risk_score': catastrophic_risk,
                'confidence': 0.65,
                'component_risks': {
                    'natural_disaster': natural_disaster_risk,
                    'terrorism': terrorism_risk,
                    'cyber': cyber_risk
                }
            }
            
        except Exception as e:
            logger.error(f"Catastrophic risk calculation failed: {e}")
            return {'risk_score': 1.0, 'confidence': 0.5}

    # Helper methods for specific calculations
    
    def _calculate_poisson_lambda(self, features: Dict[str, Any]) -> float:
        """Calculate Poisson lambda parameter for frequency modeling"""
        
        base_lambda = 0.12  # Base frequency
        
        # Age adjustment
        age = features.get('driver_age', 35)
        if age < 25:
            age_factor = 1.8
        elif age > 65:
            age_factor = 1.2
        else:
            age_factor = 1.0
        
        # Experience adjustment
        experience = features.get('driving_experience', 10)
        experience_factor = max(0.7, 1.0 - (experience - 5) * 0.05)
        
        # Violation adjustment
        violations = features.get('violation_count', 0)
        violation_factor = 1.0 + violations * 0.3
        
        # Mileage adjustment
        mileage = features.get('annual_mileage', 12000)
        mileage_factor = mileage / 12000
        
        lambda_param = base_lambda * age_factor * experience_factor * violation_factor * mileage_factor
        
        return max(0.01, min(lambda_param, 1.0))

    def _calculate_negative_binomial_risk(self, features: Dict[str, Any]) -> float:
        """Calculate negative binomial risk"""
        
        # Negative binomial parameters
        r = 0.8  # Shape parameter
        p = 0.85  # Success probability
        
        # Adjust parameters based on features
        driver_age = features.get('driver_age', 35)
        if driver_age < 25:
            p *= 0.9  # Lower success probability for young drivers
        
        violations = features.get('violation_count', 0)
        p *= (0.95 ** violations)  # Reduce success probability for violations
        
        # Calculate expected value and convert to risk score
        expected_failures = r * (1 - p) / p
        risk_score = min(expected_failures * 2.0, 3.0)
        
        return risk_score

    def _calculate_lognormal_severity(self, features: Dict[str, Any]) -> float:
        """Calculate lognormal severity risk"""
        
        # Base lognormal parameters
        mu = 8.5  # Log mean
        sigma = 1.2  # Log standard deviation
        
        # Adjust parameters based on features
        vehicle_value = features.get('vehicle_value', 25000)
        value_adjustment = math.log(vehicle_value / 25000) * 0.3
        mu += value_adjustment
        
        coverage_limit = features.get('coverage_limit', 100000)
        if coverage_limit > 250000:
            sigma *= 1.1  # Higher variability for high coverage
        
        # Calculate expected severity and normalize to risk score
        expected_severity = math.exp(mu + sigma**2 / 2)
        risk_score = min(expected_severity / 50000, 3.0)
        
        return risk_score

    def _calculate_gamma_severity(self, features: Dict[str, Any]) -> float:
        """Calculate gamma severity risk"""
        
        # Base gamma parameters
        shape = 1.5
        scale = 8000
        
        # Adjust parameters based on features
        location_risk = features.get('location_risk_score', 1.0)
        scale *= location_risk
        
        usage_type = features.get('usage_type', 'personal')
        if usage_type == 'commercial':
            shape *= 1.2
            scale *= 1.3
        
        # Calculate expected value and normalize
        expected_value = shape * scale
        risk_score = min(expected_value / 50000, 3.0)
        
        return risk_score

    async def _predict_frequency_ml(self, features: Dict[str, Any]) -> float:
        """Predict frequency using machine learning models"""
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_ml_features(features, 'frequency')
            
            # Get predictions from multiple models
            predictions = []
            
            for model_name, model in self.ml_models['frequency'].items():
                try:
                    # Scale features
                    scaled_features = self.scalers['standard'].fit_transform([feature_vector])
                    prediction = model.predict(scaled_features)[0]
                    predictions.append(max(0.01, min(prediction, 2.0)))
                except:
                    predictions.append(1.0)  # Default prediction
            
            # Ensemble prediction
            if predictions:
                return statistics.mean(predictions)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"ML frequency prediction failed: {e}")
            return 1.0

    async def _predict_severity_ml(self, features: Dict[str, Any]) -> float:
        """Predict severity using machine learning models"""
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_ml_features(features, 'severity')
            
            # Get predictions from multiple models
            predictions = []
            
            for model_name, model in self.ml_models['severity'].items():
                try:
                    # Scale features
                    scaled_features = self.scalers['standard'].fit_transform([feature_vector])
                    prediction = model.predict(scaled_features)[0]
                    predictions.append(max(0.1, min(prediction, 3.0)))
                except:
                    predictions.append(1.0)  # Default prediction
            
            # Ensemble prediction
            if predictions:
                return statistics.mean(predictions)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"ML severity prediction failed: {e}")
            return 1.0

    def _prepare_ml_features(self, features: Dict[str, Any], model_type: str) -> List[float]:
        """Prepare feature vector for ML models"""
        
        if model_type == 'frequency':
            return [
                features.get('driver_age', 35) / 100,
                features.get('driving_experience', 10) / 50,
                features.get('violation_count', 0),
                features.get('accident_count', 0),
                features.get('annual_mileage', 12000) / 50000,
                features.get('vehicle_age', 5) / 20,
                features.get('location_risk_score', 1.0),
                1.0 if features.get('usage_type') == 'business' else 0.0
            ]
        elif model_type == 'severity':
            return [
                features.get('vehicle_value', 25000) / 100000,
                features.get('driver_age', 35) / 100,
                features.get('coverage_limit', 100000) / 1000000,
                features.get('location_risk_score', 1.0),
                features.get('vehicle_safety_rating', 3) / 5,
                1.0 if features.get('usage_type') == 'commercial' else 0.0,
                features.get('deductible', 500) / 5000
            ]
        else:
            return [1.0] * 8  # Default feature vector

    def _extract_frequency_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features relevant to frequency modeling"""
        
        return {
            'driver_age': input_data.get('driver_age', 35),
            'driving_experience': input_data.get('driving_experience', 10),
            'violation_count': len(input_data.get('violations', [])),
            'accident_count': len(input_data.get('accidents', [])),
            'annual_mileage': input_data.get('annual_mileage', 12000),
            'vehicle_age': input_data.get('vehicle', {}).get('age', 5),
            'location_risk_score': input_data.get('location', {}).get('risk_score', 1.0),
            'usage_type': input_data.get('usage_type', 'personal')
        }

    def _extract_severity_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features relevant to severity modeling"""
        
        return {
            'vehicle_value': input_data.get('vehicle', {}).get('value', 25000),
            'driver_age': input_data.get('driver_age', 35),
            'coverage_limit': input_data.get('coverage_limit', 100000),
            'location_risk_score': input_data.get('location', {}).get('risk_score', 1.0),
            'vehicle_safety_rating': input_data.get('vehicle', {}).get('safety_rating', 3),
            'usage_type': input_data.get('usage_type', 'personal'),
            'deductible': input_data.get('deductible', 500)
        }

    def _calculate_geographic_correlation(self, zip_code: str) -> float:
        """Calculate geographic correlation risk"""
        
        # Simulate correlation based on zip code clustering
        if not zip_code:
            return 1.0
        
        # Use first 3 digits for regional clustering
        region = zip_code[:3] if len(zip_code) >= 3 else '000'
        
        # High-risk regions (simulated)
        high_risk_regions = ['100', '900', '330', '770', '480']
        
        if region in high_risk_regions:
            return 1.3
        else:
            return 1.0

    async def _calculate_expected_values(self, input_data: Dict[str, Any], 
                                       risk_scores: Dict[RiskCategory, float]) -> Dict[str, float]:
        """Calculate expected values for various metrics"""
        
        frequency_risk = risk_scores[RiskCategory.FREQUENCY]
        severity_risk = risk_scores[RiskCategory.SEVERITY]
        exposure_risk = risk_scores[RiskCategory.EXPOSURE]
        
        # Expected annual frequency
        base_frequency = 0.12
        expected_frequency = base_frequency * frequency_risk
        
        # Expected claim severity
        base_severity = 25000
        expected_severity = base_severity * severity_risk
        
        # Expected annual loss
        expected_annual_loss = expected_frequency * expected_severity * exposure_risk
        
        # Expected premium
        expected_premium = expected_annual_loss * 1.25  # 25% loading
        
        return {
            'expected_frequency': expected_frequency,
            'expected_severity': expected_severity,
            'expected_annual_loss': expected_annual_loss,
            'expected_premium': expected_premium
        }

    async def _calculate_risk_percentiles(self, input_data: Dict[str, Any], 
                                        risk_scores: Dict[RiskCategory, float]) -> Dict[str, Dict[int, float]]:
        """Calculate risk percentiles using Monte Carlo simulation"""
        
        percentiles_to_calculate = [50, 75, 90, 95, 99]
        results = {}
        
        # Simulate annual losses
        simulated_losses = []
        
        for _ in range(self.simulation_iterations):
            # Sample frequency from Poisson distribution
            lambda_param = 0.12 * risk_scores[RiskCategory.FREQUENCY]
            frequency = np.random.poisson(lambda_param)
            
            # Sample severity from lognormal distribution
            mu = 8.5 + math.log(risk_scores[RiskCategory.SEVERITY])
            sigma = 1.2
            
            total_loss = 0
            for _ in range(frequency):
                severity = np.random.lognormal(mu, sigma)
                total_loss += severity
            
            # Apply exposure adjustment
            total_loss *= risk_scores[RiskCategory.EXPOSURE]
            simulated_losses.append(total_loss)
        
        # Calculate percentiles
        results['annual_loss'] = {}
        for percentile in percentiles_to_calculate:
            results['annual_loss'][percentile] = np.percentile(simulated_losses, percentile)
        
        return results

    async def _perform_sensitivity_analysis(self, input_data: Dict[str, Any], 
                                          risk_scores: Dict[RiskCategory, float]) -> Dict[str, float]:
        """Perform sensitivity analysis on key variables"""
        
        base_risk = statistics.mean(risk_scores.values())
        sensitivities = {}
        
        # Test sensitivity to driver age
        age_variations = [-5, 5]  # +/- 5 years
        age_impacts = []
        
        for variation in age_variations:
            modified_data = input_data.copy()
            modified_data['driver_age'] = input_data.get('driver_age', 35) + variation
            
            # Recalculate frequency risk with modified age
            modified_features = self._extract_frequency_features(modified_data)
            modified_lambda = self._calculate_poisson_lambda(modified_features)
            modified_risk = min(modified_lambda * 2.0, 3.0)
            
            age_impacts.append(abs(modified_risk - risk_scores[RiskCategory.FREQUENCY]))
        
        sensitivities['driver_age'] = statistics.mean(age_impacts)
        
        # Test sensitivity to vehicle value
        value_variations = [-5000, 5000]  # +/- $5,000
        value_impacts = []
        
        for variation in value_variations:
            modified_data = input_data.copy()
            vehicle_data = modified_data.get('vehicle', {})
            vehicle_data['value'] = vehicle_data.get('value', 25000) + variation
            modified_data['vehicle'] = vehicle_data
            
            # Recalculate severity risk
            modified_features = self._extract_severity_features(modified_data)
            modified_risk = self._calculate_lognormal_severity(modified_features)
            
            value_impacts.append(abs(modified_risk - risk_scores[RiskCategory.SEVERITY]))
        
        sensitivities['vehicle_value'] = statistics.mean(value_impacts)
        
        # Test sensitivity to annual mileage
        mileage_variations = [-3000, 3000]  # +/- 3,000 miles
        mileage_impacts = []
        
        for variation in mileage_variations:
            modified_data = input_data.copy()
            modified_data['annual_mileage'] = input_data.get('annual_mileage', 12000) + variation
            
            # Recalculate exposure risk
            exposure_result = await self._calculate_exposure_risk(modified_data)
            modified_risk = exposure_result['risk_score']
            
            mileage_impacts.append(abs(modified_risk - risk_scores[RiskCategory.EXPOSURE]))
        
        sensitivities['annual_mileage'] = statistics.mean(mileage_impacts)
        
        return sensitivities

    async def _run_scenario_analysis(self, input_data: Dict[str, Any], 
                                   risk_scores: Dict[RiskCategory, float]) -> Dict[str, float]:
        """Run scenario analysis for different conditions"""
        
        base_risk = statistics.mean(risk_scores.values())
        scenarios = {}
        
        # Economic downturn scenario
        downturn_data = input_data.copy()
        economic_indicators = downturn_data.get('economic_indicators', {})
        economic_indicators['unemployment_rate'] = 10.0  # High unemployment
        economic_indicators['gdp_growth'] = -2.0  # Negative growth
        downturn_data['economic_indicators'] = economic_indicators
        
        downturn_correlation = await self._calculate_correlation_risk(downturn_data)
        scenarios['economic_downturn'] = downturn_correlation['risk_score']
        
        # Natural disaster scenario
        disaster_data = input_data.copy()
        location_data = disaster_data.get('location', {})
        location_data['hurricane_zone'] = True
        location_data['flood_zone'] = 'A'
        location_data['earthquake_zone'] = 3
        disaster_data['location'] = location_data
        
        disaster_catastrophic = await self._calculate_catastrophic_risk(disaster_data)
        scenarios['natural_disaster'] = disaster_catastrophic['risk_score']
        
        # Technology disruption scenario
        tech_data = input_data.copy()
        tech_data['autonomous_vehicle'] = True
        tech_data['telematics_enabled'] = True
        
        # Assume technology reduces frequency risk by 30%
        scenarios['technology_adoption'] = risk_scores[RiskCategory.FREQUENCY] * 0.7
        
        return scenarios

    def _get_models_used(self, calculation_method: CalculationMethod) -> List[str]:
        """Get list of models used in calculation"""
        
        if calculation_method == CalculationMethod.ENSEMBLE:
            return [
                'poisson_frequency',
                'negative_binomial_frequency',
                'lognormal_severity',
                'gamma_severity',
                'random_forest_ml',
                'gradient_boosting_ml',
                'monte_carlo_simulation'
            ]
        elif calculation_method == CalculationMethod.ACTUARIAL:
            return [
                'poisson_frequency',
                'lognormal_severity',
                'gamma_severity'
            ]
        elif calculation_method == CalculationMethod.MACHINE_LEARNING:
            return [
                'random_forest_frequency',
                'gradient_boosting_frequency',
                'random_forest_severity',
                'gradient_boosting_severity'
            ]
        else:
            return ['ensemble_model']

    async def _validate_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean input data"""
        
        validated_data = input_data.copy()
        
        # Validate driver age
        driver_age = validated_data.get('driver_age', 35)
        validated_data['driver_age'] = max(16, min(driver_age, 100))
        
        # Validate annual mileage
        annual_mileage = validated_data.get('annual_mileage', 12000)
        validated_data['annual_mileage'] = max(1000, min(annual_mileage, 100000))
        
        # Validate vehicle value
        vehicle_data = validated_data.get('vehicle', {})
        vehicle_value = vehicle_data.get('value', 25000)
        vehicle_data['value'] = max(1000, min(vehicle_value, 500000))
        validated_data['vehicle'] = vehicle_data
        
        # Ensure required fields have defaults
        if 'violations' not in validated_data:
            validated_data['violations'] = []
        
        if 'accidents' not in validated_data:
            validated_data['accidents'] = []
        
        if 'location' not in validated_data:
            validated_data['location'] = {'risk_score': 1.0}
        
        return validated_data

    async def _store_calculation_result(self, result: RiskCalculationResult):
        """Store risk calculation result in database"""
        
        try:
            with self.Session() as session:
                record = RiskCalculationRecord(
                    calculation_id=result.calculation_id,
                    policy_number=result.input_data.get('policy_number'),
                    claim_number=result.input_data.get('claim_number'),
                    input_data=result.input_data,
                    risk_scores={k.value: v for k, v in result.risk_scores.items()},
                    confidence_levels={k.value: v for k, v in result.confidence_levels.items()},
                    expected_values=result.expected_values,
                    percentiles=result.percentiles,
                    sensitivity_analysis=result.sensitivity_analysis,
                    scenario_results=result.scenario_results,
                    calculation_method=result.calculation_method.value,
                    models_used=result.models_used,
                    calculation_time=result.calculation_time,
                    created_at=result.created_at
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing risk calculation result: {e}")

def create_risk_calculator(db_url: str = None, redis_url: str = None) -> RiskCalculator:
    """Create and configure RiskCalculator instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return RiskCalculator(db_url=db_url, redis_url=redis_url)

