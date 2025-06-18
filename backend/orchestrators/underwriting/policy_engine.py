"""
Policy Engine - Production Ready Implementation
Comprehensive policy management, pricing, and configuration
"""

import asyncio
import json
import logging
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
import numpy as np
import pandas as pd

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

policies_processed_total = Counter('policies_processed_total', 'Total policies processed', ['policy_type'])
policy_processing_duration = Histogram('policy_processing_duration_seconds', 'Policy processing duration')
active_policies_gauge = Gauge('active_policies_current', 'Current active policies')

Base = declarative_base()

class PolicyStatus(Enum):
    QUOTED = "quoted"
    BOUND = "bound"
    ACTIVE = "active"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    RENEWED = "renewed"

class CoverageType(Enum):
    LIABILITY = "liability"
    COLLISION = "collision"
    COMPREHENSIVE = "comprehensive"
    UNINSURED_MOTORIST = "uninsured_motorist"
    PERSONAL_INJURY_PROTECTION = "personal_injury_protection"
    DWELLING = "dwelling"
    PERSONAL_PROPERTY = "personal_property"
    LIABILITY_COVERAGE = "liability_coverage"
    MEDICAL_PAYMENTS = "medical_payments"
    ADDITIONAL_LIVING_EXPENSES = "additional_living_expenses"

class PaymentFrequency(Enum):
    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"

@dataclass
class Coverage:
    coverage_id: str
    coverage_type: CoverageType
    limit: float
    deductible: float
    premium: float
    description: str
    is_required: bool
    conditions: List[str]
    exclusions: List[str]

@dataclass
class PolicyTerm:
    effective_date: datetime
    expiration_date: datetime
    term_length_months: int
    payment_frequency: PaymentFrequency
    payment_amount: float
    total_premium: float

@dataclass
class Discount:
    discount_id: str
    name: str
    description: str
    discount_type: str
    amount: float
    percentage: float
    conditions: List[str]
    applied_to: List[str]

@dataclass
class Surcharge:
    surcharge_id: str
    name: str
    description: str
    surcharge_type: str
    amount: float
    percentage: float
    reason: str
    applied_to: List[str]

@dataclass
class Policy:
    policy_id: str
    policy_number: str
    application_id: str
    policy_type: str
    status: PolicyStatus
    policyholder: Dict[str, Any]
    coverages: List[Coverage]
    policy_term: PolicyTerm
    discounts: List[Discount]
    surcharges: List[Surcharge]
    conditions: List[str]
    exclusions: List[str]
    endorsements: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

class PolicyRecord(Base):
    __tablename__ = 'policies'
    
    policy_id = Column(String, primary_key=True)
    policy_number = Column(String, unique=True, nullable=False)
    application_id = Column(String, nullable=False)
    policy_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    policyholder_data = Column(JSON)
    coverages_data = Column(JSON)
    policy_term_data = Column(JSON)
    discounts_data = Column(JSON)
    surcharges_data = Column(JSON)
    conditions = Column(JSON)
    exclusions = Column(JSON)
    endorsements = Column(JSON)
    total_premium = Column(Float)
    effective_date = Column(DateTime)
    expiration_date = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class CoverageTemplate(Base):
    __tablename__ = 'coverage_templates'
    
    template_id = Column(String, primary_key=True)
    policy_type = Column(String, nullable=False)
    coverage_type = Column(String, nullable=False)
    template_name = Column(String, nullable=False)
    default_limit = Column(Float)
    default_deductible = Column(Float)
    base_rate = Column(Float)
    rating_factors = Column(JSON)
    conditions = Column(JSON)
    exclusions = Column(JSON)
    is_required = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class RatingFactor(Base):
    __tablename__ = 'rating_factors'
    
    factor_id = Column(String, primary_key=True)
    policy_type = Column(String, nullable=False)
    factor_name = Column(String, nullable=False)
    factor_type = Column(String, nullable=False)
    factor_values = Column(JSON)
    multipliers = Column(JSON)
    effective_date = Column(DateTime, nullable=False)
    expiration_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False)

class PolicyEngine:
    """Production-ready Policy Engine for comprehensive policy management"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # Policy templates and configurations
        self.coverage_templates = {}
        self.rating_factors = {}
        self.policy_rules = {}
        
        # Pricing configurations
        self.base_rates = {}
        self.territory_factors = {}
        self.discount_rules = {}
        self.surcharge_rules = {}
        
        self._initialize_coverage_templates()
        self._initialize_rating_factors()
        self._initialize_policy_rules()
        self._initialize_pricing_configurations()
        
        logger.info("PolicyEngine initialized successfully")

    def _initialize_coverage_templates(self):
        """Initialize coverage templates for different policy types"""
        
        # Auto Personal Coverage Templates
        self.coverage_templates['auto_personal'] = {
            CoverageType.LIABILITY: {
                'name': 'Bodily Injury & Property Damage Liability',
                'description': 'Covers damages to others when you are at fault',
                'default_limits': [25000, 50000, 100000, 250000, 500000],
                'default_limit': 100000,
                'default_deductible': 0,
                'base_rate': 0.008,
                'is_required': True,
                'conditions': [
                    'Coverage applies only when insured is legally liable',
                    'Defense costs are in addition to policy limits'
                ],
                'exclusions': [
                    'Intentional acts',
                    'Racing or speed contests',
                    'Commercial use (unless specifically covered)'
                ]
            },
            CoverageType.COLLISION: {
                'name': 'Collision Coverage',
                'description': 'Covers damage to your vehicle from collision',
                'default_limits': [5000, 10000, 25000, 50000, 100000],
                'default_limit': 25000,
                'default_deductible': 500,
                'deductible_options': [250, 500, 1000, 2500],
                'base_rate': 0.012,
                'is_required': False,
                'conditions': [
                    'Coverage subject to actual cash value of vehicle',
                    'Deductible applies per occurrence'
                ],
                'exclusions': [
                    'Normal wear and tear',
                    'Mechanical breakdown',
                    'Damage from animals'
                ]
            },
            CoverageType.COMPREHENSIVE: {
                'name': 'Comprehensive Coverage',
                'description': 'Covers damage to your vehicle from non-collision events',
                'default_limits': [5000, 10000, 25000, 50000, 100000],
                'default_limit': 25000,
                'default_deductible': 500,
                'deductible_options': [100, 250, 500, 1000],
                'base_rate': 0.006,
                'is_required': False,
                'conditions': [
                    'Coverage subject to actual cash value of vehicle',
                    'Glass damage may have separate deductible'
                ],
                'exclusions': [
                    'Normal wear and tear',
                    'Mechanical breakdown',
                    'Personal property in vehicle'
                ]
            },
            CoverageType.UNINSURED_MOTORIST: {
                'name': 'Uninsured/Underinsured Motorist',
                'description': 'Covers you when hit by uninsured or underinsured driver',
                'default_limits': [25000, 50000, 100000, 250000],
                'default_limit': 50000,
                'default_deductible': 0,
                'base_rate': 0.003,
                'is_required': True,
                'conditions': [
                    'Must maintain same limits as liability coverage',
                    'Applies to bodily injury and property damage'
                ],
                'exclusions': [
                    'Hit and run without police report',
                    'Family member exclusions may apply'
                ]
            },
            CoverageType.PERSONAL_INJURY_PROTECTION: {
                'name': 'Personal Injury Protection (PIP)',
                'description': 'Covers medical expenses regardless of fault',
                'default_limits': [2500, 5000, 10000, 25000],
                'default_limit': 10000,
                'default_deductible': 0,
                'base_rate': 0.004,
                'is_required': False,
                'conditions': [
                    'No-fault coverage',
                    'Covers medical, lost wages, and essential services'
                ],
                'exclusions': [
                    'Injuries from criminal acts',
                    'Self-inflicted injuries'
                ]
            }
        }
        
        # Homeowners Coverage Templates
        self.coverage_templates['homeowners'] = {
            CoverageType.DWELLING: {
                'name': 'Dwelling Coverage',
                'description': 'Covers the structure of your home',
                'default_limits': [100000, 200000, 300000, 500000, 1000000],
                'default_limit': 300000,
                'default_deductible': 1000,
                'deductible_options': [500, 1000, 2500, 5000, 10000],
                'base_rate': 0.003,
                'is_required': True,
                'conditions': [
                    'Coverage at replacement cost',
                    'Includes attached structures'
                ],
                'exclusions': [
                    'Flood damage',
                    'Earthquake damage',
                    'Normal wear and tear'
                ]
            },
            CoverageType.PERSONAL_PROPERTY: {
                'name': 'Personal Property Coverage',
                'description': 'Covers your personal belongings',
                'default_limits': [25000, 50000, 75000, 100000, 150000],
                'default_limit': 75000,
                'default_deductible': 1000,
                'base_rate': 0.002,
                'is_required': True,
                'conditions': [
                    'Typically 50-75% of dwelling coverage',
                    'Covers property worldwide'
                ],
                'exclusions': [
                    'Business property',
                    'Motor vehicles',
                    'High-value items without scheduling'
                ]
            },
            CoverageType.LIABILITY_COVERAGE: {
                'name': 'Personal Liability Coverage',
                'description': 'Covers liability claims against you',
                'default_limits': [100000, 300000, 500000, 1000000],
                'default_limit': 300000,
                'default_deductible': 0,
                'base_rate': 0.001,
                'is_required': True,
                'conditions': [
                    'Covers bodily injury and property damage',
                    'Includes defense costs'
                ],
                'exclusions': [
                    'Business activities',
                    'Motor vehicle liability',
                    'Intentional acts'
                ]
            },
            CoverageType.MEDICAL_PAYMENTS: {
                'name': 'Medical Payments Coverage',
                'description': 'Covers medical expenses for guests injured on your property',
                'default_limits': [1000, 2500, 5000, 10000],
                'default_limit': 5000,
                'default_deductible': 0,
                'base_rate': 0.0005,
                'is_required': False,
                'conditions': [
                    'No-fault coverage',
                    'Covers reasonable medical expenses'
                ],
                'exclusions': [
                    'Injuries to household members',
                    'Business-related injuries'
                ]
            },
            CoverageType.ADDITIONAL_LIVING_EXPENSES: {
                'name': 'Additional Living Expenses',
                'description': 'Covers extra costs when home is uninhabitable',
                'default_limits': [10000, 20000, 30000, 50000],
                'default_limit': 20000,
                'default_deductible': 0,
                'base_rate': 0.0008,
                'is_required': True,
                'conditions': [
                    'Covers reasonable increase in living expenses',
                    'Time limit typically 12-24 months'
                ],
                'exclusions': [
                    'Expenses that would be incurred anyway',
                    'Luxury accommodations'
                ]
            }
        }

    def _initialize_rating_factors(self):
        """Initialize rating factors for premium calculation"""
        
        # Auto Personal Rating Factors
        self.rating_factors['auto_personal'] = {
            'age_factors': {
                'factor_type': 'age_based',
                'values': {
                    '16-20': 1.8,
                    '21-24': 1.5,
                    '25-29': 1.2,
                    '30-49': 1.0,
                    '50-64': 0.9,
                    '65-74': 1.1,
                    '75+': 1.3
                }
            },
            'gender_factors': {
                'factor_type': 'gender_based',
                'values': {
                    'male_under_25': 1.2,
                    'female_under_25': 1.1,
                    'male_25_plus': 1.0,
                    'female_25_plus': 0.95
                }
            },
            'marital_status_factors': {
                'factor_type': 'marital_status',
                'values': {
                    'single': 1.1,
                    'married': 0.9,
                    'divorced': 1.05,
                    'widowed': 1.0
                }
            },
            'credit_score_factors': {
                'factor_type': 'credit_based',
                'values': {
                    '800-850': 0.8,
                    '750-799': 0.9,
                    '700-749': 1.0,
                    '650-699': 1.1,
                    '600-649': 1.3,
                    '550-599': 1.5,
                    '300-549': 1.8
                }
            },
            'vehicle_age_factors': {
                'factor_type': 'vehicle_age',
                'values': {
                    '0-2': 1.2,
                    '3-5': 1.0,
                    '6-10': 0.9,
                    '11-15': 0.8,
                    '16+': 0.7
                }
            },
            'vehicle_type_factors': {
                'factor_type': 'vehicle_type',
                'values': {
                    'sedan': 1.0,
                    'suv': 1.1,
                    'truck': 1.15,
                    'sports_car': 1.5,
                    'luxury': 1.3,
                    'hybrid': 0.9,
                    'electric': 0.85
                }
            },
            'usage_factors': {
                'factor_type': 'vehicle_usage',
                'values': {
                    'pleasure': 0.9,
                    'commute_short': 1.0,
                    'commute_long': 1.2,
                    'business': 1.4,
                    'commercial': 1.8
                }
            },
            'mileage_factors': {
                'factor_type': 'annual_mileage',
                'values': {
                    '0-5000': 0.8,
                    '5001-10000': 0.9,
                    '10001-15000': 1.0,
                    '15001-20000': 1.1,
                    '20001-25000': 1.3,
                    '25000+': 1.5
                }
            },
            'territory_factors': {
                'factor_type': 'territory',
                'values': {
                    'rural': 0.8,
                    'suburban': 1.0,
                    'urban': 1.3,
                    'metro': 1.5
                }
            }
        }
        
        # Homeowners Rating Factors
        self.rating_factors['homeowners'] = {
            'construction_type_factors': {
                'factor_type': 'construction',
                'values': {
                    'masonry': 0.8,
                    'frame': 1.0,
                    'manufactured': 1.4,
                    'log': 1.2,
                    'other': 1.3
                }
            },
            'roof_type_factors': {
                'factor_type': 'roof_type',
                'values': {
                    'tile': 0.9,
                    'slate': 0.85,
                    'metal': 0.9,
                    'asphalt_shingle': 1.0,
                    'wood_shake': 1.3,
                    'other': 1.2
                }
            },
            'age_factors': {
                'factor_type': 'property_age',
                'values': {
                    '0-10': 0.9,
                    '11-20': 0.95,
                    '21-30': 1.0,
                    '31-40': 1.1,
                    '41-50': 1.2,
                    '51+': 1.4
                }
            },
            'protection_class_factors': {
                'factor_type': 'fire_protection',
                'values': {
                    '1': 0.7,
                    '2': 0.8,
                    '3': 0.9,
                    '4': 1.0,
                    '5': 1.1,
                    '6': 1.2,
                    '7': 1.3,
                    '8': 1.4,
                    '9': 1.6,
                    '10': 2.0
                }
            },
            'occupancy_factors': {
                'factor_type': 'occupancy',
                'values': {
                    'owner_occupied': 1.0,
                    'tenant_occupied': 1.2,
                    'seasonal': 1.3,
                    'vacant': 2.0
                }
            },
            'security_factors': {
                'factor_type': 'security_features',
                'values': {
                    'basic_alarm': 0.95,
                    'monitored_alarm': 0.9,
                    'security_system': 0.85,
                    'gated_community': 0.9,
                    'none': 1.0
                }
            },
            'claims_history_factors': {
                'factor_type': 'claims_frequency',
                'values': {
                    '0_claims': 0.9,
                    '1_claim': 1.0,
                    '2_claims': 1.3,
                    '3_claims': 1.6,
                    '4+_claims': 2.0
                }
            }
        }

    def _initialize_policy_rules(self):
        """Initialize policy rules and requirements"""
        
        self.policy_rules = {
            'auto_personal': {
                'minimum_liability_limits': {
                    'bodily_injury_per_person': 25000,
                    'bodily_injury_per_accident': 50000,
                    'property_damage': 25000
                },
                'required_coverages': [
                    CoverageType.LIABILITY,
                    CoverageType.UNINSURED_MOTORIST
                ],
                'optional_coverages': [
                    CoverageType.COLLISION,
                    CoverageType.COMPREHENSIVE,
                    CoverageType.PERSONAL_INJURY_PROTECTION
                ],
                'age_restrictions': {
                    'minimum_age': 16,
                    'maximum_age': 85
                },
                'vehicle_restrictions': {
                    'maximum_age': 25,
                    'minimum_value': 1000,
                    'maximum_value': 200000
                },
                'payment_options': [
                    PaymentFrequency.ANNUAL,
                    PaymentFrequency.SEMI_ANNUAL,
                    PaymentFrequency.QUARTERLY,
                    PaymentFrequency.MONTHLY
                ]
            },
            'homeowners': {
                'minimum_dwelling_coverage': 50000,
                'required_coverages': [
                    CoverageType.DWELLING,
                    CoverageType.PERSONAL_PROPERTY,
                    CoverageType.LIABILITY_COVERAGE,
                    CoverageType.ADDITIONAL_LIVING_EXPENSES
                ],
                'optional_coverages': [
                    CoverageType.MEDICAL_PAYMENTS
                ],
                'property_restrictions': {
                    'maximum_age': 100,
                    'minimum_value': 50000,
                    'maximum_value': 5000000
                },
                'occupancy_requirements': [
                    'owner_occupied',
                    'tenant_occupied',
                    'seasonal'
                ],
                'payment_options': [
                    PaymentFrequency.ANNUAL,
                    PaymentFrequency.SEMI_ANNUAL,
                    PaymentFrequency.QUARTERLY,
                    PaymentFrequency.MONTHLY
                ]
            }
        }

    def _initialize_pricing_configurations(self):
        """Initialize pricing configurations"""
        
        # Base rates by coverage type
        self.base_rates = {
            'auto_personal': {
                CoverageType.LIABILITY: 0.008,
                CoverageType.COLLISION: 0.012,
                CoverageType.COMPREHENSIVE: 0.006,
                CoverageType.UNINSURED_MOTORIST: 0.003,
                CoverageType.PERSONAL_INJURY_PROTECTION: 0.004
            },
            'homeowners': {
                CoverageType.DWELLING: 0.003,
                CoverageType.PERSONAL_PROPERTY: 0.002,
                CoverageType.LIABILITY_COVERAGE: 0.001,
                CoverageType.MEDICAL_PAYMENTS: 0.0005,
                CoverageType.ADDITIONAL_LIVING_EXPENSES: 0.0008
            }
        }
        
        # Territory factors
        self.territory_factors = {
            'auto_personal': {
                '001': 0.8,   # Rural
                '002': 0.9,   # Small town
                '003': 1.0,   # Suburban
                '004': 1.2,   # Urban
                '005': 1.5,   # Metro
                '006': 1.8,   # High-density urban
                '007': 0.85,  # College town
                '008': 1.1,   # Industrial
                '009': 1.3,   # High-crime urban
                '010': 2.0    # Extreme high-risk
            },
            'homeowners': {
                '001': 0.9,   # Rural low-risk
                '002': 0.95,  # Suburban low-risk
                '003': 1.0,   # Standard suburban
                '004': 1.1,   # Urban standard
                '005': 1.3,   # High-density urban
                '006': 1.5,   # High-crime urban
                '007': 0.8,   # Gated community
                '008': 1.2,   # Coastal (non-hurricane)
                '009': 1.8,   # Hurricane zone
                '010': 2.5    # Extreme catastrophe zone
            }
        }
        
        # Discount rules
        self.discount_rules = {
            'auto_personal': {
                'multi_policy': {
                    'name': 'Multi-Policy Discount',
                    'percentage': 0.05,
                    'conditions': ['has_homeowners_policy'],
                    'max_discount': 0.05
                },
                'multi_vehicle': {
                    'name': 'Multi-Vehicle Discount',
                    'percentage': 0.10,
                    'conditions': ['vehicle_count >= 2'],
                    'max_discount': 0.25
                },
                'good_driver': {
                    'name': 'Good Driver Discount',
                    'percentage': 0.10,
                    'conditions': ['no_violations_3_years', 'no_accidents_3_years'],
                    'max_discount': 0.10
                },
                'defensive_driving': {
                    'name': 'Defensive Driving Course',
                    'percentage': 0.05,
                    'conditions': ['completed_defensive_driving'],
                    'max_discount': 0.05
                },
                'good_student': {
                    'name': 'Good Student Discount',
                    'percentage': 0.10,
                    'conditions': ['student_age < 25', 'gpa >= 3.0'],
                    'max_discount': 0.10
                },
                'safety_features': {
                    'name': 'Safety Features Discount',
                    'percentage': 0.08,
                    'conditions': ['airbags', 'abs', 'stability_control'],
                    'max_discount': 0.15
                },
                'anti_theft': {
                    'name': 'Anti-Theft Device Discount',
                    'percentage': 0.03,
                    'conditions': ['anti_theft_device'],
                    'max_discount': 0.05
                }
            },
            'homeowners': {
                'multi_policy': {
                    'name': 'Multi-Policy Discount',
                    'percentage': 0.05,
                    'conditions': ['has_auto_policy'],
                    'max_discount': 0.05
                },
                'security_system': {
                    'name': 'Security System Discount',
                    'percentage': 0.10,
                    'conditions': ['monitored_security_system'],
                    'max_discount': 0.15
                },
                'fire_safety': {
                    'name': 'Fire Safety Discount',
                    'percentage': 0.08,
                    'conditions': ['smoke_detectors', 'fire_extinguisher'],
                    'max_discount': 0.10
                },
                'claims_free': {
                    'name': 'Claims-Free Discount',
                    'percentage': 0.05,
                    'conditions': ['no_claims_5_years'],
                    'max_discount': 0.15
                },
                'new_home': {
                    'name': 'New Home Discount',
                    'percentage': 0.10,
                    'conditions': ['home_age < 10'],
                    'max_discount': 0.10
                },
                'loyalty': {
                    'name': 'Loyalty Discount',
                    'percentage': 0.03,
                    'conditions': ['customer_years >= 3'],
                    'max_discount': 0.10
                }
            }
        }
        
        # Surcharge rules
        self.surcharge_rules = {
            'auto_personal': {
                'young_driver': {
                    'name': 'Young Driver Surcharge',
                    'percentage': 0.25,
                    'conditions': ['age < 25'],
                    'max_surcharge': 0.50
                },
                'violations': {
                    'name': 'Moving Violations Surcharge',
                    'percentage': 0.10,
                    'conditions': ['violations_count > 0'],
                    'max_surcharge': 0.40
                },
                'accidents': {
                    'name': 'At-Fault Accidents Surcharge',
                    'percentage': 0.20,
                    'conditions': ['at_fault_accidents > 0'],
                    'max_surcharge': 0.60
                },
                'credit_score': {
                    'name': 'Credit Score Surcharge',
                    'percentage': 0.15,
                    'conditions': ['credit_score < 650'],
                    'max_surcharge': 0.30
                },
                'high_mileage': {
                    'name': 'High Mileage Surcharge',
                    'percentage': 0.10,
                    'conditions': ['annual_mileage > 20000'],
                    'max_surcharge': 0.25
                }
            },
            'homeowners': {
                'claims_history': {
                    'name': 'Claims History Surcharge',
                    'percentage': 0.15,
                    'conditions': ['claims_count > 1'],
                    'max_surcharge': 0.50
                },
                'property_age': {
                    'name': 'Older Property Surcharge',
                    'percentage': 0.10,
                    'conditions': ['property_age > 50'],
                    'max_surcharge': 0.30
                },
                'construction_type': {
                    'name': 'Construction Type Surcharge',
                    'percentage': 0.20,
                    'conditions': ['construction_type == manufactured'],
                    'max_surcharge': 0.40
                },
                'protection_class': {
                    'name': 'Poor Fire Protection Surcharge',
                    'percentage': 0.25,
                    'conditions': ['protection_class > 6'],
                    'max_surcharge': 0.50
                },
                'catastrophe_exposure': {
                    'name': 'Catastrophe Exposure Surcharge',
                    'percentage': 0.30,
                    'conditions': ['high_cat_risk'],
                    'max_surcharge': 1.00
                }
            }
        }

    async def create_policy(self, application_data: Dict[str, Any], underwriting_result: Dict[str, Any]) -> Policy:
        """Create a new insurance policy"""
        
        policy_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        with policy_processing_duration.time():
            try:
                application_id = application_data.get('application_id')
                policy_type = application_data.get('policy_type')
                
                # Generate policy number
                policy_number = await self._generate_policy_number(policy_type)
                
                # Extract policyholder information
                policyholder = await self._extract_policyholder_info(application_data)
                
                # Create coverages
                coverages = await self._create_coverages(application_data, policy_type)
                
                # Calculate premiums
                premium_calculation = await self._calculate_policy_premiums(application_data, coverages, policy_type)
                
                # Apply discounts
                discounts = await self._apply_policy_discounts(application_data, premium_calculation, policy_type)
                
                # Apply surcharges
                surcharges = await self._apply_policy_surcharges(application_data, premium_calculation, policy_type)
                
                # Create policy term
                policy_term = await self._create_policy_term(application_data, premium_calculation, discounts, surcharges)
                
                # Determine conditions and exclusions
                conditions = await self._determine_policy_conditions(application_data, underwriting_result, policy_type)
                exclusions = await self._determine_policy_exclusions(application_data, policy_type)
                
                # Create endorsements
                endorsements = await self._create_endorsements(application_data, underwriting_result, policy_type)
                
                # Create policy object
                policy = Policy(
                    policy_id=policy_id,
                    policy_number=policy_number,
                    application_id=application_id,
                    policy_type=policy_type,
                    status=PolicyStatus.QUOTED,
                    policyholder=policyholder,
                    coverages=coverages,
                    policy_term=policy_term,
                    discounts=discounts,
                    surcharges=surcharges,
                    conditions=conditions,
                    exclusions=exclusions,
                    endorsements=endorsements,
                    created_at=start_time,
                    updated_at=start_time
                )
                
                # Store policy
                await self._store_policy(policy)
                
                # Update metrics
                policies_processed_total.labels(policy_type=policy_type).inc()
                
                return policy
                
            except Exception as e:
                logger.error(f"Policy creation failed: {e}")
                raise

    async def _generate_policy_number(self, policy_type: str) -> str:
        """Generate unique policy number"""
        
        # Policy number format: PPP-YYYYMMDD-NNNNNN
        # PPP = Policy type prefix
        # YYYYMMDD = Date
        # NNNNNN = Sequential number
        
        type_prefixes = {
            'auto_personal': 'APL',
            'homeowners': 'HOM',
            'commercial_auto': 'CAL',
            'commercial_property': 'CPL'
        }
        
        prefix = type_prefixes.get(policy_type, 'POL')
        date_part = datetime.utcnow().strftime('%Y%m%d')
        
        # Get next sequential number from Redis
        counter_key = f"policy_counter:{policy_type}:{date_part}"
        sequence = self.redis_client.incr(counter_key)
        
        # Set expiration for counter (1 day after date)
        if sequence == 1:
            self.redis_client.expire(counter_key, 86400)
        
        policy_number = f"{prefix}-{date_part}-{sequence:06d}"
        
        return policy_number

    async def _extract_policyholder_info(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format policyholder information"""
        
        applicant = application_data.get('applicant', {})
        
        policyholder = {
            'name': applicant.get('name', ''),
            'date_of_birth': applicant.get('date_of_birth', ''),
            'gender': applicant.get('gender', ''),
            'marital_status': applicant.get('marital_status', ''),
            'address': applicant.get('address', {}),
            'phone': applicant.get('phone', ''),
            'email': applicant.get('email', ''),
            'ssn': applicant.get('ssn', ''),
            'license_number': applicant.get('license_number', ''),
            'license_state': applicant.get('license_state', ''),
            'occupation': applicant.get('occupation', ''),
            'employer': applicant.get('employer', ''),
            'credit_score': applicant.get('credit_score', 0)
        }
        
        return policyholder

    async def _create_coverages(self, application_data: Dict[str, Any], policy_type: str) -> List[Coverage]:
        """Create policy coverages based on application data"""
        
        coverages = []
        coverage_data = application_data.get('coverage', {})
        templates = self.coverage_templates.get(policy_type, {})
        
        for coverage_type, template in templates.items():
            # Check if coverage is requested or required
            coverage_key = coverage_type.value
            
            if (template.get('is_required', False) or 
                coverage_data.get(coverage_key, {}).get('selected', False)):
                
                # Get limits and deductibles
                requested_limit = coverage_data.get(coverage_key, {}).get('limit', template['default_limit'])
                requested_deductible = coverage_data.get(coverage_key, {}).get('deductible', template['default_deductible'])
                
                # Validate limits
                if 'default_limits' in template:
                    if requested_limit not in template['default_limits']:
                        requested_limit = template['default_limit']
                
                # Validate deductibles
                if 'deductible_options' in template:
                    if requested_deductible not in template['deductible_options']:
                        requested_deductible = template['default_deductible']
                
                # Calculate premium (will be done in premium calculation step)
                premium = 0.0
                
                coverage = Coverage(
                    coverage_id=str(uuid.uuid4()),
                    coverage_type=coverage_type,
                    limit=requested_limit,
                    deductible=requested_deductible,
                    premium=premium,
                    description=template.get('description', ''),
                    is_required=template.get('is_required', False),
                    conditions=template.get('conditions', []),
                    exclusions=template.get('exclusions', [])
                )
                
                coverages.append(coverage)
        
        return coverages

    async def _calculate_policy_premiums(self, application_data: Dict[str, Any], 
                                       coverages: List[Coverage], 
                                       policy_type: str) -> Dict[str, Any]:
        """Calculate premiums for all coverages"""
        
        premium_calculation = {
            'base_premiums': {},
            'adjusted_premiums': {},
            'rating_factors': {},
            'total_base_premium': 0.0,
            'total_adjusted_premium': 0.0
        }
        
        try:
            base_rates = self.base_rates.get(policy_type, {})
            rating_factors = self.rating_factors.get(policy_type, {})
            
            for coverage in coverages:
                coverage_type = coverage.coverage_type
                base_rate = base_rates.get(coverage_type, 0.001)
                
                # Calculate base premium
                if coverage_type in [CoverageType.LIABILITY, CoverageType.UNINSURED_MOTORIST, 
                                   CoverageType.LIABILITY_COVERAGE, CoverageType.MEDICAL_PAYMENTS]:
                    # Liability coverages: rate per $1000 of coverage
                    base_premium = (coverage.limit / 1000) * base_rate * 1000
                else:
                    # Physical damage coverages: rate based on insured value
                    if policy_type == 'auto_personal':
                        vehicle_value = application_data.get('vehicle', {}).get('value', 25000)
                        base_premium = vehicle_value * base_rate
                    else:  # homeowners
                        dwelling_value = application_data.get('property', {}).get('dwelling_value', 300000)
                        base_premium = dwelling_value * base_rate
                
                # Apply rating factors
                adjusted_premium = base_premium
                applied_factors = {}
                
                for factor_name, factor_data in rating_factors.items():
                    factor_multiplier = await self._get_rating_factor_multiplier(
                        factor_data, application_data, policy_type
                    )
                    
                    if factor_multiplier != 1.0:
                        adjusted_premium *= factor_multiplier
                        applied_factors[factor_name] = factor_multiplier
                
                # Apply territory factor
                territory = application_data.get('territory', '003')
                territory_multiplier = self.territory_factors.get(policy_type, {}).get(territory, 1.0)
                adjusted_premium *= territory_multiplier
                applied_factors['territory'] = territory_multiplier
                
                # Store results
                premium_calculation['base_premiums'][coverage_type.value] = base_premium
                premium_calculation['adjusted_premiums'][coverage_type.value] = adjusted_premium
                premium_calculation['rating_factors'][coverage_type.value] = applied_factors
                
                # Update coverage premium
                coverage.premium = adjusted_premium
                
                # Add to totals
                premium_calculation['total_base_premium'] += base_premium
                premium_calculation['total_adjusted_premium'] += adjusted_premium
            
            return premium_calculation
            
        except Exception as e:
            logger.error(f"Premium calculation failed: {e}")
            raise

    async def _get_rating_factor_multiplier(self, factor_data: Dict[str, Any], 
                                          application_data: Dict[str, Any], 
                                          policy_type: str) -> float:
        """Get rating factor multiplier for specific application data"""
        
        factor_type = factor_data.get('factor_type')
        factor_values = factor_data.get('values', {})
        
        try:
            if factor_type == 'age_based':
                # Calculate age from date of birth
                dob = application_data.get('applicant', {}).get('date_of_birth', '1990-01-01')
                age = (datetime.utcnow() - datetime.strptime(dob, '%Y-%m-%d')).days / 365.25
                
                for age_range, multiplier in factor_values.items():
                    if '-' in age_range:
                        min_age, max_age = map(int, age_range.split('-'))
                        if min_age <= age <= max_age:
                            return multiplier
                    elif age_range.endswith('+'):
                        min_age = int(age_range[:-1])
                        if age >= min_age:
                            return multiplier
                
            elif factor_type == 'gender_based':
                gender = application_data.get('applicant', {}).get('gender', 'male')
                dob = application_data.get('applicant', {}).get('date_of_birth', '1990-01-01')
                age = (datetime.utcnow() - datetime.strptime(dob, '%Y-%m-%d')).days / 365.25
                
                if age < 25:
                    key = f"{gender}_under_25"
                else:
                    key = f"{gender}_25_plus"
                
                return factor_values.get(key, 1.0)
                
            elif factor_type == 'marital_status':
                marital_status = application_data.get('applicant', {}).get('marital_status', 'single')
                return factor_values.get(marital_status, 1.0)
                
            elif factor_type == 'credit_based':
                credit_score = application_data.get('applicant', {}).get('credit_score', 700)
                
                for score_range, multiplier in factor_values.items():
                    min_score, max_score = map(int, score_range.split('-'))
                    if min_score <= credit_score <= max_score:
                        return multiplier
                        
            elif factor_type == 'vehicle_age':
                vehicle_year = application_data.get('vehicle', {}).get('year', 2020)
                vehicle_age = datetime.utcnow().year - vehicle_year
                
                for age_range, multiplier in factor_values.items():
                    if '-' in age_range:
                        min_age, max_age = map(int, age_range.split('-'))
                        if min_age <= vehicle_age <= max_age:
                            return multiplier
                    elif age_range.endswith('+'):
                        min_age = int(age_range[:-1])
                        if vehicle_age >= min_age:
                            return multiplier
                            
            elif factor_type == 'vehicle_type':
                vehicle_type = application_data.get('vehicle', {}).get('type', 'sedan')
                return factor_values.get(vehicle_type, 1.0)
                
            elif factor_type == 'vehicle_usage':
                usage = application_data.get('vehicle', {}).get('usage', 'pleasure')
                return factor_values.get(usage, 1.0)
                
            elif factor_type == 'annual_mileage':
                mileage = application_data.get('applicant', {}).get('annual_mileage', 12000)
                
                for mileage_range, multiplier in factor_values.items():
                    if '-' in mileage_range:
                        min_miles, max_miles = map(int, mileage_range.split('-'))
                        if min_miles <= mileage <= max_miles:
                            return multiplier
                    elif mileage_range.endswith('+'):
                        min_miles = int(mileage_range[:-1])
                        if mileage >= min_miles:
                            return multiplier
                            
            elif factor_type == 'territory':
                territory = application_data.get('territory', 'suburban')
                return factor_values.get(territory, 1.0)
                
            elif factor_type == 'construction':
                construction_type = application_data.get('property', {}).get('construction_type', 'frame')
                return factor_values.get(construction_type, 1.0)
                
            elif factor_type == 'roof_type':
                roof_type = application_data.get('property', {}).get('roof_type', 'asphalt_shingle')
                return factor_values.get(roof_type, 1.0)
                
            elif factor_type == 'property_age':
                year_built = application_data.get('property', {}).get('year_built', 1990)
                property_age = datetime.utcnow().year - year_built
                
                for age_range, multiplier in factor_values.items():
                    if '-' in age_range:
                        min_age, max_age = map(int, age_range.split('-'))
                        if min_age <= property_age <= max_age:
                            return multiplier
                    elif age_range.endswith('+'):
                        min_age = int(age_range[:-1])
                        if property_age >= min_age:
                            return multiplier
                            
            elif factor_type == 'fire_protection':
                protection_class = str(application_data.get('property', {}).get('protection_class', 5))
                return factor_values.get(protection_class, 1.0)
                
            elif factor_type == 'occupancy':
                occupancy = application_data.get('property', {}).get('occupancy', 'owner_occupied')
                return factor_values.get(occupancy, 1.0)
                
            elif factor_type == 'security_features':
                security_features = application_data.get('property', {}).get('security_features', [])
                
                if 'monitored_alarm' in security_features:
                    return factor_values.get('monitored_alarm', 1.0)
                elif 'basic_alarm' in security_features:
                    return factor_values.get('basic_alarm', 1.0)
                elif 'security_system' in security_features:
                    return factor_values.get('security_system', 1.0)
                else:
                    return factor_values.get('none', 1.0)
                    
            elif factor_type == 'claims_frequency':
                claims_count = len(application_data.get('applicant', {}).get('claims_history', []))
                
                if claims_count == 0:
                    return factor_values.get('0_claims', 1.0)
                elif claims_count == 1:
                    return factor_values.get('1_claim', 1.0)
                elif claims_count == 2:
                    return factor_values.get('2_claims', 1.0)
                elif claims_count == 3:
                    return factor_values.get('3_claims', 1.0)
                else:
                    return factor_values.get('4+_claims', 1.0)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Rating factor calculation failed: {e}")
            return 1.0

    async def _apply_policy_discounts(self, application_data: Dict[str, Any], 
                                    premium_calculation: Dict[str, Any], 
                                    policy_type: str) -> List[Discount]:
        """Apply applicable discounts to policy"""
        
        discounts = []
        discount_rules = self.discount_rules.get(policy_type, {})
        
        for discount_name, discount_rule in discount_rules.items():
            if await self._check_discount_eligibility(discount_rule, application_data, policy_type):
                discount_percentage = discount_rule.get('percentage', 0.0)
                max_discount = discount_rule.get('max_discount', discount_percentage)
                
                # Calculate discount amount
                total_premium = premium_calculation.get('total_adjusted_premium', 0.0)
                discount_amount = total_premium * min(discount_percentage, max_discount)
                
                discount = Discount(
                    discount_id=str(uuid.uuid4()),
                    name=discount_rule.get('name', discount_name),
                    description=f"Discount for {discount_rule.get('name', discount_name)}",
                    discount_type=discount_name,
                    amount=discount_amount,
                    percentage=discount_percentage,
                    conditions=discount_rule.get('conditions', []),
                    applied_to=['all_coverages']
                )
                
                discounts.append(discount)
        
        return discounts

    async def _check_discount_eligibility(self, discount_rule: Dict[str, Any], 
                                        application_data: Dict[str, Any], 
                                        policy_type: str) -> bool:
        """Check if applicant is eligible for discount"""
        
        conditions = discount_rule.get('conditions', [])
        
        for condition in conditions:
            if not await self._evaluate_discount_condition(condition, application_data, policy_type):
                return False
        
        return True

    async def _evaluate_discount_condition(self, condition: str, 
                                         application_data: Dict[str, Any], 
                                         policy_type: str) -> bool:
        """Evaluate a specific discount condition"""
        
        try:
            applicant = application_data.get('applicant', {})
            vehicle = application_data.get('vehicle', {})
            property_data = application_data.get('property', {})
            
            if condition == 'has_homeowners_policy':
                return applicant.get('has_homeowners_policy', False)
            elif condition == 'has_auto_policy':
                return applicant.get('has_auto_policy', False)
            elif condition.startswith('vehicle_count >='):
                required_count = int(condition.split('>=')[1].strip())
                vehicle_count = applicant.get('vehicle_count', 1)
                return vehicle_count >= required_count
            elif condition == 'no_violations_3_years':
                violations = applicant.get('violations', [])
                recent_violations = [v for v in violations 
                                   if (datetime.utcnow() - datetime.strptime(v.get('date', '2020-01-01'), '%Y-%m-%d')).days <= 1095]
                return len(recent_violations) == 0
            elif condition == 'no_accidents_3_years':
                accidents = applicant.get('accidents', [])
                recent_accidents = [a for a in accidents 
                                  if (datetime.utcnow() - datetime.strptime(a.get('date', '2020-01-01'), '%Y-%m-%d')).days <= 1095]
                return len(recent_accidents) == 0
            elif condition == 'no_claims_5_years':
                claims = applicant.get('claims_history', [])
                recent_claims = [c for c in claims 
                               if (datetime.utcnow() - datetime.strptime(c.get('date', '2015-01-01'), '%Y-%m-%d')).days <= 1825]
                return len(recent_claims) == 0
            elif condition == 'completed_defensive_driving':
                return applicant.get('defensive_driving_completed', False)
            elif condition.startswith('student_age <'):
                max_age = int(condition.split('<')[1].strip())
                dob = applicant.get('date_of_birth', '1990-01-01')
                age = (datetime.utcnow() - datetime.strptime(dob, '%Y-%m-%d')).days / 365.25
                return age < max_age and applicant.get('is_student', False)
            elif condition.startswith('gpa >='):
                required_gpa = float(condition.split('>=')[1].strip())
                gpa = applicant.get('gpa', 0.0)
                return gpa >= required_gpa
            elif condition == 'airbags':
                safety_features = vehicle.get('safety_features', [])
                return 'airbags' in safety_features
            elif condition == 'abs':
                safety_features = vehicle.get('safety_features', [])
                return 'abs' in safety_features
            elif condition == 'stability_control':
                safety_features = vehicle.get('safety_features', [])
                return 'stability_control' in safety_features
            elif condition == 'anti_theft_device':
                return vehicle.get('anti_theft_device', False)
            elif condition == 'monitored_security_system':
                security_features = property_data.get('security_features', [])
                return 'monitored_alarm' in security_features
            elif condition == 'smoke_detectors':
                safety_features = property_data.get('safety_features', [])
                return 'smoke_detectors' in safety_features
            elif condition == 'fire_extinguisher':
                safety_features = property_data.get('safety_features', [])
                return 'fire_extinguisher' in safety_features
            elif condition.startswith('home_age <'):
                max_age = int(condition.split('<')[1].strip())
                year_built = property_data.get('year_built', 1990)
                home_age = datetime.utcnow().year - year_built
                return home_age < max_age
            elif condition.startswith('customer_years >='):
                required_years = int(condition.split('>=')[1].strip())
                customer_years = applicant.get('customer_years', 0)
                return customer_years >= required_years
            
            return False
            
        except Exception as e:
            logger.error(f"Discount condition evaluation failed: {e}")
            return False

    async def _apply_policy_surcharges(self, application_data: Dict[str, Any], 
                                     premium_calculation: Dict[str, Any], 
                                     policy_type: str) -> List[Surcharge]:
        """Apply applicable surcharges to policy"""
        
        surcharges = []
        surcharge_rules = self.surcharge_rules.get(policy_type, {})
        
        for surcharge_name, surcharge_rule in surcharge_rules.items():
            if await self._check_surcharge_applicability(surcharge_rule, application_data, policy_type):
                surcharge_percentage = await self._calculate_surcharge_percentage(
                    surcharge_rule, application_data, policy_type
                )
                max_surcharge = surcharge_rule.get('max_surcharge', surcharge_percentage)
                
                # Calculate surcharge amount
                total_premium = premium_calculation.get('total_adjusted_premium', 0.0)
                surcharge_amount = total_premium * min(surcharge_percentage, max_surcharge)
                
                surcharge = Surcharge(
                    surcharge_id=str(uuid.uuid4()),
                    name=surcharge_rule.get('name', surcharge_name),
                    description=f"Surcharge for {surcharge_rule.get('name', surcharge_name)}",
                    surcharge_type=surcharge_name,
                    amount=surcharge_amount,
                    percentage=surcharge_percentage,
                    reason=await self._get_surcharge_reason(surcharge_rule, application_data, policy_type),
                    applied_to=['all_coverages']
                )
                
                surcharges.append(surcharge)
        
        return surcharges

    async def _check_surcharge_applicability(self, surcharge_rule: Dict[str, Any], 
                                           application_data: Dict[str, Any], 
                                           policy_type: str) -> bool:
        """Check if surcharge applies to applicant"""
        
        conditions = surcharge_rule.get('conditions', [])
        
        for condition in conditions:
            if await self._evaluate_surcharge_condition(condition, application_data, policy_type):
                return True
        
        return False

    async def _evaluate_surcharge_condition(self, condition: str, 
                                          application_data: Dict[str, Any], 
                                          policy_type: str) -> bool:
        """Evaluate a specific surcharge condition"""
        
        try:
            applicant = application_data.get('applicant', {})
            property_data = application_data.get('property', {})
            
            if condition.startswith('age <'):
                max_age = int(condition.split('<')[1].strip())
                dob = applicant.get('date_of_birth', '1990-01-01')
                age = (datetime.utcnow() - datetime.strptime(dob, '%Y-%m-%d')).days / 365.25
                return age < max_age
            elif condition.startswith('violations_count >'):
                min_count = int(condition.split('>')[1].strip())
                violations_count = len(applicant.get('violations', []))
                return violations_count > min_count
            elif condition.startswith('at_fault_accidents >'):
                min_count = int(condition.split('>')[1].strip())
                accidents = applicant.get('accidents', [])
                at_fault_count = len([a for a in accidents if a.get('at_fault', False)])
                return at_fault_count > min_count
            elif condition.startswith('credit_score <'):
                max_score = int(condition.split('<')[1].strip())
                credit_score = applicant.get('credit_score', 700)
                return credit_score < max_score
            elif condition.startswith('annual_mileage >'):
                max_mileage = int(condition.split('>')[1].strip())
                annual_mileage = applicant.get('annual_mileage', 12000)
                return annual_mileage > max_mileage
            elif condition.startswith('claims_count >'):
                min_count = int(condition.split('>')[1].strip())
                claims_count = len(applicant.get('claims_history', []))
                return claims_count > min_count
            elif condition.startswith('property_age >'):
                min_age = int(condition.split('>')[1].strip())
                year_built = property_data.get('year_built', 1990)
                property_age = datetime.utcnow().year - year_built
                return property_age > min_age
            elif condition.startswith('construction_type =='):
                required_type = condition.split('==')[1].strip()
                construction_type = property_data.get('construction_type', 'frame')
                return construction_type == required_type
            elif condition.startswith('protection_class >'):
                min_class = int(condition.split('>')[1].strip())
                protection_class = property_data.get('protection_class', 5)
                return protection_class > min_class
            elif condition == 'high_cat_risk':
                cat_risk_score = property_data.get('catastrophe_risk_score', 3.0)
                return cat_risk_score > 7.0
            
            return False
            
        except Exception as e:
            logger.error(f"Surcharge condition evaluation failed: {e}")
            return False

    async def _calculate_surcharge_percentage(self, surcharge_rule: Dict[str, Any], 
                                            application_data: Dict[str, Any], 
                                            policy_type: str) -> float:
        """Calculate surcharge percentage based on severity"""
        
        base_percentage = surcharge_rule.get('percentage', 0.0)
        surcharge_type = surcharge_rule.get('name', '')
        
        try:
            if 'violations' in surcharge_type.lower():
                violations_count = len(application_data.get('applicant', {}).get('violations', []))
                return min(base_percentage * violations_count, surcharge_rule.get('max_surcharge', 0.4))
            elif 'accidents' in surcharge_type.lower():
                accidents = application_data.get('applicant', {}).get('accidents', [])
                at_fault_count = len([a for a in accidents if a.get('at_fault', False)])
                return min(base_percentage * at_fault_count, surcharge_rule.get('max_surcharge', 0.6))
            elif 'claims' in surcharge_type.lower():
                claims_count = len(application_data.get('applicant', {}).get('claims_history', []))
                return min(base_percentage * max(0, claims_count - 1), surcharge_rule.get('max_surcharge', 0.5))
            else:
                return base_percentage
                
        except Exception as e:
            logger.error(f"Surcharge percentage calculation failed: {e}")
            return base_percentage

    async def _get_surcharge_reason(self, surcharge_rule: Dict[str, Any], 
                                  application_data: Dict[str, Any], 
                                  policy_type: str) -> str:
        """Get detailed reason for surcharge"""
        
        surcharge_type = surcharge_rule.get('name', '')
        
        try:
            if 'young driver' in surcharge_type.lower():
                dob = application_data.get('applicant', {}).get('date_of_birth', '1990-01-01')
                age = (datetime.utcnow() - datetime.strptime(dob, '%Y-%m-%d')).days / 365.25
                return f"Driver age {age:.0f} is under 25"
            elif 'violations' in surcharge_type.lower():
                violations_count = len(application_data.get('applicant', {}).get('violations', []))
                return f"{violations_count} moving violations in driving record"
            elif 'accidents' in surcharge_type.lower():
                accidents = application_data.get('applicant', {}).get('accidents', [])
                at_fault_count = len([a for a in accidents if a.get('at_fault', False)])
                return f"{at_fault_count} at-fault accidents in driving record"
            elif 'credit' in surcharge_type.lower():
                credit_score = application_data.get('applicant', {}).get('credit_score', 700)
                return f"Credit score {credit_score} below acceptable threshold"
            elif 'mileage' in surcharge_type.lower():
                annual_mileage = application_data.get('applicant', {}).get('annual_mileage', 12000)
                return f"Annual mileage {annual_mileage:,} exceeds standard limits"
            elif 'claims' in surcharge_type.lower():
                claims_count = len(application_data.get('applicant', {}).get('claims_history', []))
                return f"{claims_count} claims in recent history"
            elif 'property age' in surcharge_type.lower():
                year_built = application_data.get('property', {}).get('year_built', 1990)
                property_age = datetime.utcnow().year - year_built
                return f"Property age {property_age} years increases risk"
            elif 'construction' in surcharge_type.lower():
                construction_type = application_data.get('property', {}).get('construction_type', 'frame')
                return f"Construction type {construction_type} has higher risk profile"
            elif 'protection' in surcharge_type.lower():
                protection_class = application_data.get('property', {}).get('protection_class', 5)
                return f"Fire protection class {protection_class} indicates limited fire protection"
            elif 'catastrophe' in surcharge_type.lower():
                cat_risk_score = application_data.get('property', {}).get('catastrophe_risk_score', 3.0)
                return f"High catastrophe exposure with risk score {cat_risk_score:.1f}"
            else:
                return "Risk factor identified during underwriting"
                
        except Exception as e:
            logger.error(f"Surcharge reason generation failed: {e}")
            return "Risk factor identified during underwriting"

    async def _create_policy_term(self, application_data: Dict[str, Any], 
                                premium_calculation: Dict[str, Any],
                                discounts: List[Discount],
                                surcharges: List[Surcharge]) -> PolicyTerm:
        """Create policy term with payment information"""
        
        # Calculate total premium
        base_premium = premium_calculation.get('total_adjusted_premium', 0.0)
        
        total_discount = sum(discount.amount for discount in discounts)
        total_surcharge = sum(surcharge.amount for surcharge in surcharges)
        
        total_premium = base_premium - total_discount + total_surcharge
        
        # Get payment frequency preference
        payment_frequency = PaymentFrequency(
            application_data.get('payment_frequency', 'monthly')
        )
        
        # Calculate payment amount based on frequency
        frequency_multipliers = {
            PaymentFrequency.ANNUAL: 1.0,
            PaymentFrequency.SEMI_ANNUAL: 0.51,  # Small fee for semi-annual
            PaymentFrequency.QUARTERLY: 0.26,   # Small fee for quarterly
            PaymentFrequency.MONTHLY: 0.09      # Fee for monthly payments
        }
        
        payment_multiplier = frequency_multipliers.get(payment_frequency, 0.09)
        payment_amount = total_premium * payment_multiplier
        
        # Set policy dates
        effective_date = datetime.strptime(
            application_data.get('effective_date', 
                               (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')), 
            '%Y-%m-%d'
        )
        
        expiration_date = effective_date + timedelta(days=365)
        
        policy_term = PolicyTerm(
            effective_date=effective_date,
            expiration_date=expiration_date,
            term_length_months=12,
            payment_frequency=payment_frequency,
            payment_amount=payment_amount,
            total_premium=total_premium
        )
        
        return policy_term

    async def _determine_policy_conditions(self, application_data: Dict[str, Any], 
                                         underwriting_result: Dict[str, Any],
                                         policy_type: str) -> List[str]:
        """Determine policy conditions based on underwriting results"""
        
        conditions = []
        
        # Add underwriting conditions
        underwriting_conditions = underwriting_result.get('conditions', [])
        conditions.extend(underwriting_conditions)
        
        # Add policy-specific conditions
        if policy_type == 'auto_personal':
            # Young driver conditions
            dob = application_data.get('applicant', {}).get('date_of_birth', '1990-01-01')
            age = (datetime.utcnow() - datetime.strptime(dob, '%Y-%m-%d')).days / 365.25
            
            if age < 21:
                conditions.append("Driver training course must be completed within 90 days of policy effective date")
            
            # High-risk driver conditions
            violations = len(application_data.get('applicant', {}).get('violations', []))
            if violations > 2:
                conditions.append("Defensive driving course required within 60 days")
                conditions.append("Policy subject to quarterly review")
            
            # Vehicle conditions
            vehicle_value = application_data.get('vehicle', {}).get('value', 25000)
            if vehicle_value > 75000:
                conditions.append("Agreed value coverage required for high-value vehicle")
        
        elif policy_type == 'homeowners':
            # Property age conditions
            year_built = application_data.get('property', {}).get('year_built', 1990)
            property_age = datetime.utcnow().year - year_built
            
            if property_age > 40:
                conditions.append("Property inspection required every 3 years")
            
            # Roof conditions
            roof_age = application_data.get('property', {}).get('roof_age', 10)
            if roof_age > 15:
                conditions.append("Roof inspection required within 30 days of policy effective date")
            
            # High-value property conditions
            dwelling_value = application_data.get('property', {}).get('dwelling_value', 300000)
            if dwelling_value > 750000:
                conditions.append("Annual property appraisal required")
                conditions.append("Replacement cost guarantee subject to annual review")
        
        return conditions

    async def _determine_policy_exclusions(self, application_data: Dict[str, Any], 
                                         policy_type: str) -> List[str]:
        """Determine policy exclusions"""
        
        exclusions = []
        
        # Standard exclusions by policy type
        if policy_type == 'auto_personal':
            exclusions.extend([
                "Racing or speed contests",
                "Commercial use (unless specifically covered)",
                "Intentional acts",
                "Nuclear hazard",
                "War or military action",
                "Damage from wear and tear, freezing, mechanical breakdown"
            ])
            
            # Additional exclusions based on risk factors
            vehicle_type = application_data.get('vehicle', {}).get('type', 'sedan')
            if vehicle_type == 'sports_car':
                exclusions.append("Track day or racing events excluded")
        
        elif policy_type == 'homeowners':
            exclusions.extend([
                "Flood damage (separate flood insurance required)",
                "Earthquake damage (separate earthquake insurance required)",
                "Nuclear hazard",
                "War or military action",
                "Ordinance or law coverage (unless specifically added)",
                "Business activities (unless specifically covered)",
                "Normal wear and tear, deterioration, or maintenance issues"
            ])
            
            # Location-specific exclusions
            catastrophe_exposure = application_data.get('property', {}).get('catastrophe_exposure', {})
            
            if 'hurricane' in catastrophe_exposure:
                exclusions.append("Named storm deductible applies to hurricane damage")
            
            if 'wildfire' in catastrophe_exposure:
                exclusions.append("Brush fire exclusion applies if defensible space not maintained")
        
        return exclusions

    async def _create_endorsements(self, application_data: Dict[str, Any], 
                                 underwriting_result: Dict[str, Any],
                                 policy_type: str) -> List[Dict[str, Any]]:
        """Create policy endorsements"""
        
        endorsements = []
        
        # Add endorsements based on coverage selections
        coverage_data = application_data.get('coverage', {})
        
        if policy_type == 'auto_personal':
            # Rental reimbursement endorsement
            if coverage_data.get('rental_reimbursement', {}).get('selected', False):
                endorsements.append({
                    'endorsement_id': str(uuid.uuid4()),
                    'name': 'Rental Reimbursement Coverage',
                    'description': 'Provides rental car coverage when covered vehicle is being repaired',
                    'premium': 25.0,
                    'effective_date': application_data.get('effective_date'),
                    'terms': ['$30 per day maximum', '30 day maximum period']
                })
            
            # Roadside assistance endorsement
            if coverage_data.get('roadside_assistance', {}).get('selected', False):
                endorsements.append({
                    'endorsement_id': str(uuid.uuid4()),
                    'name': 'Roadside Assistance Coverage',
                    'description': 'Provides 24/7 roadside assistance services',
                    'premium': 20.0,
                    'effective_date': application_data.get('effective_date'),
                    'terms': ['Towing, jump start, flat tire, lockout services', 'Subject to service limits']
                })
        
        elif policy_type == 'homeowners':
            # Personal property replacement cost endorsement
            if coverage_data.get('replacement_cost_personal_property', {}).get('selected', False):
                endorsements.append({
                    'endorsement_id': str(uuid.uuid4()),
                    'name': 'Personal Property Replacement Cost',
                    'description': 'Provides replacement cost coverage for personal property',
                    'premium': 50.0,
                    'effective_date': application_data.get('effective_date'),
                    'terms': ['Replaces actual cash value coverage', 'Subject to policy limits']
                })
            
            # Water backup endorsement
            if coverage_data.get('water_backup', {}).get('selected', False):
                endorsements.append({
                    'endorsement_id': str(uuid.uuid4()),
                    'name': 'Water Backup Coverage',
                    'description': 'Covers damage from sewer or drain backup',
                    'premium': 75.0,
                    'effective_date': application_data.get('effective_date'),
                    'terms': ['$10,000 limit', '$500 deductible', 'Gradual damage excluded']
                })
        
        return endorsements

    async def _store_policy(self, policy: Policy):
        """Store policy in database"""
        
        try:
            with self.Session() as session:
                policy_record = PolicyRecord(
                    policy_id=policy.policy_id,
                    policy_number=policy.policy_number,
                    application_id=policy.application_id,
                    policy_type=policy.policy_type,
                    status=policy.status.value,
                    policyholder_data=policy.policyholder,
                    coverages_data=[asdict(coverage) for coverage in policy.coverages],
                    policy_term_data=asdict(policy.policy_term),
                    discounts_data=[asdict(discount) for discount in policy.discounts],
                    surcharges_data=[asdict(surcharge) for surcharge in policy.surcharges],
                    conditions=policy.conditions,
                    exclusions=policy.exclusions,
                    endorsements=policy.endorsements,
                    total_premium=policy.policy_term.total_premium,
                    effective_date=policy.policy_term.effective_date,
                    expiration_date=policy.policy_term.expiration_date,
                    created_at=policy.created_at,
                    updated_at=policy.updated_at
                )
                
                session.add(policy_record)
                session.commit()
                
                # Cache in Redis
                cache_key = f"policy:{policy.policy_id}"
                cache_data = {
                    'policy_number': policy.policy_number,
                    'status': policy.status.value,
                    'total_premium': policy.policy_term.total_premium,
                    'effective_date': policy.policy_term.effective_date.isoformat(),
                    'expiration_date': policy.policy_term.expiration_date.isoformat()
                }
                
                self.redis_client.setex(
                    cache_key,
                    timedelta(hours=24),
                    json.dumps(cache_data)
                )
                
                # Update active policies gauge
                if policy.status == PolicyStatus.ACTIVE:
                    active_policies_gauge.inc()
                
        except Exception as e:
            logger.error(f"Error storing policy: {e}")
            raise

    async def bind_policy(self, policy_id: str) -> Policy:
        """Bind a quoted policy"""
        
        try:
            with self.Session() as session:
                policy_record = session.query(PolicyRecord).filter_by(policy_id=policy_id).first()
                
                if not policy_record:
                    raise ValueError(f"Policy {policy_id} not found")
                
                if policy_record.status != PolicyStatus.QUOTED.value:
                    raise ValueError(f"Policy {policy_id} is not in quoted status")
                
                # Update status to bound
                policy_record.status = PolicyStatus.BOUND.value
                policy_record.updated_at = datetime.utcnow()
                
                session.commit()
                
                # Update cache
                cache_key = f"policy:{policy_id}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    cache_data = json.loads(cached_data)
                    cache_data['status'] = PolicyStatus.BOUND.value
                    self.redis_client.setex(
                        cache_key,
                        timedelta(hours=24),
                        json.dumps(cache_data)
                    )
                
                # Convert back to Policy object
                policy = await self._convert_record_to_policy(policy_record)
                
                return policy
                
        except Exception as e:
            logger.error(f"Policy binding failed: {e}")
            raise

    async def _convert_record_to_policy(self, record: PolicyRecord) -> Policy:
        """Convert database record to Policy object"""
        
        # Convert coverages
        coverages = []
        for coverage_data in record.coverages_data:
            coverage = Coverage(
                coverage_id=coverage_data['coverage_id'],
                coverage_type=CoverageType(coverage_data['coverage_type']),
                limit=coverage_data['limit'],
                deductible=coverage_data['deductible'],
                premium=coverage_data['premium'],
                description=coverage_data['description'],
                is_required=coverage_data['is_required'],
                conditions=coverage_data['conditions'],
                exclusions=coverage_data['exclusions']
            )
            coverages.append(coverage)
        
        # Convert policy term
        policy_term_data = record.policy_term_data
        policy_term = PolicyTerm(
            effective_date=datetime.fromisoformat(policy_term_data['effective_date']),
            expiration_date=datetime.fromisoformat(policy_term_data['expiration_date']),
            term_length_months=policy_term_data['term_length_months'],
            payment_frequency=PaymentFrequency(policy_term_data['payment_frequency']),
            payment_amount=policy_term_data['payment_amount'],
            total_premium=policy_term_data['total_premium']
        )
        
        # Convert discounts
        discounts = []
        for discount_data in record.discounts_data:
            discount = Discount(
                discount_id=discount_data['discount_id'],
                name=discount_data['name'],
                description=discount_data['description'],
                discount_type=discount_data['discount_type'],
                amount=discount_data['amount'],
                percentage=discount_data['percentage'],
                conditions=discount_data['conditions'],
                applied_to=discount_data['applied_to']
            )
            discounts.append(discount)
        
        # Convert surcharges
        surcharges = []
        for surcharge_data in record.surcharges_data:
            surcharge = Surcharge(
                surcharge_id=surcharge_data['surcharge_id'],
                name=surcharge_data['name'],
                description=surcharge_data['description'],
                surcharge_type=surcharge_data['surcharge_type'],
                amount=surcharge_data['amount'],
                percentage=surcharge_data['percentage'],
                reason=surcharge_data['reason'],
                applied_to=surcharge_data['applied_to']
            )
            surcharges.append(surcharge)
        
        policy = Policy(
            policy_id=record.policy_id,
            policy_number=record.policy_number,
            application_id=record.application_id,
            policy_type=record.policy_type,
            status=PolicyStatus(record.status),
            policyholder=record.policyholder_data,
            coverages=coverages,
            policy_term=policy_term,
            discounts=discounts,
            surcharges=surcharges,
            conditions=record.conditions,
            exclusions=record.exclusions,
            endorsements=record.endorsements,
            created_at=record.created_at,
            updated_at=record.updated_at
        )
        
        return policy

    async def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Retrieve policy by ID"""
        
        try:
            # Try cache first
            cache_key = f"policy:{policy_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                # For full policy data, still need to query database
                pass
            
            # Query database
            with self.Session() as session:
                record = session.query(PolicyRecord).filter_by(policy_id=policy_id).first()
                
                if record:
                    return await self._convert_record_to_policy(record)
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving policy: {e}")
            return None

    async def update_policy_status(self, policy_id: str, new_status: PolicyStatus) -> bool:
        """Update policy status"""
        
        try:
            with self.Session() as session:
                policy_record = session.query(PolicyRecord).filter_by(policy_id=policy_id).first()
                
                if not policy_record:
                    return False
                
                old_status = PolicyStatus(policy_record.status)
                policy_record.status = new_status.value
                policy_record.updated_at = datetime.utcnow()
                
                session.commit()
                
                # Update metrics
                if old_status == PolicyStatus.ACTIVE and new_status != PolicyStatus.ACTIVE:
                    active_policies_gauge.dec()
                elif old_status != PolicyStatus.ACTIVE and new_status == PolicyStatus.ACTIVE:
                    active_policies_gauge.inc()
                
                return True
                
        except Exception as e:
            logger.error(f"Policy status update failed: {e}")
            return False

