"""
Liability Assessment Agent
Risk assessment and liability calculation for insurance operations
"""

from importlib import import_module

LiabilityAssessor = import_module('backend.agents.liability-assessment.liability_assessor').LiabilityAssessor
create_liability_assessor = import_module('backend.agents.liability-assessment.liability_assessor').create_liability_assessor
RiskCalculator = import_module('backend.agents.liability-assessment.risk_calculator').RiskCalculator
ActuarialEngine = import_module('backend.agents.liability-assessment.actuarial_engine').ActuarialEngine

__all__ = [
    'LiabilityAssessor',
    'create_liability_assessor',
    'RiskCalculator',
    'ActuarialEngine'
]

