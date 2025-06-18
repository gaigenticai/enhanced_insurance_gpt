"""
Decision Engine Agent
AI-powered decision making for insurance operations
"""

from importlib import import_module

DecisionEngine = import_module('backend.agents.decision-engine.decision_engine').DecisionEngine
RuleEngine = import_module('backend.agents.decision-engine.rule_engine').RuleEngine
MLModelManager = import_module('backend.agents.decision-engine.ml_models').MLModelManager
DecisionTreeProcessor = import_module('backend.agents.decision-engine.decision_tree').DecisionTreeProcessor

__all__ = [
    'DecisionEngine',
    'RuleEngine', 
    'MLModelManager',
    'DecisionTreeProcessor'
]

