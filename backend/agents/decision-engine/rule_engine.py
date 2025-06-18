"""
Rule Engine - Production Ready Implementation
Business rules engine for insurance decision making
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import re
import operator
from functools import reduce
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Monitoring
from prometheus_client import Counter, Histogram

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
rules_executed_total = Counter('rules_executed_total', 'Total rules executed', ['rule_type', 'outcome'])
rule_execution_duration = Histogram('rule_execution_duration_seconds', 'Time to execute rules')

Base = declarative_base()

class RuleType(Enum):
    VALIDATION = "validation"
    ELIGIBILITY = "eligibility"
    PRICING = "pricing"
    UNDERWRITING = "underwriting"
    CLAIMS = "claims"
    COMPLIANCE = "compliance"
    FRAUD_DETECTION = "fraud_detection"

class OperatorType(Enum):
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"

class LogicalOperator(Enum):
    AND = "and"
    OR = "or"
    NOT = "not"

class ActionType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REFER = "refer"
    SET_VALUE = "set_value"
    ADD_FLAG = "add_flag"
    CALCULATE = "calculate"
    SEND_NOTIFICATION = "send_notification"
    LOG_EVENT = "log_event"

@dataclass
class RuleCondition:
    """Represents a single rule condition"""
    field: str
    operator: OperatorType
    value: Any
    data_type: str = "string"  # string, number, boolean, date, list
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate the condition against data"""
        
        field_value = self._get_field_value(data, self.field)
        
        # Handle null checks first
        if self.operator == OperatorType.IS_NULL:
            return field_value is None
        elif self.operator == OperatorType.IS_NOT_NULL:
            return field_value is not None
        
        # If field is None and we're not checking for null, return False
        if field_value is None:
            return False
        
        # Convert values to appropriate types
        field_value = self._convert_value(field_value, self.data_type)
        comparison_value = self._convert_value(self.value, self.data_type)
        
        # Perform comparison
        try:
            if self.operator == OperatorType.EQUALS:
                return field_value == comparison_value
            elif self.operator == OperatorType.NOT_EQUALS:
                return field_value != comparison_value
            elif self.operator == OperatorType.GREATER_THAN:
                return field_value > comparison_value
            elif self.operator == OperatorType.GREATER_EQUAL:
                return field_value >= comparison_value
            elif self.operator == OperatorType.LESS_THAN:
                return field_value < comparison_value
            elif self.operator == OperatorType.LESS_EQUAL:
                return field_value <= comparison_value
            elif self.operator == OperatorType.IN:
                return field_value in comparison_value
            elif self.operator == OperatorType.NOT_IN:
                return field_value not in comparison_value
            elif self.operator == OperatorType.CONTAINS:
                return str(comparison_value) in str(field_value)
            elif self.operator == OperatorType.NOT_CONTAINS:
                return str(comparison_value) not in str(field_value)
            elif self.operator == OperatorType.STARTS_WITH:
                return str(field_value).startswith(str(comparison_value))
            elif self.operator == OperatorType.ENDS_WITH:
                return str(field_value).endswith(str(comparison_value))
            elif self.operator == OperatorType.REGEX:
                return bool(re.match(str(comparison_value), str(field_value)))
            elif self.operator == OperatorType.BETWEEN:
                if isinstance(comparison_value, (list, tuple)) and len(comparison_value) == 2:
                    return comparison_value[0] <= field_value <= comparison_value[1]
                return False
            else:
                logger.warning(f"Unknown operator: {self.operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition {self.field} {self.operator.value} {self.value}: {e}")
            return False
    
    def _get_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get field value from nested data using dot notation"""
        
        try:
            keys = field_path.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list) and key.isdigit():
                    index = int(key)
                    value = value[index] if 0 <= index < len(value) else None
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    def _convert_value(self, value: Any, data_type: str) -> Any:
        """Convert value to appropriate type"""
        
        if value is None:
            return None
        
        try:
            if data_type == "number":
                return float(value) if not isinstance(value, (int, float)) else value
            elif data_type == "boolean":
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                else:
                    return bool(value)
            elif data_type == "date":
                if isinstance(value, datetime):
                    return value
                elif isinstance(value, str):
                    # Try to parse ISO format
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                else:
                    return value
            elif data_type == "list":
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    # Try to parse JSON array
                    try:
                        parsed = json.loads(value)
                        return parsed if isinstance(parsed, list) else [parsed]
                    except:
                        return [value]
                else:
                    return [value]
            else:  # string
                return str(value)
                
        except Exception as e:
            logger.warning(f"Error converting value {value} to {data_type}: {e}")
            return value

@dataclass
class RuleAction:
    """Represents a rule action"""
    action_type: ActionType
    parameters: Dict[str, Any]
    
    def execute(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action"""
        
        result = {"action_executed": True, "action_type": self.action_type.value}
        
        try:
            if self.action_type == ActionType.APPROVE:
                result["decision"] = "approved"
                result["message"] = self.parameters.get("message", "Request approved")
                
            elif self.action_type == ActionType.REJECT:
                result["decision"] = "rejected"
                result["message"] = self.parameters.get("message", "Request rejected")
                result["reason"] = self.parameters.get("reason", "Rule violation")
                
            elif self.action_type == ActionType.REFER:
                result["decision"] = "referred"
                result["message"] = self.parameters.get("message", "Request referred for manual review")
                result["department"] = self.parameters.get("department", "underwriting")
                
            elif self.action_type == ActionType.SET_VALUE:
                field = self.parameters.get("field")
                value = self.parameters.get("value")
                if field:
                    self._set_field_value(data, field, value)
                    result["field_set"] = field
                    result["value"] = value
                    
            elif self.action_type == ActionType.ADD_FLAG:
                flag = self.parameters.get("flag")
                if flag:
                    if "flags" not in data:
                        data["flags"] = []
                    if flag not in data["flags"]:
                        data["flags"].append(flag)
                    result["flag_added"] = flag
                    
            elif self.action_type == ActionType.CALCULATE:
                formula = self.parameters.get("formula")
                target_field = self.parameters.get("target_field")
                if formula and target_field:
                    calculated_value = self._evaluate_formula(formula, data)
                    self._set_field_value(data, target_field, calculated_value)
                    result["calculated_value"] = calculated_value
                    result["target_field"] = target_field
                    
            elif self.action_type == ActionType.SEND_NOTIFICATION:
                result["notification"] = {
                    "recipient": self.parameters.get("recipient"),
                    "template": self.parameters.get("template"),
                    "data": self.parameters.get("data", {})
                }
                
            elif self.action_type == ActionType.LOG_EVENT:
                result["log_event"] = {
                    "event_type": self.parameters.get("event_type"),
                    "message": self.parameters.get("message"),
                    "severity": self.parameters.get("severity", "info")
                }
                
        except Exception as e:
            logger.error(f"Error executing action {self.action_type.value}: {e}")
            result["error"] = str(e)
            result["action_executed"] = False
        
        return result
    
    def _set_field_value(self, data: Dict[str, Any], field_path: str, value: Any):
        """Set field value in nested data using dot notation"""
        
        keys = field_path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _evaluate_formula(self, formula: str, data: Dict[str, Any]) -> float:
        """Evaluate mathematical formula with data substitution"""
        
        # Simple formula evaluation - replace field references with values
        # Format: ${field.name} gets replaced with actual values
        
        import re
        
        def replace_field(match):
            field_name = match.group(1)
            field_value = self._get_field_value(data, field_name)
            return str(field_value) if field_value is not None else "0"
        
        # Replace field references
        processed_formula = re.sub(r'\$\{([^}]+)\}', replace_field, formula)
        
        # Safe evaluation of mathematical expressions
        allowed_names = {
            "__builtins__": {},
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "sum": sum,
            "len": len
        }
        
        try:
            result = eval(processed_formula, allowed_names)
            return float(result)
        except Exception as e:
            logger.error(f"Error evaluating formula {formula}: {e}")
            return 0.0
    
    def _get_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get field value from nested data"""
        
        try:
            keys = field_path.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return None
            
            return value
            
        except Exception:
            return None

@dataclass
class BusinessRule:
    """Represents a complete business rule"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    priority: int
    is_active: bool
    conditions: List[RuleCondition]
    logical_operator: LogicalOperator
    actions: List[RuleAction]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    def evaluate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate the rule against data"""
        
        if context is None:
            context = {}
        
        start_time = datetime.utcnow()
        
        try:
            with rule_execution_duration.time():
                # Check if rule is active
                if not self.is_active:
                    return {
                        "rule_id": self.rule_id,
                        "matched": False,
                        "reason": "Rule is inactive",
                        "actions_executed": []
                    }
                
                # Evaluate conditions
                condition_results = []
                for condition in self.conditions:
                    result = condition.evaluate(data)
                    condition_results.append(result)
                
                # Apply logical operator
                if self.logical_operator == LogicalOperator.AND:
                    rule_matched = all(condition_results)
                elif self.logical_operator == LogicalOperator.OR:
                    rule_matched = any(condition_results)
                elif self.logical_operator == LogicalOperator.NOT:
                    rule_matched = not any(condition_results)
                else:
                    rule_matched = all(condition_results)  # Default to AND
                
                result = {
                    "rule_id": self.rule_id,
                    "rule_name": self.name,
                    "rule_type": self.rule_type.value,
                    "priority": self.priority,
                    "matched": rule_matched,
                    "condition_results": condition_results,
                    "logical_operator": self.logical_operator.value,
                    "actions_executed": [],
                    "processing_time": 0.0
                }
                
                # Execute actions if rule matched
                if rule_matched:
                    for action in self.actions:
                        action_result = action.execute(data, context)
                        result["actions_executed"].append(action_result)
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                result["processing_time"] = processing_time
                
                # Update metrics
                outcome = "matched" if rule_matched else "not_matched"
                rules_executed_total.labels(rule_type=self.rule_type.value, outcome=outcome).inc()
                
                return result
                
        except Exception as e:
            logger.error(f"Error evaluating rule {self.rule_id}: {e}")
            return {
                "rule_id": self.rule_id,
                "matched": False,
                "error": str(e),
                "actions_executed": []
            }

class BusinessRuleModel(Base):
    """SQLAlchemy model for persisting business rules"""
    __tablename__ = 'business_rules'
    
    rule_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    rule_type = Column(String, nullable=False)
    priority = Column(Integer, default=5)
    is_active = Column(Boolean, default=True)
    conditions = Column(JSON)
    logical_operator = Column(String, default="and")
    actions = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class RuleEngine:
    """
    Production-ready Business Rules Engine
    Executes business rules for insurance operations
    """
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis setup
        self.redis_client = redis.from_url(redis_url)
        
        # Rule cache
        self.rule_cache: Dict[str, BusinessRule] = {}
        self.rules_by_type: Dict[RuleType, List[BusinessRule]] = {}
        
        # Load rules
        self._load_rules()
        
        logger.info("RuleEngine initialized successfully")

    def _load_rules(self):
        """Load rules from database"""
        
        try:
            with self.Session() as session:
                rule_models = session.query(BusinessRuleModel).filter(
                    BusinessRuleModel.is_active == True
                ).order_by(BusinessRuleModel.priority.desc()).all()
                
                self.rule_cache.clear()
                self.rules_by_type.clear()
                
                for rule_model in rule_models:
                    rule = self._model_to_rule(rule_model)
                    self.rule_cache[rule.rule_id] = rule
                    
                    if rule.rule_type not in self.rules_by_type:
                        self.rules_by_type[rule.rule_type] = []
                    self.rules_by_type[rule.rule_type].append(rule)
                
                logger.info(f"Loaded {len(self.rule_cache)} business rules")
                
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            self._load_default_rules()

    def _model_to_rule(self, model: BusinessRuleModel) -> BusinessRule:
        """Convert database model to BusinessRule object"""
        
        # Convert conditions
        conditions = []
        for cond_data in model.conditions or []:
            condition = RuleCondition(
                field=cond_data["field"],
                operator=OperatorType(cond_data["operator"]),
                value=cond_data["value"],
                data_type=cond_data.get("data_type", "string")
            )
            conditions.append(condition)
        
        # Convert actions
        actions = []
        for action_data in model.actions or []:
            action = RuleAction(
                action_type=ActionType(action_data["action_type"]),
                parameters=action_data.get("parameters", {})
            )
            actions.append(action)
        
        return BusinessRule(
            rule_id=model.rule_id,
            name=model.name,
            description=model.description or "",
            rule_type=RuleType(model.rule_type),
            priority=model.priority,
            is_active=model.is_active,
            conditions=conditions,
            logical_operator=LogicalOperator(model.logical_operator),
            actions=actions,
            metadata=model.metadata or {},
            created_at=model.created_at,
            updated_at=model.updated_at
        )

    def _load_default_rules(self):
        """Load default rules if database is empty"""
        
        default_rules = [
            {
                "rule_id": "underwriting_age_check",
                "name": "Age Eligibility Check",
                "description": "Check if applicant age is within acceptable range",
                "rule_type": "underwriting",
                "priority": 10,
                "conditions": [
                    {"field": "age", "operator": "gte", "value": 18, "data_type": "number"},
                    {"field": "age", "operator": "lte", "value": 75, "data_type": "number"}
                ],
                "logical_operator": "and",
                "actions": [
                    {"action_type": "add_flag", "parameters": {"flag": "age_eligible"}}
                ]
            },
            {
                "rule_id": "underwriting_income_check",
                "name": "Minimum Income Check",
                "description": "Check if applicant meets minimum income requirement",
                "rule_type": "underwriting",
                "priority": 9,
                "conditions": [
                    {"field": "annual_income", "operator": "gte", "value": 25000, "data_type": "number"}
                ],
                "logical_operator": "and",
                "actions": [
                    {"action_type": "add_flag", "parameters": {"flag": "income_eligible"}}
                ]
            },
            {
                "rule_id": "claims_policy_active",
                "name": "Active Policy Check",
                "description": "Verify policy is active for claims processing",
                "rule_type": "claims",
                "priority": 10,
                "conditions": [
                    {"field": "policy_status", "operator": "eq", "value": "active", "data_type": "string"}
                ],
                "logical_operator": "and",
                "actions": [
                    {"action_type": "add_flag", "parameters": {"flag": "policy_active"}}
                ]
            },
            {
                "rule_id": "fraud_multiple_claims",
                "name": "Multiple Claims Detection",
                "description": "Flag potential fraud for multiple claims",
                "rule_type": "fraud_detection",
                "priority": 8,
                "conditions": [
                    {"field": "claims_last_year", "operator": "gt", "value": 3, "data_type": "number"}
                ],
                "logical_operator": "and",
                "actions": [
                    {"action_type": "add_flag", "parameters": {"flag": "potential_fraud"}},
                    {"action_type": "refer", "parameters": {"department": "fraud_investigation", "message": "Multiple claims detected"}}
                ]
            },
            {
                "rule_id": "pricing_high_risk_adjustment",
                "name": "High Risk Pricing Adjustment",
                "description": "Apply pricing adjustment for high-risk applicants",
                "rule_type": "pricing",
                "priority": 7,
                "conditions": [
                    {"field": "risk_score", "operator": "gte", "value": 8, "data_type": "number"}
                ],
                "logical_operator": "and",
                "actions": [
                    {"action_type": "calculate", "parameters": {"formula": "${base_premium} * 1.5", "target_field": "adjusted_premium"}},
                    {"action_type": "add_flag", "parameters": {"flag": "high_risk_pricing"}}
                ]
            }
        ]
        
        for rule_data in default_rules:
            self.create_rule(rule_data)
        
        logger.info(f"Created {len(default_rules)} default rules")

    def execute_rules(self, rule_type: RuleType, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute all rules of a specific type"""
        
        if context is None:
            context = {}
        
        start_time = datetime.utcnow()
        
        # Get rules for this type
        rules = self.rules_by_type.get(rule_type, [])
        
        # Sort by priority (highest first)
        rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        
        results = {
            "rule_type": rule_type.value,
            "total_rules": len(rules),
            "rules_matched": 0,
            "rules_executed": [],
            "final_decision": None,
            "flags": [],
            "processing_time": 0.0,
            "data_modifications": {}
        }
        
        # Track data modifications
        original_data = json.loads(json.dumps(data))  # Deep copy
        
        try:
            for rule in rules:
                rule_result = rule.evaluate(data, context)
                results["rules_executed"].append(rule_result)
                
                if rule_result["matched"]:
                    results["rules_matched"] += 1
                    
                    # Process actions
                    for action_result in rule_result.get("actions_executed", []):
                        if action_result.get("decision"):
                            if not results["final_decision"]:  # First decision wins
                                results["final_decision"] = action_result["decision"]
                        
                        if action_result.get("flag_added"):
                            results["flags"].append(action_result["flag_added"])
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            results["processing_time"] = processing_time
            
            # Track data modifications
            for key, value in data.items():
                if key not in original_data or original_data[key] != value:
                    results["data_modifications"][key] = {
                        "original": original_data.get(key),
                        "modified": value
                    }
            
            logger.info(f"Executed {len(rules)} rules of type {rule_type.value}, {results['rules_matched']} matched")
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing rules for type {rule_type.value}: {e}")
            results["error"] = str(e)
            return results

    def execute_single_rule(self, rule_id: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single rule by ID"""
        
        if rule_id not in self.rule_cache:
            return {
                "error": f"Rule {rule_id} not found",
                "rule_id": rule_id,
                "matched": False
            }
        
        rule = self.rule_cache[rule_id]
        return rule.evaluate(data, context)

    def create_rule(self, rule_data: Dict[str, Any]) -> str:
        """Create a new business rule"""
        
        try:
            rule_id = rule_data.get("rule_id", str(uuid.uuid4()))
            
            # Convert to database model
            model = BusinessRuleModel(
                rule_id=rule_id,
                name=rule_data["name"],
                description=rule_data.get("description", ""),
                rule_type=rule_data["rule_type"],
                priority=rule_data.get("priority", 5),
                is_active=rule_data.get("is_active", True),
                conditions=rule_data.get("conditions", []),
                logical_operator=rule_data.get("logical_operator", "and"),
                actions=rule_data.get("actions", []),
                metadata=rule_data.get("metadata", {}),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Save to database
            with self.Session() as session:
                session.merge(model)
                session.commit()
            
            # Reload rules to update cache
            self._load_rules()
            
            logger.info(f"Created rule {rule_id}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Error creating rule: {e}")
            raise

    def update_rule(self, rule_id: str, rule_data: Dict[str, Any]) -> bool:
        """Update an existing business rule"""
        
        try:
            with self.Session() as session:
                model = session.query(BusinessRuleModel).filter(
                    BusinessRuleModel.rule_id == rule_id
                ).first()
                
                if not model:
                    return False
                
                # Update fields
                for field, value in rule_data.items():
                    if hasattr(model, field) and field != "rule_id":
                        setattr(model, field, value)
                
                model.updated_at = datetime.utcnow()
                session.commit()
            
            # Reload rules to update cache
            self._load_rules()
            
            logger.info(f"Updated rule {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating rule {rule_id}: {e}")
            return False

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a business rule"""
        
        try:
            with self.Session() as session:
                model = session.query(BusinessRuleModel).filter(
                    BusinessRuleModel.rule_id == rule_id
                ).first()
                
                if not model:
                    return False
                
                session.delete(model)
                session.commit()
            
            # Reload rules to update cache
            self._load_rules()
            
            logger.info(f"Deleted rule {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting rule {rule_id}: {e}")
            return False

    def get_rules(self, rule_type: RuleType = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get rules with optional filtering"""
        
        try:
            with self.Session() as session:
                query = session.query(BusinessRuleModel)
                
                if rule_type:
                    query = query.filter(BusinessRuleModel.rule_type == rule_type.value)
                
                if active_only:
                    query = query.filter(BusinessRuleModel.is_active == True)
                
                rules = query.order_by(BusinessRuleModel.priority.desc()).all()
                
                result = []
                for rule in rules:
                    result.append({
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "description": rule.description,
                        "rule_type": rule.rule_type,
                        "priority": rule.priority,
                        "is_active": rule.is_active,
                        "conditions": rule.conditions,
                        "logical_operator": rule.logical_operator,
                        "actions": rule.actions,
                        "metadata": rule.metadata,
                        "created_at": rule.created_at.isoformat(),
                        "updated_at": rule.updated_at.isoformat()
                    })
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting rules: {e}")
            return []

    def validate_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate rule structure and syntax"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Required fields
        required_fields = ["name", "rule_type", "conditions", "actions"]
        for field in required_fields:
            if field not in rule_data:
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Validate rule type
        if "rule_type" in rule_data:
            try:
                RuleType(rule_data["rule_type"])
            except ValueError:
                validation_result["errors"].append(f"Invalid rule_type: {rule_data['rule_type']}")
        
        # Validate conditions
        if "conditions" in rule_data:
            for i, condition in enumerate(rule_data["conditions"]):
                if not isinstance(condition, dict):
                    validation_result["errors"].append(f"Condition {i} must be a dictionary")
                    continue
                
                required_condition_fields = ["field", "operator", "value"]
                for field in required_condition_fields:
                    if field not in condition:
                        validation_result["errors"].append(f"Condition {i} missing field: {field}")
                
                if "operator" in condition:
                    try:
                        OperatorType(condition["operator"])
                    except ValueError:
                        validation_result["errors"].append(f"Condition {i} invalid operator: {condition['operator']}")
        
        # Validate actions
        if "actions" in rule_data:
            for i, action in enumerate(rule_data["actions"]):
                if not isinstance(action, dict):
                    validation_result["errors"].append(f"Action {i} must be a dictionary")
                    continue
                
                if "action_type" not in action:
                    validation_result["errors"].append(f"Action {i} missing action_type")
                else:
                    try:
                        ActionType(action["action_type"])
                    except ValueError:
                        validation_result["errors"].append(f"Action {i} invalid action_type: {action['action_type']}")
        
        # Validate logical operator
        if "logical_operator" in rule_data:
            try:
                LogicalOperator(rule_data["logical_operator"])
            except ValueError:
                validation_result["errors"].append(f"Invalid logical_operator: {rule_data['logical_operator']}")
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        return validation_result

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule engine statistics"""
        
        try:
            with self.Session() as session:
                total_rules = session.query(BusinessRuleModel).count()
                active_rules = session.query(BusinessRuleModel).filter(
                    BusinessRuleModel.is_active == True
                ).count()
                
                # Rules by type
                type_stats = {}
                for rule_type in RuleType:
                    count = session.query(BusinessRuleModel).filter(
                        BusinessRuleModel.rule_type == rule_type.value,
                        BusinessRuleModel.is_active == True
                    ).count()
                    type_stats[rule_type.value] = count
                
                return {
                    "total_rules": total_rules,
                    "active_rules": active_rules,
                    "inactive_rules": total_rules - active_rules,
                    "rules_by_type": type_stats,
                    "cached_rules": len(self.rule_cache)
                }
                
        except Exception as e:
            logger.error(f"Error getting rule statistics: {e}")
            return {}

    def reload_rules(self):
        """Reload rules from database"""
        
        self._load_rules()
        logger.info("Rules reloaded from database")

# Factory function
def create_rule_engine(db_url: str = None, redis_url: str = None) -> RuleEngine:
    """Create and configure a RuleEngine instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return RuleEngine(db_url=db_url, redis_url=redis_url)

# Example usage
if __name__ == "__main__":
    def test_rule_engine():
        """Test the rule engine functionality"""
        
        engine = create_rule_engine()
        
        # Test data
        test_data = {
            "age": 35,
            "annual_income": 75000,
            "credit_score": 720,
            "policy_status": "active",
            "claims_last_year": 1,
            "risk_score": 6
        }
        
        # Execute underwriting rules
        result = engine.execute_rules(RuleType.UNDERWRITING, test_data)
        
        print(f"Underwriting Rules Result:")
        print(f"Total rules: {result['total_rules']}")
        print(f"Rules matched: {result['rules_matched']}")
        print(f"Final decision: {result['final_decision']}")
        print(f"Flags: {result['flags']}")
        
        # Get statistics
        stats = engine.get_rule_statistics()
        print(f"Rule Statistics: {stats}")
    
    # Run test
    # test_rule_engine()

