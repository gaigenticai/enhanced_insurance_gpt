"""
Decision Tree Processor - Production Ready Implementation
Advanced decision tree processing for insurance operations
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import graphviz
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
decision_trees_executed_total = Counter('decision_trees_executed_total', 'Total decision trees executed', ['tree_type'])
decision_tree_execution_duration = Histogram('decision_tree_execution_duration_seconds', 'Time to execute decision trees')

Base = declarative_base()

class DecisionTreeType(Enum):
    UNDERWRITING = "underwriting"
    CLAIMS = "claims"
    PRICING = "pricing"
    RISK_ASSESSMENT = "risk_assessment"
    FRAUD_DETECTION = "fraud_detection"
    ELIGIBILITY = "eligibility"
    COVERAGE_RECOMMENDATION = "coverage_recommendation"

class NodeType(Enum):
    CONDITION = "condition"
    ACTION = "action"
    CALCULATION = "calculation"
    REFERENCE = "reference"

class OperatorType(Enum):
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    CONTAINS = "contains"

@dataclass
class DecisionNode:
    """Represents a node in the decision tree"""
    node_id: str
    node_type: NodeType
    name: str
    description: str
    condition: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None
    calculation: Optional[Dict[str, Any]] = None
    children: List[str] = None
    parent: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DecisionTreeResult:
    """Result of decision tree execution"""
    tree_id: str
    execution_id: str
    path_taken: List[str]
    final_node: str
    final_action: Dict[str, Any]
    calculations: Dict[str, Any]
    processing_time: float
    input_data: Dict[str, Any]
    timestamp: datetime

class DecisionTreeModel(Base):
    """SQLAlchemy model for decision trees"""
    __tablename__ = 'decision_trees'
    
    tree_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    tree_type = Column(String, nullable=False)
    version = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    tree_structure = Column(JSON)
    root_node_id = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    metadata = Column(JSON)

class DecisionTreeExecutionModel(Base):
    """SQLAlchemy model for decision tree executions"""
    __tablename__ = 'decision_tree_executions'
    
    execution_id = Column(String, primary_key=True)
    tree_id = Column(String, nullable=False)
    path_taken = Column(JSON)
    final_node = Column(String)
    final_action = Column(JSON)
    calculations = Column(JSON)
    processing_time = Column(Float)
    input_data = Column(JSON)
    executed_at = Column(DateTime, nullable=False)

class DecisionTreeProcessor:
    """
    Production-ready Decision Tree Processor
    Executes complex decision trees for insurance operations
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
        
        # Tree cache
        self.tree_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load active trees
        self._load_active_trees()
        
        logger.info("DecisionTreeProcessor initialized successfully")

    def _load_active_trees(self):
        """Load active decision trees from database"""
        
        try:
            with self.Session() as session:
                trees = session.query(DecisionTreeModel).filter(
                    DecisionTreeModel.is_active == True
                ).all()
                
                self.tree_cache.clear()
                
                for tree in trees:
                    self.tree_cache[tree.tree_id] = {
                        "name": tree.name,
                        "description": tree.description,
                        "tree_type": tree.tree_type,
                        "version": tree.version,
                        "structure": tree.tree_structure,
                        "root_node_id": tree.root_node_id,
                        "metadata": tree.metadata or {}
                    }
                
                logger.info(f"Loaded {len(self.tree_cache)} active decision trees")
                
        except Exception as e:
            logger.error(f"Error loading decision trees: {e}")
            self._create_default_trees()

    def _create_default_trees(self):
        """Create default decision trees"""
        
        # Underwriting decision tree
        underwriting_tree = {
            "tree_id": "underwriting_basic",
            "name": "Basic Underwriting Decision Tree",
            "description": "Standard underwriting decision process",
            "tree_type": DecisionTreeType.UNDERWRITING.value,
            "version": "1.0",
            "nodes": {
                "root": {
                    "node_id": "root",
                    "node_type": NodeType.CONDITION.value,
                    "name": "Age Check",
                    "description": "Check if applicant age is within acceptable range",
                    "condition": {
                        "field": "age",
                        "operator": OperatorType.BETWEEN.value,
                        "value": [18, 75]
                    },
                    "children": ["age_pass", "age_fail"]
                },
                "age_pass": {
                    "node_id": "age_pass",
                    "node_type": NodeType.CONDITION.value,
                    "name": "Income Check",
                    "description": "Check minimum income requirement",
                    "condition": {
                        "field": "annual_income",
                        "operator": OperatorType.GREATER_EQUAL.value,
                        "value": 25000
                    },
                    "children": ["income_pass", "income_fail"]
                },
                "age_fail": {
                    "node_id": "age_fail",
                    "node_type": NodeType.ACTION.value,
                    "name": "Reject - Age",
                    "description": "Reject application due to age",
                    "action": {
                        "decision": "reject",
                        "reason": "Age outside acceptable range",
                        "code": "AGE_REJECT"
                    }
                },
                "income_pass": {
                    "node_id": "income_pass",
                    "node_type": NodeType.CONDITION.value,
                    "name": "Credit Score Check",
                    "description": "Check credit score requirement",
                    "condition": {
                        "field": "credit_score",
                        "operator": OperatorType.GREATER_EQUAL.value,
                        "value": 650
                    },
                    "children": ["credit_pass", "credit_refer"]
                },
                "income_fail": {
                    "node_id": "income_fail",
                    "node_type": NodeType.ACTION.value,
                    "name": "Reject - Income",
                    "description": "Reject application due to insufficient income",
                    "action": {
                        "decision": "reject",
                        "reason": "Insufficient annual income",
                        "code": "INCOME_REJECT"
                    }
                },
                "credit_pass": {
                    "node_id": "credit_pass",
                    "node_type": NodeType.CALCULATION.value,
                    "name": "Calculate Premium",
                    "description": "Calculate insurance premium",
                    "calculation": {
                        "formula": "base_premium * risk_multiplier",
                        "variables": {
                            "base_premium": 1000,
                            "risk_multiplier": "${risk_score} / 10"
                        },
                        "result_field": "calculated_premium"
                    },
                    "children": ["approve"]
                },
                "credit_refer": {
                    "node_id": "credit_refer",
                    "node_type": NodeType.ACTION.value,
                    "name": "Refer for Review",
                    "description": "Refer application for manual review",
                    "action": {
                        "decision": "refer",
                        "reason": "Credit score below threshold",
                        "code": "CREDIT_REFER",
                        "department": "underwriting"
                    }
                },
                "approve": {
                    "node_id": "approve",
                    "node_type": NodeType.ACTION.value,
                    "name": "Approve Application",
                    "description": "Approve insurance application",
                    "action": {
                        "decision": "approve",
                        "reason": "All criteria met",
                        "code": "APPROVED",
                        "premium": "${calculated_premium}"
                    }
                }
            },
            "root_node_id": "root"
        }
        
        # Claims decision tree
        claims_tree = {
            "tree_id": "claims_basic",
            "name": "Basic Claims Decision Tree",
            "description": "Standard claims processing decision tree",
            "tree_type": DecisionTreeType.CLAIMS.value,
            "version": "1.0",
            "nodes": {
                "root": {
                    "node_id": "root",
                    "node_type": NodeType.CONDITION.value,
                    "name": "Policy Status Check",
                    "description": "Check if policy is active",
                    "condition": {
                        "field": "policy_status",
                        "operator": OperatorType.EQUALS.value,
                        "value": "active"
                    },
                    "children": ["policy_active", "policy_inactive"]
                },
                "policy_active": {
                    "node_id": "policy_active",
                    "node_type": NodeType.CONDITION.value,
                    "name": "Coverage Check",
                    "description": "Check if claim is within coverage limits",
                    "condition": {
                        "field": "claim_amount",
                        "operator": OperatorType.LESS_EQUAL.value,
                        "value": "${coverage_limit}"
                    },
                    "children": ["coverage_ok", "coverage_exceed"]
                },
                "policy_inactive": {
                    "node_id": "policy_inactive",
                    "node_type": NodeType.ACTION.value,
                    "name": "Reject - Policy Inactive",
                    "description": "Reject claim due to inactive policy",
                    "action": {
                        "decision": "reject",
                        "reason": "Policy is not active",
                        "code": "POLICY_INACTIVE"
                    }
                },
                "coverage_ok": {
                    "node_id": "coverage_ok",
                    "node_type": NodeType.CONDITION.value,
                    "name": "Fraud Check",
                    "description": "Check for potential fraud indicators",
                    "condition": {
                        "field": "fraud_score",
                        "operator": OperatorType.LESS_THAN.value,
                        "value": 7
                    },
                    "children": ["fraud_low", "fraud_high"]
                },
                "coverage_exceed": {
                    "node_id": "coverage_exceed",
                    "node_type": NodeType.ACTION.value,
                    "name": "Partial Approval",
                    "description": "Approve claim up to coverage limit",
                    "action": {
                        "decision": "partial_approve",
                        "reason": "Claim exceeds coverage limit",
                        "code": "PARTIAL_APPROVE",
                        "approved_amount": "${coverage_limit}"
                    }
                },
                "fraud_low": {
                    "node_id": "fraud_low",
                    "node_type": NodeType.CALCULATION.value,
                    "name": "Calculate Payout",
                    "description": "Calculate claim payout amount",
                    "calculation": {
                        "formula": "claim_amount - deductible",
                        "variables": {
                            "claim_amount": "${claim_amount}",
                            "deductible": "${deductible}"
                        },
                        "result_field": "payout_amount"
                    },
                    "children": ["approve_claim"]
                },
                "fraud_high": {
                    "node_id": "fraud_high",
                    "node_type": NodeType.ACTION.value,
                    "name": "Investigate Fraud",
                    "description": "Refer claim for fraud investigation",
                    "action": {
                        "decision": "investigate",
                        "reason": "High fraud score detected",
                        "code": "FRAUD_INVESTIGATE",
                        "department": "fraud_investigation"
                    }
                },
                "approve_claim": {
                    "node_id": "approve_claim",
                    "node_type": NodeType.ACTION.value,
                    "name": "Approve Claim",
                    "description": "Approve claim for payment",
                    "action": {
                        "decision": "approve",
                        "reason": "Claim meets all criteria",
                        "code": "CLAIM_APPROVED",
                        "payout_amount": "${payout_amount}"
                    }
                }
            },
            "root_node_id": "root"
        }
        
        # Create trees
        self.create_decision_tree(underwriting_tree)
        self.create_decision_tree(claims_tree)
        
        logger.info("Created default decision trees")

    async def execute_decision_tree(self, tree_id: str, input_data: Dict[str, Any]) -> DecisionTreeResult:
        """Execute a decision tree with input data"""
        
        start_time = datetime.utcnow()
        execution_id = str(uuid.uuid4())
        
        with decision_tree_execution_duration.time():
            try:
                if tree_id not in self.tree_cache:
                    raise ValueError(f"Decision tree {tree_id} not found")
                
                tree = self.tree_cache[tree_id]
                nodes = tree["structure"]["nodes"]
                root_node_id = tree["root_node_id"]
                
                # Initialize execution context
                context = {
                    "input_data": input_data.copy(),
                    "calculations": {},
                    "path_taken": [],
                    "variables": input_data.copy()
                }
                
                # Execute tree starting from root
                final_node, final_action = await self._execute_node(root_node_id, nodes, context)
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Create result
                result = DecisionTreeResult(
                    tree_id=tree_id,
                    execution_id=execution_id,
                    path_taken=context["path_taken"],
                    final_node=final_node,
                    final_action=final_action,
                    calculations=context["calculations"],
                    processing_time=processing_time,
                    input_data=input_data,
                    timestamp=datetime.utcnow()
                )
                
                # Store execution record
                await self._store_execution(result)
                
                # Update metrics
                decision_trees_executed_total.labels(tree_type=tree["tree_type"]).inc()
                
                logger.info(f"Decision tree {tree_id} executed successfully in {processing_time:.3f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing decision tree {tree_id}: {e}")
                raise

    async def _execute_node(self, node_id: str, nodes: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Execute a single node in the decision tree"""
        
        if node_id not in nodes:
            raise ValueError(f"Node {node_id} not found in tree")
        
        node = nodes[node_id]
        context["path_taken"].append(node_id)
        
        node_type = NodeType(node["node_type"])
        
        if node_type == NodeType.CONDITION:
            # Evaluate condition
            condition_result = self._evaluate_condition(node["condition"], context)
            
            # Get next node based on condition result
            children = node.get("children", [])
            if len(children) >= 2:
                next_node_id = children[0] if condition_result else children[1]
                return await self._execute_node(next_node_id, nodes, context)
            else:
                raise ValueError(f"Condition node {node_id} must have at least 2 children")
        
        elif node_type == NodeType.ACTION:
            # Execute action and return
            action = node["action"]
            resolved_action = self._resolve_variables(action, context)
            return node_id, resolved_action
        
        elif node_type == NodeType.CALCULATION:
            # Perform calculation
            calculation = node["calculation"]
            result = self._perform_calculation(calculation, context)
            
            # Store calculation result
            result_field = calculation.get("result_field", "calculation_result")
            context["calculations"][result_field] = result
            context["variables"][result_field] = result
            
            # Continue to next node
            children = node.get("children", [])
            if children:
                next_node_id = children[0]
                return await self._execute_node(next_node_id, nodes, context)
            else:
                # No children, return calculation as action
                return node_id, {"calculation_result": result}
        
        elif node_type == NodeType.REFERENCE:
            # Reference to another tree or external system
            reference = node.get("reference", {})
            # For now, just continue to next node
            children = node.get("children", [])
            if children:
                next_node_id = children[0]
                return await self._execute_node(next_node_id, nodes, context)
            else:
                return node_id, {"reference": reference}
        
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a condition"""
        
        field = condition["field"]
        operator = OperatorType(condition["operator"])
        value = condition["value"]
        
        # Get field value from context
        field_value = self._get_field_value(context["variables"], field)
        
        # Resolve value if it contains variables
        resolved_value = self._resolve_variables(value, context)
        
        # Perform comparison
        try:
            if operator == OperatorType.EQUALS:
                return field_value == resolved_value
            elif operator == OperatorType.NOT_EQUALS:
                return field_value != resolved_value
            elif operator == OperatorType.GREATER_THAN:
                return float(field_value) > float(resolved_value)
            elif operator == OperatorType.GREATER_EQUAL:
                return float(field_value) >= float(resolved_value)
            elif operator == OperatorType.LESS_THAN:
                return float(field_value) < float(resolved_value)
            elif operator == OperatorType.LESS_EQUAL:
                return float(field_value) <= float(resolved_value)
            elif operator == OperatorType.IN:
                return field_value in resolved_value
            elif operator == OperatorType.NOT_IN:
                return field_value not in resolved_value
            elif operator == OperatorType.BETWEEN:
                if isinstance(resolved_value, list) and len(resolved_value) == 2:
                    return resolved_value[0] <= float(field_value) <= resolved_value[1]
                return False
            elif operator == OperatorType.CONTAINS:
                return str(resolved_value) in str(field_value)
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition {field} {operator.value} {value}: {e}")
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

    def _resolve_variables(self, value: Any, context: Dict[str, Any]) -> Any:
        """Resolve variables in values (e.g., ${variable_name})"""
        
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            # Variable reference
            var_name = value[2:-1]
            return self._get_field_value(context["variables"], var_name)
        elif isinstance(value, dict):
            # Recursively resolve dictionary values
            resolved = {}
            for k, v in value.items():
                resolved[k] = self._resolve_variables(v, context)
            return resolved
        elif isinstance(value, list):
            # Recursively resolve list values
            return [self._resolve_variables(v, context) for v in value]
        else:
            return value

    def _perform_calculation(self, calculation: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Perform mathematical calculation"""
        
        formula = calculation["formula"]
        variables = calculation.get("variables", {})
        
        # Resolve variables
        resolved_variables = {}
        for var_name, var_value in variables.items():
            resolved_variables[var_name] = self._resolve_variables(var_value, context)
        
        # Replace variables in formula
        processed_formula = formula
        for var_name, var_value in resolved_variables.items():
            processed_formula = processed_formula.replace(var_name, str(var_value))
        
        # Also replace direct variable references
        import re
        def replace_var_ref(match):
            var_name = match.group(1)
            var_value = self._get_field_value(context["variables"], var_name)
            return str(var_value) if var_value is not None else "0"
        
        processed_formula = re.sub(r'\$\{([^}]+)\}', replace_var_ref, processed_formula)
        
        # Safe evaluation
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

    async def _store_execution(self, result: DecisionTreeResult):
        """Store execution result in database"""
        
        try:
            with self.Session() as session:
                execution = DecisionTreeExecutionModel(
                    execution_id=result.execution_id,
                    tree_id=result.tree_id,
                    path_taken=result.path_taken,
                    final_node=result.final_node,
                    final_action=result.final_action,
                    calculations=result.calculations,
                    processing_time=result.processing_time,
                    input_data=result.input_data,
                    executed_at=result.timestamp
                )
                
                session.add(execution)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing execution result: {e}")

    def create_decision_tree(self, tree_data: Dict[str, Any]) -> str:
        """Create a new decision tree"""
        
        try:
            tree_id = tree_data.get("tree_id", str(uuid.uuid4()))
            
            # Validate tree structure
            self._validate_tree_structure(tree_data)
            
            # Create database record
            with self.Session() as session:
                tree_model = DecisionTreeModel(
                    tree_id=tree_id,
                    name=tree_data["name"],
                    description=tree_data.get("description", ""),
                    tree_type=tree_data["tree_type"],
                    version=tree_data.get("version", "1.0"),
                    is_active=tree_data.get("is_active", True),
                    tree_structure=tree_data,
                    root_node_id=tree_data["root_node_id"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    metadata=tree_data.get("metadata", {})
                )
                
                session.merge(tree_model)
                session.commit()
            
            # Update cache
            self._load_active_trees()
            
            logger.info(f"Created decision tree {tree_id}")
            return tree_id
            
        except Exception as e:
            logger.error(f"Error creating decision tree: {e}")
            raise

    def _validate_tree_structure(self, tree_data: Dict[str, Any]):
        """Validate decision tree structure"""
        
        required_fields = ["name", "tree_type", "nodes", "root_node_id"]
        for field in required_fields:
            if field not in tree_data:
                raise ValueError(f"Missing required field: {field}")
        
        nodes = tree_data["nodes"]
        root_node_id = tree_data["root_node_id"]
        
        if root_node_id not in nodes:
            raise ValueError(f"Root node {root_node_id} not found in nodes")
        
        # Validate each node
        for node_id, node in nodes.items():
            required_node_fields = ["node_id", "node_type", "name"]
            for field in required_node_fields:
                if field not in node:
                    raise ValueError(f"Node {node_id} missing field: {field}")
            
            # Validate node type specific requirements
            node_type = NodeType(node["node_type"])
            
            if node_type == NodeType.CONDITION and "condition" not in node:
                raise ValueError(f"Condition node {node_id} missing condition")
            elif node_type == NodeType.ACTION and "action" not in node:
                raise ValueError(f"Action node {node_id} missing action")
            elif node_type == NodeType.CALCULATION and "calculation" not in node:
                raise ValueError(f"Calculation node {node_id} missing calculation")

    def update_decision_tree(self, tree_id: str, tree_data: Dict[str, Any]) -> bool:
        """Update an existing decision tree"""
        
        try:
            with self.Session() as session:
                tree_model = session.query(DecisionTreeModel).filter(
                    DecisionTreeModel.tree_id == tree_id
                ).first()
                
                if not tree_model:
                    return False
                
                # Validate new structure
                self._validate_tree_structure(tree_data)
                
                # Update fields
                tree_model.name = tree_data.get("name", tree_model.name)
                tree_model.description = tree_data.get("description", tree_model.description)
                tree_model.tree_type = tree_data.get("tree_type", tree_model.tree_type)
                tree_model.version = tree_data.get("version", tree_model.version)
                tree_model.is_active = tree_data.get("is_active", tree_model.is_active)
                tree_model.tree_structure = tree_data
                tree_model.root_node_id = tree_data.get("root_node_id", tree_model.root_node_id)
                tree_model.updated_at = datetime.utcnow()
                tree_model.metadata = tree_data.get("metadata", tree_model.metadata)
                
                session.commit()
            
            # Update cache
            self._load_active_trees()
            
            logger.info(f"Updated decision tree {tree_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating decision tree {tree_id}: {e}")
            return False

    def get_decision_tree(self, tree_id: str) -> Optional[Dict[str, Any]]:
        """Get decision tree by ID"""
        
        try:
            with self.Session() as session:
                tree_model = session.query(DecisionTreeModel).filter(
                    DecisionTreeModel.tree_id == tree_id
                ).first()
                
                if not tree_model:
                    return None
                
                return {
                    "tree_id": tree_model.tree_id,
                    "name": tree_model.name,
                    "description": tree_model.description,
                    "tree_type": tree_model.tree_type,
                    "version": tree_model.version,
                    "is_active": tree_model.is_active,
                    "tree_structure": tree_model.tree_structure,
                    "root_node_id": tree_model.root_node_id,
                    "created_at": tree_model.created_at.isoformat(),
                    "updated_at": tree_model.updated_at.isoformat(),
                    "metadata": tree_model.metadata
                }
                
        except Exception as e:
            logger.error(f"Error getting decision tree {tree_id}: {e}")
            return None

    def list_decision_trees(self, tree_type: DecisionTreeType = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """List decision trees with optional filtering"""
        
        try:
            with self.Session() as session:
                query = session.query(DecisionTreeModel)
                
                if tree_type:
                    query = query.filter(DecisionTreeModel.tree_type == tree_type.value)
                
                if active_only:
                    query = query.filter(DecisionTreeModel.is_active == True)
                
                trees = query.order_by(DecisionTreeModel.created_at.desc()).all()
                
                result = []
                for tree in trees:
                    result.append({
                        "tree_id": tree.tree_id,
                        "name": tree.name,
                        "description": tree.description,
                        "tree_type": tree.tree_type,
                        "version": tree.version,
                        "is_active": tree.is_active,
                        "created_at": tree.created_at.isoformat(),
                        "updated_at": tree.updated_at.isoformat()
                    })
                
                return result
                
        except Exception as e:
            logger.error(f"Error listing decision trees: {e}")
            return []

    def get_execution_history(self, tree_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get decision tree execution history"""
        
        try:
            with self.Session() as session:
                query = session.query(DecisionTreeExecutionModel)
                
                if tree_id:
                    query = query.filter(DecisionTreeExecutionModel.tree_id == tree_id)
                
                executions = query.order_by(DecisionTreeExecutionModel.executed_at.desc()).limit(limit).all()
                
                result = []
                for execution in executions:
                    result.append({
                        "execution_id": execution.execution_id,
                        "tree_id": execution.tree_id,
                        "path_taken": execution.path_taken,
                        "final_node": execution.final_node,
                        "final_action": execution.final_action,
                        "processing_time": execution.processing_time,
                        "executed_at": execution.executed_at.isoformat()
                    })
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting execution history: {e}")
            return []

    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get decision tree statistics"""
        
        try:
            with self.Session() as session:
                total_trees = session.query(DecisionTreeModel).count()
                active_trees = session.query(DecisionTreeModel).filter(
                    DecisionTreeModel.is_active == True
                ).count()
                
                # Trees by type
                type_stats = {}
                for tree_type in DecisionTreeType:
                    count = session.query(DecisionTreeModel).filter(
                        DecisionTreeModel.tree_type == tree_type.value,
                        DecisionTreeModel.is_active == True
                    ).count()
                    type_stats[tree_type.value] = count
                
                # Execution statistics
                total_executions = session.query(DecisionTreeExecutionModel).count()
                
                # Recent executions (last 24 hours)
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                recent_executions = session.query(DecisionTreeExecutionModel).filter(
                    DecisionTreeExecutionModel.executed_at >= recent_cutoff
                ).count()
                
                return {
                    "total_trees": total_trees,
                    "active_trees": active_trees,
                    "trees_by_type": type_stats,
                    "total_executions": total_executions,
                    "recent_executions_24h": recent_executions,
                    "cached_trees": len(self.tree_cache)
                }
                
        except Exception as e:
            logger.error(f"Error getting tree statistics: {e}")
            return {}

# Factory function
def create_decision_tree_processor(db_url: str = None, redis_url: str = None) -> DecisionTreeProcessor:
    """Create and configure DecisionTreeProcessor instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return DecisionTreeProcessor(db_url=db_url, redis_url=redis_url)

# Example usage
if __name__ == "__main__":
    async def test_decision_tree_processor():
        """Test decision tree processor functionality"""
        
        processor = create_decision_tree_processor()
        
        # Test underwriting decision tree
        test_data = {
            "age": 35,
            "annual_income": 75000,
            "credit_score": 720,
            "risk_score": 6,
            "coverage_limit": 500000
        }
        
        result = await processor.execute_decision_tree("underwriting_basic", test_data)
        
        print(f"Decision Tree Result:")
        print(f"Path taken: {result.path_taken}")
        print(f"Final action: {result.final_action}")
        print(f"Calculations: {result.calculations}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        # Get statistics
        stats = processor.get_tree_statistics()
        print(f"Statistics: {stats}")
    
    # Run test
    # asyncio.run(test_decision_tree_processor())

