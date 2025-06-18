"""
Automation Agent - Production Ready Implementation
Handles workflow automation, task scheduling, and process orchestration for insurance operations
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import schedule
import time
import threading

# Database and monitoring imports
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
automation_tasks_total = Counter('automation_tasks_total', 'Total automation tasks executed', ['task_type', 'status'])
automation_duration = Histogram('automation_task_duration_seconds', 'Time spent executing automation tasks')
active_workflows = Gauge('active_workflows_total', 'Number of active workflows')

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowType(Enum):
    UNDERWRITING = "underwriting"
    CLAIMS_PROCESSING = "claims_processing"
    DOCUMENT_REVIEW = "document_review"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    CUSTOMER_NOTIFICATION = "customer_notification"

@dataclass
class AutomationTask:
    """Represents an automation task with all necessary metadata"""
    task_id: str
    workflow_type: WorkflowType
    task_name: str
    parameters: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 5  # 1-10, 10 being highest priority

@dataclass
class WorkflowDefinition:
    """Defines a workflow with its steps and configuration"""
    workflow_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    steps: List[Dict[str, Any]]
    triggers: List[str]
    conditions: Dict[str, Any]
    timeout_minutes: int = 30
    is_active: bool = True

class AutomationAgent:
    """
    Production-ready Automation Agent for Insurance AI System
    Handles workflow automation, task scheduling, and process orchestration
    """
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.redis_client = redis.from_url(redis_url)
        
        # Task management
        self.active_tasks: Dict[str, AutomationTask] = {}
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Scheduling
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Load workflow definitions
        self._load_workflow_definitions()
        
        logger.info("AutomationAgent initialized successfully")

    def _load_workflow_definitions(self):
        """Load predefined workflow definitions"""
        workflows = [
            WorkflowDefinition(
                workflow_id="underwriting_auto",
                name="Automated Underwriting Process",
                description="Automated underwriting workflow for standard policies",
                workflow_type=WorkflowType.UNDERWRITING,
                steps=[
                    {"step": "document_analysis", "agent": "document_analysis", "timeout": 300},
                    {"step": "risk_assessment", "agent": "risk_assessment", "timeout": 600},
                    {"step": "compliance_check", "agent": "compliance", "timeout": 180},
                    {"step": "decision_generation", "agent": "decision_engine", "timeout": 120}
                ],
                triggers=["policy_application_submitted"],
                conditions={"policy_value": {"max": 100000}, "risk_score": {"max": 7}}
            ),
            WorkflowDefinition(
                workflow_id="claims_auto",
                name="Automated Claims Processing",
                description="Automated claims processing for standard claims",
                workflow_type=WorkflowType.CLAIMS_PROCESSING,
                steps=[
                    {"step": "evidence_processing", "agent": "evidence_processing", "timeout": 900},
                    {"step": "damage_assessment", "agent": "liability_assessment", "timeout": 600},
                    {"step": "fraud_detection", "agent": "validation", "timeout": 300},
                    {"step": "settlement_calculation", "agent": "decision_engine", "timeout": 180}
                ],
                triggers=["claim_submitted"],
                conditions={"claim_amount": {"max": 50000}, "claim_type": {"in": ["auto", "property"]}}
            ),
            WorkflowDefinition(
                workflow_id="document_review_auto",
                name="Automated Document Review",
                description="Automated document review and validation",
                workflow_type=WorkflowType.DOCUMENT_REVIEW,
                steps=[
                    {"step": "document_extraction", "agent": "document_analysis", "timeout": 300},
                    {"step": "content_validation", "agent": "validation", "timeout": 180},
                    {"step": "compliance_check", "agent": "compliance", "timeout": 120}
                ],
                triggers=["document_uploaded"],
                conditions={}
            )
        ]
        
        for workflow in workflows:
            self.workflow_definitions[workflow.workflow_id] = workflow
            
        logger.info(f"Loaded {len(workflows)} workflow definitions")

    async def create_task(self, 
                         workflow_type: WorkflowType,
                         task_name: str,
                         parameters: Dict[str, Any],
                         scheduled_at: Optional[datetime] = None,
                         priority: int = 5) -> str:
        """Create a new automation task"""
        
        task_id = str(uuid.uuid4())
        task = AutomationTask(
            task_id=task_id,
            workflow_type=workflow_type,
            task_name=task_name,
            parameters=parameters,
            status=TaskStatus.PENDING,
            created_at=datetime.utcnow(),
            scheduled_at=scheduled_at,
            priority=priority
        )
        
        self.active_tasks[task_id] = task
        
        # Store in Redis for persistence
        await self._store_task_in_redis(task)
        
        # Add to queue if not scheduled
        if scheduled_at is None or scheduled_at <= datetime.utcnow():
            await self.task_queue.put(task_id)
        else:
            # Schedule for later execution
            self._schedule_task(task)
        
        automation_tasks_total.labels(task_type=workflow_type.value, status='created').inc()
        logger.info(f"Created automation task {task_id} for {workflow_type.value}")
        
        return task_id

    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any]) -> str:
        """Execute a predefined workflow"""
        
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflow_definitions[workflow_id]
        
        # Check conditions
        if not self._check_workflow_conditions(workflow, parameters):
            raise ValueError(f"Workflow conditions not met for {workflow_id}")
        
        task_id = await self.create_task(
            workflow_type=workflow.workflow_type,
            task_name=f"Execute {workflow.name}",
            parameters={
                "workflow_id": workflow_id,
                "workflow_steps": workflow.steps,
                "input_parameters": parameters,
                "timeout_minutes": workflow.timeout_minutes
            }
        )
        
        logger.info(f"Started workflow execution {workflow_id} with task {task_id}")
        return task_id

    def _check_workflow_conditions(self, workflow: WorkflowDefinition, parameters: Dict[str, Any]) -> bool:
        """Check if workflow conditions are met"""
        
        if not workflow.conditions:
            return True
        
        for condition_key, condition_value in workflow.conditions.items():
            if condition_key not in parameters:
                continue
                
            param_value = parameters[condition_key]
            
            if isinstance(condition_value, dict):
                if "max" in condition_value and param_value > condition_value["max"]:
                    return False
                if "min" in condition_value and param_value < condition_value["min"]:
                    return False
                if "in" in condition_value and param_value not in condition_value["in"]:
                    return False
                if "not_in" in condition_value and param_value in condition_value["not_in"]:
                    return False
        
        return True

    async def process_task_queue(self):
        """Process tasks from the queue"""
        
        while True:
            try:
                # Get task from queue with timeout
                task_id = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    await self._execute_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing task queue: {e}")
                await asyncio.sleep(1)

    async def _execute_task(self, task: AutomationTask):
        """Execute a single automation task"""
        
        with automation_duration.time():
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                active_workflows.inc()
                
                await self._store_task_in_redis(task)
                
                logger.info(f"Executing task {task.task_id}: {task.task_name}")
                
                # Execute based on task type
                if "workflow_id" in task.parameters:
                    result = await self._execute_workflow_steps(task)
                else:
                    result = await self._execute_single_task(task)
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                task.result = result
                
                automation_tasks_total.labels(task_type=task.workflow_type.value, status='completed').inc()
                logger.info(f"Task {task.task_id} completed successfully")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                task.completed_at = datetime.utcnow()
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                    await self.task_queue.put(task.task_id)
                    logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                else:
                    automation_tasks_total.labels(task_type=task.workflow_type.value, status='failed').inc()
                    logger.error(f"Task {task.task_id} failed permanently: {e}")
                
            finally:
                active_workflows.dec()
                await self._store_task_in_redis(task)

    async def _execute_workflow_steps(self, task: AutomationTask) -> Dict[str, Any]:
        """Execute workflow steps sequentially"""
        
        workflow_id = task.parameters["workflow_id"]
        steps = task.parameters["workflow_steps"]
        input_params = task.parameters["input_parameters"]
        
        results = {}
        current_data = input_params.copy()
        
        for step in steps:
            step_name = step["step"]
            agent_name = step["agent"]
            timeout = step.get("timeout", 300)
            
            logger.info(f"Executing workflow step: {step_name}")
            
            try:
                # Simulate agent execution (in production, this would call actual agents)
                step_result = await self._call_agent(agent_name, current_data, timeout)
                results[step_name] = step_result
                
                # Pass results to next step
                current_data.update(step_result)
                
            except Exception as e:
                logger.error(f"Workflow step {step_name} failed: {e}")
                raise
        
        return {
            "workflow_id": workflow_id,
            "step_results": results,
            "final_data": current_data,
            "execution_time": (datetime.utcnow() - task.started_at).total_seconds()
        }

    async def _execute_single_task(self, task: AutomationTask) -> Dict[str, Any]:
        """Execute a single automation task"""
        
        # Simulate task execution based on task name
        if "notification" in task.task_name.lower():
            return await self._send_notification(task.parameters)
        elif "validation" in task.task_name.lower():
            return await self._validate_data(task.parameters)
        elif "calculation" in task.task_name.lower():
            return await self._perform_calculation(task.parameters)
        else:
            return await self._generic_task_execution(task.parameters)

    async def _call_agent(self, agent_name: str, data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Call another agent for processing via HTTP API"""
        
        import aiohttp
        import asyncio
        
        # Agent endpoint mapping
        agent_endpoints = {
            "document_analysis": "http://localhost:8001/api/v1/analyze-documents",
            "risk_assessment": "http://localhost:8002/api/v1/assess-risk", 
            "compliance": "http://localhost:8003/api/v1/check-compliance",
            "evidence_processing": "http://localhost:8004/api/v1/process-evidence",
            "decision_engine": "http://localhost:8005/api/v1/make-decision",
            "validation": "http://localhost:8006/api/v1/validate-data",
            "communication": "http://localhost:8007/api/v1/send-notification",
            "liability_assessment": "http://localhost:8008/api/v1/assess-liability"
        }
        
        endpoint = agent_endpoints.get(agent_name)
        if not endpoint:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AutomationAgent/1.0',
            'X-Request-ID': str(uuid.uuid4())
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(endpoint, json=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Agent {agent_name} returned {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            raise Exception(f"Agent {agent_name} timed out after {timeout} seconds")
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP error calling agent {agent_name}: {e}")

    async def _send_notification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send automated notifications via Communication Agent"""
        
        import aiohttp
        
        notification_endpoint = "http://localhost:8007/api/v1/send-notification"
        
        payload = {
            "recipient_id": parameters.get("recipient_id"),
            "channel": parameters.get("channel", "email"),
            "subject": parameters.get("subject", "Automated Notification"),
            "content": parameters.get("message", ""),
            "priority": parameters.get("priority", "normal"),
            "template_id": parameters.get("template_id"),
            "template_data": parameters.get("template_data", {})
        }
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AutomationAgent/1.0'
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(notification_endpoint, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "notification_sent": True,
                            "message_id": result.get("message_id"),
                            "channel": payload["channel"],
                            "recipient": payload["recipient_id"],
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Notification service error {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return {
                "notification_sent": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _validate_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data using Validation Agent"""
        
        import aiohttp
        
        validation_endpoint = "http://localhost:8006/api/v1/validate-data"
        
        payload = {
            "data": parameters.get("data", {}),
            "validation_rules": parameters.get("validation_rules", []),
            "schema": parameters.get("schema"),
            "strict_mode": parameters.get("strict_mode", True)
        }
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AutomationAgent/1.0'
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(validation_endpoint, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "validation_status": result.get("status", "passed"),
                            "results": result.get("validation_results", []),
                            "errors": result.get("errors", []),
                            "warnings": result.get("warnings", []),
                            "validated_fields": result.get("field_count", 0),
                            "processing_time": result.get("processing_time", 0)
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Validation service error {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {
                "validation_status": "failed",
                "error": str(e),
                "validated_fields": 0
            }

    async def _perform_calculation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform automated calculations using business rules"""
        
        calculation_type = parameters.get("type", "premium")
        values = parameters.get("values", {})
        
        try:
            if calculation_type == "premium":
                # Premium calculation with real business logic
                base_amount = float(values.get("base_amount", 1000))
                coverage_amount = float(values.get("coverage_amount", 100000))
                risk_score = float(values.get("risk_score", 5))
                age = int(values.get("age", 30))
                location_factor = float(values.get("location_factor", 1.0))
                
                # Base premium calculation
                coverage_factor = coverage_amount / 100000  # Normalize to 100k
                age_factor = 1.0 + (max(0, age - 25) * 0.01)  # 1% per year over 25
                risk_factor = 1.0 + (risk_score / 10)  # Risk score 1-10
                
                calculated_premium = base_amount * coverage_factor * age_factor * risk_factor * location_factor
                
                # Apply discounts
                discounts = values.get("discounts", [])
                total_discount = sum(float(d.get("amount", 0)) for d in discounts)
                final_premium = max(calculated_premium - total_discount, base_amount * 0.5)  # Min 50% of base
                
                return {
                    "calculation_type": calculation_type,
                    "base_premium": round(base_amount, 2),
                    "calculated_premium": round(calculated_premium, 2),
                    "total_discounts": round(total_discount, 2),
                    "final_premium": round(final_premium, 2),
                    "factors": {
                        "coverage_factor": round(coverage_factor, 3),
                        "age_factor": round(age_factor, 3),
                        "risk_factor": round(risk_factor, 3),
                        "location_factor": round(location_factor, 3)
                    },
                    "input_values": values
                }
                
            elif calculation_type == "claim_settlement":
                # Claim settlement calculation
                claim_amount = float(values.get("claim_amount", 0))
                coverage_limit = float(values.get("coverage_limit", 100000))
                deductible = float(values.get("deductible", 500))
                depreciation_rate = float(values.get("depreciation_rate", 0))
                
                # Calculate settlement
                covered_amount = min(claim_amount, coverage_limit)
                depreciated_amount = covered_amount * (1 - depreciation_rate)
                settlement_amount = max(0, depreciated_amount - deductible)
                
                return {
                    "calculation_type": calculation_type,
                    "claim_amount": round(claim_amount, 2),
                    "coverage_limit": round(coverage_limit, 2),
                    "deductible": round(deductible, 2),
                    "depreciation_rate": round(depreciation_rate, 3),
                    "covered_amount": round(covered_amount, 2),
                    "settlement_amount": round(settlement_amount, 2),
                    "input_values": values
                }
                
            elif calculation_type == "risk_score":
                # Risk score calculation
                factors = values.get("risk_factors", {})
                weights = {
                    "age": 0.2,
                    "location": 0.3,
                    "history": 0.25,
                    "coverage_type": 0.15,
                    "credit_score": 0.1
                }
                
                total_score = 0
                factor_scores = {}
                
                for factor, weight in weights.items():
                    factor_value = factors.get(factor, 5)  # Default to medium risk
                    normalized_score = min(10, max(1, float(factor_value)))
                    weighted_score = normalized_score * weight
                    total_score += weighted_score
                    factor_scores[factor] = {
                        "raw_score": factor_value,
                        "normalized_score": normalized_score,
                        "weight": weight,
                        "weighted_score": round(weighted_score, 2)
                    }
                
                risk_level = "low" if total_score <= 3 else "medium" if total_score <= 7 else "high"
                
                return {
                    "calculation_type": calculation_type,
                    "total_risk_score": round(total_score, 2),
                    "risk_level": risk_level,
                    "factor_scores": factor_scores,
                    "input_values": values
                }
                
            else:
                # Generic calculation
                result = sum(float(v) for v in values.values() if isinstance(v, (int, float, str)) and str(v).replace('.', '').isdigit())
                
                return {
                    "calculation_type": calculation_type,
                    "result": round(result, 2),
                    "input_values": values
                }
                
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return {
                "calculation_type": calculation_type,
                "error": str(e),
                "result": 0,
                "input_values": values
            }

    async def _generic_task_execution(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generic task execution for undefined task types"""
        
        await asyncio.sleep(0.1)  # Simulate processing
        
        return {
            "status": "completed",
            "parameters_processed": len(parameters),
            "execution_time": 0.1
        }

    def _schedule_task(self, task: AutomationTask):
        """Schedule a task for future execution"""
        
        if not self.scheduler_running:
            self.start_scheduler()
        
        def execute_scheduled_task():
            asyncio.create_task(self.task_queue.put(task.task_id))
        
        schedule.every().day.at(task.scheduled_at.strftime("%H:%M")).do(execute_scheduled_task)
        logger.info(f"Scheduled task {task.task_id} for {task.scheduled_at}")

    def start_scheduler(self):
        """Start the task scheduler"""
        
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        
        def run_scheduler():
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(1)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Task scheduler started")

    def stop_scheduler(self):
        """Stop the task scheduler"""
        
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Task scheduler stopped")

    async def _store_task_in_redis(self, task: AutomationTask):
        """Store task in Redis for persistence"""
        
        try:
            task_data = asdict(task)
            # Convert datetime objects to ISO strings
            for key, value in task_data.items():
                if isinstance(value, datetime):
                    task_data[key] = value.isoformat() if value else None
            
            self.redis_client.setex(
                f"automation_task:{task.task_id}",
                3600 * 24,  # 24 hours TTL
                json.dumps(task_data, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to store task in Redis: {e}")

    async def get_task_status(self, task_id: str) -> Optional[AutomationTask]:
        """Get task status by ID"""
        
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Try to load from Redis
        try:
            task_data = self.redis_client.get(f"automation_task:{task_id}")
            if task_data:
                data = json.loads(task_data)
                # Convert ISO strings back to datetime objects
                for key in ['created_at', 'scheduled_at', 'started_at', 'completed_at']:
                    if data.get(key):
                        data[key] = datetime.fromisoformat(data[key])
                
                return AutomationTask(**data)
        except Exception as e:
            logger.error(f"Failed to load task from Redis: {e}")
        
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        
        await self._store_task_in_redis(task)
        automation_tasks_total.labels(task_type=task.workflow_type.value, status='cancelled').inc()
        
        logger.info(f"Task {task_id} cancelled")
        return True

    def get_workflow_definitions(self) -> Dict[str, WorkflowDefinition]:
        """Get all available workflow definitions"""
        return self.workflow_definitions.copy()

    async def get_active_tasks(self) -> List[AutomationTask]:
        """Get all active tasks"""
        return [task for task in self.active_tasks.values() 
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]]

    async def get_task_statistics(self) -> Dict[str, Any]:
        """Get automation task statistics"""
        
        total_tasks = len(self.active_tasks)
        status_counts = {}
        
        for task in self.active_tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tasks": total_tasks,
            "status_breakdown": status_counts,
            "active_workflows": len([t for t in self.active_tasks.values() 
                                   if t.status == TaskStatus.RUNNING]),
            "workflow_definitions": len(self.workflow_definitions)
        }

    async def cleanup_completed_tasks(self, older_than_hours: int = 24):
        """Clean up completed tasks older than specified hours"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        tasks_to_remove = []
        
        for task_id, task in self.active_tasks.items():
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task.completed_at and task.completed_at < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]
            # Remove from Redis
            self.redis_client.delete(f"automation_task:{task_id}")
        
        logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
        return len(tasks_to_remove)

    async def shutdown(self):
        """Graceful shutdown of the automation agent"""
        
        logger.info("Shutting down AutomationAgent...")
        
        # Stop scheduler
        self.stop_scheduler()
        
        # Cancel all pending tasks
        for task_id, task in self.active_tasks.items():
            if task.status == TaskStatus.PENDING:
                await self.cancel_task(task_id)
        
        # Wait for running tasks to complete (with timeout)
        timeout = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            running_tasks = [t for t in self.active_tasks.values() if t.status == TaskStatus.RUNNING]
            if not running_tasks:
                break
            await asyncio.sleep(1)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("AutomationAgent shutdown complete")

# Factory function for easy instantiation
def create_automation_agent(db_url: str = None, redis_url: str = None) -> AutomationAgent:
    """Create and configure an AutomationAgent instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return AutomationAgent(db_url=db_url, redis_url=redis_url)

# Example usage and testing
if __name__ == "__main__":
    async def test_automation_agent():
        """Test the automation agent functionality"""
        
        agent = create_automation_agent()
        
        # Test workflow execution
        task_id = await agent.execute_workflow(
            "underwriting_auto",
            {
                "policy_value": 50000,
                "risk_score": 5,
                "documents": ["application.pdf", "id_copy.pdf"],
                "customer_id": "CUST123"
            }
        )
        
        print(f"Started workflow with task ID: {task_id}")
        
        # Start task processing
        await agent.process_task_queue()
    
    # Run test
    # asyncio.run(test_automation_agent())

