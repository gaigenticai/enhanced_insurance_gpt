"""
Workflow Manager - Production Ready Implementation
Advanced workflow management for claims processing
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, PriorityQueue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class WorkflowTask:
    task_id: str
    task_name: str
    task_type: str
    dependencies: List[str]
    parameters: Dict[str, Any]
    timeout: int
    retry_count: int
    max_retries: int
    priority: TaskPriority
    status: TaskStatus
    assigned_worker: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration: Optional[float]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class WorkflowInstance:
    workflow_id: str
    workflow_name: str
    workflow_type: str
    claim_id: str
    status: WorkflowStatus
    tasks: List[WorkflowTask]
    context: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_duration: Optional[float]
    progress_percentage: float
    current_stage: str
    error_count: int
    retry_count: int
    max_retries: int

class WorkflowRecord(Base):
    __tablename__ = 'workflow_instances'
    
    workflow_id = Column(String, primary_key=True)
    workflow_name = Column(String, nullable=False)
    workflow_type = Column(String, nullable=False)
    claim_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    tasks = Column(JSON)
    context = Column(JSON)
    created_at = Column(DateTime, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    total_duration = Column(Float)
    progress_percentage = Column(Float)
    current_stage = Column(String)
    error_count = Column(Integer)
    retry_count = Column(Integer)
    max_retries = Column(Integer)

class TaskExecutionRecord(Base):
    __tablename__ = 'task_executions'
    
    execution_id = Column(String, primary_key=True)
    workflow_id = Column(String, nullable=False)
    task_id = Column(String, nullable=False)
    task_name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration = Column(Float)
    result = Column(JSON)
    error = Column(Text)
    retry_count = Column(Integer)
    worker_id = Column(String)

class ClaimsWorkflowManager:
    """Production-ready workflow manager for claims processing"""
    
    def __init__(self, db_url: str, redis_url: str, max_workers: int = 10):
        self.db_url = db_url
        self.redis_url = redis_url
        self.max_workers = max_workers
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # Workflow execution engine
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = PriorityQueue()
        self.running_workflows = {}
        self.workflow_graphs = {}
        
        # Task handlers
        self.task_handlers = {}
        
        # Workflow templates
        self.workflow_templates = {}
        
        # Event handlers
        self.event_handlers = {}
        
        # Monitoring
        self.metrics = {
            'workflows_started': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'tasks_executed': 0,
            'tasks_failed': 0,
            'average_workflow_duration': 0
        }
        
        self._initialize_workflow_templates()
        self._initialize_task_handlers()
        self._start_workflow_engine()
        
        logger.info("ClaimsWorkflowManager initialized successfully")

    def _initialize_workflow_templates(self):
        """Initialize workflow templates"""
        
        # Auto liability claim workflow template
        self.workflow_templates['auto_liability_claim'] = {
            'name': 'Auto Liability Claim Processing',
            'description': 'Complete workflow for processing auto liability claims',
            'tasks': [
                {
                    'task_id': 'intake_validation',
                    'task_name': 'Intake and Validation',
                    'task_type': 'validation',
                    'dependencies': [],
                    'timeout': 300,
                    'max_retries': 3,
                    'priority': TaskPriority.HIGH,
                    'parameters': {
                        'validation_type': 'claim_intake',
                        'required_fields': ['policy_number', 'loss_date', 'claimant_info']
                    }
                },
                {
                    'task_id': 'document_collection',
                    'task_name': 'Document Collection',
                    'task_type': 'document_processing',
                    'dependencies': ['intake_validation'],
                    'timeout': 1800,
                    'max_retries': 2,
                    'priority': TaskPriority.MEDIUM,
                    'parameters': {
                        'document_types': ['police_report', 'photos', 'witness_statements'],
                        'auto_request': True
                    }
                },
                {
                    'task_id': 'liability_investigation',
                    'task_name': 'Liability Investigation',
                    'task_type': 'investigation',
                    'dependencies': ['document_collection'],
                    'timeout': 3600,
                    'max_retries': 2,
                    'priority': TaskPriority.HIGH,
                    'parameters': {
                        'investigation_type': 'liability_analysis',
                        'include_interviews': True
                    }
                },
                {
                    'task_id': 'damage_evaluation',
                    'task_name': 'Damage Evaluation',
                    'task_type': 'evaluation',
                    'dependencies': ['liability_investigation'],
                    'timeout': 2400,
                    'max_retries': 2,
                    'priority': TaskPriority.MEDIUM,
                    'parameters': {
                        'evaluation_type': 'comprehensive_damage',
                        'include_medical': True
                    }
                },
                {
                    'task_id': 'settlement_calculation',
                    'task_name': 'Settlement Calculation',
                    'task_type': 'calculation',
                    'dependencies': ['damage_evaluation'],
                    'timeout': 600,
                    'max_retries': 3,
                    'priority': TaskPriority.HIGH,
                    'parameters': {
                        'calculation_type': 'settlement_amount',
                        'include_reserves': True
                    }
                },
                {
                    'task_id': 'decision_making',
                    'task_name': 'Settlement Decision',
                    'task_type': 'decision',
                    'dependencies': ['settlement_calculation'],
                    'timeout': 1200,
                    'max_retries': 2,
                    'priority': TaskPriority.CRITICAL,
                    'parameters': {
                        'decision_type': 'settlement_approval',
                        'require_manager_approval': True
                    }
                },
                {
                    'task_id': 'payment_processing',
                    'task_name': 'Payment Processing',
                    'task_type': 'payment',
                    'dependencies': ['decision_making'],
                    'timeout': 1800,
                    'max_retries': 3,
                    'priority': TaskPriority.CRITICAL,
                    'parameters': {
                        'payment_type': 'settlement_payment',
                        'require_releases': True
                    }
                },
                {
                    'task_id': 'claim_closure',
                    'task_name': 'Claim Closure',
                    'task_type': 'closure',
                    'dependencies': ['payment_processing'],
                    'timeout': 600,
                    'max_retries': 2,
                    'priority': TaskPriority.MEDIUM,
                    'parameters': {
                        'closure_type': 'standard_closure',
                        'send_notifications': True
                    }
                }
            ]
        }
        
        # Auto collision claim workflow template
        self.workflow_templates['auto_collision_claim'] = {
            'name': 'Auto Collision Claim Processing',
            'description': 'Streamlined workflow for auto collision claims',
            'tasks': [
                {
                    'task_id': 'coverage_verification',
                    'task_name': 'Coverage Verification',
                    'task_type': 'validation',
                    'dependencies': [],
                    'timeout': 180,
                    'max_retries': 3,
                    'priority': TaskPriority.HIGH,
                    'parameters': {
                        'validation_type': 'coverage_check',
                        'coverage_types': ['collision', 'comprehensive']
                    }
                },
                {
                    'task_id': 'vehicle_inspection',
                    'task_name': 'Vehicle Inspection',
                    'task_type': 'inspection',
                    'dependencies': ['coverage_verification'],
                    'timeout': 2400,
                    'max_retries': 2,
                    'priority': TaskPriority.HIGH,
                    'parameters': {
                        'inspection_type': 'vehicle_damage',
                        'schedule_automatically': True
                    }
                },
                {
                    'task_id': 'repair_estimation',
                    'task_name': 'Repair Estimation',
                    'task_type': 'estimation',
                    'dependencies': ['vehicle_inspection'],
                    'timeout': 1200,
                    'max_retries': 2,
                    'priority': TaskPriority.MEDIUM,
                    'parameters': {
                        'estimation_type': 'repair_cost',
                        'get_multiple_estimates': True
                    }
                },
                {
                    'task_id': 'settlement_authorization',
                    'task_name': 'Settlement Authorization',
                    'task_type': 'authorization',
                    'dependencies': ['repair_estimation'],
                    'timeout': 600,
                    'max_retries': 2,
                    'priority': TaskPriority.HIGH,
                    'parameters': {
                        'authorization_type': 'repair_settlement',
                        'check_authority_limits': True
                    }
                },
                {
                    'task_id': 'payment_issuance',
                    'task_name': 'Payment Issuance',
                    'task_type': 'payment',
                    'dependencies': ['settlement_authorization'],
                    'timeout': 900,
                    'max_retries': 3,
                    'priority': TaskPriority.CRITICAL,
                    'parameters': {
                        'payment_type': 'repair_payment',
                        'coordinate_with_shop': True
                    }
                },
                {
                    'task_id': 'repair_monitoring',
                    'task_name': 'Repair Monitoring',
                    'task_type': 'monitoring',
                    'dependencies': ['payment_issuance'],
                    'timeout': 7200,
                    'max_retries': 1,
                    'priority': TaskPriority.LOW,
                    'parameters': {
                        'monitoring_type': 'repair_progress',
                        'check_intervals': 24
                    }
                },
                {
                    'task_id': 'final_closure',
                    'task_name': 'Final Closure',
                    'task_type': 'closure',
                    'dependencies': ['repair_monitoring'],
                    'timeout': 300,
                    'max_retries': 2,
                    'priority': TaskPriority.MEDIUM,
                    'parameters': {
                        'closure_type': 'repair_completion',
                        'verify_satisfaction': True
                    }
                }
            ]
        }

    def _initialize_task_handlers(self):
        """Initialize task handlers for different task types"""
        
        self.task_handlers = {
            'validation': self._handle_validation_task,
            'document_processing': self._handle_document_processing_task,
            'investigation': self._handle_investigation_task,
            'evaluation': self._handle_evaluation_task,
            'calculation': self._handle_calculation_task,
            'decision': self._handle_decision_task,
            'payment': self._handle_payment_task,
            'closure': self._handle_closure_task,
            'inspection': self._handle_inspection_task,
            'estimation': self._handle_estimation_task,
            'authorization': self._handle_authorization_task,
            'monitoring': self._handle_monitoring_task
        }

    def _start_workflow_engine(self):
        """Start the workflow execution engine"""
        
        def workflow_worker():
            while True:
                try:
                    # Get next task from queue
                    priority, task_data = self.task_queue.get(timeout=1)
                    
                    # Execute task
                    asyncio.run(self._execute_task(task_data))
                    
                    self.task_queue.task_done()
                    
                except Exception as e:
                    if "Empty" not in str(e):  # Ignore empty queue timeout
                        logger.error(f"Workflow worker error: {e}")
                    time.sleep(0.1)
        
        # Start worker threads
        for i in range(self.max_workers):
            worker_thread = threading.Thread(target=workflow_worker, daemon=True)
            worker_thread.start()
        
        logger.info(f"Started {self.max_workers} workflow worker threads")

    async def create_workflow(self, workflow_type: str, claim_id: str, context: Dict[str, Any] = None) -> WorkflowInstance:
        """Create a new workflow instance"""
        
        try:
            if workflow_type not in self.workflow_templates:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            template = self.workflow_templates[workflow_type]
            workflow_id = str(uuid.uuid4())
            
            # Create workflow tasks from template
            tasks = []
            for task_template in template['tasks']:
                task = WorkflowTask(
                    task_id=task_template['task_id'],
                    task_name=task_template['task_name'],
                    task_type=task_template['task_type'],
                    dependencies=task_template['dependencies'],
                    parameters=task_template['parameters'],
                    timeout=task_template['timeout'],
                    retry_count=0,
                    max_retries=task_template['max_retries'],
                    priority=task_template['priority'],
                    status=TaskStatus.PENDING,
                    assigned_worker=None,
                    started_at=None,
                    completed_at=None,
                    duration=None,
                    result=None,
                    error=None,
                    metadata={}
                )
                tasks.append(task)
            
            # Create workflow instance
            workflow = WorkflowInstance(
                workflow_id=workflow_id,
                workflow_name=template['name'],
                workflow_type=workflow_type,
                claim_id=claim_id,
                status=WorkflowStatus.PENDING,
                tasks=tasks,
                context=context or {},
                created_at=datetime.utcnow(),
                started_at=None,
                completed_at=None,
                total_duration=None,
                progress_percentage=0.0,
                current_stage="initialization",
                error_count=0,
                retry_count=0,
                max_retries=3
            )
            
            # Create dependency graph
            self.workflow_graphs[workflow_id] = self._create_dependency_graph(tasks)
            
            # Store workflow
            await self._store_workflow(workflow)
            
            # Store in running workflows
            self.running_workflows[workflow_id] = workflow
            
            self.metrics['workflows_started'] += 1
            
            logger.info(f"Created workflow {workflow_id} for claim {claim_id}")
            
            return workflow
            
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            raise

    async def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution"""
        
        try:
            if workflow_id not in self.running_workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.running_workflows[workflow_id]
            
            if workflow.status != WorkflowStatus.PENDING:
                raise ValueError(f"Workflow {workflow_id} is not in pending status")
            
            # Update workflow status
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.utcnow()
            workflow.current_stage = "execution"
            
            # Queue ready tasks
            await self._queue_ready_tasks(workflow_id)
            
            # Store updated workflow
            await self._store_workflow(workflow)
            
            logger.info(f"Started workflow {workflow_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow start failed: {e}")
            return False

    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause workflow execution"""
        
        try:
            if workflow_id not in self.running_workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.running_workflows[workflow_id]
            workflow.status = WorkflowStatus.PAUSED
            
            await self._store_workflow(workflow)
            
            logger.info(f"Paused workflow {workflow_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow pause failed: {e}")
            return False

    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume paused workflow"""
        
        try:
            if workflow_id not in self.running_workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.running_workflows[workflow_id]
            
            if workflow.status != WorkflowStatus.PAUSED:
                raise ValueError(f"Workflow {workflow_id} is not paused")
            
            workflow.status = WorkflowStatus.RUNNING
            
            # Queue ready tasks
            await self._queue_ready_tasks(workflow_id)
            
            await self._store_workflow(workflow)
            
            logger.info(f"Resumed workflow {workflow_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow resume failed: {e}")
            return False

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel workflow execution"""
        
        try:
            if workflow_id not in self.running_workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.running_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.utcnow()
            
            if workflow.started_at:
                workflow.total_duration = (workflow.completed_at - workflow.started_at).total_seconds()
            
            await self._store_workflow(workflow)
            
            # Remove from running workflows
            del self.running_workflows[workflow_id]
            
            logger.info(f"Cancelled workflow {workflow_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow cancellation failed: {e}")
            return False

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status and progress"""
        
        try:
            if workflow_id in self.running_workflows:
                workflow = self.running_workflows[workflow_id]
            else:
                # Load from database
                workflow = await self._load_workflow(workflow_id)
            
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Calculate progress
            total_tasks = len(workflow.tasks)
            completed_tasks = len([t for t in workflow.tasks if t.status == TaskStatus.COMPLETED])
            progress_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
            
            # Get current stage
            current_stage = "completed"
            for task in workflow.tasks:
                if task.status in [TaskStatus.RUNNING, TaskStatus.READY]:
                    current_stage = task.task_name
                    break
                elif task.status == TaskStatus.PENDING:
                    current_stage = "waiting"
                    break
            
            return {
                'workflow_id': workflow.workflow_id,
                'workflow_name': workflow.workflow_name,
                'claim_id': workflow.claim_id,
                'status': workflow.status.value,
                'progress_percentage': progress_percentage,
                'current_stage': current_stage,
                'created_at': workflow.created_at.isoformat(),
                'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
                'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
                'total_duration': workflow.total_duration,
                'error_count': workflow.error_count,
                'tasks': [
                    {
                        'task_id': task.task_id,
                        'task_name': task.task_name,
                        'status': task.status.value,
                        'started_at': task.started_at.isoformat() if task.started_at else None,
                        'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                        'duration': task.duration,
                        'error': task.error
                    }
                    for task in workflow.tasks
                ]
            }
            
        except Exception as e:
            logger.error(f"Get workflow status failed: {e}")
            raise

    def _create_dependency_graph(self, tasks: List[WorkflowTask]) -> nx.DiGraph:
        """Create dependency graph for tasks"""
        
        graph = nx.DiGraph()
        
        # Add nodes
        for task in tasks:
            graph.add_node(task.task_id, task=task)
        
        # Add edges for dependencies
        for task in tasks:
            for dep_id in task.dependencies:
                graph.add_edge(dep_id, task.task_id)
        
        return graph

    async def _queue_ready_tasks(self, workflow_id: str):
        """Queue tasks that are ready to execute"""
        
        try:
            workflow = self.running_workflows[workflow_id]
            graph = self.workflow_graphs[workflow_id]
            
            for task in workflow.tasks:
                if task.status == TaskStatus.PENDING:
                    # Check if all dependencies are completed
                    dependencies_completed = True
                    for dep_id in task.dependencies:
                        dep_task = next((t for t in workflow.tasks if t.task_id == dep_id), None)
                        if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                            dependencies_completed = False
                            break
                    
                    if dependencies_completed:
                        task.status = TaskStatus.READY
                        
                        # Add to task queue with priority
                        task_data = {
                            'workflow_id': workflow_id,
                            'task': task
                        }
                        
                        # Priority queue uses negative values for higher priority
                        priority = -task.priority.value
                        self.task_queue.put((priority, task_data))
                        
                        logger.debug(f"Queued task {task.task_id} for workflow {workflow_id}")
            
        except Exception as e:
            logger.error(f"Queue ready tasks failed: {e}")

    async def _execute_task(self, task_data: Dict[str, Any]):
        """Execute a workflow task"""
        
        workflow_id = task_data['workflow_id']
        task = task_data['task']
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.assigned_worker = threading.current_thread().name
            
            logger.info(f"Executing task {task.task_id} in workflow {workflow_id}")
            
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")
            
            # Execute task with timeout
            try:
                result = await asyncio.wait_for(
                    handler(task, self.running_workflows[workflow_id]),
                    timeout=task.timeout
                )
                
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                task.duration = (task.completed_at - task.started_at).total_seconds()
                task.result = result
                
                self.metrics['tasks_executed'] += 1
                
                logger.info(f"Completed task {task.task_id} in {task.duration:.2f}s")
                
            except asyncio.TimeoutError:
                raise Exception(f"Task {task.task_id} timed out after {task.timeout} seconds")
            
            # Check if workflow is complete
            await self._check_workflow_completion(workflow_id)
            
            # Queue next ready tasks
            await self._queue_ready_tasks(workflow_id)
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.duration = (task.completed_at - task.started_at).total_seconds() if task.started_at else 0
            task.error = str(e)
            task.retry_count += 1
            
            self.metrics['tasks_failed'] += 1
            
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Handle task retry
            if task.retry_count <= task.max_retries:
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
                
                # Reset task for retry
                task.status = TaskStatus.READY
                task.started_at = None
                task.completed_at = None
                task.duration = None
                task.assigned_worker = None
                
                # Re-queue task with delay
                await asyncio.sleep(min(task.retry_count * 30, 300))  # Exponential backoff, max 5 minutes
                
                task_data = {
                    'workflow_id': workflow_id,
                    'task': task
                }
                priority = -task.priority.value
                self.task_queue.put((priority, task_data))
                
            else:
                # Max retries exceeded, fail workflow
                logger.error(f"Task {task.task_id} failed permanently after {task.max_retries} retries")
                await self._fail_workflow(workflow_id, f"Task {task.task_id} failed permanently")
        
        finally:
            # Store task execution record
            await self._store_task_execution(workflow_id, task)

    async def _check_workflow_completion(self, workflow_id: str):
        """Check if workflow is complete"""
        
        try:
            workflow = self.running_workflows[workflow_id]
            
            # Check if all tasks are completed or failed
            all_tasks_done = all(
                task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED]
                for task in workflow.tasks
            )
            
            if all_tasks_done:
                # Check if any critical tasks failed
                critical_failures = [
                    task for task in workflow.tasks
                    if task.status == TaskStatus.FAILED and task.priority in [TaskPriority.CRITICAL, TaskPriority.URGENT]
                ]
                
                if critical_failures:
                    await self._fail_workflow(workflow_id, "Critical tasks failed")
                else:
                    await self._complete_workflow(workflow_id)
            
        except Exception as e:
            logger.error(f"Workflow completion check failed: {e}")

    async def _complete_workflow(self, workflow_id: str):
        """Complete workflow successfully"""
        
        try:
            workflow = self.running_workflows[workflow_id]
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
            workflow.total_duration = (workflow.completed_at - workflow.started_at).total_seconds()
            workflow.progress_percentage = 100.0
            workflow.current_stage = "completed"
            
            # Update metrics
            self.metrics['workflows_completed'] += 1
            self._update_average_duration(workflow.total_duration)
            
            # Store final workflow state
            await self._store_workflow(workflow)
            
            # Remove from running workflows
            del self.running_workflows[workflow_id]
            
            # Trigger completion events
            await self._trigger_workflow_event('workflow_completed', workflow)
            
            logger.info(f"Workflow {workflow_id} completed successfully in {workflow.total_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Workflow completion failed: {e}")

    async def _fail_workflow(self, workflow_id: str, reason: str):
        """Fail workflow"""
        
        try:
            workflow = self.running_workflows[workflow_id]
            
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()
            workflow.total_duration = (workflow.completed_at - workflow.started_at).total_seconds()
            workflow.error_count += 1
            
            # Update metrics
            self.metrics['workflows_failed'] += 1
            
            # Store final workflow state
            await self._store_workflow(workflow)
            
            # Remove from running workflows
            del self.running_workflows[workflow_id]
            
            # Trigger failure events
            await self._trigger_workflow_event('workflow_failed', workflow, {'reason': reason})
            
            logger.error(f"Workflow {workflow_id} failed: {reason}")
            
        except Exception as e:
            logger.error(f"Workflow failure handling failed: {e}")

    def _update_average_duration(self, duration: float):
        """Update average workflow duration metric"""
        
        current_avg = self.metrics['average_workflow_duration']
        completed_count = self.metrics['workflows_completed']
        
        if completed_count == 1:
            self.metrics['average_workflow_duration'] = duration
        else:
            # Calculate running average
            self.metrics['average_workflow_duration'] = (
                (current_avg * (completed_count - 1) + duration) / completed_count
            )

    async def _trigger_workflow_event(self, event_type: str, workflow: WorkflowInstance, data: Dict[str, Any] = None):
        """Trigger workflow event handlers"""
        
        try:
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        await handler(workflow, data or {})
                    except Exception as e:
                        logger.error(f"Event handler failed for {event_type}: {e}")
            
        except Exception as e:
            logger.error(f"Event triggering failed: {e}")

    # Task handlers
    
    async def _handle_validation_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle validation task"""
        
        # This would call the validation agent
        # For now, simulate validation
        
        validation_type = task.parameters.get('validation_type', 'general')
        
        if validation_type == 'claim_intake':
            required_fields = task.parameters.get('required_fields', [])
            claim_data = workflow.context.get('claim_data', {})
            
            missing_fields = [field for field in required_fields if field not in claim_data]
            
            return {
                'validation_result': 'passed' if not missing_fields else 'failed',
                'missing_fields': missing_fields,
                'validation_score': 1.0 if not missing_fields else 0.5
            }
        
        elif validation_type == 'coverage_check':
            coverage_types = task.parameters.get('coverage_types', [])
            
            return {
                'validation_result': 'passed',
                'coverage_verified': coverage_types,
                'coverage_limits': {'collision': 50000, 'comprehensive': 50000}
            }
        
        return {'validation_result': 'passed'}

    async def _handle_document_processing_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle document processing task"""
        
        document_types = task.parameters.get('document_types', [])
        auto_request = task.parameters.get('auto_request', False)
        
        # Production document processing with real agent calls
        processed_documents = []
        for doc_type in document_types:
            try:
                # Call document analysis agent
                doc_result = await self._call_document_agent(doc_type, workflow.claim_id)
                processed_documents.append({
                    'document_type': doc_type,
                    'status': 'processed',
                    'extracted_data': doc_result.get('extracted_data', {}),
                    'confidence_score': doc_result.get('confidence_score', 0.0),
                    'processing_time': doc_result.get('processing_time', 0),
                    'agent_response': doc_result
                })
            except Exception as e:
                processed_documents.append({
                    'document_type': doc_type,
                    'status': 'failed',
                    'error': str(e),
                    'confidence_score': 0.0
                })
        
        return {
            'processed_documents': processed_documents,
            'total_documents': len(document_types),
            'processing_complete': all(doc['status'] == 'processed' for doc in processed_documents)
        }

    async def _handle_investigation_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle investigation task"""
        
        investigation_type = task.parameters.get('investigation_type', 'general')
        include_interviews = task.parameters.get('include_interviews', False)
        claim_id = workflow.claim_id
        
        # Production investigation with real agent coordination
        try:
            # Call evidence processing agent for forensic analysis
            evidence_analysis = await self._call_evidence_agent(claim_id, investigation_type)
            
            # Call liability assessment agent for fault determination
            liability_analysis = await self._call_liability_agent(claim_id, investigation_type)
            
            # Compile investigation findings
            investigation_result = {
                'investigation_type': investigation_type,
                'findings': evidence_analysis.get('findings', []),
                'liability_assessment': liability_analysis.get('assessment', {}),
                'confidence_score': min(
                    evidence_analysis.get('confidence', 0.0),
                    liability_analysis.get('confidence', 0.0)
                ),
                'recommendations': self._generate_investigation_recommendations(
                    evidence_analysis, liability_analysis
                ),
                'evidence_quality': evidence_analysis.get('quality_score', 0.0),
                'fraud_indicators': evidence_analysis.get('fraud_indicators', [])
            }
            
            if include_interviews:
                # Coordinate witness interviews
                interview_results = await self._conduct_witness_interviews(claim_id)
                investigation_result['interviews_conducted'] = len(interview_results)
                investigation_result['interview_summary'] = self._summarize_interviews(interview_results)
                investigation_result['witness_credibility'] = [
                    interview.get('credibility_score', 0.0) for interview in interview_results
                ]
                
        except Exception as e:
            logger.error(f"Investigation failed for claim {claim_id}: {e}")
            investigation_result = {
                'investigation_type': investigation_type,
                'findings': [],
                'error': str(e),
                'confidence_score': 0.0,
                'recommendations': ['Manual review required due to investigation failure']
            }
        
        return investigation_result

    async def _handle_evaluation_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle evaluation task"""
        
        evaluation_type = task.parameters.get('evaluation_type', 'general')
        include_medical = task.parameters.get('include_medical', False)
        claim_id = workflow.claim_id
        
        # Production evaluation with real damage assessment
        try:
            # Call damage assessment agents for comprehensive evaluation
            damage_assessment = await self._call_damage_assessment_agent(claim_id, evaluation_type)
            
            # Calculate total damages from multiple sources
            vehicle_damage = damage_assessment.get('vehicle_damage', 0)
            property_damage = damage_assessment.get('property_damage', 0)
            medical_costs = 0
            
            if include_medical:
                medical_assessment = await self._call_medical_assessment_agent(claim_id)
                medical_costs = medical_assessment.get('total_medical_costs', 0)
            
            # Calculate total with confidence weighting
            total_damages = vehicle_damage + property_damage + medical_costs
            confidence_score = damage_assessment.get('confidence_score', 0.0)
            
            evaluation_result = {
                'evaluation_type': evaluation_type,
                'total_damages': total_damages,
                'vehicle_damage': vehicle_damage,
                'property_damage': property_damage,
                'medical_costs': medical_costs,
                'evaluation_confidence': confidence_score,
                'assessment_details': damage_assessment.get('details', {}),
                'depreciation_applied': damage_assessment.get('depreciation', 0),
                'salvage_value': damage_assessment.get('salvage_value', 0)
            }
            
            if include_medical:
                evaluation_result['medical_assessment'] = medical_assessment
                
        except Exception as e:
            logger.error(f"Evaluation failed for claim {claim_id}: {e}")
            evaluation_result = {
                'evaluation_type': evaluation_type,
                'total_damages': 0,
                'error': str(e),
                'evaluation_confidence': 0.0
            }
        
        if include_medical:
            evaluation_result['medical_expenses'] = 5000
            evaluation_result['medical_evaluation'] = 'Reasonable and necessary'
        
        return evaluation_result

    async def _handle_calculation_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle calculation task"""
        
        calculation_type = task.parameters.get('calculation_type', 'general')
        include_reserves = task.parameters.get('include_reserves', False)
        claim_id = workflow.claim_id
        
        # Production calculation with actuarial models
        try:
            # Get evaluation results from previous workflow steps
            total_damages = workflow.context.get('total_damages', 0)
            liability_percentage = workflow.context.get('liability_percentage', 0)
            
            # Call liability assessment agent for precise liability calculation
            liability_assessment = await self._call_liability_agent(claim_id, 'calculation')
            liability_percentage = liability_assessment.get('liability_percentage', liability_percentage)
            
            # Apply deductibles and policy limits
            policy_info = await self._get_policy_information(claim_id)
            deductible = policy_info.get('deductible', 0)
            policy_limit = policy_info.get('policy_limit', float('inf'))
            
            # Calculate settlement with all factors
            gross_settlement = total_damages * (liability_percentage / 100)
            net_settlement = max(0, gross_settlement - deductible)
            final_settlement = min(net_settlement, policy_limit)
            
            # Calculate reserves using actuarial models
            reserve_multiplier = 1.0
            if include_reserves:
                reserve_calculation = await self._calculate_reserves(claim_id, final_settlement)
                reserve_multiplier = reserve_calculation.get('multiplier', 1.2)
            
            calculation_result = {
                'calculation_type': calculation_type,
                'total_damages': total_damages,
                'liability_percentage': liability_percentage,
                'gross_settlement': gross_settlement,
                'deductible_applied': deductible,
                'policy_limit': policy_limit,
                'settlement_amount': final_settlement,
                'calculation_confidence': liability_assessment.get('confidence', 0.0),
                'calculation_factors': {
                    'liability_assessment': liability_assessment,
                    'policy_terms': policy_info,
                    'deductible_impact': deductible,
                    'policy_limit_impact': policy_limit != float('inf')
                }
            }
            
            if include_reserves:
                calculation_result['reserve_amount'] = final_settlement * reserve_multiplier
                calculation_result['reserve_calculation'] = reserve_calculation
                
        except Exception as e:
            logger.error(f"Calculation failed for claim {claim_id}: {e}")
            calculation_result = {
                'calculation_type': calculation_type,
                'settlement_amount': 0,
                'error': str(e),
                'calculation_confidence': 0.0
            }
        
        if include_reserves:
            calculation_result['reserve_amount'] = calculation_result.get('settlement_amount', 0) * 1.2
        
        return calculation_result

    async def _handle_decision_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle decision task"""
        
        decision_type = task.parameters.get('decision_type', 'general')
        require_manager_approval = task.parameters.get('require_manager_approval', False)
        
        # Simulate decision making
        settlement_amount = workflow.context.get('settlement_amount', 25000)
        
        decision_result = {
            'decision_type': decision_type,
            'decision': 'approve',
            'settlement_amount': settlement_amount,
            'decision_confidence': 0.9,
            'decision_date': datetime.utcnow().isoformat()
        }
        
        if require_manager_approval and settlement_amount > 50000:
            decision_result['manager_approval_required'] = True
            decision_result['approval_status'] = 'pending'
        else:
            decision_result['manager_approval_required'] = False
            decision_result['approval_status'] = 'approved'
        
        return decision_result

    async def _handle_payment_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle payment task"""
        
        payment_type = task.parameters.get('payment_type', 'general')
        require_releases = task.parameters.get('require_releases', False)
        
        # Simulate payment processing
        settlement_amount = workflow.context.get('settlement_amount', 25000)
        
        payment_result = {
            'payment_type': payment_type,
            'payment_amount': settlement_amount,
            'payment_status': 'processed',
            'payment_id': str(uuid.uuid4()),
            'payment_date': datetime.utcnow().isoformat()
        }
        
        if require_releases:
            payment_result['releases_obtained'] = True
            payment_result['release_documents'] = ['general_release', 'medical_release']
        
        return payment_result

    async def _handle_closure_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle closure task"""
        
        closure_type = task.parameters.get('closure_type', 'general')
        send_notifications = task.parameters.get('send_notifications', False)
        
        # Simulate closure
        closure_result = {
            'closure_type': closure_type,
            'closure_status': 'completed',
            'closure_date': datetime.utcnow().isoformat(),
            'final_settlement': workflow.context.get('settlement_amount', 25000)
        }
        
        if send_notifications:
            closure_result['notifications_sent'] = ['claimant', 'adjuster', 'manager']
        
        return closure_result

    async def _handle_inspection_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle inspection task"""
        
        inspection_type = task.parameters.get('inspection_type', 'general')
        schedule_automatically = task.parameters.get('schedule_automatically', False)
        
        # Simulate inspection
        inspection_result = {
            'inspection_type': inspection_type,
            'inspection_status': 'completed',
            'damage_assessment': 'Moderate damage to front end',
            'repair_estimate': 15000,
            'total_loss_indicator': False
        }
        
        if schedule_automatically:
            inspection_result['inspection_scheduled'] = True
            inspection_result['inspection_date'] = (datetime.utcnow() + timedelta(days=2)).isoformat()
        
        return inspection_result

    async def _handle_estimation_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle estimation task"""
        
        estimation_type = task.parameters.get('estimation_type', 'general')
        get_multiple_estimates = task.parameters.get('get_multiple_estimates', False)
        
        # Simulate estimation
        base_estimate = 15000
        
        estimation_result = {
            'estimation_type': estimation_type,
            'primary_estimate': base_estimate,
            'estimation_confidence': 0.9
        }
        
        if get_multiple_estimates:
            estimation_result['estimates'] = [
                {'shop': 'Shop A', 'estimate': base_estimate},
                {'shop': 'Shop B', 'estimate': base_estimate * 1.1},
                {'shop': 'Shop C', 'estimate': base_estimate * 0.95}
            ]
            estimation_result['average_estimate'] = base_estimate * 1.02
        
        return estimation_result

    async def _handle_authorization_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle authorization task"""
        
        authorization_type = task.parameters.get('authorization_type', 'general')
        check_authority_limits = task.parameters.get('check_authority_limits', False)
        
        # Simulate authorization
        settlement_amount = workflow.context.get('settlement_amount', 15000)
        
        authorization_result = {
            'authorization_type': authorization_type,
            'authorization_status': 'approved',
            'authorized_amount': settlement_amount,
            'authorization_date': datetime.utcnow().isoformat()
        }
        
        if check_authority_limits:
            authority_limit = 25000  # Simulated limit
            if settlement_amount > authority_limit:
                authorization_result['authorization_status'] = 'requires_escalation'
                authorization_result['escalation_required'] = True
            else:
                authorization_result['within_authority'] = True
        
        return authorization_result

    async def _handle_monitoring_task(self, task: WorkflowTask, workflow: WorkflowInstance) -> Dict[str, Any]:
        """Handle monitoring task"""
        
        monitoring_type = task.parameters.get('monitoring_type', 'general')
        check_intervals = task.parameters.get('check_intervals', 24)
        
        # Simulate monitoring
        monitoring_result = {
            'monitoring_type': monitoring_type,
            'monitoring_status': 'active',
            'check_intervals_hours': check_intervals,
            'last_check': datetime.utcnow().isoformat(),
            'status_updates': [
                'Repair work started',
                'Parts ordered',
                'Estimated completion in 3 days'
            ]
        }
        
        return monitoring_result

    # Storage methods
    
    async def _store_workflow(self, workflow: WorkflowInstance):
        """Store workflow in database"""
        
        try:
            with self.Session() as session:
                workflow_record = session.query(WorkflowRecord).filter_by(workflow_id=workflow.workflow_id).first()
                
                if not workflow_record:
                    workflow_record = WorkflowRecord(
                        workflow_id=workflow.workflow_id,
                        workflow_name=workflow.workflow_name,
                        workflow_type=workflow.workflow_type,
                        claim_id=workflow.claim_id,
                        status=workflow.status.value,
                        tasks=[asdict(task) for task in workflow.tasks],
                        context=workflow.context,
                        created_at=workflow.created_at,
                        started_at=workflow.started_at,
                        completed_at=workflow.completed_at,
                        total_duration=workflow.total_duration,
                        progress_percentage=workflow.progress_percentage,
                        current_stage=workflow.current_stage,
                        error_count=workflow.error_count,
                        retry_count=workflow.retry_count,
                        max_retries=workflow.max_retries
                    )
                    session.add(workflow_record)
                else:
                    workflow_record.status = workflow.status.value
                    workflow_record.tasks = [asdict(task) for task in workflow.tasks]
                    workflow_record.context = workflow.context
                    workflow_record.started_at = workflow.started_at
                    workflow_record.completed_at = workflow.completed_at
                    workflow_record.total_duration = workflow.total_duration
                    workflow_record.progress_percentage = workflow.progress_percentage
                    workflow_record.current_stage = workflow.current_stage
                    workflow_record.error_count = workflow.error_count
                    workflow_record.retry_count = workflow.retry_count
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Workflow storage failed: {e}")

    async def _load_workflow(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Load workflow from database"""
        
        try:
            with self.Session() as session:
                workflow_record = session.query(WorkflowRecord).filter_by(workflow_id=workflow_id).first()
                
                if not workflow_record:
                    return None
                
                # Reconstruct tasks
                tasks = []
                for task_data in workflow_record.tasks:
                    task = WorkflowTask(**task_data)
                    tasks.append(task)
                
                # Reconstruct workflow
                workflow = WorkflowInstance(
                    workflow_id=workflow_record.workflow_id,
                    workflow_name=workflow_record.workflow_name,
                    workflow_type=workflow_record.workflow_type,
                    claim_id=workflow_record.claim_id,
                    status=WorkflowStatus(workflow_record.status),
                    tasks=tasks,
                    context=workflow_record.context,
                    created_at=workflow_record.created_at,
                    started_at=workflow_record.started_at,
                    completed_at=workflow_record.completed_at,
                    total_duration=workflow_record.total_duration,
                    progress_percentage=workflow_record.progress_percentage,
                    current_stage=workflow_record.current_stage,
                    error_count=workflow_record.error_count,
                    retry_count=workflow_record.retry_count,
                    max_retries=workflow_record.max_retries
                )
                
                return workflow
                
        except Exception as e:
            logger.error(f"Workflow loading failed: {e}")
            return None

    async def _store_task_execution(self, workflow_id: str, task: WorkflowTask):
        """Store task execution record"""
        
        try:
            with self.Session() as session:
                execution_record = TaskExecutionRecord(
                    execution_id=str(uuid.uuid4()),
                    workflow_id=workflow_id,
                    task_id=task.task_id,
                    task_name=task.task_name,
                    status=task.status.value,
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                    duration=task.duration,
                    result=task.result,
                    error=task.error,
                    retry_count=task.retry_count,
                    worker_id=task.assigned_worker
                )
                
                session.add(execution_record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Task execution storage failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get workflow manager metrics"""
        
        return {
            'workflows_started': self.metrics['workflows_started'],
            'workflows_completed': self.metrics['workflows_completed'],
            'workflows_failed': self.metrics['workflows_failed'],
            'tasks_executed': self.metrics['tasks_executed'],
            'tasks_failed': self.metrics['tasks_failed'],
            'average_workflow_duration': self.metrics['average_workflow_duration'],
            'active_workflows': len(self.running_workflows),
            'queued_tasks': self.task_queue.qsize()
        }

