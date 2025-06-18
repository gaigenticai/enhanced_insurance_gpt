"""
Workflow Executor - Production Ready Implementation
Handles execution of complex multi-step workflows with error handling and monitoring
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import time

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, Summary
import redis

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
workflow_executions_total = Counter('workflow_executions_total', 'Total workflow executions', ['workflow_type', 'status'])
workflow_duration = Histogram('workflow_execution_duration_seconds', 'Time spent executing workflows')
workflow_step_duration = Summary('workflow_step_duration_seconds', 'Time spent executing workflow steps', ['step_name'])
active_workflow_executions = Gauge('active_workflow_executions', 'Number of active workflow executions')

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"

@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    step_id: str
    name: str
    agent_endpoint: str
    input_mapping: Dict[str, str]
    output_mapping: Dict[str, str]
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: int = 5
    required: bool = True
    condition: Optional[str] = None
    parallel_group: Optional[str] = None
    depends_on: List[str] = None

@dataclass
class StepExecution:
    """Tracks execution of a workflow step"""
    step_id: str
    status: StepStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    retry_count: int = 0

@dataclass
class WorkflowExecution:
    """Tracks execution of an entire workflow"""
    execution_id: str
    workflow_id: str
    status: StepStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = None
    output_data: Optional[Dict[str, Any]] = None
    step_executions: Dict[str, StepExecution] = None
    error_message: Optional[str] = None
    total_execution_time: Optional[float] = None
    
    def __post_init__(self):
        if self.step_executions is None:
            self.step_executions = {}

class WorkflowExecutor:
    """
    Production-ready Workflow Executor
    Handles complex multi-step workflow execution with error handling, retries, and monitoring
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.redis_client = redis.from_url(self.redis_url)
        
        # Execution tracking
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # HTTP session for agent communication
        self.session = None
        
        logger.info("WorkflowExecutor initialized successfully")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def execute_workflow(self, 
                             workflow_id: str,
                             steps: List[WorkflowStep],
                             input_data: Dict[str, Any],
                             execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> str:
        """
        Execute a workflow with the given steps and input data
        Returns execution ID for tracking
        """
        
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow(),
            input_data=input_data
        )
        
        self.active_executions[execution_id] = execution
        active_workflow_executions.inc()
        
        logger.info(f"Starting workflow execution {execution_id} for workflow {workflow_id}")
        
        try:
            with workflow_duration.time():
                if execution_mode == ExecutionMode.SEQUENTIAL:
                    await self._execute_sequential(execution, steps)
                elif execution_mode == ExecutionMode.PARALLEL:
                    await self._execute_parallel(execution, steps)
                elif execution_mode == ExecutionMode.CONDITIONAL:
                    await self._execute_conditional(execution, steps)
                else:
                    raise ValueError(f"Unsupported execution mode: {execution_mode}")
            
            execution.status = StepStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.total_execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            workflow_executions_total.labels(workflow_type=workflow_id, status='completed').inc()
            logger.info(f"Workflow execution {execution_id} completed successfully in {execution.total_execution_time:.2f}s")
            
        except Exception as e:
            execution.status = StepStatus.FAILED
            execution.completed_at = datetime.utcnow()
            execution.error_message = str(e)
            execution.total_execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            workflow_executions_total.labels(workflow_type=workflow_id, status='failed').inc()
            logger.error(f"Workflow execution {execution_id} failed: {e}")
            logger.error(traceback.format_exc())
            
        finally:
            active_workflow_executions.dec()
            await self._store_execution_result(execution)
        
        return execution_id

    async def _execute_sequential(self, execution: WorkflowExecution, steps: List[WorkflowStep]):
        """Execute workflow steps sequentially"""
        
        current_data = execution.input_data.copy()
        
        for step in steps:
            if not self._should_execute_step(step, current_data, execution):
                execution.step_executions[step.step_id] = StepExecution(
                    step_id=step.step_id,
                    status=StepStatus.SKIPPED
                )
                continue
            
            step_result = await self._execute_step(step, current_data, execution)
            
            if step_result.status == StepStatus.COMPLETED and step_result.output_data:
                # Merge step output into current data for next steps
                current_data.update(step_result.output_data)
            elif step_result.status == StepStatus.FAILED and step.required:
                raise Exception(f"Required step {step.step_id} failed: {step_result.error_message}")
        
        execution.output_data = current_data

    async def _execute_parallel(self, execution: WorkflowExecution, steps: List[WorkflowStep]):
        """Execute workflow steps in parallel groups"""
        
        current_data = execution.input_data.copy()
        
        # Group steps by parallel_group
        parallel_groups = {}
        sequential_steps = []
        
        for step in steps:
            if step.parallel_group:
                if step.parallel_group not in parallel_groups:
                    parallel_groups[step.parallel_group] = []
                parallel_groups[step.parallel_group].append(step)
            else:
                sequential_steps.append(step)
        
        # Execute sequential steps first
        for step in sequential_steps:
            if not self._should_execute_step(step, current_data, execution):
                execution.step_executions[step.step_id] = StepExecution(
                    step_id=step.step_id,
                    status=StepStatus.SKIPPED
                )
                continue
            
            step_result = await self._execute_step(step, current_data, execution)
            
            if step_result.status == StepStatus.COMPLETED and step_result.output_data:
                current_data.update(step_result.output_data)
            elif step_result.status == StepStatus.FAILED and step.required:
                raise Exception(f"Required step {step.step_id} failed: {step_result.error_message}")
        
        # Execute parallel groups
        for group_name, group_steps in parallel_groups.items():
            logger.info(f"Executing parallel group: {group_name}")
            
            # Filter steps that should be executed
            executable_steps = [s for s in group_steps 
                              if self._should_execute_step(s, current_data, execution)]
            
            if not executable_steps:
                continue
            
            # Execute steps in parallel
            tasks = [self._execute_step(step, current_data, execution) for step in executable_steps]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(executable_steps, results):
                if isinstance(result, Exception):
                    if step.required:
                        raise result
                    else:
                        logger.warning(f"Non-required step {step.step_id} failed: {result}")
                elif result.status == StepStatus.COMPLETED and result.output_data:
                    current_data.update(result.output_data)
        
        execution.output_data = current_data

    async def _execute_conditional(self, execution: WorkflowExecution, steps: List[WorkflowStep]):
        """Execute workflow steps with conditional logic"""
        
        current_data = execution.input_data.copy()
        executed_steps = set()
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(steps)
        
        # Execute steps based on dependencies and conditions
        while len(executed_steps) < len(steps):
            ready_steps = []
            
            for step in steps:
                if step.step_id in executed_steps:
                    continue
                
                # Check if dependencies are satisfied
                if step.depends_on:
                    if not all(dep in executed_steps for dep in step.depends_on):
                        continue
                
                # Check if step should be executed
                if self._should_execute_step(step, current_data, execution):
                    ready_steps.append(step)
                else:
                    execution.step_executions[step.step_id] = StepExecution(
                        step_id=step.step_id,
                        status=StepStatus.SKIPPED
                    )
                    executed_steps.add(step.step_id)
            
            if not ready_steps:
                # No more steps can be executed
                break
            
            # Execute ready steps
            for step in ready_steps:
                step_result = await self._execute_step(step, current_data, execution)
                executed_steps.add(step.step_id)
                
                if step_result.status == StepStatus.COMPLETED and step_result.output_data:
                    current_data.update(step_result.output_data)
                elif step_result.status == StepStatus.FAILED and step.required:
                    raise Exception(f"Required step {step.step_id} failed: {step_result.error_message}")
        
        execution.output_data = current_data

    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for conditional execution"""
        
        graph = {}
        for step in steps:
            graph[step.step_id] = step.depends_on or []
        
        return graph

    def _should_execute_step(self, step: WorkflowStep, current_data: Dict[str, Any], 
                           execution: WorkflowExecution) -> bool:
        """Determine if a step should be executed based on conditions"""
        
        if not step.condition:
            return True
        
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            condition = step.condition
            
            # Replace variables in condition with actual values
            for key, value in current_data.items():
                condition = condition.replace(f"${key}", str(value))
            
            # Evaluate simple conditions
            if "==" in condition:
                left, right = condition.split("==")
                return left.strip() == right.strip()
            elif "!=" in condition:
                left, right = condition.split("!=")
                return left.strip() != right.strip()
            elif ">" in condition:
                left, right = condition.split(">")
                return float(left.strip()) > float(right.strip())
            elif "<" in condition:
                left, right = condition.split("<")
                return float(left.strip()) < float(right.strip())
            else:
                # Default to True for unknown conditions
                return True
                
        except Exception as e:
            logger.warning(f"Error evaluating condition '{step.condition}': {e}")
            return True

    async def _execute_step(self, step: WorkflowStep, input_data: Dict[str, Any], 
                          execution: WorkflowExecution) -> StepExecution:
        """Execute a single workflow step"""
        
        step_execution = StepExecution(
            step_id=step.step_id,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow(),
            input_data=input_data.copy()
        )
        
        execution.step_executions[step.step_id] = step_execution
        
        logger.info(f"Executing step {step.step_id}: {step.name}")
        
        with workflow_step_duration.labels(step_name=step.name).time():
            try:
                # Map input data according to step configuration
                mapped_input = self._map_step_input(step, input_data)
                
                # Call agent endpoint
                output_data = await self._call_agent_endpoint(
                    step.agent_endpoint,
                    mapped_input,
                    step.timeout_seconds
                )
                
                # Map output data according to step configuration
                mapped_output = self._map_step_output(step, output_data)
                
                step_execution.status = StepStatus.COMPLETED
                step_execution.output_data = mapped_output
                
                logger.info(f"Step {step.step_id} completed successfully")
                
            except Exception as e:
                step_execution.status = StepStatus.FAILED
                step_execution.error_message = str(e)
                
                # Retry logic
                if step_execution.retry_count < step.max_retries:
                    step_execution.retry_count += 1
                    step_execution.status = StepStatus.RETRYING
                    
                    logger.warning(f"Step {step.step_id} failed, retrying ({step_execution.retry_count}/{step.max_retries})")
                    
                    await asyncio.sleep(step.retry_delay * step_execution.retry_count)
                    return await self._execute_step(step, input_data, execution)
                else:
                    logger.error(f"Step {step.step_id} failed permanently: {e}")
            
            finally:
                step_execution.completed_at = datetime.utcnow()
                if step_execution.started_at:
                    step_execution.execution_time = (
                        step_execution.completed_at - step_execution.started_at
                    ).total_seconds()
        
        return step_execution

    def _map_step_input(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map input data according to step input mapping"""
        
        if not step.input_mapping:
            return input_data
        
        mapped_data = {}
        
        for target_key, source_key in step.input_mapping.items():
            if source_key in input_data:
                mapped_data[target_key] = input_data[source_key]
            else:
                logger.warning(f"Input mapping key '{source_key}' not found in input data for step {step.step_id}")
        
        return mapped_data

    def _map_step_output(self, step: WorkflowStep, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map output data according to step output mapping"""
        
        if not step.output_mapping:
            return output_data
        
        mapped_data = {}
        
        for target_key, source_key in step.output_mapping.items():
            if source_key in output_data:
                mapped_data[target_key] = output_data[source_key]
            else:
                logger.warning(f"Output mapping key '{source_key}' not found in output data for step {step.step_id}")
        
        return mapped_data

    async def _call_agent_endpoint(self, endpoint: str, data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Call an agent endpoint with the given data"""
        
        if not self.session:
            raise RuntimeError("HTTP session not initialized. Use async context manager.")
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'WorkflowExecutor/1.0'
            }
            
            async with self.session.post(
                endpoint,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Agent endpoint returned {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            raise Exception(f"Agent endpoint {endpoint} timed out after {timeout} seconds")
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP error calling agent endpoint {endpoint}: {e}")

    async def _store_execution_result(self, execution: WorkflowExecution):
        """Store execution result in Redis for persistence"""
        
        try:
            execution_data = asdict(execution)
            
            # Convert datetime objects to ISO strings
            def convert_datetime(obj):
                if isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(v) for v in obj]
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                else:
                    return obj
            
            execution_data = convert_datetime(execution_data)
            
            self.redis_client.setex(
                f"workflow_execution:{execution.execution_id}",
                3600 * 24 * 7,  # 7 days TTL
                json.dumps(execution_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to store execution result in Redis: {e}")

    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status by ID"""
        
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        
        # Try to load from Redis
        try:
            execution_data = self.redis_client.get(f"workflow_execution:{execution_id}")
            if execution_data:
                data = json.loads(execution_data)
                
                # Convert ISO strings back to datetime objects
                def convert_datetime(obj):
                    if isinstance(obj, dict):
                        result = {}
                        for k, v in obj.items():
                            if k.endswith('_at') and isinstance(v, str):
                                try:
                                    result[k] = datetime.fromisoformat(v)
                                except:
                                    result[k] = v
                            else:
                                result[k] = convert_datetime(v)
                        return result
                    elif isinstance(obj, list):
                        return [convert_datetime(v) for v in obj]
                    else:
                        return obj
                
                data = convert_datetime(data)
                
                # Reconstruct step executions
                if 'step_executions' in data and data['step_executions']:
                    step_executions = {}
                    for step_id, step_data in data['step_executions'].items():
                        step_executions[step_id] = StepExecution(**step_data)
                    data['step_executions'] = step_executions
                
                return WorkflowExecution(**data)
                
        except Exception as e:
            logger.error(f"Failed to load execution from Redis: {e}")
        
        return None

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution"""
        
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        
        if execution.status not in [StepStatus.RUNNING, StepStatus.PENDING]:
            return False
        
        execution.status = StepStatus.FAILED
        execution.completed_at = datetime.utcnow()
        execution.error_message = "Execution cancelled by user"
        
        # Cancel running steps
        for step_execution in execution.step_executions.values():
            if step_execution.status == StepStatus.RUNNING:
                step_execution.status = StepStatus.FAILED
                step_execution.error_message = "Cancelled"
                step_execution.completed_at = datetime.utcnow()
        
        await self._store_execution_result(execution)
        workflow_executions_total.labels(workflow_type=execution.workflow_id, status='cancelled').inc()
        
        logger.info(f"Workflow execution {execution_id} cancelled")
        return True

    async def get_execution_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        
        active_count = len([e for e in self.active_executions.values() 
                          if e.status == StepStatus.RUNNING])
        
        status_counts = {}
        for execution in self.active_executions.values():
            status = execution.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "active_executions": active_count,
            "total_executions": len(self.active_executions),
            "status_breakdown": status_counts
        }

    async def cleanup_completed_executions(self, older_than_hours: int = 24):
        """Clean up completed executions older than specified hours"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        executions_to_remove = []
        
        for execution_id, execution in self.active_executions.items():
            if (execution.status in [StepStatus.COMPLETED, StepStatus.FAILED] and
                execution.completed_at and execution.completed_at < cutoff_time):
                executions_to_remove.append(execution_id)
        
        for execution_id in executions_to_remove:
            del self.active_executions[execution_id]
        
        logger.info(f"Cleaned up {len(executions_to_remove)} completed executions")
        return len(executions_to_remove)

    async def shutdown(self):
        """Graceful shutdown of the workflow executor"""
        
        logger.info("Shutting down WorkflowExecutor...")
        
        # Cancel all running executions
        for execution_id in list(self.active_executions.keys()):
            await self.cancel_execution(execution_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        logger.info("WorkflowExecutor shutdown complete")

# Factory function for easy instantiation
def create_workflow_executor(redis_url: str = None) -> WorkflowExecutor:
    """Create and configure a WorkflowExecutor instance"""
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return WorkflowExecutor(redis_url=redis_url)

# Example usage
if __name__ == "__main__":
    async def test_workflow_executor():
        """Test the workflow executor functionality"""
        
        # Define workflow steps
        steps = [
            WorkflowStep(
                step_id="document_analysis",
                name="Analyze Documents",
                agent_endpoint="http://localhost:8001/analyze-documents",
                input_mapping={"documents": "input_documents"},
                output_mapping={"extracted_data": "document_data"},
                timeout_seconds=300
            ),
            WorkflowStep(
                step_id="risk_assessment",
                name="Assess Risk",
                agent_endpoint="http://localhost:8002/assess-risk",
                input_mapping={"policy_data": "document_data"},
                output_mapping={"risk_score": "risk_score"},
                timeout_seconds=180,
                depends_on=["document_analysis"]
            ),
            WorkflowStep(
                step_id="decision_making",
                name="Make Decision",
                agent_endpoint="http://localhost:8003/make-decision",
                input_mapping={"risk_score": "risk_score", "policy_data": "document_data"},
                output_mapping={"decision": "final_decision"},
                timeout_seconds=120,
                depends_on=["risk_assessment"],
                condition="$risk_score < 8"
            )
        ]
        
        # Test workflow execution
        async with create_workflow_executor() as executor:
            execution_id = await executor.execute_workflow(
                workflow_id="test_underwriting",
                steps=steps,
                input_data={
                    "input_documents": ["policy_app.pdf", "id_copy.pdf"],
                    "customer_id": "CUST123"
                },
                execution_mode=ExecutionMode.CONDITIONAL
            )
            
            print(f"Started workflow execution: {execution_id}")
            
            # Monitor execution
            while True:
                execution = await executor.get_execution_status(execution_id)
                if execution and execution.status in [StepStatus.COMPLETED, StepStatus.FAILED]:
                    break
                await asyncio.sleep(1)
            
            print(f"Workflow completed with status: {execution.status}")
    
    # Run test
    # asyncio.run(test_workflow_executor())

