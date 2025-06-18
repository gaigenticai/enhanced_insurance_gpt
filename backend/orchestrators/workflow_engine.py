"""
Insurance AI Agent System - Workflow Management System
Production-ready workflow engine with state management, retry logic, and monitoring
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import json
from dataclasses import dataclass, asdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, update, func
import structlog
import redis.asyncio as redis
from contextlib import asynccontextmanager

from backend.shared.models import Workflow, AgentExecution, Organization
from backend.shared.schemas import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    WorkflowStatus, AgentExecutionStatus
)
from backend.shared.services import BaseService, ServiceException, EventService
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils

logger = structlog.get_logger(__name__)

class WorkflowState(str, Enum):
    """Workflow state enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class WorkflowPriority(int, Enum):
    """Workflow priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    NORMAL = 5

@dataclass
class WorkflowStep:
    """Workflow step definition"""
    name: str
    agent_name: str
    input_mapping: Dict[str, str]
    output_mapping: Dict[str, str]
    retry_count: int = 3
    timeout_seconds: int = 300
    required: bool = True
    condition: Optional[str] = None  # Python expression to evaluate

@dataclass
class WorkflowDefinition:
    """Workflow definition with steps and configuration"""
    name: str
    version: str
    description: str
    steps: List[WorkflowStep]
    timeout_seconds: int = 3600
    max_retries: int = 3
    retry_delay_seconds: int = 60

class WorkflowEngine:
    """
    Core workflow execution engine with state management and monitoring
    """
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
        self.event_service = EventService(redis_client)
        self.logger = structlog.get_logger("workflow_engine")
        
        # Workflow definitions registry
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        
        # Active workflow instances
        self.active_workflows: Dict[uuid.UUID, Dict[str, Any]] = {}
        
        # Circuit breaker for agent failures
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Initialize built-in workflow definitions
        self._initialize_workflow_definitions()
    
    def _initialize_workflow_definitions(self):
        """Initialize built-in workflow definitions"""
        
        # Underwriting workflow definition
        underwriting_steps = [
            WorkflowStep(
                name="document_analysis",
                agent_name="document_analysis_agent",
                input_mapping={"documents": "input_data.documents"},
                output_mapping={"extracted_data": "document_analysis_result"}
            ),
            WorkflowStep(
                name="validation",
                agent_name="validation_agent",
                input_mapping={"submission_data": "input_data", "extracted_data": "document_analysis_result"},
                output_mapping={"validation_result": "validation_result"}
            ),
            WorkflowStep(
                name="risk_assessment",
                agent_name="decision_engine_agent",
                input_mapping={"submission_data": "input_data", "validation_result": "validation_result"},
                output_mapping={"risk_assessment": "risk_assessment"}
            ),
            WorkflowStep(
                name="decision",
                agent_name="decision_engine_agent",
                input_mapping={"risk_assessment": "risk_assessment"},
                output_mapping={"decision": "underwriting_decision"},
                condition="risk_assessment.risk_score is not None"
            ),
            WorkflowStep(
                name="communication",
                agent_name="communication_agent",
                input_mapping={"decision": "underwriting_decision", "submission_data": "input_data"},
                output_mapping={"communication_result": "communication_result"}
            )
        ]
        
        self.workflow_definitions["underwriting"] = WorkflowDefinition(
            name="underwriting",
            version="1.0.0",
            description="Standard underwriting workflow",
            steps=underwriting_steps,
            timeout_seconds=1800,  # 30 minutes
            max_retries=3
        )
        
        # Claims processing workflow definition
        claims_steps = [
            WorkflowStep(
                name="document_analysis",
                agent_name="document_analysis_agent",
                input_mapping={"documents": "input_data.documents"},
                output_mapping={"extracted_data": "document_analysis_result"}
            ),
            WorkflowStep(
                name="evidence_processing",
                agent_name="evidence_processing_agent",
                input_mapping={"claim_data": "input_data", "extracted_data": "document_analysis_result"},
                output_mapping={"evidence_analysis": "evidence_analysis"}
            ),
            WorkflowStep(
                name="liability_assessment",
                agent_name="liability_assessment_agent",
                input_mapping={"claim_data": "input_data", "evidence_analysis": "evidence_analysis"},
                output_mapping={"liability_result": "liability_assessment"}
            ),
            WorkflowStep(
                name="validation",
                agent_name="validation_agent",
                input_mapping={"claim_data": "input_data", "liability_result": "liability_assessment"},
                output_mapping={"validation_result": "validation_result"}
            ),
            WorkflowStep(
                name="automation_check",
                agent_name="automation_agent",
                input_mapping={"claim_data": "input_data", "validation_result": "validation_result"},
                output_mapping={"automation_result": "automation_result"}
            ),
            WorkflowStep(
                name="communication",
                agent_name="communication_agent",
                input_mapping={"automation_result": "automation_result", "claim_data": "input_data"},
                output_mapping={"communication_result": "communication_result"},
                condition="automation_result.requires_communication == True"
            )
        ]
        
        self.workflow_definitions["claims"] = WorkflowDefinition(
            name="claims",
            version="1.0.0",
            description="Standard claims processing workflow",
            steps=claims_steps,
            timeout_seconds=3600,  # 1 hour
            max_retries=3
        )
    
    async def start_workflow(
        self,
        workflow_id: uuid.UUID,
        workflow_type: str,
        input_data: Dict[str, Any],
        organization_id: uuid.UUID,
        priority: int = 5
    ) -> bool:
        """Start workflow execution"""
        
        try:
            # Get workflow definition
            if workflow_type not in self.workflow_definitions:
                raise ServiceException(f"Unknown workflow type: {workflow_type}")
            
            definition = self.workflow_definitions[workflow_type]
            
            # Initialize workflow state
            workflow_state = {
                "workflow_id": str(workflow_id),
                "workflow_type": workflow_type,
                "definition": asdict(definition),
                "input_data": input_data,
                "organization_id": str(organization_id),
                "priority": priority,
                "current_step": 0,
                "step_results": {},
                "context": input_data.copy(),
                "started_at": datetime.utcnow().isoformat(),
                "status": WorkflowState.RUNNING.value,
                "retry_count": 0,
                "error_history": []
            }
            
            # Store in active workflows
            self.active_workflows[workflow_id] = workflow_state
            
            # Store in Redis for persistence
            await self._save_workflow_state(workflow_id, workflow_state)
            
            # Update database
            workflow_service = BaseService(Workflow, self.db_session)
            await workflow_service.update(
                workflow_id,
                WorkflowUpdate(
                    status=WorkflowStatus.RUNNING,
                    started_at=datetime.utcnow(),
                    timeout_at=datetime.utcnow() + timedelta(seconds=definition.timeout_seconds)
                )
            )
            
            # Publish workflow started event
            await self.event_service.publish(
                "workflow_started",
                {
                    "workflow_id": str(workflow_id),
                    "workflow_type": workflow_type,
                    "organization_id": str(organization_id)
                },
                "workflows"
            )
            
            # Start execution
            asyncio.create_task(self._execute_workflow(workflow_id))
            
            self.logger.info(
                "Workflow started",
                workflow_id=str(workflow_id),
                workflow_type=workflow_type
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to start workflow",
                workflow_id=str(workflow_id),
                error=str(e)
            )
            raise
    
    async def _execute_workflow(self, workflow_id: uuid.UUID):
        """Execute workflow steps"""
        
        try:
            workflow_state = await self._get_workflow_state(workflow_id)
            if not workflow_state:
                raise ServiceException(f"Workflow state not found: {workflow_id}")
            
            definition = WorkflowDefinition(**workflow_state["definition"])
            
            # Execute steps sequentially
            while workflow_state["current_step"] < len(definition.steps):
                step = definition.steps[workflow_state["current_step"]]
                
                try:
                    # Check if step should be executed (condition evaluation)
                    if step.condition and not self._evaluate_condition(step.condition, workflow_state["context"]):
                        self.logger.info(
                            "Skipping step due to condition",
                            workflow_id=str(workflow_id),
                            step_name=step.name,
                            condition=step.condition
                        )
                        workflow_state["current_step"] += 1
                        continue
                    
                    # Check circuit breaker
                    if self._is_circuit_breaker_open(step.agent_name):
                        raise ServiceException(f"Circuit breaker open for agent: {step.agent_name}")
                    
                    # Execute step
                    step_result = await self._execute_workflow_step(workflow_id, step, workflow_state)
                    
                    # Update workflow state
                    workflow_state["step_results"][step.name] = step_result
                    workflow_state["current_step"] += 1
                    
                    # Update context with step outputs
                    self._update_context_from_step(workflow_state["context"], step, step_result)
                    
                    # Save state
                    await self._save_workflow_state(workflow_id, workflow_state)
                    
                    # Reset circuit breaker on success
                    self._reset_circuit_breaker(step.agent_name)
                    
                except Exception as e:
                    self.logger.error(
                        "Workflow step failed",
                        workflow_id=str(workflow_id),
                        step_name=step.name,
                        error=str(e)
                    )
                    
                    # Record error
                    workflow_state["error_history"].append({
                        "step": step.name,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Update circuit breaker
                    self._record_circuit_breaker_failure(step.agent_name)
                    
                    # Handle step failure
                    if step.required:
                        # Retry if possible
                        if workflow_state["retry_count"] < definition.max_retries:
                            workflow_state["retry_count"] += 1
                            await asyncio.sleep(definition.retry_delay_seconds)
                            continue
                        else:
                            # Fail workflow
                            await self._fail_workflow(workflow_id, f"Required step failed: {step.name}")
                            return
                    else:
                        # Skip optional step
                        workflow_state["current_step"] += 1
                        continue
            
            # All steps completed successfully
            await self._complete_workflow(workflow_id, workflow_state["context"])
            
        except Exception as e:
            await self._fail_workflow(workflow_id, str(e))
    
    async def _execute_workflow_step(
        self,
        workflow_id: uuid.UUID,
        step: WorkflowStep,
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        # Prepare input data for agent
        agent_input = self._prepare_agent_input(step, workflow_state["context"])
        
        # Create agent execution record
        execution = AgentExecution(
            workflow_id=workflow_id,
            agent_name=step.agent_name,
            agent_version="1.0.0",
            input_data=agent_input,
            status=AgentExecutionStatus.RUNNING
        )
        
        self.db_session.add(execution)
        await self.db_session.commit()
        await self.db_session.refresh(execution)
        
        start_time = datetime.utcnow()
        
        try:
            # Execute agent with timeout
            agent_result = await asyncio.wait_for(
                self._call_agent(step.agent_name, agent_input),
                timeout=step.timeout_seconds
            )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update execution record
            execution.status = AgentExecutionStatus.COMPLETED
            execution.output_data = agent_result
            execution.execution_time_ms = int(execution_time)
            execution.completed_at = datetime.utcnow()
            
            await self.db_session.commit()
            
            # Record metrics
            metrics.record_agent_execution(step.agent_name, execution_time / 1000, success=True)
            
            self.logger.info(
                "Workflow step completed",
                workflow_id=str(workflow_id),
                step_name=step.name,
                agent_name=step.agent_name,
                execution_time_ms=execution.execution_time_ms
            )
            
            return agent_result
            
        except asyncio.TimeoutError:
            # Handle timeout
            execution.status = AgentExecutionStatus.FAILED
            execution.error_message = f"Agent execution timeout after {step.timeout_seconds} seconds"
            execution.completed_at = datetime.utcnow()
            
            await self.db_session.commit()
            
            # Record metrics
            metrics.record_agent_execution(step.agent_name, 0, success=False)
            
            raise ServiceException(f"Agent {step.agent_name} execution timeout")
            
        except Exception as e:
            # Handle execution error
            execution.status = AgentExecutionStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            
            await self.db_session.commit()
            
            # Record metrics
            metrics.record_agent_execution(step.agent_name, 0, success=False)
            
            raise
    
    async def _call_agent(self, agent_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call agent with production HTTP API integration"""
        
        try:
            # Production agent API calls
            import httpx
            
            # Map agent names to service endpoints
            agent_endpoints = {
                "document_analysis": f"{self.config.get('DOCUMENT_SERVICE_URL', 'http://localhost:8006')}/analyze",
                "validation": f"{self.config.get('VALIDATION_SERVICE_URL', 'http://localhost:8007')}/validate", 
                "decision_engine": f"{self.config.get('DECISION_SERVICE_URL', 'http://localhost:8008')}/decide",
                "communication": f"{self.config.get('COMMUNICATION_SERVICE_URL', 'http://localhost:8009')}/send",
                "evidence_processing": f"{self.config.get('EVIDENCE_SERVICE_URL', 'http://localhost:8010')}/process",
                "risk_assessment": f"{self.config.get('RISK_SERVICE_URL', 'http://localhost:8011')}/assess",
                "compliance": f"{self.config.get('COMPLIANCE_SERVICE_URL', 'http://localhost:8012')}/check"
            }
            
            # Find matching endpoint
            endpoint = None
            for agent_key, url in agent_endpoints.items():
                if agent_key in agent_name.lower():
                    endpoint = url
                    break
            
            if not endpoint:
                # Default to generic agent endpoint
                endpoint = f"{self.config.get('AGENT_SERVICE_URL', 'http://localhost:8000')}/agents/{agent_name}"
            
            # Make HTTP request to agent service
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint,
                    json=input_data,
                    headers={
                        "Content-Type": "application/json",
                        "X-Workflow-ID": self.workflow_id,
                        "X-Request-Source": "workflow-engine"
                    }
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Agent call failed: {response.status_code} - {response.text}")
                    return {
                        "error": f"Agent call failed with status {response.status_code}",
                        "status": "failed"
                    }
                    
        except Exception as e:
            logger.error(f"Error calling agent {agent_name}: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _prepare_agent_input(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for agent based on input mapping"""
        
        agent_input = {}
        
        for input_key, context_path in step.input_mapping.items():
            try:
                # Advanced path resolution with nested object support
                value = self._resolve_context_path(context, context_path)
                agent_input[input_key] = value
            except KeyError:
                self.logger.warning(
                    "Input mapping path not found",
                    step_name=step.name,
                    input_key=input_key,
                    context_path=context_path
                )
        
        return agent_input
    
    def _resolve_context_path(self, context: Dict[str, Any], path: str) -> Any:
        """Resolve context path (e.g., 'input_data.documents')"""
        
        parts = path.split('.')
        current = context
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise KeyError(f"Path not found: {path}")
        
        return current
    
    def _update_context_from_step(
        self,
        context: Dict[str, Any],
        step: WorkflowStep,
        step_result: Dict[str, Any]
    ):
        """Update context with step output based on output mapping"""
        
        for output_key, context_key in step.output_mapping.items():
            if output_key in step_result:
                context[context_key] = step_result[output_key]
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate step condition"""
        
        try:
            # Production-ready condition evaluation with comprehensive operators
            import re
            import operator
            
            # Define safe operators for condition evaluation
            operators = {
                '==': operator.eq,
                '!=': operator.ne,
                '<': operator.lt,
                '<=': operator.le,
                '>': operator.gt,
                '>=': operator.ge,
                'in': lambda x, y: x in y,
                'not in': lambda x, y: x not in y,
                'is': operator.is_,
                'is not': operator.is_not,
                'and': operator.and_,
                'or': operator.or_,
                'not': operator.not_
            }
            
            # Handle null checks
            if "is not None" in condition:
                path = condition.split(" is not None")[0].strip()
                try:
                    value = self._resolve_context_path(context, path)
                    return value is not None
                except KeyError:
                    return False
            
            if "is None" in condition:
                path = condition.split(" is None")[0].strip()
                try:
                    value = self._resolve_context_path(context, path)
                    return value is None
                except KeyError:
                    return True
            
            # Handle boolean conditions
            if condition.lower() in ['true', 'false']:
                return condition.lower() == 'true'
            
            # Handle comparison operations
            for op_str, op_func in operators.items():
                if op_str in condition and op_str not in ['and', 'or', 'not', 'is', 'is not']:
                    parts = condition.split(op_str, 1)
                    if len(parts) == 2:
                        left_path = parts[0].strip()
                        right_value = parts[1].strip()
                        
                        try:
                            left_val = self._resolve_context_path(context, left_path)
                            
                            # Try to resolve right side as path, otherwise use as literal
                            try:
                                right_val = self._resolve_context_path(context, right_value)
                            except KeyError:
                                # Parse as literal value
                                if right_value.startswith('"') and right_value.endswith('"'):
                                    right_val = right_value[1:-1]  # String literal
                                elif right_value.startswith("'") and right_value.endswith("'"):
                                    right_val = right_value[1:-1]  # String literal
                                elif right_value.lower() in ['true', 'false']:
                                    right_val = right_value.lower() == 'true'  # Boolean literal
                                elif right_value.replace('.', '').replace('-', '').isdigit():
                                    right_val = float(right_value) if '.' in right_value else int(right_value)
                                else:
                                    right_val = right_value  # String value
                            
                            return op_func(left_val, right_val)
                            
                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(f"Error evaluating condition '{condition}': {e}")
                            return False
            
            # Handle logical operations (simplified)
            if ' and ' in condition:
                parts = condition.split(' and ')
                return all(self._evaluate_condition(part.strip(), context) for part in parts)
            
            if ' or ' in condition:
                parts = condition.split(' or ')
                return any(self._evaluate_condition(part.strip(), context) for part in parts)
            
            # Try to resolve as a path (for boolean values)
            try:
                value = self._resolve_context_path(context, condition)
                return bool(value)
            except KeyError:
                pass
            
            # Default to True for unknown conditions
            logger.warning(f"Unknown condition format: {condition}")
            return True
            
        except Exception:
            return False
    
    async def _complete_workflow(self, workflow_id: uuid.UUID, final_context: Dict[str, Any]):
        """Complete workflow successfully"""
        
        try:
            # Update workflow state
            workflow_state = await self._get_workflow_state(workflow_id)
            workflow_state["status"] = WorkflowState.COMPLETED.value
            workflow_state["completed_at"] = datetime.utcnow().isoformat()
            workflow_state["final_context"] = final_context
            
            await self._save_workflow_state(workflow_id, workflow_state)
            
            # Update database
            workflow_service = BaseService(Workflow, self.db_session)
            await workflow_service.update(
                workflow_id,
                WorkflowUpdate(
                    status=WorkflowStatus.COMPLETED,
                    completed_at=datetime.utcnow(),
                    output_data=final_context
                )
            )
            
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            # Publish workflow completed event
            await self.event_service.publish(
                "workflow_completed",
                {
                    "workflow_id": str(workflow_id),
                    "final_context": final_context
                },
                "workflows"
            )
            
            # Record metrics
            metrics.record_workflow(workflow_state["workflow_type"], "completed")
            
            self.logger.info(
                "Workflow completed successfully",
                workflow_id=str(workflow_id)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to complete workflow",
                workflow_id=str(workflow_id),
                error=str(e)
            )
    
    async def _fail_workflow(self, workflow_id: uuid.UUID, error_message: str):
        """Fail workflow with error"""
        
        try:
            # Update workflow state
            workflow_state = await self._get_workflow_state(workflow_id)
            workflow_state["status"] = WorkflowState.FAILED.value
            workflow_state["completed_at"] = datetime.utcnow().isoformat()
            workflow_state["error_message"] = error_message
            
            await self._save_workflow_state(workflow_id, workflow_state)
            
            # Update database
            workflow_service = BaseService(Workflow, self.db_session)
            await workflow_service.update(
                workflow_id,
                WorkflowUpdate(
                    status=WorkflowStatus.FAILED,
                    completed_at=datetime.utcnow(),
                    error_message=error_message
                )
            )
            
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            # Publish workflow failed event
            await self.event_service.publish(
                "workflow_failed",
                {
                    "workflow_id": str(workflow_id),
                    "error_message": error_message
                },
                "workflows"
            )
            
            # Record metrics
            metrics.record_workflow(workflow_state["workflow_type"], "failed")
            
            self.logger.error(
                "Workflow failed",
                workflow_id=str(workflow_id),
                error_message=error_message
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to fail workflow",
                workflow_id=str(workflow_id),
                error=str(e)
            )
    
    async def pause_workflow(self, workflow_id: uuid.UUID) -> bool:
        """Pause workflow execution"""
        
        try:
            workflow_state = await self._get_workflow_state(workflow_id)
            if not workflow_state:
                return False
            
            workflow_state["status"] = WorkflowState.PAUSED.value
            workflow_state["paused_at"] = datetime.utcnow().isoformat()
            
            await self._save_workflow_state(workflow_id, workflow_state)
            
            # Update database
            workflow_service = BaseService(Workflow, self.db_session)
            await workflow_service.update(
                workflow_id,
                WorkflowUpdate(status=WorkflowStatus.PENDING)  # Use PENDING for paused state
            )
            
            self.logger.info("Workflow paused", workflow_id=str(workflow_id))
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to pause workflow",
                workflow_id=str(workflow_id),
                error=str(e)
            )
            return False
    
    async def resume_workflow(self, workflow_id: uuid.UUID) -> bool:
        """Resume paused workflow"""
        
        try:
            workflow_state = await self._get_workflow_state(workflow_id)
            if not workflow_state or workflow_state["status"] != WorkflowState.PAUSED.value:
                return False
            
            workflow_state["status"] = WorkflowState.RUNNING.value
            workflow_state["resumed_at"] = datetime.utcnow().isoformat()
            
            await self._save_workflow_state(workflow_id, workflow_state)
            
            # Update database
            workflow_service = BaseService(Workflow, self.db_session)
            await workflow_service.update(
                workflow_id,
                WorkflowUpdate(status=WorkflowStatus.RUNNING)
            )
            
            # Resume execution
            asyncio.create_task(self._execute_workflow(workflow_id))
            
            self.logger.info("Workflow resumed", workflow_id=str(workflow_id))
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to resume workflow",
                workflow_id=str(workflow_id),
                error=str(e)
            )
            return False
    
    async def cancel_workflow(self, workflow_id: uuid.UUID) -> bool:
        """Cancel workflow execution"""
        
        try:
            workflow_state = await self._get_workflow_state(workflow_id)
            if not workflow_state:
                return False
            
            workflow_state["status"] = WorkflowState.CANCELLED.value
            workflow_state["cancelled_at"] = datetime.utcnow().isoformat()
            
            await self._save_workflow_state(workflow_id, workflow_state)
            
            # Update database
            workflow_service = BaseService(Workflow, self.db_session)
            await workflow_service.update(
                workflow_id,
                WorkflowUpdate(
                    status=WorkflowStatus.CANCELLED,
                    completed_at=datetime.utcnow()
                )
            )
            
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            # Publish workflow cancelled event
            await self.event_service.publish(
                "workflow_cancelled",
                {"workflow_id": str(workflow_id)},
                "workflows"
            )
            
            self.logger.info("Workflow cancelled", workflow_id=str(workflow_id))
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to cancel workflow",
                workflow_id=str(workflow_id),
                error=str(e)
            )
            return False
    
    async def get_workflow_status(self, workflow_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""
        
        try:
            workflow_state = await self._get_workflow_state(workflow_id)
            if not workflow_state:
                return None
            
            return {
                "workflow_id": workflow_id,
                "status": workflow_state["status"],
                "current_step": workflow_state["current_step"],
                "total_steps": len(workflow_state["definition"]["steps"]),
                "progress_percentage": (workflow_state["current_step"] / len(workflow_state["definition"]["steps"]) * 100),
                "started_at": workflow_state.get("started_at"),
                "completed_at": workflow_state.get("completed_at"),
                "error_message": workflow_state.get("error_message"),
                "retry_count": workflow_state.get("retry_count", 0)
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to get workflow status",
                workflow_id=str(workflow_id),
                error=str(e)
            )
            return None
    
    # Circuit breaker methods
    
    def _is_circuit_breaker_open(self, agent_name: str) -> bool:
        """Check if circuit breaker is open for agent"""
        
        if agent_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[agent_name]
        
        # Check if breaker is open and cooldown period has passed
        if breaker["state"] == "open":
            if datetime.utcnow() > breaker["next_attempt"]:
                breaker["state"] = "half_open"
                return False
            return True
        
        return False
    
    def _record_circuit_breaker_failure(self, agent_name: str):
        """Record circuit breaker failure"""
        
        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = {
                "failure_count": 0,
                "state": "closed",
                "next_attempt": None
            }
        
        breaker = self.circuit_breakers[agent_name]
        breaker["failure_count"] += 1
        
        # Open circuit breaker if failure threshold reached
        if breaker["failure_count"] >= 5:  # Threshold of 5 failures
            breaker["state"] = "open"
            breaker["next_attempt"] = datetime.utcnow() + timedelta(minutes=5)  # 5 minute cooldown
            
            self.logger.warning(
                "Circuit breaker opened",
                agent_name=agent_name,
                failure_count=breaker["failure_count"]
            )
    
    def _reset_circuit_breaker(self, agent_name: str):
        """Reset circuit breaker on success"""
        
        if agent_name in self.circuit_breakers:
            self.circuit_breakers[agent_name] = {
                "failure_count": 0,
                "state": "closed",
                "next_attempt": None
            }
    
    # State persistence methods
    
    async def _save_workflow_state(self, workflow_id: uuid.UUID, state: Dict[str, Any]):
        """Save workflow state to Redis"""
        
        try:
            key = f"workflow_state:{workflow_id}"
            await self.redis_client.setex(
                key,
                timedelta(hours=24),  # 24 hour TTL
                json.dumps(state, default=str)
            )
        except Exception as e:
            self.logger.error(
                "Failed to save workflow state",
                workflow_id=str(workflow_id),
                error=str(e)
            )
    
    async def _get_workflow_state(self, workflow_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get workflow state from Redis"""
        
        try:
            # Check active workflows first
            if workflow_id in self.active_workflows:
                return self.active_workflows[workflow_id]
            
            # Check Redis
            key = f"workflow_state:{workflow_id}"
            state_json = await self.redis_client.get(key)
            
            if state_json:
                state = json.loads(state_json)
                self.active_workflows[workflow_id] = state
                return state
            
            return None
            
        except Exception as e:
            self.logger.error(
                "Failed to get workflow state",
                workflow_id=str(workflow_id),
                error=str(e)
            )
            return None
    
    async def cleanup_completed_workflows(self, older_than_hours: int = 24):
        """Cleanup completed workflow states"""
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            
            # Get all workflow state keys
            keys = await self.redis_client.keys("workflow_state:*")
            
            for key in keys:
                try:
                    state_json = await self.redis_client.get(key)
                    if state_json:
                        state = json.loads(state_json)
                        
                        # Check if workflow is completed and old enough
                        if state["status"] in ["completed", "failed", "cancelled"]:
                            completed_at = datetime.fromisoformat(state.get("completed_at", ""))
                            if completed_at < cutoff_time:
                                await self.redis_client.delete(key)
                                
                                # Remove from active workflows if present
                                workflow_id = uuid.UUID(state["workflow_id"])
                                if workflow_id in self.active_workflows:
                                    del self.active_workflows[workflow_id]
                                
                except Exception as e:
                    self.logger.warning(
                        "Failed to cleanup workflow state",
                        key=key,
                        error=str(e)
                    )
            
            self.logger.info(
                "Workflow cleanup completed",
                older_than_hours=older_than_hours
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to cleanup workflows",
                error=str(e)
            )

# Global workflow engine instance
workflow_engine: Optional[WorkflowEngine] = None

async def get_workflow_engine() -> WorkflowEngine:
    """Get global workflow engine instance"""
    global workflow_engine
    
    if workflow_engine is None:
        async with get_db_session() as db_session:
            async with get_redis_client() as redis_client:
                workflow_engine = WorkflowEngine(db_session, redis_client)
    
    return workflow_engine

@asynccontextmanager
async def workflow_engine_context():
    """Context manager for workflow engine"""
    engine = await get_workflow_engine()
    try:
        yield engine
    finally:
        # Cleanup if needed
        pass

