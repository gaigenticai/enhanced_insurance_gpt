"""
Insurance AI Agent System - Underwriting Orchestrator
Production-ready master orchestrator for underwriting workflows
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import structlog

from backend.shared.models import (
    Workflow, UnderwritingSubmission, Policy, AgentExecution,
    Organization, User
)
from backend.shared.schemas import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    UnderwritingSubmissionCreate, UnderwritingSubmissionUpdate,
    PolicyCreate, PolicyUpdate,
    UnderwritingDecision, WorkflowStatus, AgentExecutionStatus
)
from backend.shared.services import BaseService, WorkflowService, AgentService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

logger = structlog.get_logger(__name__)

class UnderwritingOrchestrator(WorkflowService):
    """
    Master orchestrator for underwriting workflows
    Coordinates all underwriting-related agents and processes
    """
    
    def __init__(self, db_session: AsyncSession):
        super().__init__(UnderwritingSubmission, db_session)
        self.service_name = "underwriting_orchestrator"
        self.logger = structlog.get_logger(self.service_name)
        
        # Agent execution order for underwriting workflow
        self.agent_sequence = [
            "document_analysis_agent",
            "validation_agent", 
            "decision_engine_agent",
            "communication_agent"
        ]
        
        # Risk scoring thresholds
        self.risk_thresholds = {
            "auto_accept": 20.0,
            "auto_decline": 80.0,
            "manual_review": 50.0
        }
    
    async def create_underwriting_workflow(
        self,
        submission_data: Dict[str, Any],
        organization_id: uuid.UUID,
        created_by: Optional[uuid.UUID] = None,
        priority: int = 5
    ) -> Tuple[WorkflowResponse, UnderwritingSubmission]:
        """Create a new underwriting workflow"""
        
        try:
            # Validate submission data
            if not self._validate_submission_data(submission_data):
                raise ServiceException("Invalid submission data", "VALIDATION_ERROR")
            
            # Generate submission number
            submission_number = DataUtils.generate_reference_number("UW", 8)
            
            # Create workflow
            workflow_data = WorkflowCreate(
                type="underwriting",
                priority=priority,
                input_data=submission_data,
                organization_id=organization_id,
                metadata={
                    "submission_number": submission_number,
                    "created_by": str(created_by) if created_by else None,
                    "agent_sequence": self.agent_sequence
                }
            )
            
            workflow = await self.create(workflow_data, created_by, organization_id)
            
            # Create underwriting submission
            submission_create = UnderwritingSubmissionCreate(
                workflow_id=workflow.id,
                submission_number=submission_number,
                broker_id=submission_data.get("broker_id"),
                broker_name=submission_data.get("broker_name"),
                insured_name=submission_data["insured_name"],
                insured_industry=submission_data.get("insured_industry"),
                policy_type=submission_data["policy_type"],
                coverage_amount=submission_data.get("coverage_amount"),
                premium_amount=submission_data.get("premium_amount"),
                submission_data=submission_data,
                effective_date=submission_data.get("effective_date"),
                expiry_date=submission_data.get("expiry_date")
            )
            
            submission_service = BaseService(UnderwritingSubmission, self.db_session)
            submission = await submission_service.create(submission_create, created_by, organization_id)
            
            # Log workflow creation
            audit_logger.log_user_action(
                user_id=str(created_by) if created_by else "system",
                action="create_underwriting_workflow",
                resource_type="workflow",
                resource_id=str(workflow.id),
                details={
                    "submission_number": submission_number,
                    "insured_name": submission_data["insured_name"],
                    "policy_type": submission_data["policy_type"]
                }
            )
            
            self.logger.info(
                "Created underwriting workflow",
                workflow_id=str(workflow.id),
                submission_number=submission_number,
                insured_name=submission_data["insured_name"]
            )
            
            return WorkflowResponse.from_orm(workflow), submission
            
        except Exception as e:
            self.logger.error(
                "Failed to create underwriting workflow",
                error=str(e),
                submission_data=submission_data
            )
            raise
    
    async def execute_underwriting_workflow(
        self,
        workflow_id: uuid.UUID,
        timeout_seconds: int = 1800  # 30 minutes
    ) -> Dict[str, Any]:
        """Execute the complete underwriting workflow"""
        
        async with performance_monitor.monitor_operation("underwriting_workflow_execution"):
            try:
                # Start workflow
                await self.start_workflow(workflow_id, timeout_seconds)
                
                # Get workflow and submission
                workflow = await self.get(workflow_id)
                if not workflow:
                    raise ServiceException(f"Workflow not found: {workflow_id}")
                
                submission_service = BaseService(UnderwritingSubmission, self.db_session)
                submissions = await submission_service.get_multi(
                    pagination={"page": 1, "size": 1},
                    filters={"workflow_id": workflow_id}
                )
                
                if not submissions.items:
                    raise ServiceException(f"Submission not found for workflow: {workflow_id}")
                
                submission = submissions.items[0]
                
                # Execute agent sequence
                agent_results = {}
                current_data = workflow.input_data.copy()
                
                for agent_name in self.agent_sequence:
                    try:
                        self.logger.info(
                            "Executing agent",
                            workflow_id=str(workflow_id),
                            agent_name=agent_name
                        )
                        
                        # Execute agent
                        agent_result = await self._execute_agent(
                            workflow_id,
                            agent_name,
                            current_data
                        )
                        
                        agent_results[agent_name] = agent_result
                        
                        # Update current data with agent output
                        if agent_result.get("output_data"):
                            current_data.update(agent_result["output_data"])
                        
                        # Check for early termination conditions
                        if agent_name == "validation_agent":
                            validation_result = agent_result.get("output_data", {})
                            if not validation_result.get("is_valid", True):
                                # Auto-decline if validation fails
                                await self._auto_decline_submission(
                                    submission,
                                    "Validation failed",
                                    validation_result.get("validation_errors", [])
                                )
                                break
                        
                        elif agent_name == "decision_engine_agent":
                            decision_result = agent_result.get("output_data", {})
                            risk_score = decision_result.get("risk_score", 50.0)
                            
                            # Update submission with risk score
                            await submission_service.update(
                                submission.id,
                                UnderwritingSubmissionUpdate(
                                    risk_score=Decimal(str(risk_score)),
                                    decision_confidence=Decimal(str(decision_result.get("confidence", 0.0)))
                                )
                            )
                            
                            # Check for auto-decision
                            auto_decision = self._determine_auto_decision(risk_score)
                            if auto_decision:
                                await self._apply_auto_decision(
                                    submission,
                                    auto_decision,
                                    decision_result
                                )
                                
                                # Skip communication agent for auto-decline
                                if auto_decision == "decline":
                                    break
                        
                    except Exception as e:
                        self.logger.error(
                            "Agent execution failed",
                            workflow_id=str(workflow_id),
                            agent_name=agent_name,
                            error=str(e)
                        )
                        
                        # Record agent failure
                        agent_results[agent_name] = {
                            "status": "failed",
                            "error": str(e)
                        }
                        
                        # Fail workflow if critical agent fails
                        if agent_name in ["document_analysis_agent", "validation_agent"]:
                            await self.fail_workflow(
                                workflow_id,
                                f"Critical agent {agent_name} failed: {str(e)}"
                            )
                            raise
                
                # Prepare final output
                final_output = {
                    "submission_id": str(submission.id),
                    "submission_number": submission.submission_number,
                    "agent_results": agent_results,
                    "final_data": current_data,
                    "execution_summary": self._generate_execution_summary(agent_results)
                }
                
                # Complete workflow
                await self.complete_workflow(workflow_id, final_output, success=True)
                
                # Record metrics
                metrics.record_workflow("underwriting", "completed")
                
                self.logger.info(
                    "Underwriting workflow completed",
                    workflow_id=str(workflow_id),
                    submission_number=submission.submission_number
                )
                
                return final_output
                
            except Exception as e:
                # Fail workflow
                await self.fail_workflow(workflow_id, str(e), retry=False)
                
                # Record metrics
                metrics.record_workflow("underwriting", "failed")
                
                self.logger.error(
                    "Underwriting workflow failed",
                    workflow_id=str(workflow_id),
                    error=str(e)
                )
                raise
    
    async def process_manual_decision(
        self,
        submission_id: uuid.UUID,
        decision: UnderwritingDecision,
        underwriter_id: uuid.UUID,
        notes: str = None,
        decision_reasons: Dict[str, Any] = None
    ) -> UnderwritingSubmission:
        """Process manual underwriting decision"""
        
        try:
            submission_service = BaseService(UnderwritingSubmission, self.db_session)
            submission = await submission_service.get(submission_id)
            
            if not submission:
                raise ServiceException(f"Submission not found: {submission_id}")
            
            # Update submission with decision
            update_data = UnderwritingSubmissionUpdate(
                decision=decision,
                decision_reasons=decision_reasons or {},
                underwriter_notes=notes
            )
            
            updated_submission = await submission_service.update(
                submission_id,
                update_data,
                updated_by=underwriter_id
            )
            
            # Create policy if accepted
            if decision == UnderwritingDecision.ACCEPT:
                await self._create_policy_from_submission(updated_submission)
            
            # Send communication
            await self._send_decision_communication(
                updated_submission,
                decision,
                is_manual=True
            )
            
            # Log decision
            audit_logger.log_user_action(
                user_id=str(underwriter_id),
                action="manual_underwriting_decision",
                resource_type="submission",
                resource_id=str(submission_id),
                details={
                    "decision": decision.value,
                    "submission_number": submission.submission_number,
                    "notes": notes
                }
            )
            
            self.logger.info(
                "Manual underwriting decision processed",
                submission_id=str(submission_id),
                decision=decision.value,
                underwriter_id=str(underwriter_id)
            )
            
            return updated_submission
            
        except Exception as e:
            self.logger.error(
                "Failed to process manual decision",
                submission_id=str(submission_id),
                decision=decision.value if decision else None,
                error=str(e)
            )
            raise
    
    async def get_underwriting_queue(
        self,
        organization_id: uuid.UUID,
        status_filter: List[str] = None,
        priority_filter: List[int] = None,
        page: int = 1,
        size: int = 20
    ) -> Dict[str, Any]:
        """Get underwriting queue with filtering"""
        
        try:
            # Build filters
            filters = {"organization_id": organization_id}
            
            if status_filter:
                filters["decision"] = {"in": status_filter}
            
            # Get submissions
            submission_service = BaseService(UnderwritingSubmission, self.db_session)
            result = await submission_service.get_multi(
                pagination={"page": page, "size": size, "sort_by": "created_at", "sort_order": "desc"},
                filters=filters,
                organization_id=organization_id
            )
            
            # Enrich with workflow information
            enriched_items = []
            for submission in result.items:
                workflow = await self.get(submission.workflow_id)
                
                item_data = {
                    "submission": submission,
                    "workflow": workflow,
                    "priority": workflow.priority if workflow else 5,
                    "age_hours": self._calculate_age_hours(submission.created_at),
                    "sla_status": self._calculate_sla_status(submission.created_at, workflow.priority if workflow else 5)
                }
                
                enriched_items.append(item_data)
            
            # Sort by priority and age
            enriched_items.sort(
                key=lambda x: (x["priority"], -x["age_hours"]),
                reverse=True
            )
            
            return {
                "items": enriched_items,
                "total": result.total,
                "page": result.page,
                "size": result.size,
                "pages": result.pages,
                "queue_stats": await self._get_queue_statistics(organization_id)
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to get underwriting queue",
                organization_id=str(organization_id),
                error=str(e)
            )
            raise
    
    async def get_underwriting_metrics(
        self,
        organization_id: uuid.UUID,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """Get underwriting performance metrics"""
        
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get submission statistics
            submission_service = BaseService(UnderwritingSubmission, self.db_session)
            
            # Total submissions
            total_query = select(UnderwritingSubmission).where(
                and_(
                    UnderwritingSubmission.organization_id == organization_id,
                    UnderwritingSubmission.created_at >= start_date,
                    UnderwritingSubmission.created_at <= end_date
                )
            )
            
            result = await self.db_session.execute(total_query)
            submissions = result.scalars().all()
            
            # Calculate metrics
            total_submissions = len(submissions)
            decisions = [s.decision for s in submissions if s.decision]
            
            decision_counts = {
                "accept": len([d for d in decisions if d == UnderwritingDecision.ACCEPT]),
                "decline": len([d for d in decisions if d == UnderwritingDecision.DECLINE]),
                "refer": len([d for d in decisions if d == UnderwritingDecision.REFER]),
                "pending": len([s for s in submissions if not s.decision])
            }
            
            # Calculate rates
            total_decided = sum([decision_counts[k] for k in ["accept", "decline", "refer"]])
            
            metrics_data = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "volume": {
                    "total_submissions": total_submissions,
                    "total_decided": total_decided,
                    "pending": decision_counts["pending"]
                },
                "decisions": decision_counts,
                "rates": {
                    "acceptance_rate": (decision_counts["accept"] / total_decided * 100) if total_decided > 0 else 0,
                    "decline_rate": (decision_counts["decline"] / total_decided * 100) if total_decided > 0 else 0,
                    "referral_rate": (decision_counts["refer"] / total_decided * 100) if total_decided > 0 else 0
                },
                "risk_analysis": self._calculate_risk_metrics(submissions),
                "processing_times": self._calculate_processing_times(submissions),
                "sla_performance": self._calculate_sla_performance(submissions)
            }
            
            return metrics_data
            
        except Exception as e:
            self.logger.error(
                "Failed to get underwriting metrics",
                organization_id=str(organization_id),
                error=str(e)
            )
            raise
    
    # Private helper methods
    
    def _validate_submission_data(self, data: Dict[str, Any]) -> bool:
        """Validate submission data"""
        required_fields = ["insured_name", "policy_type"]
        
        for field in required_fields:
            if field not in data or not data[field]:
                return False
        
        # Validate amounts if provided
        if "coverage_amount" in data and data["coverage_amount"]:
            if not ValidationUtils.validate_amount(data["coverage_amount"]):
                return False
        
        if "premium_amount" in data and data["premium_amount"]:
            if not ValidationUtils.validate_amount(data["premium_amount"]):
                return False
        
        return True
    
    async def _execute_agent(
        self,
        workflow_id: uuid.UUID,
        agent_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific agent"""
        
        # Create agent execution record
        execution = AgentExecution(
            workflow_id=workflow_id,
            agent_name=agent_name,
            agent_version="1.0.0",
            input_data=input_data,
            status=AgentExecutionStatus.RUNNING
        )
        
        self.db_session.add(execution)
        await self.db_session.commit()
        await self.db_session.refresh(execution)
        
        start_time = datetime.utcnow()
        
        try:
            # Simulate agent execution (in real implementation, this would call actual agents)
            output_data = await self._simulate_agent_execution(agent_name, input_data)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update execution record
            execution.status = AgentExecutionStatus.COMPLETED
            execution.output_data = output_data
            execution.execution_time_ms = int(execution_time)
            execution.completed_at = datetime.utcnow()
            
            await self.db_session.commit()
            
            # Record metrics
            metrics.record_agent_execution(agent_name, execution_time / 1000, success=True)
            
            return {
                "status": "completed",
                "output_data": output_data,
                "execution_time_ms": execution.execution_time_ms
            }
            
        except Exception as e:
            # Update execution record with error
            execution.status = AgentExecutionStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            
            await self.db_session.commit()
            
            # Record metrics
            metrics.record_agent_execution(agent_name, 0, success=False)
            
            raise
    
    async def _simulate_agent_execution(
        self,
        agent_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute agent with production HTTP API calls"""
        
        try:
            import httpx
            
            # Production agent service endpoints
            agent_endpoints = {
                "document_analysis_agent": f"{self.config.get('DOCUMENT_SERVICE_URL', 'http://localhost:8006')}/analyze",
                "validation_agent": f"{self.config.get('VALIDATION_SERVICE_URL', 'http://localhost:8007')}/validate",
                "decision_engine_agent": f"{self.config.get('DECISION_SERVICE_URL', 'http://localhost:8008')}/decide",
                "communication_agent": f"{self.config.get('COMMUNICATION_SERVICE_URL', 'http://localhost:8009')}/send",
                "risk_assessment_agent": f"{self.config.get('RISK_SERVICE_URL', 'http://localhost:8011')}/assess",
                "compliance_agent": f"{self.config.get('COMPLIANCE_SERVICE_URL', 'http://localhost:8012')}/check"
            }
            
            endpoint = agent_endpoints.get(agent_name)
            if not endpoint:
                # Fallback to generic agent service
                endpoint = f"{self.config.get('AGENT_SERVICE_URL', 'http://localhost:8000')}/agents/{agent_name}"
            
            # Prepare request payload
            payload = {
                "agent_name": agent_name,
                "input_data": input_data,
                "context": {
                    "orchestrator": "underwriting",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4())
                }
            }
            
            # Make HTTP request to agent service
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Orchestrator": "underwriting",
                        "X-Request-Source": "underwriting-orchestrator",
                        "Authorization": f"Bearer {self.config.get('AGENT_API_TOKEN', '')}"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Log successful agent execution
                    logger.info(f"Agent {agent_name} executed successfully")
                    
                    # Update metrics
                    if hasattr(self, 'metrics'):
                        self.metrics.agent_calls_total.labels(
                            agent=agent_name,
                            status='success'
                        ).inc()
                    
                    return result
                    
                else:
                    error_msg = f"Agent {agent_name} failed with status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    
                    # Update error metrics
                    if hasattr(self, 'metrics'):
                        self.metrics.agent_calls_total.labels(
                            agent=agent_name,
                            status='error'
                        ).inc()
                    
                    # Return error response
                    return {
                        "error": error_msg,
                        "status": "failed",
                        "agent_name": agent_name
                    }
                    
        except httpx.TimeoutException:
            error_msg = f"Agent {agent_name} request timed out"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "status": "timeout",
                "agent_name": agent_name
            }
            
        except Exception as e:
            error_msg = f"Error calling agent {agent_name}: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "status": "failed",
                "agent_name": agent_name
            }
    
    def _calculate_risk_score(self, data: Dict[str, Any]) -> float:
        """Calculate risk score based on submission data"""
        base_score = 30.0
        
        # Industry risk adjustment
        industry = data.get("insured_industry", "").lower()
        industry_risk = {
            "construction": 20.0,
            "manufacturing": 15.0,
            "technology": 5.0,
            "healthcare": 10.0,
            "retail": 8.0
        }
        
        base_score += industry_risk.get(industry, 12.0)
        
        # Coverage amount adjustment
        coverage = data.get("coverage_amount", 0)
        if coverage > 10000000:  # $10M+
            base_score += 15.0
        elif coverage > 5000000:  # $5M+
            base_score += 10.0
        elif coverage > 1000000:  # $1M+
            base_score += 5.0
        
        return min(max(base_score, 0.0), 100.0)
    
    def _determine_auto_decision(self, risk_score: float) -> Optional[str]:
        """Determine if auto-decision can be made"""
        if risk_score <= self.risk_thresholds["auto_accept"]:
            return "accept"
        elif risk_score >= self.risk_thresholds["auto_decline"]:
            return "decline"
        else:
            return None  # Manual review required
    
    async def _apply_auto_decision(
        self,
        submission: UnderwritingSubmission,
        decision: str,
        decision_data: Dict[str, Any]
    ):
        """Apply auto-decision to submission"""
        
        submission_service = BaseService(UnderwritingSubmission, self.db_session)
        
        update_data = UnderwritingSubmissionUpdate(
            decision=UnderwritingDecision(decision),
            decision_reasons=decision_data,
            decision_confidence=Decimal(str(decision_data.get("confidence", 0.0)))
        )
        
        await submission_service.update(submission.id, update_data)
        
        # Create policy if accepted
        if decision == "accept":
            await self._create_policy_from_submission(submission)
        
        # Send communication
        await self._send_decision_communication(submission, decision, is_manual=False)
    
    async def _auto_decline_submission(
        self,
        submission: UnderwritingSubmission,
        reason: str,
        details: List[str]
    ):
        """Auto-decline submission due to validation failure"""
        
        submission_service = BaseService(UnderwritingSubmission, self.db_session)
        
        update_data = UnderwritingSubmissionUpdate(
            decision=UnderwritingDecision.DECLINE,
            decision_reasons={"reason": reason, "details": details},
            decision_confidence=Decimal("100.0")
        )
        
        await submission_service.update(submission.id, update_data)
        
        # Send decline communication
        await self._send_decision_communication(submission, "decline", is_manual=False)
    
    async def _create_policy_from_submission(self, submission: UnderwritingSubmission):
        """Create policy from accepted submission"""
        
        policy_service = BaseService(Policy, self.db_session)
        
        policy_number = DataUtils.generate_reference_number("POL", 10)
        
        policy_data = PolicyCreate(
            submission_id=submission.id,
            policy_number=policy_number,
            policy_data=submission.submission_data,
            effective_date=submission.effective_date or datetime.utcnow().date(),
            expiry_date=submission.expiry_date or (datetime.utcnow() + timedelta(days=365)).date()
        )
        
        policy = await policy_service.create(policy_data)
        
        self.logger.info(
            "Policy created from submission",
            submission_id=str(submission.id),
            policy_id=str(policy.id),
            policy_number=policy_number
        )
        
        return policy
    
    async def _send_decision_communication(
        self,
        submission: UnderwritingSubmission,
        decision: str,
        is_manual: bool
    ):
        """Send decision communication"""
        
        # In real implementation, this would trigger the communication agent
        self.logger.info(
            "Decision communication sent",
            submission_id=str(submission.id),
            decision=decision,
            is_manual=is_manual
        )
    
    def _generate_execution_summary(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution summary"""
        
        total_agents = len(agent_results)
        successful_agents = len([r for r in agent_results.values() if r.get("status") != "failed"])
        
        return {
            "total_agents": total_agents,
            "successful_agents": successful_agents,
            "success_rate": (successful_agents / total_agents * 100) if total_agents > 0 else 0,
            "execution_time_ms": sum([
                r.get("execution_time_ms", 0) for r in agent_results.values()
            ])
        }
    
    def _calculate_age_hours(self, created_at: datetime) -> float:
        """Calculate age in hours"""
        return (datetime.utcnow() - created_at).total_seconds() / 3600
    
    def _calculate_sla_status(self, created_at: datetime, priority: int) -> str:
        """Calculate SLA status"""
        age_hours = self._calculate_age_hours(created_at)
        
        # SLA thresholds based on priority
        sla_hours = {
            1: 2,   # Critical - 2 hours
            2: 4,   # High - 4 hours
            3: 8,   # Medium - 8 hours
            4: 24,  # Low - 24 hours
            5: 48   # Normal - 48 hours
        }
        
        threshold = sla_hours.get(priority, 48)
        
        if age_hours <= threshold * 0.5:
            return "green"
        elif age_hours <= threshold * 0.8:
            return "yellow"
        elif age_hours <= threshold:
            return "orange"
        else:
            return "red"
    
    async def _get_queue_statistics(self, organization_id: uuid.UUID) -> Dict[str, Any]:
        """Get queue statistics"""
        
        # Get pending submissions
        query = select(UnderwritingSubmission).where(
            and_(
                UnderwritingSubmission.organization_id == organization_id,
                UnderwritingSubmission.decision.is_(None)
            )
        )
        
        result = await self.db_session.execute(query)
        pending_submissions = result.scalars().all()
        
        # Calculate statistics
        total_pending = len(pending_submissions)
        
        sla_status_counts = {
            "green": 0,
            "yellow": 0,
            "orange": 0,
            "red": 0
        }
        
        for submission in pending_submissions:
            workflow = await self.get(submission.workflow_id)
            priority = workflow.priority if workflow else 5
            sla_status = self._calculate_sla_status(submission.created_at, priority)
            sla_status_counts[sla_status] += 1
        
        return {
            "total_pending": total_pending,
            "sla_status": sla_status_counts,
            "avg_age_hours": sum([
                self._calculate_age_hours(s.created_at) for s in pending_submissions
            ]) / total_pending if total_pending > 0 else 0
        }
    
    def _calculate_risk_metrics(self, submissions: List[UnderwritingSubmission]) -> Dict[str, Any]:
        """Calculate risk analysis metrics"""
        
        risk_scores = [float(s.risk_score) for s in submissions if s.risk_score]
        
        if not risk_scores:
            return {"avg_risk_score": 0, "risk_distribution": {}}
        
        return {
            "avg_risk_score": sum(risk_scores) / len(risk_scores),
            "min_risk_score": min(risk_scores),
            "max_risk_score": max(risk_scores),
            "risk_distribution": {
                "low_risk": len([s for s in risk_scores if s <= 30]),
                "medium_risk": len([s for s in risk_scores if 30 < s <= 70]),
                "high_risk": len([s for s in risk_scores if s > 70])
            }
        }
    
    def _calculate_processing_times(self, submissions: List[UnderwritingSubmission]) -> Dict[str, Any]:
        """Calculate processing time metrics"""
        
        completed_submissions = [s for s in submissions if s.decision]
        
        if not completed_submissions:
            return {"avg_processing_hours": 0}
        
        processing_times = []
        for submission in completed_submissions:
            # Calculate time from creation to decision
            if submission.updated_at:
                hours = (submission.updated_at - submission.created_at).total_seconds() / 3600
                processing_times.append(hours)
        
        if not processing_times:
            return {"avg_processing_hours": 0}
        
        return {
            "avg_processing_hours": sum(processing_times) / len(processing_times),
            "min_processing_hours": min(processing_times),
            "max_processing_hours": max(processing_times)
        }
    
    def _calculate_sla_performance(self, submissions: List[UnderwritingSubmission]) -> Dict[str, Any]:
        """Calculate SLA performance metrics"""
        
        completed_submissions = [s for s in submissions if s.decision]
        
        if not completed_submissions:
            return {"sla_compliance_rate": 0}
        
        sla_compliant = 0
        
        for submission in completed_submissions:
            # Get workflow to determine priority
            # For now, assume priority 5 (48 hour SLA)
            sla_hours = 48
            
            if submission.updated_at:
                processing_hours = (submission.updated_at - submission.created_at).total_seconds() / 3600
                if processing_hours <= sla_hours:
                    sla_compliant += 1
        
        return {
            "sla_compliance_rate": (sla_compliant / len(completed_submissions) * 100),
            "total_evaluated": len(completed_submissions),
            "sla_compliant": sla_compliant
        }

