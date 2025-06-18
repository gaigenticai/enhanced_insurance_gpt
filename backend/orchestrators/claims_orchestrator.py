"""
Insurance AI Agent System - Claims Orchestrator
Production-ready master orchestrator for claims processing workflows
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
import structlog

from backend.shared.models import (
    Workflow, Claim, Evidence, ClaimActivity, Policy,
    Organization, User, AgentExecution
)
from backend.shared.schemas import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    ClaimCreate, ClaimUpdate, ClaimResponse,
    EvidenceCreate, EvidenceUpdate,
    ClaimActivityCreate,
    ClaimStatus, ClaimPriority, WorkflowStatus, AgentExecutionStatus
)
from backend.shared.services import BaseService, WorkflowService, AgentService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

logger = structlog.get_logger(__name__)

class ClaimsOrchestrator(WorkflowService):
    """
    Master orchestrator for claims processing workflows
    Coordinates all claims-related agents and processes
    """
    
    def __init__(self, db_session: AsyncSession):
        super().__init__(Claim, db_session)
        self.service_name = "claims_orchestrator"
        self.logger = structlog.get_logger(self.service_name)
        
        # Agent execution order for claims workflow
        self.agent_sequence = [
            "document_analysis_agent",
            "evidence_processing_agent",
            "liability_assessment_agent",
            "validation_agent",
            "automation_agent",
            "communication_agent"
        ]
        
        # STP (Straight Through Processing) thresholds
        self.stp_thresholds = {
            "max_amount": Decimal("10000.00"),
            "min_confidence": 0.85,
            "max_fraud_score": 20.0
        }
        
        # Priority assignment rules
        self.priority_rules = {
            "urgent": {"amount_threshold": 100000, "keywords": ["death", "injury", "emergency"]},
            "high": {"amount_threshold": 50000, "keywords": ["fire", "theft", "accident"]},
            "medium": {"amount_threshold": 10000, "keywords": ["damage", "loss"]},
            "low": {"amount_threshold": 0, "keywords": []}
        }
    
    async def create_claims_workflow(
        self,
        claim_data: Dict[str, Any],
        organization_id: uuid.UUID,
        created_by: Optional[uuid.UUID] = None,
        policy_id: Optional[uuid.UUID] = None
    ) -> Tuple[WorkflowResponse, ClaimResponse]:
        """Create a new claims processing workflow"""
        
        try:
            # Validate claim data
            if not self._validate_claim_data(claim_data):
                raise ServiceException("Invalid claim data", "VALIDATION_ERROR")
            
            # Generate claim number
            claim_number = DataUtils.generate_reference_number("CLM", 10)
            
            # Determine priority
            priority = self._determine_claim_priority(claim_data)
            
            # Create workflow
            workflow_data = WorkflowCreate(
                type="claims",
                priority=priority,
                input_data=claim_data,
                organization_id=organization_id,
                metadata={
                    "claim_number": claim_number,
                    "created_by": str(created_by) if created_by else None,
                    "policy_id": str(policy_id) if policy_id else None,
                    "agent_sequence": self.agent_sequence,
                    "auto_priority": priority
                }
            )
            
            workflow = await self.create(workflow_data, created_by, organization_id)
            
            # Create claim record
            claim_create = ClaimCreate(
                workflow_id=workflow.id,
                policy_id=policy_id,
                claim_number=claim_number,
                policy_number=claim_data.get("policy_number"),
                claim_type=claim_data["claim_type"],
                incident_date=claim_data.get("incident_date"),
                reported_date=claim_data.get("reported_date", datetime.utcnow().date()),
                claim_amount=claim_data.get("claim_amount"),
                reserve_amount=claim_data.get("reserve_amount"),
                claim_data=claim_data,
                priority=ClaimPriority(self._priority_int_to_string(priority))
            )
            
            claim_service = BaseService(Claim, self.db_session)
            claim = await claim_service.create(claim_create, created_by, organization_id)
            
            # Create initial activity
            await self._create_claim_activity(
                claim.id,
                "claim_created",
                f"Claim {claim_number} created and workflow initiated",
                {"workflow_id": str(workflow.id)},
                created_by
            )
            
            # Log workflow creation
            audit_logger.log_user_action(
                user_id=str(created_by) if created_by else "system",
                action="create_claims_workflow",
                resource_type="workflow",
                resource_id=str(workflow.id),
                details={
                    "claim_number": claim_number,
                    "claim_type": claim_data["claim_type"],
                    "claim_amount": str(claim_data.get("claim_amount", 0))
                }
            )
            
            self.logger.info(
                "Created claims workflow",
                workflow_id=str(workflow.id),
                claim_number=claim_number,
                claim_type=claim_data["claim_type"]
            )
            
            return WorkflowResponse.from_orm(workflow), ClaimResponse.from_orm(claim)
            
        except Exception as e:
            self.logger.error(
                "Failed to create claims workflow",
                error=str(e),
                claim_data=claim_data
            )
            raise
    
    async def execute_claims_workflow(
        self,
        workflow_id: uuid.UUID,
        timeout_seconds: int = 3600  # 1 hour
    ) -> Dict[str, Any]:
        """Execute the complete claims processing workflow"""
        
        async with performance_monitor.monitor_operation("claims_workflow_execution"):
            try:
                # Start workflow
                await self.start_workflow(workflow_id, timeout_seconds)
                
                # Get workflow and claim
                workflow = await self.get(workflow_id)
                if not workflow:
                    raise ServiceException(f"Workflow not found: {workflow_id}")
                
                claim_service = BaseService(Claim, self.db_session)
                claims = await claim_service.get_multi(
                    pagination={"page": 1, "size": 1},
                    filters={"workflow_id": workflow_id}
                )
                
                if not claims.items:
                    raise ServiceException(f"Claim not found for workflow: {workflow_id}")
                
                claim = claims.items[0]
                
                # Execute agent sequence
                agent_results = {}
                current_data = workflow.input_data.copy()
                stp_eligible = True
                
                for agent_name in self.agent_sequence:
                    try:
                        self.logger.info(
                            "Executing agent",
                            workflow_id=str(workflow_id),
                            agent_name=agent_name,
                            claim_number=claim.claim_number
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
                        
                        # Check STP eligibility after each agent
                        if agent_name == "evidence_processing_agent":
                            evidence_result = agent_result.get("output_data", {})
                            if evidence_result.get("fraud_indicators"):
                                stp_eligible = False
                        
                        elif agent_name == "liability_assessment_agent":
                            liability_result = agent_result.get("output_data", {})
                            liability_score = liability_result.get("liability_score", 0.0)
                            
                            # Update claim with liability assessment
                            await claim_service.update(
                                claim.id,
                                ClaimUpdate(
                                    liability_assessment=liability_result,
                                    fraud_score=Decimal(str(current_data.get("fraud_score", 0.0)))
                                )
                            )
                            
                            # Check if liability is clear
                            if liability_score < 0.7:  # Low liability confidence
                                stp_eligible = False
                        
                        elif agent_name == "validation_agent":
                            validation_result = agent_result.get("output_data", {})
                            if not validation_result.get("is_valid", True):
                                stp_eligible = False
                                
                                # Create activity for validation issues
                                await self._create_claim_activity(
                                    claim.id,
                                    "validation_issues",
                                    "Validation issues identified",
                                    validation_result.get("validation_errors", {}),
                                    None
                                )
                        
                        elif agent_name == "automation_agent":
                            automation_result = agent_result.get("output_data", {})
                            
                            # Check STP eligibility
                            final_stp_eligible = (
                                stp_eligible and
                                self._check_stp_eligibility(claim, current_data)
                            )
                            
                            # Update claim with STP status
                            await claim_service.update(
                                claim.id,
                                ClaimUpdate(stp_eligible=final_stp_eligible)
                            )
                            
                            # Process STP if eligible
                            if final_stp_eligible:
                                await self._process_stp_claim(claim, current_data)
                                
                                # Skip manual communication for STP
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
                        
                        # Create activity for agent failure
                        await self._create_claim_activity(
                            claim.id,
                            "agent_failure",
                            f"Agent {agent_name} execution failed",
                            {"error": str(e)},
                            None
                        )
                        
                        # Continue with next agent for non-critical failures
                        if agent_name not in ["document_analysis_agent"]:
                            continue
                        else:
                            # Fail workflow for critical agent failures
                            await self.fail_workflow(
                                workflow_id,
                                f"Critical agent {agent_name} failed: {str(e)}"
                            )
                            raise
                
                # Prepare final output
                final_output = {
                    "claim_id": str(claim.id),
                    "claim_number": claim.claim_number,
                    "agent_results": agent_results,
                    "final_data": current_data,
                    "stp_processed": current_data.get("stp_processed", False),
                    "execution_summary": self._generate_execution_summary(agent_results)
                }
                
                # Complete workflow
                await self.complete_workflow(workflow_id, final_output, success=True)
                
                # Record metrics
                metrics.record_workflow("claims", "completed")
                
                # Create completion activity
                await self._create_claim_activity(
                    claim.id,
                    "workflow_completed",
                    "Claims processing workflow completed",
                    final_output["execution_summary"],
                    None
                )
                
                self.logger.info(
                    "Claims workflow completed",
                    workflow_id=str(workflow_id),
                    claim_number=claim.claim_number,
                    stp_processed=final_output["stp_processed"]
                )
                
                return final_output
                
            except Exception as e:
                # Fail workflow
                await self.fail_workflow(workflow_id, str(e), retry=False)
                
                # Record metrics
                metrics.record_workflow("claims", "failed")
                
                self.logger.error(
                    "Claims workflow failed",
                    workflow_id=str(workflow_id),
                    error=str(e)
                )
                raise
    
    async def process_manual_settlement(
        self,
        claim_id: uuid.UUID,
        settlement_amount: Decimal,
        adjuster_id: uuid.UUID,
        notes: str = None,
        settlement_details: Dict[str, Any] = None
    ) -> ClaimResponse:
        """Process manual claim settlement"""
        
        try:
            claim_service = BaseService(Claim, self.db_session)
            claim = await claim_service.get(claim_id)
            
            if not claim:
                raise ServiceException(f"Claim not found: {claim_id}")
            
            if claim.status in [ClaimStatus.SETTLED, ClaimStatus.CLOSED]:
                raise ServiceException(f"Claim already settled/closed: {claim_id}")
            
            # Update claim with settlement
            update_data = ClaimUpdate(
                status=ClaimStatus.SETTLED,
                settlement_amount=settlement_amount,
                settlement_date=datetime.utcnow().date(),
                adjuster_id=adjuster_id,
                adjuster_notes=notes
            )
            
            updated_claim = await claim_service.update(
                claim_id,
                update_data,
                updated_by=adjuster_id
            )
            
            # Create settlement activity
            await self._create_claim_activity(
                claim_id,
                "manual_settlement",
                f"Claim settled manually for {DataUtils.format_currency(settlement_amount)}",
                {
                    "settlement_amount": str(settlement_amount),
                    "settlement_details": settlement_details or {},
                    "adjuster_notes": notes
                },
                adjuster_id
            )
            
            # Send settlement communication
            await self._send_settlement_communication(updated_claim, is_manual=True)
            
            # Log settlement
            audit_logger.log_user_action(
                user_id=str(adjuster_id),
                action="manual_claim_settlement",
                resource_type="claim",
                resource_id=str(claim_id),
                details={
                    "claim_number": claim.claim_number,
                    "settlement_amount": str(settlement_amount),
                    "notes": notes
                }
            )
            
            self.logger.info(
                "Manual claim settlement processed",
                claim_id=str(claim_id),
                settlement_amount=str(settlement_amount),
                adjuster_id=str(adjuster_id)
            )
            
            return ClaimResponse.from_orm(updated_claim)
            
        except Exception as e:
            self.logger.error(
                "Failed to process manual settlement",
                claim_id=str(claim_id),
                settlement_amount=str(settlement_amount) if settlement_amount else None,
                error=str(e)
            )
            raise
    
    async def add_evidence(
        self,
        claim_id: uuid.UUID,
        file_key: str,
        file_name: str,
        file_type: str,
        uploaded_by: uuid.UUID,
        metadata: Dict[str, Any] = None
    ) -> Evidence:
        """Add evidence to a claim"""
        
        try:
            # Verify claim exists
            claim_service = BaseService(Claim, self.db_session)
            claim = await claim_service.get(claim_id)
            
            if not claim:
                raise ServiceException(f"Claim not found: {claim_id}")
            
            # Create evidence record
            evidence_create = EvidenceCreate(
                claim_id=claim_id,
                file_key=file_key,
                file_name=file_name,
                file_type=file_type,
                metadata=metadata or {}
            )
            
            evidence_service = BaseService(Evidence, self.db_session)
            evidence = await evidence_service.create(evidence_create, uploaded_by)
            
            # Create activity
            await self._create_claim_activity(
                claim_id,
                "evidence_added",
                f"Evidence file '{file_name}' added to claim",
                {
                    "evidence_id": str(evidence.id),
                    "file_name": file_name,
                    "file_type": file_type
                },
                uploaded_by
            )
            
            # Trigger evidence processing if workflow is still active
            workflow = await self.get(claim.workflow_id)
            if workflow and workflow.status == WorkflowStatus.RUNNING:
                await self._trigger_evidence_processing(claim_id, evidence.id)
            
            self.logger.info(
                "Evidence added to claim",
                claim_id=str(claim_id),
                evidence_id=str(evidence.id),
                file_name=file_name
            )
            
            return evidence
            
        except Exception as e:
            self.logger.error(
                "Failed to add evidence",
                claim_id=str(claim_id),
                file_name=file_name,
                error=str(e)
            )
            raise
    
    async def get_claims_queue(
        self,
        organization_id: uuid.UUID,
        status_filter: List[str] = None,
        priority_filter: List[str] = None,
        adjuster_id: Optional[uuid.UUID] = None,
        page: int = 1,
        size: int = 20
    ) -> Dict[str, Any]:
        """Get claims processing queue with filtering"""
        
        try:
            # Build filters
            filters = {"organization_id": organization_id}
            
            if status_filter:
                filters["status"] = {"in": status_filter}
            
            if priority_filter:
                filters["priority"] = {"in": priority_filter}
            
            if adjuster_id:
                filters["adjuster_id"] = adjuster_id
            
            # Get claims
            claim_service = BaseService(Claim, self.db_session)
            result = await claim_service.get_multi(
                pagination={"page": page, "size": size, "sort_by": "created_at", "sort_order": "desc"},
                filters=filters,
                organization_id=organization_id
            )
            
            # Enrich with workflow and activity information
            enriched_items = []
            for claim in result.items:
                workflow = await self.get(claim.workflow_id)
                
                # Get latest activity
                latest_activity = await self._get_latest_activity(claim.id)
                
                item_data = {
                    "claim": claim,
                    "workflow": workflow,
                    "latest_activity": latest_activity,
                    "age_hours": self._calculate_age_hours(claim.created_at),
                    "sla_status": self._calculate_sla_status(claim.created_at, claim.priority),
                    "estimated_reserve": self._calculate_estimated_reserve(claim)
                }
                
                enriched_items.append(item_data)
            
            # Sort by priority and age
            priority_order = {"urgent": 4, "high": 3, "medium": 2, "low": 1}
            enriched_items.sort(
                key=lambda x: (
                    priority_order.get(x["claim"].priority.value, 0),
                    -x["age_hours"]
                ),
                reverse=True
            )
            
            return {
                "items": enriched_items,
                "total": result.total,
                "page": result.page,
                "size": result.size,
                "pages": result.pages,
                "queue_stats": await self._get_claims_queue_statistics(organization_id)
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to get claims queue",
                organization_id=str(organization_id),
                error=str(e)
            )
            raise
    
    async def get_claims_metrics(
        self,
        organization_id: uuid.UUID,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """Get claims processing performance metrics"""
        
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get claims statistics
            claim_service = BaseService(Claim, self.db_session)
            
            # Get claims in date range
            query = select(Claim).where(
                and_(
                    Claim.organization_id == organization_id,
                    Claim.created_at >= start_date,
                    Claim.created_at <= end_date
                )
            )
            
            result = await self.db_session.execute(query)
            claims = result.scalars().all()
            
            # Calculate metrics
            total_claims = len(claims)
            
            # Status distribution
            status_counts = {}
            for status in ClaimStatus:
                status_counts[status.value] = len([c for c in claims if c.status == status])
            
            # Priority distribution
            priority_counts = {}
            for priority in ClaimPriority:
                priority_counts[priority.value] = len([c for c in claims if c.priority == priority])
            
            # Financial metrics
            total_claimed = sum([float(c.claim_amount or 0) for c in claims])
            total_paid = sum([float(c.paid_amount or 0) for c in claims])
            total_reserved = sum([float(c.reserve_amount or 0) for c in claims])
            total_settled = sum([float(c.settlement_amount or 0) for c in claims if c.settlement_amount])
            
            # STP metrics
            stp_eligible = len([c for c in claims if c.stp_eligible])
            stp_processed = len([c for c in claims if c.stp_eligible and c.status == ClaimStatus.SETTLED])
            
            metrics_data = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "volume": {
                    "total_claims": total_claims,
                    "open_claims": status_counts.get("open", 0),
                    "settled_claims": status_counts.get("settled", 0),
                    "closed_claims": status_counts.get("closed", 0)
                },
                "status_distribution": status_counts,
                "priority_distribution": priority_counts,
                "financial": {
                    "total_claimed": total_claimed,
                    "total_paid": total_paid,
                    "total_reserved": total_reserved,
                    "total_settled": total_settled,
                    "average_claim_amount": total_claimed / total_claims if total_claims > 0 else 0,
                    "settlement_ratio": (total_settled / total_claimed * 100) if total_claimed > 0 else 0
                },
                "stp_metrics": {
                    "stp_eligible": stp_eligible,
                    "stp_processed": stp_processed,
                    "stp_rate": (stp_eligible / total_claims * 100) if total_claims > 0 else 0,
                    "stp_success_rate": (stp_processed / stp_eligible * 100) if stp_eligible > 0 else 0
                },
                "processing_times": self._calculate_claims_processing_times(claims),
                "fraud_metrics": self._calculate_fraud_metrics(claims),
                "sla_performance": self._calculate_claims_sla_performance(claims)
            }
            
            return metrics_data
            
        except Exception as e:
            self.logger.error(
                "Failed to get claims metrics",
                organization_id=str(organization_id),
                error=str(e)
            )
            raise
    
    # Private helper methods
    
    def _validate_claim_data(self, data: Dict[str, Any]) -> bool:
        """Validate claim data"""
        required_fields = ["claim_type"]
        
        for field in required_fields:
            if field not in data or not data[field]:
                return False
        
        # Validate amounts if provided
        if "claim_amount" in data and data["claim_amount"]:
            if not ValidationUtils.validate_amount(data["claim_amount"]):
                return False
        
        return True
    
    def _determine_claim_priority(self, claim_data: Dict[str, Any]) -> int:
        """Determine claim priority based on rules"""
        
        claim_amount = float(claim_data.get("claim_amount", 0))
        description = claim_data.get("description", "").lower()
        claim_type = claim_data.get("claim_type", "").lower()
        
        # Check for urgent conditions
        urgent_keywords = self.priority_rules["urgent"]["keywords"]
        if (claim_amount >= self.priority_rules["urgent"]["amount_threshold"] or
            any(keyword in description or keyword in claim_type for keyword in urgent_keywords)):
            return 1  # Urgent
        
        # Check for high priority
        high_keywords = self.priority_rules["high"]["keywords"]
        if (claim_amount >= self.priority_rules["high"]["amount_threshold"] or
            any(keyword in description or keyword in claim_type for keyword in high_keywords)):
            return 2  # High
        
        # Check for medium priority
        medium_keywords = self.priority_rules["medium"]["keywords"]
        if (claim_amount >= self.priority_rules["medium"]["amount_threshold"] or
            any(keyword in description or keyword in claim_type for keyword in medium_keywords)):
            return 3  # Medium
        
        return 5  # Low/Normal
    
    def _priority_int_to_string(self, priority_int: int) -> str:
        """Convert priority integer to string"""
        mapping = {1: "urgent", 2: "high", 3: "medium", 4: "low", 5: "low"}
        return mapping.get(priority_int, "low")
    
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
            output_data = await self._simulate_claims_agent_execution(agent_name, input_data)
            
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
    
    async def _simulate_claims_agent_execution(
        self,
        agent_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute claims agent with production HTTP API calls"""
        
        try:
            import httpx
            
            # Production agent service endpoints
            agent_endpoints = {
                "document_analysis_agent": f"{self.config.get('DOCUMENT_SERVICE_URL', 'http://localhost:8006')}/analyze",
                "evidence_processing_agent": f"{self.config.get('EVIDENCE_SERVICE_URL', 'http://localhost:8010')}/process",
                "liability_assessment_agent": f"{self.config.get('LIABILITY_SERVICE_URL', 'http://localhost:8013')}/assess",
                "validation_agent": f"{self.config.get('VALIDATION_SERVICE_URL', 'http://localhost:8007')}/validate",
                "automation_agent": f"{self.config.get('AUTOMATION_SERVICE_URL', 'http://localhost:8014')}/automate",
                "communication_agent": f"{self.config.get('COMMUNICATION_SERVICE_URL', 'http://localhost:8009')}/send",
                "fraud_detection_agent": f"{self.config.get('FRAUD_SERVICE_URL', 'http://localhost:8015')}/detect"
            }
            
            endpoint = agent_endpoints.get(agent_name)
            if not endpoint:
                # Fallback to generic agent service
                endpoint = f"{self.config.get('AGENT_SERVICE_URL', 'http://localhost:8000')}/agents/{agent_name}"
            
            # Prepare request payload with claims context
            payload = {
                "agent_name": agent_name,
                "input_data": input_data,
                "context": {
                    "orchestrator": "claims",
                    "claim_id": input_data.get("claim_id"),
                    "policy_id": input_data.get("policy_id"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4()),
                    "workflow_stage": input_data.get("workflow_stage", "processing")
                }
            }
            
            # Make HTTP request to agent service
            async with httpx.AsyncClient(timeout=45.0) as client:  # Longer timeout for claims processing
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Orchestrator": "claims",
                        "X-Claim-ID": str(input_data.get("claim_id", "")),
                        "X-Request-Source": "claims-orchestrator",
                        "Authorization": f"Bearer {self.config.get('AGENT_API_TOKEN', '')}"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Log successful agent execution
                    logger.info(f"Claims agent {agent_name} executed successfully for claim {input_data.get('claim_id')}")
                    
                    # Update metrics
                    if hasattr(self, 'metrics'):
                        self.metrics.agent_calls_total.labels(
                            agent=agent_name,
                            status='success',
                            orchestrator='claims'
                        ).inc()
                        
                        # Track processing time
                        if 'processing_time' in result:
                            self.metrics.agent_processing_time.labels(
                                agent=agent_name
                            ).observe(result['processing_time'])
                    
                    return result
                    
                elif response.status_code == 429:
                    # Rate limiting - implement retry logic
                    error_msg = f"Agent {agent_name} rate limited, retrying..."
                    logger.warning(error_msg)
                    
                    # Wait and retry once
                    await asyncio.sleep(2)
                    retry_response = await client.post(endpoint, json=payload, headers=response.request.headers)
                    
                    if retry_response.status_code == 200:
                        return retry_response.json()
                    else:
                        return {
                            "error": f"Agent {agent_name} rate limited and retry failed",
                            "status": "rate_limited",
                            "agent_name": agent_name
                        }
                        
                else:
                    error_msg = f"Claims agent {agent_name} failed with status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    
                    # Update error metrics
                    if hasattr(self, 'metrics'):
                        self.metrics.agent_calls_total.labels(
                            agent=agent_name,
                            status='error',
                            orchestrator='claims'
                        ).inc()
                    
                    # Return error response
                    return {
                        "error": error_msg,
                        "status": "failed",
                        "agent_name": agent_name,
                        "http_status": response.status_code
                    }
                    
        except httpx.TimeoutException:
            error_msg = f"Claims agent {agent_name} request timed out"
            logger.error(error_msg)
            
            # Update timeout metrics
            if hasattr(self, 'metrics'):
                self.metrics.agent_calls_total.labels(
                    agent=agent_name,
                    status='timeout',
                    orchestrator='claims'
                ).inc()
            
            return {
                "error": error_msg,
                "status": "timeout",
                "agent_name": agent_name
            }
            
        except Exception as e:
            error_msg = f"Error calling claims agent {agent_name}: {str(e)}"
            logger.error(error_msg)
            
            # Update error metrics
            if hasattr(self, 'metrics'):
                self.metrics.agent_calls_total.labels(
                    agent=agent_name,
                    status='exception',
                    orchestrator='claims'
                ).inc()
            
            return {
                "error": error_msg,
                "status": "failed",
                "agent_name": agent_name,
                "exception_type": type(e).__name__
            }
        
        else:
            return {"message": f"Agent {agent_name} executed successfully"}
    
    def _check_stp_eligibility(self, claim: Claim, data: Dict[str, Any]) -> bool:
        """Check if claim is eligible for Straight Through Processing"""
        
        # Amount check
        if claim.claim_amount and claim.claim_amount > self.stp_thresholds["max_amount"]:
            return False
        
        # Fraud score check
        fraud_score = data.get("fraud_score", 0)
        if fraud_score > self.stp_thresholds["max_fraud_score"]:
            return False
        
        # Confidence check
        confidence = data.get("automation_confidence", 0)
        if confidence < self.stp_thresholds["min_confidence"]:
            return False
        
        # Policy validation check
        policy_validation = data.get("policy_validation", {})
        if not all([
            policy_validation.get("coverage_valid", False),
            policy_validation.get("policy_active", False),
            policy_validation.get("deductible_met", False)
        ]):
            return False
        
        return True
    
    async def _process_stp_claim(self, claim: Claim, data: Dict[str, Any]):
        """Process claim through Straight Through Processing"""
        
        try:
            settlement_amount = data.get("settlement_amount", float(claim.claim_amount or 0))
            
            # Update claim status
            claim_service = BaseService(Claim, self.db_session)
            await claim_service.update(
                claim.id,
                ClaimUpdate(
                    status=ClaimStatus.SETTLED,
                    settlement_amount=Decimal(str(settlement_amount)),
                    settlement_date=datetime.utcnow().date(),
                    paid_amount=Decimal(str(settlement_amount))
                )
            )
            
            # Create STP activity
            await self._create_claim_activity(
                claim.id,
                "stp_settlement",
                f"Claim auto-settled through STP for {DataUtils.format_currency(settlement_amount)}",
                {
                    "settlement_amount": settlement_amount,
                    "stp_processed": True,
                    "automation_data": data
                },
                None
            )
            
            # Send settlement communication
            await self._send_settlement_communication(claim, is_manual=False)
            
            # Mark as STP processed
            data["stp_processed"] = True
            
            self.logger.info(
                "Claim processed through STP",
                claim_id=str(claim.id),
                claim_number=claim.claim_number,
                settlement_amount=settlement_amount
            )
            
        except Exception as e:
            self.logger.error(
                "STP processing failed",
                claim_id=str(claim.id),
                error=str(e)
            )
            raise
    
    async def _create_claim_activity(
        self,
        claim_id: uuid.UUID,
        activity_type: str,
        description: str,
        details: Dict[str, Any],
        performed_by: Optional[uuid.UUID]
    ):
        """Create claim activity record"""
        
        activity_create = ClaimActivityCreate(
            claim_id=claim_id,
            activity_type=activity_type,
            description=description,
            details=details,
            performed_by=performed_by
        )
        
        activity_service = BaseService(ClaimActivity, self.db_session)
        await activity_service.create(activity_create, performed_by)
    
    async def _send_settlement_communication(self, claim: Claim, is_manual: bool):
        """Send settlement communication"""
        
        # In real implementation, this would trigger the communication agent
        self.logger.info(
            "Settlement communication sent",
            claim_id=str(claim.id),
            claim_number=claim.claim_number,
            is_manual=is_manual
        )
    
    async def _trigger_evidence_processing(self, claim_id: uuid.UUID, evidence_id: uuid.UUID):
        """Trigger evidence processing for new evidence"""
        
        # In real implementation, this would trigger the evidence processing agent
        self.logger.info(
            "Evidence processing triggered",
            claim_id=str(claim_id),
            evidence_id=str(evidence_id)
        )
    
    async def _get_latest_activity(self, claim_id: uuid.UUID) -> Optional[ClaimActivity]:
        """Get latest activity for a claim"""
        
        query = select(ClaimActivity).where(
            ClaimActivity.claim_id == claim_id
        ).order_by(ClaimActivity.created_at.desc()).limit(1)
        
        result = await self.db_session.execute(query)
        return result.scalar_one_or_none()
    
    def _calculate_age_hours(self, created_at: datetime) -> float:
        """Calculate age in hours"""
        return (datetime.utcnow() - created_at).total_seconds() / 3600
    
    def _calculate_sla_status(self, created_at: datetime, priority: ClaimPriority) -> str:
        """Calculate SLA status based on priority"""
        age_hours = self._calculate_age_hours(created_at)
        
        # SLA thresholds based on priority
        sla_hours = {
            ClaimPriority.URGENT: 4,   # 4 hours
            ClaimPriority.HIGH: 12,    # 12 hours
            ClaimPriority.MEDIUM: 24,  # 24 hours
            ClaimPriority.LOW: 72      # 72 hours
        }
        
        threshold = sla_hours.get(priority, 72)
        
        if age_hours <= threshold * 0.5:
            return "green"
        elif age_hours <= threshold * 0.8:
            return "yellow"
        elif age_hours <= threshold:
            return "orange"
        else:
            return "red"
    
    def _calculate_estimated_reserve(self, claim: Claim) -> Decimal:
        """Calculate estimated reserve for claim"""
        
        if claim.reserve_amount:
            return claim.reserve_amount
        
        if claim.claim_amount:
            # Estimate reserve as 120% of claim amount
            return claim.claim_amount * Decimal("1.2")
        
        return Decimal("0")
    
    async def _get_claims_queue_statistics(self, organization_id: uuid.UUID) -> Dict[str, Any]:
        """Get claims queue statistics"""
        
        # Get open claims
        query = select(Claim).where(
            and_(
                Claim.organization_id == organization_id,
                Claim.status.in_([ClaimStatus.OPEN, ClaimStatus.INVESTIGATING])
            )
        )
        
        result = await self.db_session.execute(query)
        open_claims = result.scalars().all()
        
        # Calculate statistics
        total_open = len(open_claims)
        
        priority_counts = {}
        for priority in ClaimPriority:
            priority_counts[priority.value] = len([c for c in open_claims if c.priority == priority])
        
        sla_status_counts = {
            "green": 0,
            "yellow": 0,
            "orange": 0,
            "red": 0
        }
        
        for claim in open_claims:
            sla_status = self._calculate_sla_status(claim.created_at, claim.priority)
            sla_status_counts[sla_status] += 1
        
        return {
            "total_open": total_open,
            "priority_distribution": priority_counts,
            "sla_status": sla_status_counts,
            "avg_age_hours": sum([
                self._calculate_age_hours(c.created_at) for c in open_claims
            ]) / total_open if total_open > 0 else 0
        }
    
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
    
    def _calculate_claims_processing_times(self, claims: List[Claim]) -> Dict[str, Any]:
        """Calculate claims processing time metrics"""
        
        settled_claims = [c for c in claims if c.status == ClaimStatus.SETTLED and c.settlement_date]
        
        if not settled_claims:
            return {"avg_processing_days": 0}
        
        processing_times = []
        for claim in settled_claims:
            if claim.settlement_date:
                # Calculate days from creation to settlement
                settlement_datetime = datetime.combine(claim.settlement_date, datetime.min.time())
                days = (settlement_datetime - claim.created_at).days
                processing_times.append(days)
        
        if not processing_times:
            return {"avg_processing_days": 0}
        
        return {
            "avg_processing_days": sum(processing_times) / len(processing_times),
            "min_processing_days": min(processing_times),
            "max_processing_days": max(processing_times),
            "median_processing_days": sorted(processing_times)[len(processing_times) // 2]
        }
    
    def _calculate_fraud_metrics(self, claims: List[Claim]) -> Dict[str, Any]:
        """Calculate fraud detection metrics"""
        
        claims_with_fraud_score = [c for c in claims if c.fraud_score is not None]
        
        if not claims_with_fraud_score:
            return {"avg_fraud_score": 0, "high_risk_claims": 0}
        
        fraud_scores = [float(c.fraud_score) for c in claims_with_fraud_score]
        high_risk_threshold = 70.0
        
        return {
            "avg_fraud_score": sum(fraud_scores) / len(fraud_scores),
            "high_risk_claims": len([s for s in fraud_scores if s >= high_risk_threshold]),
            "fraud_detection_rate": (len([s for s in fraud_scores if s >= high_risk_threshold]) / len(fraud_scores) * 100)
        }
    
    def _calculate_claims_sla_performance(self, claims: List[Claim]) -> Dict[str, Any]:
        """Calculate claims SLA performance metrics"""
        
        completed_claims = [c for c in claims if c.status in [ClaimStatus.SETTLED, ClaimStatus.CLOSED]]
        
        if not completed_claims:
            return {"sla_compliance_rate": 0}
        
        sla_compliant = 0
        
        for claim in completed_claims:
            # Get SLA threshold for priority
            sla_hours = {
                ClaimPriority.URGENT: 4,
                ClaimPriority.HIGH: 12,
                ClaimPriority.MEDIUM: 24,
                ClaimPriority.LOW: 72
            }
            
            threshold = sla_hours.get(claim.priority, 72)
            
            # Calculate processing time
            if claim.settlement_date:
                settlement_datetime = datetime.combine(claim.settlement_date, datetime.min.time())
                processing_hours = (settlement_datetime - claim.created_at).total_seconds() / 3600
                
                if processing_hours <= threshold:
                    sla_compliant += 1
        
        return {
            "sla_compliance_rate": (sla_compliant / len(completed_claims) * 100),
            "total_evaluated": len(completed_claims),
            "sla_compliant": sla_compliant
        }

