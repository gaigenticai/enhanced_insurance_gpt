"""
Insurance AI Agent System - Compliance Agent
Production-ready agent for regulatory compliance and audit management
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from decimal import Decimal
import json
import re
from dataclasses import dataclass
from enum import Enum

# Compliance and regulatory libraries
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

# Database and utilities
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
import structlog
import redis.asyncio as redis

from backend.shared.models import (
    Claim, Policy, User, Organization, AgentExecution,
    AuditLog
)
from backend.shared.schemas import (
    ComplianceCheckCreate, ComplianceCheckUpdate, ComplianceStatus,
    RegulatoryFramework, ComplianceViolationType, AgentExecutionStatus
)
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

logger = structlog.get_logger(__name__)

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NAIC = "naic"
    STATE_INSURANCE = "state_insurance"

class ViolationSeverity(Enum):
    """Violation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    id: str
    framework: ComplianceFramework
    name: str
    description: str
    severity: ViolationSeverity
    check_function: str
    parameters: Dict[str, Any]
    remediation_steps: List[str]

class ComplianceAgent:
    """
    Advanced compliance agent for regulatory compliance and audit management
    Supports multiple regulatory frameworks and automated compliance checking
    """
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
        self.agent_name = "compliance_agent"
        self.agent_version = "1.0.0"
        self.logger = structlog.get_logger(self.agent_name)
        
        # Initialize compliance rules
        self._initialize_compliance_rules()
        
        # Compliance configuration
        self.compliance_config = {
            "audit_retention_days": 2555,  # 7 years
            "violation_escalation_hours": 24,
            "critical_violation_escalation_hours": 4,
            "compliance_check_frequency_hours": 24,
            "report_generation_frequency_days": 30
        }
        
        # Regulatory frameworks configuration
        self.frameworks_config = {
            ComplianceFramework.GDPR: {
                "data_retention_days": 2555,  # 7 years
                "consent_required": True,
                "right_to_erasure": True,
                "data_portability": True,
                "breach_notification_hours": 72
            },
            ComplianceFramework.CCPA: {
                "data_retention_days": 1095,  # 3 years
                "opt_out_required": True,
                "data_disclosure": True,
                "deletion_rights": True
            },
            ComplianceFramework.NAIC: {
                "claim_handling_standards": True,
                "market_conduct_compliance": True,
                "financial_reporting": True,
                "consumer_protection": True
            },
            ComplianceFramework.STATE_INSURANCE: {
                "licensing_requirements": True,
                "rate_filing_compliance": True,
                "claim_settlement_timeframes": True,
                "unfair_practices_prevention": True
            }
        }
        
        # Violation tracking
        self.violation_cache = {}
        
        # Compliance metrics
        self.compliance_metrics = {
            "checks_performed": 0,
            "violations_detected": 0,
            "violations_resolved": 0,
            "compliance_score": 100.0
        }
    
    def _initialize_compliance_rules(self):
        """Initialize compliance rules for different frameworks"""
        
        self.compliance_rules = {
            # GDPR Rules
            "gdpr_data_retention": ComplianceRule(
                id="gdpr_data_retention",
                framework=ComplianceFramework.GDPR,
                name="Data Retention Compliance",
                description="Ensure personal data is not retained beyond legal requirements",
                severity=ViolationSeverity.HIGH,
                check_function="check_data_retention",
                parameters={"max_retention_days": 2555},
                remediation_steps=[
                    "Review data retention policies",
                    "Implement automated data deletion",
                    "Update privacy notices"
                ]
            ),
            
            "gdpr_consent_tracking": ComplianceRule(
                id="gdpr_consent_tracking",
                framework=ComplianceFramework.GDPR,
                name="Consent Tracking",
                description="Verify proper consent is obtained and tracked for data processing",
                severity=ViolationSeverity.CRITICAL,
                check_function="check_consent_tracking",
                parameters={"require_explicit_consent": True},
                remediation_steps=[
                    "Implement consent management system",
                    "Update data collection forms",
                    "Train staff on consent requirements"
                ]
            ),
            
            "gdpr_breach_notification": ComplianceRule(
                id="gdpr_breach_notification",
                framework=ComplianceFramework.GDPR,
                name="Breach Notification Timing",
                description="Ensure data breaches are reported within 72 hours",
                severity=ViolationSeverity.CRITICAL,
                check_function="check_breach_notification",
                parameters={"notification_hours": 72},
                remediation_steps=[
                    "Implement breach detection system",
                    "Create incident response procedures",
                    "Establish notification workflows"
                ]
            ),
            
            # NAIC Rules
            "naic_claim_handling": ComplianceRule(
                id="naic_claim_handling",
                framework=ComplianceFramework.NAIC,
                name="Claim Handling Standards",
                description="Ensure claims are handled according to NAIC standards",
                severity=ViolationSeverity.HIGH,
                check_function="check_claim_handling_standards",
                parameters={"max_processing_days": 30},
                remediation_steps=[
                    "Review claim processing procedures",
                    "Implement automated workflow tracking",
                    "Train adjusters on standards"
                ]
            ),
            
            "naic_unfair_practices": ComplianceRule(
                id="naic_unfair_practices",
                framework=ComplianceFramework.NAIC,
                name="Unfair Claims Practices Prevention",
                description="Prevent unfair or deceptive claims practices",
                severity=ViolationSeverity.CRITICAL,
                check_function="check_unfair_practices",
                parameters={"denial_rate_threshold": 0.15},
                remediation_steps=[
                    "Review denial patterns",
                    "Implement fairness audits",
                    "Update training programs"
                ]
            ),
            
            # State Insurance Rules
            "state_claim_settlement": ComplianceRule(
                id="state_claim_settlement",
                framework=ComplianceFramework.STATE_INSURANCE,
                name="Claim Settlement Timeframes",
                description="Ensure claims are settled within state-mandated timeframes",
                severity=ViolationSeverity.HIGH,
                check_function="check_settlement_timeframes",
                parameters={"max_settlement_days": 45},
                remediation_steps=[
                    "Implement settlement tracking",
                    "Automate settlement processes",
                    "Monitor compliance metrics"
                ]
            ),
            
            # Financial Compliance
            "financial_reporting": ComplianceRule(
                id="financial_reporting",
                framework=ComplianceFramework.SOX,
                name="Financial Reporting Accuracy",
                description="Ensure accurate and timely financial reporting",
                severity=ViolationSeverity.CRITICAL,
                check_function="check_financial_reporting",
                parameters={"variance_threshold": 0.05},
                remediation_steps=[
                    "Review financial controls",
                    "Implement automated reconciliation",
                    "Enhance audit procedures"
                ]
            )
        }
    
    async def perform_compliance_check(
        self,
        entity_type: str,
        entity_id: uuid.UUID,
        frameworks: Optional[List[ComplianceFramework]] = None,
        check_types: Optional[List[str]] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive compliance check
        
        Args:
            entity_type: Type of entity to check (claim, policy, user, organization)
            entity_id: UUID of the entity
            frameworks: Specific frameworks to check against
            check_types: Specific check types to perform
            context_data: Additional context for compliance checking
            
        Returns:
            Dictionary containing compliance check results
        """
        
        async with performance_monitor.monitor_operation("compliance_check"):
            try:
                # Create agent execution record
                execution = AgentExecution(
                    agent_name=self.agent_name,
                    agent_version=self.agent_version,
                    input_data={
                        "entity_type": entity_type,
                        "entity_id": str(entity_id),
                        "frameworks": [f.value for f in frameworks] if frameworks else None,
                        "check_types": check_types,
                        "context_data": context_data or {}
                    },
                    status=AgentExecutionStatus.RUNNING
                )
                
                self.db_session.add(execution)
                await self.db_session.commit()
                await self.db_session.refresh(execution)
                
                start_time = datetime.utcnow()
                
                # Gather entity data
                entity_data = await self._gather_entity_data(entity_type, entity_id)
                
                # Determine applicable frameworks
                if not frameworks:
                    frameworks = self._determine_applicable_frameworks(entity_type, entity_data)
                
                # Perform compliance checks
                compliance_results = {}
                violations = []
                overall_compliance_score = 100.0
                
                for framework in frameworks:
                    framework_results = await self._check_framework_compliance(
                        framework, entity_type, entity_data, check_types
                    )
                    
                    compliance_results[framework.value] = framework_results
                    violations.extend(framework_results.get("violations", []))
                    
                    # Update overall compliance score
                    framework_score = framework_results.get("compliance_score", 100.0)
                    overall_compliance_score = min(overall_compliance_score, framework_score)
                
                # Generate compliance status
                compliance_status = self._determine_compliance_status(overall_compliance_score, violations)
                
                # Generate recommendations
                recommendations = await self._generate_compliance_recommendations(
                    violations, entity_type, entity_data
                )
                
                # Prepare final result
                compliance_result = {
                    "entity_type": entity_type,
                    "entity_id": str(entity_id),
                    "check_timestamp": datetime.utcnow().isoformat(),
                    "overall_compliance_score": overall_compliance_score,
                    "compliance_status": compliance_status.value,
                    "frameworks_checked": [f.value for f in frameworks],
                    "framework_results": compliance_results,
                    "violations": violations,
                    "recommendations": recommendations,
                    "next_check_due": (datetime.utcnow() + timedelta(
                        hours=self.compliance_config["compliance_check_frequency_hours"]
                    )).isoformat(),
                    "check_metadata": {
                        "agent_name": self.agent_name,
                        "agent_version": self.agent_version,
                        "processing_time_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
                        "rules_checked": len([rule for framework in frameworks 
                                            for rule in self._get_framework_rules(framework)]),
                        "data_sources": list(entity_data.keys())
                    }
                }
                
                # Save compliance check results
                await self._save_compliance_results(entity_type, entity_id, compliance_result)
                
                # Handle violations
                if violations:
                    await self._handle_violations(violations, entity_type, entity_id)
                
                # Update execution record
                execution.status = AgentExecutionStatus.COMPLETED
                execution.output_data = compliance_result
                execution.execution_time_ms = compliance_result["check_metadata"]["processing_time_ms"]
                execution.completed_at = datetime.utcnow()
                
                await self.db_session.commit()
                
                # Record metrics
                metrics.record_agent_execution(
                    self.agent_name, 
                    execution.execution_time_ms / 1000, 
                    success=True
                )
                
                # Update compliance metrics
                self.compliance_metrics["checks_performed"] += 1
                self.compliance_metrics["violations_detected"] += len(violations)
                
                # Log compliance check
                audit_logger.log_user_action(
                    user_id="system",
                    action="compliance_check_completed",
                    resource_type=entity_type,
                    resource_id=str(entity_id),
                    details={
                        "frameworks_checked": [f.value for f in frameworks],
                        "compliance_score": overall_compliance_score,
                        "violations_count": len(violations),
                        "processing_time_ms": execution.execution_time_ms
                    }
                )
                
                self.logger.info(
                    "Compliance check completed",
                    entity_type=entity_type,
                    entity_id=str(entity_id),
                    compliance_score=overall_compliance_score,
                    violations_count=len(violations),
                    processing_time_ms=execution.execution_time_ms
                )
                
                return compliance_result
                
            except Exception as e:
                # Update execution record with error
                execution.status = AgentExecutionStatus.FAILED
                execution.error_message = str(e)
                execution.completed_at = datetime.utcnow()
                
                await self.db_session.commit()
                
                # Record metrics
                metrics.record_agent_execution(self.agent_name, 0, success=False)
                
                self.logger.error(
                    "Compliance check failed",
                    entity_type=entity_type,
                    entity_id=str(entity_id),
                    error=str(e)
                )
                raise ServiceException(f"Compliance check failed: {str(e)}")
    
    async def _gather_entity_data(self, entity_type: str, entity_id: uuid.UUID) -> Dict[str, Any]:
        """Gather comprehensive data for compliance checking"""
        
        try:
            entity_data = {}
            
            if entity_type == "claim":
                entity_data = await self._gather_claim_compliance_data(entity_id)
            elif entity_type == "policy":
                entity_data = await self._gather_policy_compliance_data(entity_id)
            elif entity_type == "user":
                entity_data = await self._gather_user_compliance_data(entity_id)
            elif entity_type == "organization":
                entity_data = await self._gather_organization_compliance_data(entity_id)
            else:
                raise ServiceException(f"Unsupported entity type: {entity_type}")
            
            return entity_data
            
        except Exception as e:
            self.logger.error("Failed to gather entity data for compliance", error=str(e))
            raise
    
    async def _gather_claim_compliance_data(self, claim_id: uuid.UUID) -> Dict[str, Any]:
        """Gather claim data for compliance checking"""
        
        try:
            # Get claim details
            claim_service = BaseService(Claim, self.db_session)
            claim = await claim_service.get(claim_id)
            
            if not claim:
                raise ServiceException(f"Claim not found: {claim_id}")
            
            # Get related policy and user
            policy_service = BaseService(Policy, self.db_session)
            policy = await policy_service.get(claim.policy_id) if claim.policy_id else None
            
            user_service = BaseService(User, self.db_session)
            user = None
            if policy and policy.user_id:
                user = await user_service.get(policy.user_id)
            
            # Get audit logs for this claim
            audit_query = select(AuditLog).where(
                and_(
                    AuditLog.resource_type == "claim",
                    AuditLog.resource_id == str(claim_id)
                )
            ).order_by(AuditLog.created_at.desc())
            
            audit_result = await self.db_session.execute(audit_query)
            audit_logs = audit_result.scalars().all()
            
            # Calculate processing times
            processing_time_days = None
            if claim.reported_date and claim.status_updated_at:
                processing_time_days = (claim.status_updated_at - claim.reported_date).days
            
            claim_data = {
                "claim": {
                    "id": str(claim.id),
                    "claim_number": claim.claim_number,
                    "status": claim.status.value,
                    "claim_type": claim.claim_type,
                    "incident_date": claim.incident_date.isoformat() if claim.incident_date else None,
                    "reported_date": claim.reported_date.isoformat() if claim.reported_date else None,
                    "status_updated_at": claim.status_updated_at.isoformat() if claim.status_updated_at else None,
                    "amount_claimed": float(claim.amount_claimed) if claim.amount_claimed else 0.0,
                    "amount_approved": float(claim.amount_approved) if claim.amount_approved else 0.0,
                    "processing_time_days": processing_time_days,
                    "created_at": claim.created_at.isoformat(),
                    "updated_at": claim.updated_at.isoformat()
                },
                "policy": {
                    "id": str(policy.id) if policy else None,
                    "policy_number": policy.policy_number if policy else None,
                    "policy_type": policy.policy_type if policy else None,
                    "status": policy.status.value if policy else None,
                    "effective_date": policy.effective_date.isoformat() if policy and policy.effective_date else None,
                    "expiry_date": policy.expiry_date.isoformat() if policy and policy.expiry_date else None
                },
                "user": {
                    "id": str(user.id) if user else None,
                    "email": user.email if user else None,
                    "created_at": user.created_at.isoformat() if user else None,
                    "consent_given": getattr(user, 'consent_given', None) if user else None,
                    "consent_date": getattr(user, 'consent_date', None) if user else None
                },
                "audit_logs": [
                    {
                        "id": str(log.id),
                        "action": log.action,
                        "user_id": log.user_id,
                        "timestamp": log.created_at.isoformat(),
                        "details": log.details
                    }
                    for log in audit_logs[:50]  # Limit to recent 50 logs
                ]
            }
            
            return claim_data
            
        except Exception as e:
            self.logger.error("Failed to gather claim compliance data", error=str(e))
            raise
    
    async def _gather_policy_compliance_data(self, policy_id: uuid.UUID) -> Dict[str, Any]:
        """Gather policy data for compliance checking"""
        
        try:
            # Get policy details
            policy_service = BaseService(Policy, self.db_session)
            policy = await policy_service.get(policy_id)
            
            if not policy:
                raise ServiceException(f"Policy not found: {policy_id}")
            
            # Get related user and claims
            user_service = BaseService(User, self.db_session)
            user = await user_service.get(policy.user_id) if policy.user_id else None
            
            claims_query = select(Claim).where(Claim.policy_id == policy_id)
            claims_result = await self.db_session.execute(claims_query)
            claims = claims_result.scalars().all()
            
            policy_data = {
                "policy": {
                    "id": str(policy.id),
                    "policy_number": policy.policy_number,
                    "policy_type": policy.policy_type,
                    "status": policy.status.value,
                    "coverage_amount": float(policy.coverage_amount) if policy.coverage_amount else 0.0,
                    "premium_amount": float(policy.premium_amount) if policy.premium_amount else 0.0,
                    "effective_date": policy.effective_date.isoformat() if policy.effective_date else None,
                    "expiry_date": policy.expiry_date.isoformat() if policy.expiry_date else None,
                    "created_at": policy.created_at.isoformat(),
                    "updated_at": policy.updated_at.isoformat()
                },
                "user": {
                    "id": str(user.id) if user else None,
                    "email": user.email if user else None,
                    "created_at": user.created_at.isoformat() if user else None,
                    "consent_given": getattr(user, 'consent_given', None) if user else None,
                    "consent_date": getattr(user, 'consent_date', None) if user else None
                },
                "claims": [
                    {
                        "id": str(claim.id),
                        "claim_number": claim.claim_number,
                        "status": claim.status.value,
                        "amount_claimed": float(claim.amount_claimed) if claim.amount_claimed else 0.0,
                        "amount_approved": float(claim.amount_approved) if claim.amount_approved else 0.0,
                        "reported_date": claim.reported_date.isoformat() if claim.reported_date else None
                    }
                    for claim in claims
                ]
            }
            
            return policy_data
            
        except Exception as e:
            self.logger.error("Failed to gather policy compliance data", error=str(e))
            raise
    
    async def _gather_user_compliance_data(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """Gather user data for compliance checking"""
        
        try:
            # Get user details
            user_service = BaseService(User, self.db_session)
            user = await user_service.get(user_id)
            
            if not user:
                raise ServiceException(f"User not found: {user_id}")
            
            # Get user's policies and claims
            policies_query = select(Policy).where(Policy.user_id == user_id)
            policies_result = await self.db_session.execute(policies_query)
            policies = policies_result.scalars().all()
            
            # Get user's data access logs
            access_logs_query = select(AuditLog).where(
                and_(
                    AuditLog.user_id == str(user_id),
                    AuditLog.action.like('%data_access%')
                )
            ).order_by(AuditLog.created_at.desc()).limit(100)
            
            access_result = await self.db_session.execute(access_logs_query)
            access_logs = access_result.scalars().all()
            
            user_data = {
                "user": {
                    "id": str(user.id),
                    "email": user.email,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat(),
                    "updated_at": user.updated_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "consent_given": getattr(user, 'consent_given', None),
                    "consent_date": getattr(user, 'consent_date', None),
                    "data_retention_until": getattr(user, 'data_retention_until', None)
                },
                "policies": [
                    {
                        "id": str(policy.id),
                        "policy_number": policy.policy_number,
                        "policy_type": policy.policy_type,
                        "status": policy.status.value,
                        "created_at": policy.created_at.isoformat()
                    }
                    for policy in policies
                ],
                "data_access_logs": [
                    {
                        "id": str(log.id),
                        "action": log.action,
                        "timestamp": log.created_at.isoformat(),
                        "details": log.details
                    }
                    for log in access_logs
                ]
            }
            
            return user_data
            
        except Exception as e:
            self.logger.error("Failed to gather user compliance data", error=str(e))
            raise
    
    async def _gather_organization_compliance_data(self, org_id: uuid.UUID) -> Dict[str, Any]:
        """Gather organization data for compliance checking"""
        
        try:
            # Get organization details
            org_service = BaseService(Organization, self.db_session)
            organization = await org_service.get(org_id)
            
            if not organization:
                raise ServiceException(f"Organization not found: {org_id}")
            
            # Get organization's users and policies
            users_query = select(User).where(User.organization_id == org_id)
            users_result = await self.db_session.execute(users_query)
            users = users_result.scalars().all()
            
            policies_query = select(Policy).where(Policy.organization_id == org_id)
            policies_result = await self.db_session.execute(policies_query)
            policies = policies_result.scalars().all()
            
            org_data = {
                "organization": {
                    "id": str(organization.id),
                    "name": organization.name,
                    "organization_type": organization.organization_type,
                    "is_active": organization.is_active,
                    "created_at": organization.created_at.isoformat(),
                    "updated_at": organization.updated_at.isoformat()
                },
                "users": [
                    {
                        "id": str(user.id),
                        "email": user.email,
                        "is_active": user.is_active,
                        "created_at": user.created_at.isoformat()
                    }
                    for user in users
                ],
                "policies": [
                    {
                        "id": str(policy.id),
                        "policy_number": policy.policy_number,
                        "policy_type": policy.policy_type,
                        "status": policy.status.value
                    }
                    for policy in policies
                ]
            }
            
            return org_data
            
        except Exception as e:
            self.logger.error("Failed to gather organization compliance data", error=str(e))
            raise
    
    def _determine_applicable_frameworks(self, entity_type: str, entity_data: Dict[str, Any]) -> List[ComplianceFramework]:
        """Determine applicable compliance frameworks"""
        
        frameworks = []
        
        # Always check GDPR for EU-related data
        frameworks.append(ComplianceFramework.GDPR)
        
        # Insurance-specific frameworks
        if entity_type in ["claim", "policy"]:
            frameworks.extend([
                ComplianceFramework.NAIC,
                ComplianceFramework.STATE_INSURANCE
            ])
        
        # Financial frameworks for high-value transactions
        if entity_type == "claim":
            claim_amount = entity_data.get("claim", {}).get("amount_claimed", 0.0)
            if claim_amount > 100000:  # High-value claims
                frameworks.append(ComplianceFramework.SOX)
        
        return frameworks
    
    async def _check_framework_compliance(
        self, 
        framework: ComplianceFramework, 
        entity_type: str, 
        entity_data: Dict[str, Any],
        check_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Check compliance for specific framework"""
        
        try:
            framework_rules = self._get_framework_rules(framework)
            
            # Filter rules by check types if specified
            if check_types:
                framework_rules = [rule for rule in framework_rules if rule.id in check_types]
            
            violations = []
            compliance_score = 100.0
            checks_performed = 0
            
            for rule in framework_rules:
                try:
                    # Perform rule check
                    check_result = await self._perform_rule_check(rule, entity_type, entity_data)
                    checks_performed += 1
                    
                    if not check_result["compliant"]:
                        violation = {
                            "rule_id": rule.id,
                            "rule_name": rule.name,
                            "framework": framework.value,
                            "severity": rule.severity.value,
                            "description": check_result.get("violation_description", rule.description),
                            "details": check_result.get("details", {}),
                            "remediation_steps": rule.remediation_steps,
                            "detected_at": datetime.utcnow().isoformat()
                        }
                        violations.append(violation)
                        
                        # Reduce compliance score based on severity
                        severity_impact = {
                            ViolationSeverity.CRITICAL: 25,
                            ViolationSeverity.HIGH: 15,
                            ViolationSeverity.MEDIUM: 10,
                            ViolationSeverity.LOW: 5,
                            ViolationSeverity.INFO: 1
                        }
                        compliance_score -= severity_impact.get(rule.severity, 5)
                
                except Exception as e:
                    self.logger.warning(f"Rule check failed for {rule.id}", error=str(e))
            
            compliance_score = max(0.0, compliance_score)
            
            return {
                "framework": framework.value,
                "compliance_score": compliance_score,
                "checks_performed": checks_performed,
                "violations": violations,
                "rules_checked": [rule.id for rule in framework_rules]
            }
            
        except Exception as e:
            self.logger.error(f"Framework compliance check failed for {framework.value}", error=str(e))
            return {
                "framework": framework.value,
                "compliance_score": 0.0,
                "checks_performed": 0,
                "violations": [],
                "error": str(e)
            }
    
    def _get_framework_rules(self, framework: ComplianceFramework) -> List[ComplianceRule]:
        """Get compliance rules for framework"""
        
        return [rule for rule in self.compliance_rules.values() if rule.framework == framework]
    
    async def _perform_rule_check(
        self, 
        rule: ComplianceRule, 
        entity_type: str, 
        entity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform individual rule check"""
        
        try:
            # Route to specific check function
            check_function = getattr(self, rule.check_function, None)
            if not check_function:
                return {
                    "compliant": False,
                    "violation_description": f"Check function {rule.check_function} not implemented",
                    "details": {}
                }
            
            return await check_function(rule, entity_type, entity_data)
            
        except Exception as e:
            self.logger.error(f"Rule check failed for {rule.id}", error=str(e))
            return {
                "compliant": False,
                "violation_description": f"Check execution failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    # Specific compliance check functions
    
    async def check_data_retention(self, rule: ComplianceRule, entity_type: str, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data retention compliance"""
        
        try:
            max_retention_days = rule.parameters.get("max_retention_days", 2555)
            current_date = datetime.utcnow()
            
            # Check user data retention
            if "user" in entity_data and entity_data["user"]:
                user_created = datetime.fromisoformat(entity_data["user"]["created_at"])
                retention_days = (current_date - user_created).days
                
                if retention_days > max_retention_days:
                    return {
                        "compliant": False,
                        "violation_description": f"User data retained for {retention_days} days, exceeds limit of {max_retention_days} days",
                        "details": {
                            "user_id": entity_data["user"]["id"],
                            "retention_days": retention_days,
                            "max_allowed": max_retention_days
                        }
                    }
            
            # Check policy data retention
            if "policy" in entity_data and entity_data["policy"]:
                if entity_data["policy"].get("expiry_date"):
                    expiry_date = datetime.fromisoformat(entity_data["policy"]["expiry_date"])
                    days_since_expiry = (current_date - expiry_date).days
                    
                    if days_since_expiry > max_retention_days:
                        return {
                            "compliant": False,
                            "violation_description": f"Expired policy data retained for {days_since_expiry} days after expiry",
                            "details": {
                                "policy_id": entity_data["policy"]["id"],
                                "days_since_expiry": days_since_expiry,
                                "max_allowed": max_retention_days
                            }
                        }
            
            return {"compliant": True, "details": {}}
            
        except Exception as e:
            return {
                "compliant": False,
                "violation_description": f"Data retention check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_consent_tracking(self, rule: ComplianceRule, entity_type: str, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check consent tracking compliance"""
        
        try:
            require_explicit_consent = rule.parameters.get("require_explicit_consent", True)
            
            if "user" in entity_data and entity_data["user"]:
                user_data = entity_data["user"]
                
                # Check if consent is recorded
                consent_given = user_data.get("consent_given")
                consent_date = user_data.get("consent_date")
                
                if require_explicit_consent:
                    if consent_given is None or not consent_given:
                        return {
                            "compliant": False,
                            "violation_description": "No explicit consent recorded for data processing",
                            "details": {
                                "user_id": user_data["id"],
                                "consent_given": consent_given,
                                "consent_date": consent_date
                            }
                        }
                    
                    if not consent_date:
                        return {
                            "compliant": False,
                            "violation_description": "Consent date not recorded",
                            "details": {
                                "user_id": user_data["id"],
                                "consent_given": consent_given
                            }
                        }
            
            return {"compliant": True, "details": {}}
            
        except Exception as e:
            return {
                "compliant": False,
                "violation_description": f"Consent tracking check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_breach_notification(self, rule: ComplianceRule, entity_type: str, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check breach notification compliance"""
        
        try:
            notification_hours = rule.parameters.get("notification_hours", 72)
            
            # Check audit logs for security incidents
            if "audit_logs" in entity_data:
                for log in entity_data["audit_logs"]:
                    if "security_incident" in log.get("action", "").lower():
                        incident_time = datetime.fromisoformat(log["timestamp"])
                        hours_since_incident = (datetime.utcnow() - incident_time).total_seconds() / 3600
                        
                        # Check if notification was sent within required timeframe
                        notification_sent = any(
                            "breach_notification" in audit_log.get("action", "").lower()
                            for audit_log in entity_data["audit_logs"]
                            if datetime.fromisoformat(audit_log["timestamp"]) > incident_time
                        )
                        
                        if hours_since_incident > notification_hours and not notification_sent:
                            return {
                                "compliant": False,
                                "violation_description": f"Security incident not reported within {notification_hours} hours",
                                "details": {
                                    "incident_id": log["id"],
                                    "incident_time": log["timestamp"],
                                    "hours_since_incident": hours_since_incident,
                                    "notification_required_within": notification_hours
                                }
                            }
            
            return {"compliant": True, "details": {}}
            
        except Exception as e:
            return {
                "compliant": False,
                "violation_description": f"Breach notification check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_claim_handling_standards(self, rule: ComplianceRule, entity_type: str, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check NAIC claim handling standards"""
        
        try:
            max_processing_days = rule.parameters.get("max_processing_days", 30)
            
            if "claim" in entity_data and entity_data["claim"]:
                claim_data = entity_data["claim"]
                processing_time_days = claim_data.get("processing_time_days")
                
                if processing_time_days and processing_time_days > max_processing_days:
                    return {
                        "compliant": False,
                        "violation_description": f"Claim processing time ({processing_time_days} days) exceeds NAIC standard ({max_processing_days} days)",
                        "details": {
                            "claim_id": claim_data["id"],
                            "processing_time_days": processing_time_days,
                            "max_allowed_days": max_processing_days,
                            "reported_date": claim_data.get("reported_date"),
                            "status_updated_at": claim_data.get("status_updated_at")
                        }
                    }
                
                # Check for proper documentation
                if not claim_data.get("claim_number"):
                    return {
                        "compliant": False,
                        "violation_description": "Claim missing required claim number",
                        "details": {"claim_id": claim_data["id"]}
                    }
            
            return {"compliant": True, "details": {}}
            
        except Exception as e:
            return {
                "compliant": False,
                "violation_description": f"Claim handling standards check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_unfair_practices(self, rule: ComplianceRule, entity_type: str, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for unfair claims practices"""
        
        try:
            denial_rate_threshold = rule.parameters.get("denial_rate_threshold", 0.15)
            
            if "claims" in entity_data and entity_data["claims"]:
                claims = entity_data["claims"]
                
                if len(claims) >= 10:  # Only check if sufficient sample size
                    denied_claims = [
                        claim for claim in claims 
                        if claim.get("amount_approved", 0) == 0 and claim.get("amount_claimed", 0) > 0
                    ]
                    
                    denial_rate = len(denied_claims) / len(claims)
                    
                    if denial_rate > denial_rate_threshold:
                        return {
                            "compliant": False,
                            "violation_description": f"High claim denial rate ({denial_rate:.1%}) exceeds threshold ({denial_rate_threshold:.1%})",
                            "details": {
                                "total_claims": len(claims),
                                "denied_claims": len(denied_claims),
                                "denial_rate": denial_rate,
                                "threshold": denial_rate_threshold
                            }
                        }
            
            return {"compliant": True, "details": {}}
            
        except Exception as e:
            return {
                "compliant": False,
                "violation_description": f"Unfair practices check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_settlement_timeframes(self, rule: ComplianceRule, entity_type: str, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check state-mandated settlement timeframes"""
        
        try:
            max_settlement_days = rule.parameters.get("max_settlement_days", 45)
            
            if "claim" in entity_data and entity_data["claim"]:
                claim_data = entity_data["claim"]
                
                # Check if claim is settled
                if claim_data.get("status") == "settled" and claim_data.get("amount_approved", 0) > 0:
                    reported_date = claim_data.get("reported_date")
                    status_updated_at = claim_data.get("status_updated_at")
                    
                    if reported_date and status_updated_at:
                        settlement_days = (
                            datetime.fromisoformat(status_updated_at) - 
                            datetime.fromisoformat(reported_date)
                        ).days
                        
                        if settlement_days > max_settlement_days:
                            return {
                                "compliant": False,
                                "violation_description": f"Claim settlement time ({settlement_days} days) exceeds state requirement ({max_settlement_days} days)",
                                "details": {
                                    "claim_id": claim_data["id"],
                                    "settlement_days": settlement_days,
                                    "max_allowed_days": max_settlement_days,
                                    "reported_date": reported_date,
                                    "settled_date": status_updated_at
                                }
                            }
            
            return {"compliant": True, "details": {}}
            
        except Exception as e:
            return {
                "compliant": False,
                "violation_description": f"Settlement timeframes check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_financial_reporting(self, rule: ComplianceRule, entity_type: str, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check financial reporting accuracy"""
        
        try:
            variance_threshold = rule.parameters.get("variance_threshold", 0.05)
            
            if "claims" in entity_data and entity_data["claims"]:
                claims = entity_data["claims"]
                
                # Check for significant variances between claimed and approved amounts
                for claim in claims:
                    amount_claimed = claim.get("amount_claimed", 0.0)
                    amount_approved = claim.get("amount_approved", 0.0)
                    
                    if amount_claimed > 0:
                        variance = abs(amount_approved - amount_claimed) / amount_claimed
                        
                        # Large variances without proper documentation could indicate issues
                        if variance > variance_threshold and amount_claimed > 10000:
                            return {
                                "compliant": False,
                                "violation_description": f"Large variance ({variance:.1%}) between claimed and approved amounts requires documentation",
                                "details": {
                                    "claim_id": claim.get("id"),
                                    "amount_claimed": amount_claimed,
                                    "amount_approved": amount_approved,
                                    "variance": variance,
                                    "threshold": variance_threshold
                                }
                            }
            
            return {"compliant": True, "details": {}}
            
        except Exception as e:
            return {
                "compliant": False,
                "violation_description": f"Financial reporting check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _determine_compliance_status(self, compliance_score: float, violations: List[Dict[str, Any]]) -> ComplianceStatus:
        """Determine overall compliance status"""
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.get("severity") == "critical"]
        if critical_violations:
            return ComplianceStatus.NON_COMPLIANT
        
        # Check compliance score
        if compliance_score >= 95:
            return ComplianceStatus.COMPLIANT
        elif compliance_score >= 80:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT
    
    async def _generate_compliance_recommendations(
        self, 
        violations: List[Dict[str, Any]], 
        entity_type: str, 
        entity_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate compliance recommendations"""
        
        try:
            recommendations = []
            
            # Group violations by severity
            critical_violations = [v for v in violations if v.get("severity") == "critical"]
            high_violations = [v for v in violations if v.get("severity") == "high"]
            
            # Critical violation recommendations
            for violation in critical_violations:
                recommendations.append({
                    "priority": "critical",
                    "category": "compliance",
                    "title": f"Address Critical Violation: {violation['rule_name']}",
                    "description": violation["description"],
                    "remediation_steps": violation["remediation_steps"],
                    "estimated_effort": "high",
                    "deadline": (datetime.utcnow() + timedelta(
                        hours=self.compliance_config["critical_violation_escalation_hours"]
                    )).isoformat()
                })
            
            # High violation recommendations
            for violation in high_violations:
                recommendations.append({
                    "priority": "high",
                    "category": "compliance",
                    "title": f"Address High Priority Violation: {violation['rule_name']}",
                    "description": violation["description"],
                    "remediation_steps": violation["remediation_steps"],
                    "estimated_effort": "medium",
                    "deadline": (datetime.utcnow() + timedelta(
                        hours=self.compliance_config["violation_escalation_hours"]
                    )).isoformat()
                })
            
            # General compliance improvements
            if len(violations) > 5:
                recommendations.append({
                    "priority": "medium",
                    "category": "process_improvement",
                    "title": "Implement Comprehensive Compliance Review",
                    "description": f"Multiple violations detected ({len(violations)}), consider systematic compliance review",
                    "remediation_steps": [
                        "Conduct compliance audit",
                        "Update policies and procedures",
                        "Implement automated compliance monitoring",
                        "Train staff on compliance requirements"
                    ],
                    "estimated_effort": "high"
                })
            
            # Framework-specific recommendations
            frameworks_with_violations = set(v.get("framework") for v in violations)
            
            if ComplianceFramework.GDPR.value in frameworks_with_violations:
                recommendations.append({
                    "priority": "high",
                    "category": "data_protection",
                    "title": "Enhance GDPR Compliance Program",
                    "description": "GDPR violations detected, strengthen data protection measures",
                    "remediation_steps": [
                        "Review data processing activities",
                        "Update privacy notices",
                        "Implement consent management",
                        "Conduct privacy impact assessments"
                    ],
                    "estimated_effort": "high"
                })
            
            if ComplianceFramework.NAIC.value in frameworks_with_violations:
                recommendations.append({
                    "priority": "high",
                    "category": "insurance_compliance",
                    "title": "Improve NAIC Compliance",
                    "description": "NAIC violations detected, review claim handling procedures",
                    "remediation_steps": [
                        "Review claim processing workflows",
                        "Implement automated compliance checks",
                        "Train adjusters on NAIC standards",
                        "Monitor compliance metrics"
                    ],
                    "estimated_effort": "medium"
                })
            
            return recommendations[:10]  # Limit to top 10 recommendations
            
        except Exception as e:
            self.logger.error("Compliance recommendations generation failed", error=str(e))
            return []
    
    async def _save_compliance_results(
        self,
        entity_type: str,
        entity_id: uuid.UUID,
        compliance_result: Dict[str, Any]
    ):
        """Persist or log compliance results."""

        try:
            self.logger.info(
                "Compliance results ready",
                entity_type=entity_type,
                entity_id=str(entity_id),
                results=compliance_result,
            )
            # TODO: persist results via a proper service once model exists
        except Exception as e:
            self.logger.error("Failed to save compliance results", error=str(e))
            # Don't raise exception as this is not critical for the check itself
    
    async def _handle_violations(
        self, 
        violations: List[Dict[str, Any]], 
        entity_type: str, 
        entity_id: uuid.UUID
    ):
        """Handle compliance violations"""
        
        try:
            for violation in violations:
                # Log violation
                audit_logger.log_user_action(
                    user_id="system",
                    action="compliance_violation_detected",
                    resource_type=entity_type,
                    resource_id=str(entity_id),
                    details=violation
                )
                
                # Escalate critical violations
                if violation.get("severity") == "critical":
                    await self._escalate_critical_violation(violation, entity_type, entity_id)
                
                # Update violation cache
                violation_key = f"{violation['rule_id']}:{entity_type}:{entity_id}"
                self.violation_cache[violation_key] = violation
            
        except Exception as e:
            self.logger.error("Failed to handle violations", error=str(e))
    
    async def _escalate_critical_violation(
        self, 
        violation: Dict[str, Any], 
        entity_type: str, 
        entity_id: uuid.UUID
    ):
        """Escalate critical compliance violations"""
        
        try:
            # Create escalation notification
            escalation_data = {
                "violation": violation,
                "entity_type": entity_type,
                "entity_id": str(entity_id),
                "escalation_timestamp": datetime.utcnow().isoformat(),
                "requires_immediate_action": True
            }
            
            # Send to compliance team (would integrate with notification system)
            await self.redis_client.lpush("critical_compliance_violations", json.dumps(escalation_data))
            
            self.logger.critical(
                "Critical compliance violation escalated",
                violation_rule=violation["rule_id"],
                entity_type=entity_type,
                entity_id=str(entity_id)
            )
            
        except Exception as e:
            self.logger.error("Failed to escalate critical violation", error=str(e))
    
    async def generate_compliance_report(
        self,
        report_type: str = "comprehensive",
        date_range: Optional[Tuple[datetime, datetime]] = None,
        frameworks: Optional[List[ComplianceFramework]] = None,
        organization_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """Generate compliance report"""

        try:
            if not date_range:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)
                date_range = (start_date, end_date)

            start_date, end_date = date_range
            # TODO: retrieve stored compliance results once model exists
            compliance_checks: List[Dict[str, Any]] = []

            # Analyze compliance data
            total_checks = len(compliance_checks)
            compliant_checks = len([
                c for c in compliance_checks
                if c.get("compliance_status") == ComplianceStatus.COMPLIANT
            ])

            compliance_rate = (compliant_checks / total_checks * 100) if total_checks > 0 else 0

            # Violation analysis
            all_violations = []
            for check in compliance_checks:
                violations = check.get("violations", [])
                all_violations.extend(violations)
            
            violations_by_severity = {}
            violations_by_framework = {}
            
            for violation in all_violations:
                severity = violation.get("severity", "unknown")
                framework = violation.get("framework", "unknown")
                
                violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
                violations_by_framework[framework] = violations_by_framework.get(framework, 0) + 1
            
            # Generate report
            report = {
                "report_type": report_type,
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "generated_at": datetime.utcnow().isoformat(),
                "summary": {
                    "total_checks_performed": total_checks,
                    "compliant_checks": compliant_checks,
                    "compliance_rate_percent": round(compliance_rate, 2),
                    "total_violations": len(all_violations),
                    "critical_violations": violations_by_severity.get("critical", 0),
                    "high_violations": violations_by_severity.get("high", 0)
                },
                "violations_by_severity": violations_by_severity,
                "violations_by_framework": violations_by_framework,
                "compliance_trends": await self._analyze_compliance_trends(start_date, end_date),
                "recommendations": await self._generate_report_recommendations(all_violations),
                "report_metadata": {
                    "agent_name": self.agent_name,
                    "agent_version": self.agent_version,
                    "frameworks_included": [f.value for f in frameworks] if frameworks else "all"
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error("Compliance report generation failed", error=str(e))
            raise ServiceException(f"Report generation failed: {str(e)}")
    
    async def _analyze_compliance_trends(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze compliance trends over time"""

        try:
            self.logger.info(
                "Compliance trend analysis not implemented", start=str(start_date), end=str(end_date)
            )
            return {"daily_trends": [], "trend_analysis": {}}

        except Exception as e:
            self.logger.warning("Compliance trends analysis failed", error=str(e))
            return {"daily_trends": [], "trend_analysis": {}}
    
    async def _generate_report_recommendations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations for compliance report"""
        
        try:
            recommendations = []
            
            # Analyze violation patterns
            violation_counts = {}
            for violation in violations:
                rule_id = violation.get("rule_id", "unknown")
                violation_counts[rule_id] = violation_counts.get(rule_id, 0) + 1
            
            # Recommend addressing frequent violations
            frequent_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for rule_id, count in frequent_violations:
                if count > 1:
                    recommendations.append({
                        "priority": "high",
                        "category": "process_improvement",
                        "title": f"Address Recurring Violation: {rule_id}",
                        "description": f"Rule {rule_id} violated {count} times, indicating systematic issue",
                        "action": "systematic_review",
                        "estimated_impact": "high"
                    })
            
            # Framework-specific recommendations
            frameworks_with_violations = set(v.get("framework") for v in violations)
            
            for framework in frameworks_with_violations:
                framework_violations = [v for v in violations if v.get("framework") == framework]
                if len(framework_violations) > 3:
                    recommendations.append({
                        "priority": "medium",
                        "category": "framework_compliance",
                        "title": f"Strengthen {framework.upper()} Compliance",
                        "description": f"Multiple {framework} violations detected, review compliance program",
                        "action": "compliance_program_review",
                        "estimated_impact": "medium"
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error("Report recommendations generation failed", error=str(e))
            return []

# Agent factory function
async def create_compliance_agent(db_session: AsyncSession, redis_client: redis.Redis) -> ComplianceAgent:
    """Create compliance agent instance"""
    return ComplianceAgent(db_session, redis_client)

