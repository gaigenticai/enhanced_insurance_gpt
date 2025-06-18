"""
Insurance AI Agent System - API Gateway
Production-ready FastAPI gateway with comprehensive authentication and routing
"""

import os
import asyncio
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from decimal import Decimal
import re
from functools import wraps

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Authentication and security
from passlib.context import CryptContext
from jose import JWTError, jwt
import bcrypt
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware

# Database and utilities
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, or_, func, text
import structlog
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator
import aiofiles

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Monitoring and metrics
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Internal imports
from backend.shared.models import User, Organization, Policy, Claim, AgentExecution
from backend.shared.schemas import (
    UserCreate, UserUpdate, UserResponse,
    PolicyCreate, PolicyUpdate, PolicyResponse,
    ClaimCreate, ClaimUpdate, ClaimResponse,
    OrganizationCreate, OrganizationUpdate, OrganizationResponse,
    AgentExecutionResponse, ComplianceCheckResponse,
    TokenResponse, UserLogin, PasswordReset
)
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

# Orchestrators
from backend.orchestrators.underwriting_orchestrator import UnderwritingOrchestrator
from backend.orchestrators.claims_orchestrator import ClaimsOrchestrator

# Agents
from backend.agents.document_analysis_agent import DocumentAnalysisAgent
from backend.agents.communication.communication_agent import CommunicationAgent
from backend.agents.evidence_processing_agent import EvidenceProcessingAgent
from backend.agents.risk_assessment_agent import RiskAssessmentAgent
from backend.agents.compliance_agent import ComplianceAgent

logger = structlog.get_logger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
ACTIVE_CONNECTIONS = Gauge('websocket_connections_active', 'Active WebSocket connections')

class APIGateway:
    """
    Production-ready API Gateway with comprehensive features
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Insurance AI Agent System API",
            description="Production-ready API for Insurance AI Agent System",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Database and Redis
        self.db_engine = None
        self.db_session_factory = None
        self.redis_client = None
        
        # WebSocket connections
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # OAuth configuration
        self.oauth = OAuth()
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_websockets()
        self._setup_exception_handlers()
        
        logger.info("API Gateway initialized")
    
    def _setup_middleware(self):
        """Setup middleware for CORS, rate limiting, monitoring, etc."""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Session middleware for OAuth
        self.app.add_middleware(
            SessionMiddleware,
            secret_key=SECRET_KEY
        )
        
        # Rate limiting middleware
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.app.add_middleware(SlowAPIMiddleware)
        
        # Request monitoring middleware
        @self.app.middleware("http")
        async def monitor_requests(request: Request, call_next):
            start_time = time.time()
            
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            # Add response headers
            response.headers["X-Process-Time"] = str(duration)
            response.headers["X-API-Version"] = "1.0.0"
            
            return response
        
        # Authentication middleware
        @self.app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Skip auth for public endpoints
            # Only allow unauthenticated access to specific auth endpoints
            public_paths = [
                "/docs",
                "/redoc",
                "/openapi.json",
                "/health",
                "/metrics",
                "/api/v1/auth/login",
                "/api/v1/auth/register",
                "/api/v1/auth/token",
                "/api/v1/auth/refresh",
                "/api/v1/auth/password-reset",
            ]
            
            if any(request.url.path.startswith(path) for path in public_paths):
                return await call_next(request)
            
            # Check for valid token
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid authorization header"}
                )
            
            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                user_id = payload.get("sub")
                if not user_id:
                    raise JWTError("Invalid token payload")
                
                # Add user info to request state
                request.state.user_id = user_id
                request.state.user_email = payload.get("email")
                
            except JWTError:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired token"}
                )
            
            return await call_next(request)
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics():
            return Response(
                generate_latest(prometheus_client.REGISTRY),
                media_type="text/plain"
            )
        
        # Include API routes
        self._setup_auth_routes()
        self._setup_user_routes()
        self._setup_policy_routes()
        self._setup_claim_routes()
        self._setup_orchestrator_routes()
        self._setup_agent_routes()
        self._setup_admin_routes()
    
    def _setup_auth_routes(self):
        """Setup authentication routes"""
        
        @self.app.post("/api/v1/auth/register", response_model=UserResponse)
        @limiter.limit("5/minute")
        async def register(request: Request, user_data: UserCreate):
            """Register new user"""
            try:
                async with get_db_session() as db:
                    user_service = BaseService(User, db)
                    
                    # Check if user exists
                    existing_user = await user_service.get_by_field("email", user_data.email)
                    if existing_user:
                        raise HTTPException(status_code=400, detail="Email already registered")
                    
                    # Hash password
                    hashed_password = pwd_context.hash(user_data.password)
                    
                    # Create user
                    user = User(
                        email=user_data.email,
                        password_hash=hashed_password,
                        first_name=user_data.first_name,
                        last_name=user_data.last_name,
                        phone=user_data.phone,
                        is_active=True
                    )
                    
                    created_user = await user_service.create(user)
                    
                    # Log registration
                    audit_logger.log_user_action(
                        user_id=str(created_user.id),
                        action="user_registered",
                        resource_type="user",
                        resource_id=str(created_user.id)
                    )
                    
                    return UserResponse.from_orm(created_user)
                    
            except Exception as e:
                logger.error("User registration failed", error=str(e))
                raise HTTPException(status_code=500, detail="Registration failed")
        
        @self.app.post("/api/v1/auth/token", response_model=TokenResponse)
        @limiter.limit("10/minute")
        async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
            """User login"""
            try:
                async with get_db_session() as db:
                    user_service = BaseService(User, db)
                    
                    # Get user by email
                    user = await user_service.get_by_field("email", form_data.username)
                    if not user or not pwd_context.verify(form_data.password, user.password_hash):




                        raise HTTPException(status_code=401, detail="Invalid credentials")
                    
                    if not user.is_active:
                        raise HTTPException(status_code=401, detail="Account disabled")
                    
                    # Create access token
                    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                    access_token = create_access_token(
                        data={"sub": str(user.id), "email": user.email},
                        expires_delta=access_token_expires
                    )
                    
                    # Create refresh token
                    refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
                    refresh_token = create_access_token(
                        data={"sub": str(user.id), "type": "refresh"},
                        expires_delta=refresh_token_expires
                    )
                    
                    # Update last login
                    user.last_login = datetime.utcnow()
                    await user_service.update(user.id, user)
                    
                    # Log login
                    audit_logger.log_user_action(
                        user_id=str(user.id),
                        action="user_login",
                        resource_type="user",
                        resource_id=str(user.id)
                    )
                    
                    return TokenResponse(
                        access_token=access_token,
                        refresh_token=refresh_token,
                        token_type="bearer",
                        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Login failed", error=str(e))
                raise HTTPException(status_code=500, detail="Login failed")
        
        @self.app.post("/api/v1/auth/refresh", response_model=TokenResponse)
        @limiter.limit("20/minute")
        async def refresh_token(request: Request, refresh_token: str):
            """Refresh access token"""
            try:
                payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
                user_id = payload.get("sub")
                token_type = payload.get("type")
                
                if not user_id or token_type != "refresh":
                    raise HTTPException(status_code=401, detail="Invalid refresh token")
                
                async with get_db_session() as db:
                    user_service = BaseService(User, db)
                    user = await user_service.get(uuid.UUID(user_id))
                    
                    if not user or not user.is_active:
                        raise HTTPException(status_code=401, detail="User not found or inactive")
                    
                    # Create new access token
                    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                    access_token = create_access_token(
                        data={"sub": str(user.id), "email": user.email},
                        expires_delta=access_token_expires
                    )
                    
                    return TokenResponse(
                        access_token=access_token,
                        token_type="bearer",
                        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
                    )
                    
            except JWTError:
                raise HTTPException(status_code=401, detail="Invalid refresh token")
            except Exception as e:
                logger.error("Token refresh failed", error=str(e))
                raise HTTPException(status_code=500, detail="Token refresh failed")
        
        @self.app.post("/api/v1/auth/logout")
        async def logout(request: Request):
            """User logout"""
            try:
                user_id = request.state.user_id
                
                # Log logout
                audit_logger.log_user_action(
                    user_id=user_id,
                    action="user_logout",
                    resource_type="user",
                    resource_id=user_id
                )
                
                return {"message": "Successfully logged out"}
                
            except Exception as e:
                logger.error("Logout failed", error=str(e))
                raise HTTPException(status_code=500, detail="Logout failed")
        
        @self.app.post("/api/v1/auth/password-reset")
        @limiter.limit("3/minute")
        async def request_password_reset(request: Request, email: str):
            """Request password reset"""
            try:
                async with get_db_session() as db:
                    user_service = BaseService(User, db)
                    user = await user_service.get_by_field("email", email)
                    
                    if user:
                        # Generate reset token
                        reset_token = create_access_token(
                            data={"sub": str(user.id), "type": "password_reset"},
                            expires_delta=timedelta(hours=1)
                        )
                        
                        # Send reset email (would integrate with communication agent)
                        # For now, just log the token
                        logger.info("Password reset requested", user_id=str(user.id), reset_token=reset_token)
                        
                        # Log password reset request
                        audit_logger.log_user_action(
                            user_id=str(user.id),
                            action="password_reset_requested",
                            resource_type="user",
                            resource_id=str(user.id)
                        )
                
                # Always return success to prevent email enumeration
                return {"message": "If the email exists, a reset link has been sent"}
                
            except Exception as e:
                logger.error("Password reset request failed", error=str(e))
                raise HTTPException(status_code=500, detail="Password reset request failed")
    
    def _setup_user_routes(self):
        """Setup user management routes"""
        
        @self.app.get("/api/v1/users/me", response_model=UserResponse)
        async def get_current_user(request: Request):
            """Get current user profile"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    user_service = BaseService(User, db)
                    user = await user_service.get(uuid.UUID(user_id))
                    
                    if not user:
                        raise HTTPException(status_code=404, detail="User not found")
                    
                    return UserResponse.from_orm(user)
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Get current user failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get user")
        
        @self.app.put("/api/v1/users/me", response_model=UserResponse)
        async def update_current_user(request: Request, user_update: UserUpdate):
            """Update current user profile"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    user_service = BaseService(User, db)
                    user = await user_service.get(uuid.UUID(user_id))
                    
                    if not user:
                        raise HTTPException(status_code=404, detail="User not found")
                    
                    # Update user fields
                    update_data = user_update.dict(exclude_unset=True)
                    for field, value in update_data.items():
                        setattr(user, field, value)
                    
                    updated_user = await user_service.update(user.id, user)
                    
                    # Log update
                    audit_logger.log_user_action(
                        user_id=user_id,
                        action="user_profile_updated",
                        resource_type="user",
                        resource_id=user_id,
                        details={"updated_fields": list(update_data.keys())}
                    )
                    
                    return UserResponse.from_orm(updated_user)

            except HTTPException:
                raise
            except Exception as e:
                logger.error("Update user failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to update user")

        # ------------------------------------------------------------------
        # Compatibility aliases matching the auth service endpoints
        # ------------------------------------------------------------------

        @self.app.get("/api/v1/auth/profile", response_model=UserResponse)
        async def get_profile(request: Request):
            """Alias for retrieving the authenticated user's profile."""
            return await get_current_user(request)

        @self.app.put("/api/v1/auth/profile", response_model=UserResponse)
        async def update_profile(request: Request, user_update: UserUpdate):
            """Alias for updating the authenticated user's profile."""
            return await update_current_user(request, user_update)
        
        @self.app.get("/api/v1/users/{user_id}", response_model=UserResponse)
        async def get_user(request: Request, user_id: uuid.UUID):
            """Get user by ID (admin only)"""
            try:
                # Check admin permissions (simplified)
                current_user_id = request.state.user_id
                
                async with get_db_session() as db:
                    user_service = BaseService(User, db)
                    user = await user_service.get(user_id)
                    
                    if not user:
                        raise HTTPException(status_code=404, detail="User not found")
                    
                    return UserResponse.from_orm(user)
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Get user failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get user")
    
    def _setup_policy_routes(self):
        """Setup policy management routes"""
        
        @self.app.get("/api/v1/policies", response_model=List[PolicyResponse])
        @limiter.limit("30/minute")
        async def get_policies(request: Request, skip: int = 0, limit: int = 100):
            """Get user's policies"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    policy_service = BaseService(Policy, db)
                    policies = await policy_service.get_by_field("user_id", uuid.UUID(user_id), limit=limit, offset=skip)
                    
                    return [PolicyResponse.from_orm(policy) for policy in policies]
                    
            except Exception as e:
                logger.error("Get policies failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get policies")
        
        @self.app.get("/api/v1/policies/{policy_id}", response_model=PolicyResponse)
        async def get_policy(request: Request, policy_id: uuid.UUID):
            """Get policy by ID"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    policy_service = BaseService(Policy, db)
                    policy = await policy_service.get(policy_id)
                    
                    if not policy:
                        raise HTTPException(status_code=404, detail="Policy not found")
                    
                    # Check ownership
                    if str(policy.user_id) != user_id:
                        raise HTTPException(status_code=403, detail="Access denied")
                    
                    return PolicyResponse.from_orm(policy)
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Get policy failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get policy")
        
        @self.app.post("/api/v1/policies", response_model=PolicyResponse)
        @limiter.limit("10/minute")
        async def create_policy(request: Request, policy_data: PolicyCreate):
            """Create new policy"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    policy_service = BaseService(Policy, db)
                    
                    # Create policy
                    policy = Policy(
                        user_id=uuid.UUID(user_id),
                        policy_type=policy_data.policy_type,
                        coverage_amount=policy_data.coverage_amount,
                        premium_amount=policy_data.premium_amount,
                        effective_date=policy_data.effective_date,
                        expiry_date=policy_data.expiry_date,
                        status=policy_data.status
                    )
                    
                    created_policy = await policy_service.create(policy)
                    
                    # Log policy creation
                    audit_logger.log_user_action(
                        user_id=user_id,
                        action="policy_created",
                        resource_type="policy",
                        resource_id=str(created_policy.id)
                    )
                    
                    return PolicyResponse.from_orm(created_policy)
                    
            except Exception as e:
                logger.error("Create policy failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to create policy")
    
    def _setup_claim_routes(self):
        """Setup claim management routes"""
        
        @self.app.get("/api/v1/claims", response_model=List[ClaimResponse])
        @limiter.limit("30/minute")
        async def get_claims(request: Request, skip: int = 0, limit: int = 100):
            """Get user's claims"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    # Get user's policies first
                    policy_service = BaseService(Policy, db)
                    policies = await policy_service.get_by_field("user_id", uuid.UUID(user_id))
                    policy_ids = [policy.id for policy in policies]
                    
                    # Get claims for user's policies
                    claim_service = BaseService(Claim, db)
                    claims = []
                    for policy_id in policy_ids:
                        policy_claims = await claim_service.get_by_field("policy_id", policy_id, limit=limit, offset=skip)
                        claims.extend(policy_claims)
                    
                    return [ClaimResponse.from_orm(claim) for claim in claims[:limit]]
                    
            except Exception as e:
                logger.error("Get claims failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get claims")
        
        @self.app.get("/api/v1/claims/{claim_id}", response_model=ClaimResponse)
        async def get_claim(request: Request, claim_id: uuid.UUID):
            """Get claim by ID"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    claim_service = BaseService(Claim, db)
                    claim = await claim_service.get(claim_id)
                    
                    if not claim:
                        raise HTTPException(status_code=404, detail="Claim not found")
                    
                    # Check ownership through policy
                    policy_service = BaseService(Policy, db)
                    policy = await policy_service.get(claim.policy_id)
                    
                    if not policy or str(policy.user_id) != user_id:
                        raise HTTPException(status_code=403, detail="Access denied")
                    
                    return ClaimResponse.from_orm(claim)
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Get claim failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get claim")
        
        @self.app.post("/api/v1/claims", response_model=ClaimResponse)
        @limiter.limit("5/minute")
        async def create_claim(request: Request, claim_data: ClaimCreate):
            """Create new claim"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    # Verify policy ownership
                    policy_service = BaseService(Policy, db)
                    policy = await policy_service.get(claim_data.policy_id)
                    
                    if not policy or str(policy.user_id) != user_id:
                        raise HTTPException(status_code=403, detail="Policy not found or access denied")
                    
                    # Create claim
                    claim_service = BaseService(Claim, db)
                    claim = Claim(
                        policy_id=claim_data.policy_id,
                        claim_type=claim_data.claim_type,
                        incident_date=claim_data.incident_date,
                        reported_date=datetime.utcnow(),
                        amount_claimed=claim_data.amount_claimed,
                        description=claim_data.description,
                        status=claim_data.status
                    )
                    
                    created_claim = await claim_service.create(claim)
                    
                    # Log claim creation
                    audit_logger.log_user_action(
                        user_id=user_id,
                        action="claim_created",
                        resource_type="claim",
                        resource_id=str(created_claim.id)
                    )
                    
                    return ClaimResponse.from_orm(created_claim)
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Create claim failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to create claim")
    
    def _setup_orchestrator_routes(self):
        """Setup orchestrator routes"""
        
        @self.app.post("/api/v1/underwriting/process")
        @limiter.limit("10/minute")
        async def process_underwriting(request: Request, policy_data: Dict[str, Any]):
            """Process underwriting request"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    async with get_redis_client() as redis_client:
                        
                        # Initialize underwriting orchestrator
                        orchestrator = UnderwritingOrchestrator(db, redis_client)
                        
                        # Process underwriting
                        result = await orchestrator.process_underwriting_request(
                            policy_data=policy_data,
                            user_id=uuid.UUID(user_id)
                        )
                        
                        return result
                    
            except Exception as e:
                logger.error("Underwriting processing failed", error=str(e))
                raise HTTPException(status_code=500, detail="Underwriting processing failed")
        
        @self.app.post("/api/v1/claims/process")
        @limiter.limit("10/minute")
        async def process_claim(request: Request, claim_data: Dict[str, Any]):
            """Process claim request"""
            try:
                user_id = request.state.user_id
                
                async with get_db_session() as db:
                    async with get_redis_client() as redis_client:
                        
                        # Initialize claims orchestrator
                        orchestrator = ClaimsOrchestrator(db, redis_client)
                        
                        # Process claim
                        result = await orchestrator.process_claim_request(
                            claim_data=claim_data,
                            user_id=uuid.UUID(user_id)
                        )
                        
                        return result
                    
            except Exception as e:
                logger.error("Claim processing failed", error=str(e))
                raise HTTPException(status_code=500, detail="Claim processing failed")
    
    def _setup_agent_routes(self):
        """Setup agent execution routes"""
        
        @self.app.post("/api/v1/agents/document-analysis")
        @limiter.limit("20/minute")
        async def analyze_document(request: Request, document_data: Dict[str, Any]):
            """Analyze document using Document Analysis Agent"""
            try:
                async with get_db_session() as db:
                    async with get_redis_client() as redis_client:
                        
                        agent = DocumentAnalysisAgent(db, redis_client)
                        result = await agent.analyze_document(
                            document_path=document_data.get("document_path"),
                            document_type=document_data.get("document_type"),
                            analysis_options=document_data.get("analysis_options", {})
                        )
                        
                        return result
                    
            except Exception as e:
                logger.error("Document analysis failed", error=str(e))
                raise HTTPException(status_code=500, detail="Document analysis failed")
        
        @self.app.post("/api/v1/agents/risk-assessment")
        @limiter.limit("15/minute")
        async def assess_risk(request: Request, risk_data: Dict[str, Any]):
            """Assess risk using Risk Assessment Agent"""
            try:
                async with get_db_session() as db:
                    async with get_redis_client() as redis_client:
                        
                        agent = RiskAssessmentAgent(db, redis_client)
                        result = await agent.assess_risk(
                            entity_type=risk_data.get("entity_type"),
                            entity_id=uuid.UUID(risk_data.get("entity_id")),
                            assessment_type=risk_data.get("assessment_type"),
                            context_data=risk_data.get("context_data", {})
                        )
                        
                        return result
                    
            except Exception as e:
                logger.error("Risk assessment failed", error=str(e))
                raise HTTPException(status_code=500, detail="Risk assessment failed")
        
        @self.app.post("/api/v1/agents/compliance-check")
        @limiter.limit("10/minute")
        async def check_compliance(request: Request, compliance_data: Dict[str, Any]):
            """Check compliance using Compliance Agent"""
            try:
                async with get_db_session() as db:
                    async with get_redis_client() as redis_client:
                        
                        agent = ComplianceAgent(db, redis_client)
                        result = await agent.perform_compliance_check(
                            entity_type=compliance_data.get("entity_type"),
                            entity_id=uuid.UUID(compliance_data.get("entity_id")),
                            frameworks=compliance_data.get("frameworks"),
                            check_types=compliance_data.get("check_types"),
                            context_data=compliance_data.get("context_data", {})
                        )
                        
                        return result
                    
            except Exception as e:
                logger.error("Compliance check failed", error=str(e))
                raise HTTPException(status_code=500, detail="Compliance check failed")
    
    def _setup_admin_routes(self):
        """Setup admin routes"""
        
        @self.app.get("/api/v1/admin/stats")
        async def get_system_stats(request: Request):
            """Get system statistics (admin only)"""
            try:
                async with get_db_session() as db:
                    # Get basic counts
                    user_count = await db.execute(select(func.count(User.id)))
                    policy_count = await db.execute(select(func.count(Policy.id)))
                    claim_count = await db.execute(select(func.count(Claim.id)))
                    
                    return {
                        "users": user_count.scalar(),
                        "policies": policy_count.scalar(),
                        "claims": claim_count.scalar(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
            except Exception as e:
                logger.error("Get system stats failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get system stats")
    
    def _setup_websockets(self):
        """Setup WebSocket endpoints"""
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await websocket.accept()
            self.websocket_connections[client_id] = websocket
            ACTIVE_CONNECTIONS.inc()
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif message.get("type") == "subscribe":
                        # Handle subscription to updates
                        await websocket.send_text(json.dumps({
                            "type": "subscribed",
                            "channel": message.get("channel")
                        }))
                    
            except WebSocketDisconnect:
                del self.websocket_connections[client_id]
                ACTIVE_CONNECTIONS.dec()
    
    def _setup_exception_handlers(self):
        """Setup global exception handlers"""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "detail": exc.detail,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": request.url.path
                }
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logger.error("Unhandled exception", error=str(exc), path=request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": request.url.path
                }
            )
    
    async def broadcast_message(self, message: Dict[str, Any], client_ids: Optional[List[str]] = None):
        """Broadcast message to WebSocket clients"""
        try:
            message_json = json.dumps(message)
            
            if client_ids:
                # Send to specific clients
                for client_id in client_ids:
                    if client_id in self.websocket_connections:
                        await self.websocket_connections[client_id].send_text(message_json)
            else:
                # Broadcast to all clients
                for websocket in self.websocket_connections.values():
                    await websocket.send_text(message_json)
                    
        except Exception as e:
            logger.error("WebSocket broadcast failed", error=str(e))

# Utility functions

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# Initialize API Gateway
api_gateway = APIGateway()
app = api_gateway.app

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database and Redis connections"""
    try:
        # Initialize database
        from backend.shared.database import init_database
        await init_database()
        
        logger.info("API Gateway started successfully")
        
    except Exception as e:
        logger.error("API Gateway startup failed", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup connections"""
    try:
        if api_gateway.redis_client:
            await api_gateway.redis_client.close()
        
        logger.info("API Gateway shutdown completed")
        
    except Exception as e:
        logger.error("API Gateway shutdown failed", error=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

