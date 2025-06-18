"""
API Gateway Application - Production Ready
Central entry point for all Insurance AI Agent System APIs
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time
import hashlib
from jose import jwt
from jose.exceptions import JWTError
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import httpx
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_gateway_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('api_gateway_active_connections', 'Active connections')
RATE_LIMIT_HITS = Counter('api_gateway_rate_limit_hits_total', 'Rate limit hits', ['user_id', 'endpoint'])
BACKEND_HEALTH = Gauge('api_gateway_backend_health', 'Backend service health', ['service'])

# Configuration
class Config:
    """Application configuration"""
    
    # Server configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt-secret-key")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    
    # CORS configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    CORS_HEADERS = ["*"]
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # Redis configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Backend services
    BACKEND_SERVICES = {
        "auth": os.getenv("AUTH_SERVICE_URL", "http://localhost:8001"),
        "users": os.getenv("USERS_SERVICE_URL", "http://localhost:8002"),
        "customers": os.getenv("CUSTOMERS_SERVICE_URL", "http://localhost:8003"),
        "policies": os.getenv("POLICIES_SERVICE_URL", "http://localhost:8004"),
        "claims": os.getenv("CLAIMS_SERVICE_URL", "http://localhost:8005"),
        "documents": os.getenv("DOCUMENTS_SERVICE_URL", "http://localhost:8006"),
        "evidence": os.getenv("EVIDENCE_SERVICE_URL", "http://localhost:8007"),
        "ai": os.getenv("AI_SERVICE_URL", "http://localhost:8008"),
        "workflows": os.getenv("WORKFLOWS_SERVICE_URL", "http://localhost:8009"),
        "notifications": os.getenv("NOTIFICATIONS_SERVICE_URL", "http://localhost:8010"),
        "reports": os.getenv("REPORTS_SERVICE_URL", "http://localhost:8011"),
        "admin": os.getenv("ADMIN_SERVICE_URL", "http://localhost:8012")
    }
    
    # Health check configuration
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))

config = Config()

# Global variables
redis_client: Optional[redis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None
service_health: Dict[str, bool] = {}

# Security
security = HTTPBearer(auto_error=False)

class RateLimiter:
    """Redis-based rate limiter"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, limit: int, window: int) -> tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limit"""
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window)
            
            results = await pipe.execute()
            current_requests = results[1]
            
            allowed = current_requests < limit
            
            return allowed, {
                "allowed": allowed,
                "limit": limit,
                "remaining": max(0, limit - current_requests - 1),
                "reset_time": current_time + window,
                "retry_after": window if not allowed else None
            }
            
        except Exception as e:
            logger.error("Rate limiting error", error=str(e))
            # Fail open - allow request if rate limiter fails
            return True, {"allowed": True, "limit": limit, "remaining": limit}

class AuthManager:
    """JWT authentication manager"""
    
    @staticmethod
    def decode_token(token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(
                token,
                config.JWT_SECRET_KEY,
                algorithms=[config.JWT_ALGORITHM]
            )
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return None
                
            return payload
            
        except JWTError:
            return None
    
    @staticmethod
    def extract_user_info(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user information from token payload"""
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", {}),
            "session_id": payload.get("session_id")
        }

class ServiceRouter:
    """Route requests to appropriate backend services"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        
        # Route mapping
        self.routes = {
            "/auth": "auth",
            "/users": "users", 
            "/customers": "customers",
            "/policies": "policies",
            "/claims": "claims",
            "/documents": "documents",
            "/evidence": "evidence",
            "/ai": "ai",
            "/workflows": "workflows",
            "/notifications": "notifications",
            "/reports": "reports",
            "/admin": "admin"
        }
    
    def get_service_for_path(self, path: str) -> Optional[str]:
        """Determine which service should handle the request"""
        for route_prefix, service_name in self.routes.items():
            if path.startswith(route_prefix):
                return service_name
        return None
    
    async def forward_request(
        self,
        service_name: str,
        method: str,
        path: str,
        headers: Dict[str, str],
        query_params: str,
        body: Optional[bytes] = None
    ) -> httpx.Response:
        """Forward request to backend service"""
        
        service_url = config.BACKEND_SERVICES.get(service_name)
        if not service_url:
            raise HTTPException(
                status_code=503,
                detail=f"Service {service_name} not available"
            )
        
        # Construct full URL
        url = f"{service_url}{path}"
        if query_params:
            url += f"?{query_params}"
        
        # Prepare headers (remove hop-by-hop headers)
        forward_headers = {
            k: v for k, v in headers.items()
            if k.lower() not in [
                "host", "connection", "upgrade", "proxy-authenticate",
                "proxy-authorization", "te", "trailers", "transfer-encoding"
            ]
        }
        
        # Add service identification
        forward_headers["X-Forwarded-By"] = "insurance-ai-gateway"
        forward_headers["X-Request-ID"] = headers.get("X-Request-ID", "")
        
        try:
            response = await self.http_client.request(
                method=method,
                url=url,
                headers=forward_headers,
                content=body,
                timeout=30.0
            )
            return response
            
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"Service {service_name} timeout"
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Service {service_name} unavailable"
            )

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Extract current user from JWT token"""
    if not credentials:
        return None
    
    payload = AuthManager.decode_token(credentials.credentials)
    if not payload:
        return None
    
    return AuthManager.extract_user_info(payload)

async def require_authentication(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require valid authentication"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user

async def check_rate_limit(request: Request, user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Check rate limiting"""
    if not redis_client:
        return  # Skip if Redis not available
    
    # Use user ID if authenticated, otherwise IP address
    identifier = user.get("user_id") if user else request.client.host
    key = f"rate_limit:{identifier}:{request.url.path}"
    
    rate_limiter = RateLimiter(redis_client)
    allowed, info = await rate_limiter.is_allowed(
        key,
        config.RATE_LIMIT_REQUESTS,
        config.RATE_LIMIT_WINDOW
    )
    
    if not allowed:
        RATE_LIMIT_HITS.labels(
            user_id=identifier,
            endpoint=request.url.path
        ).inc()
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset_time"]),
                "Retry-After": str(info["retry_after"])
            }
        )
    
    # Add rate limit headers to response
    request.state.rate_limit_info = info

# Health check functions
async def check_service_health():
    """Periodically check backend service health"""
    while True:
        try:
            for service_name, service_url in config.BACKEND_SERVICES.items():
                try:
                    async with httpx.AsyncClient(timeout=config.HEALTH_CHECK_TIMEOUT) as client:
                        response = await client.get(f"{service_url}/health")
                        healthy = response.status_code == 200
                        service_health[service_name] = healthy
                        BACKEND_HEALTH.labels(service=service_name).set(1 if healthy else 0)
                        
                except Exception:
                    service_health[service_name] = False
                    BACKEND_HEALTH.labels(service=service_name).set(0)
            
            await asyncio.sleep(config.HEALTH_CHECK_INTERVAL)
            
        except Exception as e:
            logger.error("Health check error", error=str(e))
            await asyncio.sleep(config.HEALTH_CHECK_INTERVAL)

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    global redis_client, http_client
    
    # Startup
    logger.info("Starting API Gateway")
    
    # Initialize Redis
    try:
        redis_client = redis.from_url(config.REDIS_URL)
        await redis_client.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.error("Redis connection failed", error=str(e))
        redis_client = None
    
    # Initialize HTTP client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    
    # Start health check task
    health_task = asyncio.create_task(check_service_health())
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway")
    
    # Cancel health check
    health_task.cancel()
    
    # Close connections
    if redis_client:
        await redis_client.close()
    
    if http_client:
        await http_client.aclose()

# Create FastAPI application
app = FastAPI(
    title="Insurance AI Agent System - API Gateway",
    description="Central API Gateway for Insurance AI Agent System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=config.CORS_METHODS,
    allow_headers=config.CORS_HEADERS,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

if not config.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )

# Request/Response middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Request processing middleware"""
    start_time = time.time()
    
    # Generate request ID
    request_id = hashlib.md5(
        f"{time.time()}{request.client.host}{request.url}".encode()
    ).hexdigest()[:16]
    
    request.state.request_id = request_id
    
    # Add request ID to headers
    request.headers.__dict__["_list"].append(
        (b"x-request-id", request_id.encode())
    )
    
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Gateway-Version"] = "1.0.0"
        
        # Add rate limit headers if available
        if hasattr(request.state, "rate_limit_info"):
            info = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset_time"])
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        logger.info(
            "Request processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            request_id=request_id
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            request_id=request_id
        )
        raise
    finally:
        ACTIVE_CONNECTIONS.dec()

# Health check endpoints
@app.get("/health")
async def health_check():
    """Gateway health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": service_health
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service status"""
    redis_healthy = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_healthy = True
        except Exception:
            pass
    
    return {
        "status": "healthy" if redis_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "redis": {
                "status": "healthy" if redis_healthy else "unhealthy",
                "url": config.REDIS_URL
            },
            "services": {
                name: {
                    "status": "healthy" if healthy else "unhealthy",
                    "url": config.BACKEND_SERVICES[name]
                }
                for name, healthy in service_health.items()
            }
        }
    }

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Authentication endpoints (direct implementation)
@app.post("/auth/login")
async def login(request: Request):
    """Forward login request to auth service"""
    return await forward_to_service(request, "auth")

@app.post("/auth/refresh")
async def refresh_token(request: Request):
    """Forward token refresh to auth service"""
    return await forward_to_service(request, "auth")

@app.post("/auth/logout")
async def logout(request: Request):
    """Forward logout request to auth service"""
    return await forward_to_service(request, "auth")

# Generic request forwarding
async def forward_to_service(request: Request, service_name: str = None):
    """Forward request to appropriate backend service"""
    
    # Determine service if not specified
    if not service_name:
        service_name = ServiceRouter(http_client).get_service_for_path(request.url.path)
    
    if not service_name:
        raise HTTPException(
            status_code=404,
            detail="Service not found for this endpoint"
        )
    
    # Check service health
    if service_name in service_health and not service_health[service_name]:
        raise HTTPException(
            status_code=503,
            detail=f"Service {service_name} is currently unavailable"
        )
    
    # Read request body
    body = await request.body()
    
    # Forward request
    router = ServiceRouter(http_client)
    response = await router.forward_request(
        service_name=service_name,
        method=request.method,
        path=request.url.path,
        headers=dict(request.headers),
        query_params=str(request.url.query),
        body=body if body else None
    )
    
    # Return response
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("content-type")
    )

# Catch-all route for service forwarding
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_request(request: Request, _: str = Depends(check_rate_limit)):
    """Proxy all requests to appropriate backend services"""
    return await forward_to_service(request)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        request_id=getattr(request.state, "request_id", None)
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        workers=1 if config.DEBUG else 4,
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )

