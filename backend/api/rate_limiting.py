"""
Insurance AI Agent System - Rate Limiting and API Versioning
Production-ready rate limiting and API versioning utilities
"""

import asyncio
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from decimal import Decimal
import re
from functools import wraps
from enum import Enum

# FastAPI and dependencies
from fastapi import HTTPException, Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis.asyncio as redis

# Database and utilities
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
import structlog
from pydantic import BaseModel, Field, validator

# Internal imports
from backend.shared.models import User, Organization, RateLimitRule, APIUsage
from backend.shared.schemas import RateLimitConfig, APIVersionInfo
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

logger = structlog.get_logger(__name__)

class RateLimitType(Enum):
    """Rate limit types"""
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    PER_MONTH = "per_month"
    BURST = "burst"
    SLIDING_WINDOW = "sliding_window"

class APIVersion(Enum):
    """Supported API versions"""
    V1 = "v1"
    V2 = "v2"
    BETA = "beta"

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

class AdvancedRateLimiter:
    """
    Advanced rate limiter with multiple strategies and Redis backend
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = structlog.get_logger("rate_limiter")
        
        # Default rate limit configurations
        self.default_limits = {
            "anonymous": {
                RateLimitType.PER_MINUTE: 10,
                RateLimitType.PER_HOUR: 100,
                RateLimitType.PER_DAY: 1000
            },
            "authenticated": {
                RateLimitType.PER_MINUTE: 60,
                RateLimitType.PER_HOUR: 1000,
                RateLimitType.PER_DAY: 10000
            },
            "premium": {
                RateLimitType.PER_MINUTE: 200,
                RateLimitType.PER_HOUR: 5000,
                RateLimitType.PER_DAY: 50000
            },
            "admin": {
                RateLimitType.PER_MINUTE: 1000,
                RateLimitType.PER_HOUR: 10000,
                RateLimitType.PER_DAY: 100000
            }
        }
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/v1/auth/login": {
                RateLimitType.PER_MINUTE: 5,
                RateLimitType.PER_HOUR: 20
            },
            "/api/v1/auth/register": {
                RateLimitType.PER_MINUTE: 3,
                RateLimitType.PER_HOUR: 10
            },
            "/api/v1/claims": {
                RateLimitType.PER_MINUTE: 10,
                RateLimitType.PER_HOUR: 100
            },
            "/api/v1/agents/": {
                RateLimitType.PER_MINUTE: 20,
                RateLimitType.PER_HOUR: 200
            }
        }
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        user_tier: str = "anonymous",
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limits
        
        Args:
            identifier: Unique identifier (IP, user ID, API key)
            endpoint: API endpoint being accessed
            user_tier: User tier for rate limiting
            strategy: Rate limiting strategy to use
            
        Returns:
            Dictionary with rate limit status and metadata
        """
        
        try:
            current_time = time.time()
            
            # Get applicable limits
            limits = self._get_applicable_limits(endpoint, user_tier)
            
            # Check each limit type
            limit_results = {}
            overall_allowed = True
            
            for limit_type, limit_value in limits.items():
                if strategy == RateLimitStrategy.SLIDING_WINDOW:
                    result = await self._check_sliding_window_limit(
                        identifier, endpoint, limit_type, limit_value, current_time
                    )
                elif strategy == RateLimitStrategy.TOKEN_BUCKET:
                    result = await self._check_token_bucket_limit(
                        identifier, endpoint, limit_type, limit_value, current_time
                    )
                else:  # Fixed window
                    result = await self._check_fixed_window_limit(
                        identifier, endpoint, limit_type, limit_value, current_time
                    )
                
                limit_results[limit_type.value] = result
                
                if not result["allowed"]:
                    overall_allowed = False
            
            # Record API usage
            await self._record_api_usage(identifier, endpoint, user_tier, overall_allowed)
            
            # Prepare response
            rate_limit_info = {
                "allowed": overall_allowed,
                "identifier": identifier,
                "endpoint": endpoint,
                "user_tier": user_tier,
                "strategy": strategy.value,
                "limits": limit_results,
                "timestamp": current_time,
                "reset_times": self._calculate_reset_times(current_time)
            }
            
            if not overall_allowed:
                # Find the most restrictive limit for retry-after
                retry_after = min(
                    result.get("retry_after", 60)
                    for result in limit_results.values()
                    if not result["allowed"]
                )
                rate_limit_info["retry_after"] = retry_after
            
            return rate_limit_info
            
        except Exception as e:
            self.logger.error("Rate limit check failed", error=str(e))
            # Fail open - allow request if rate limiting fails
            return {
                "allowed": True,
                "error": str(e),
                "fallback": True
            }
    
    def _get_applicable_limits(self, endpoint: str, user_tier: str) -> Dict[RateLimitType, int]:
        """Get applicable rate limits for endpoint and user tier"""
        
        # Start with user tier defaults
        limits = self.default_limits.get(user_tier, self.default_limits["anonymous"]).copy()
        
        # Apply endpoint-specific limits
        for endpoint_pattern, endpoint_limits in self.endpoint_limits.items():
            if endpoint.startswith(endpoint_pattern):
                # Use more restrictive limits
                for limit_type, limit_value in endpoint_limits.items():
                    if limit_type in limits:
                        limits[limit_type] = min(limits[limit_type], limit_value)
                    else:
                        limits[limit_type] = limit_value
                break
        
        return limits
    
    async def _check_sliding_window_limit(
        self,
        identifier: str,
        endpoint: str,
        limit_type: RateLimitType,
        limit_value: int,
        current_time: float
    ) -> Dict[str, Any]:
        """Check sliding window rate limit"""
        
        try:
            window_size = self._get_window_size(limit_type)
            key = f"rate_limit:sliding:{identifier}:{endpoint}:{limit_type.value}"
            
            # Remove old entries
            cutoff_time = current_time - window_size
            await self.redis_client.zremrangebyscore(key, 0, cutoff_time)
            
            # Count current requests
            current_count = await self.redis_client.zcard(key)
            
            if current_count >= limit_value:
                # Get oldest request time for retry-after calculation
                oldest_requests = await self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest_requests:
                    oldest_time = oldest_requests[0][1]
                    retry_after = int(oldest_time + window_size - current_time + 1)
                else:
                    retry_after = int(window_size)
                
                return {
                    "allowed": False,
                    "current_count": current_count,
                    "limit": limit_value,
                    "window_size": window_size,
                    "retry_after": retry_after
                }
            
            # Add current request
            await self.redis_client.zadd(key, {str(uuid.uuid4()): current_time})
            await self.redis_client.expire(key, int(window_size) + 1)
            
            return {
                "allowed": True,
                "current_count": current_count + 1,
                "limit": limit_value,
                "window_size": window_size,
                "remaining": limit_value - current_count - 1
            }
            
        except Exception as e:
            self.logger.error("Sliding window rate limit check failed", error=str(e))
            return {"allowed": True, "error": str(e)}
    
    async def _check_token_bucket_limit(
        self,
        identifier: str,
        endpoint: str,
        limit_type: RateLimitType,
        limit_value: int,
        current_time: float
    ) -> Dict[str, Any]:
        """Check token bucket rate limit"""
        
        try:
            key = f"rate_limit:bucket:{identifier}:{endpoint}:{limit_type.value}"
            refill_rate = self._get_refill_rate(limit_type)
            bucket_size = limit_value
            
            # Get current bucket state
            bucket_data = await self.redis_client.hmget(key, "tokens", "last_refill")
            
            if bucket_data[0] is None:
                # Initialize bucket
                tokens = bucket_size - 1  # Consume one token for current request
                last_refill = current_time
            else:
                tokens = float(bucket_data[0])
                last_refill = float(bucket_data[1])
                
                # Calculate tokens to add
                time_passed = current_time - last_refill
                tokens_to_add = time_passed * refill_rate
                tokens = min(bucket_size, tokens + tokens_to_add)
                
                # Check if request can be served
                if tokens < 1:
                    retry_after = int((1 - tokens) / refill_rate + 1)
                    return {
                        "allowed": False,
                        "tokens": tokens,
                        "bucket_size": bucket_size,
                        "refill_rate": refill_rate,
                        "retry_after": retry_after
                    }
                
                # Consume token
                tokens -= 1
            
            # Update bucket state
            await self.redis_client.hmset(key, {
                "tokens": tokens,
                "last_refill": current_time
            })
            await self.redis_client.expire(key, int(self._get_window_size(limit_type)) + 1)
            
            return {
                "allowed": True,
                "tokens": tokens,
                "bucket_size": bucket_size,
                "refill_rate": refill_rate
            }
            
        except Exception as e:
            self.logger.error("Token bucket rate limit check failed", error=str(e))
            return {"allowed": True, "error": str(e)}
    
    async def _check_fixed_window_limit(
        self,
        identifier: str,
        endpoint: str,
        limit_type: RateLimitType,
        limit_value: int,
        current_time: float
    ) -> Dict[str, Any]:
        """Check fixed window rate limit"""
        
        try:
            window_size = self._get_window_size(limit_type)
            window_start = int(current_time // window_size) * window_size
            key = f"rate_limit:fixed:{identifier}:{endpoint}:{limit_type.value}:{window_start}"
            
            # Get current count
            current_count = await self.redis_client.get(key)
            current_count = int(current_count) if current_count else 0
            
            if current_count >= limit_value:
                retry_after = int(window_start + window_size - current_time + 1)
                return {
                    "allowed": False,
                    "current_count": current_count,
                    "limit": limit_value,
                    "window_start": window_start,
                    "window_size": window_size,
                    "retry_after": retry_after
                }
            
            # Increment counter
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, int(window_size) + 1)
            
            return {
                "allowed": True,
                "current_count": current_count + 1,
                "limit": limit_value,
                "window_start": window_start,
                "window_size": window_size,
                "remaining": limit_value - current_count - 1
            }
            
        except Exception as e:
            self.logger.error("Fixed window rate limit check failed", error=str(e))
            return {"allowed": True, "error": str(e)}
    
    def _get_window_size(self, limit_type: RateLimitType) -> float:
        """Get window size in seconds for limit type"""
        
        window_sizes = {
            RateLimitType.PER_MINUTE: 60,
            RateLimitType.PER_HOUR: 3600,
            RateLimitType.PER_DAY: 86400,
            RateLimitType.PER_MONTH: 2592000  # 30 days
        }
        
        return window_sizes.get(limit_type, 60)
    
    def _get_refill_rate(self, limit_type: RateLimitType) -> float:
        """Get token refill rate for limit type"""
        
        refill_rates = {
            RateLimitType.PER_MINUTE: 1.0 / 60,  # 1 token per second
            RateLimitType.PER_HOUR: 1.0 / 3600,  # 1 token per hour
            RateLimitType.PER_DAY: 1.0 / 86400,  # 1 token per day
            RateLimitType.PER_MONTH: 1.0 / 2592000  # 1 token per month
        }
        
        return refill_rates.get(limit_type, 1.0 / 60)
    
    def _calculate_reset_times(self, current_time: float) -> Dict[str, int]:
        """Calculate reset times for different window types"""
        
        return {
            "minute": int((int(current_time // 60) + 1) * 60),
            "hour": int((int(current_time // 3600) + 1) * 3600),
            "day": int((int(current_time // 86400) + 1) * 86400)
        }
    
    async def _record_api_usage(
        self,
        identifier: str,
        endpoint: str,
        user_tier: str,
        allowed: bool
    ):
        """Record API usage for analytics"""
        
        try:
            usage_key = f"api_usage:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
            usage_data = {
                "identifier": identifier,
                "endpoint": endpoint,
                "user_tier": user_tier,
                "allowed": allowed,
                "timestamp": time.time()
            }
            
            await self.redis_client.lpush(usage_key, json.dumps(usage_data))
            await self.redis_client.expire(usage_key, 86400 * 7)  # Keep for 7 days
            
        except Exception as e:
            self.logger.warning("Failed to record API usage", error=str(e))

class APIVersionManager:
    """
    API version management with backward compatibility
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("api_version_manager")
        
        # Version configurations
        self.version_configs = {
            APIVersion.V1: {
                "supported": True,
                "deprecated": False,
                "sunset_date": None,
                "features": [
                    "basic_auth",
                    "user_management",
                    "policy_management",
                    "claim_management",
                    "basic_orchestration"
                ],
                "breaking_changes": [],
                "migration_guide": None
            },
            APIVersion.V2: {
                "supported": True,
                "deprecated": False,
                "sunset_date": None,
                "features": [
                    "oauth2_auth",
                    "advanced_user_management",
                    "enhanced_policy_management",
                    "advanced_claim_management",
                    "full_orchestration",
                    "real_time_updates",
                    "advanced_analytics"
                ],
                "breaking_changes": [
                    "Authentication now requires OAuth2",
                    "Response format changed for user endpoints",
                    "New required fields in policy creation"
                ],
                "migration_guide": "/docs/migration/v1-to-v2"
            },
            APIVersion.BETA: {
                "supported": True,
                "deprecated": False,
                "sunset_date": None,
                "features": [
                    "experimental_ai_features",
                    "advanced_ml_models",
                    "predictive_analytics",
                    "automated_decision_making"
                ],
                "breaking_changes": [
                    "Beta features may change without notice",
                    "No backward compatibility guarantees"
                ],
                "migration_guide": "/docs/beta-features"
            }
        }
        
        # Default version
        self.default_version = APIVersion.V1
        
        # Version detection patterns
        self.version_patterns = {
            r"/api/v1/": APIVersion.V1,
            r"/api/v2/": APIVersion.V2,
            r"/api/beta/": APIVersion.BETA
        }
    
    def detect_api_version(self, request: Request) -> APIVersion:
        """Detect API version from request"""
        
        try:
            # Check URL path
            path = request.url.path
            for pattern, version in self.version_patterns.items():
                if re.match(pattern, path):
                    return version
            
            # Check Accept header
            accept_header = request.headers.get("Accept", "")
            if "application/vnd.insurance-ai.v2+json" in accept_header:
                return APIVersion.V2
            elif "application/vnd.insurance-ai.beta+json" in accept_header:
                return APIVersion.BETA
            elif "application/vnd.insurance-ai.v1+json" in accept_header:
                return APIVersion.V1
            
            # Check custom header
            version_header = request.headers.get("API-Version", "")
            if version_header:
                try:
                    return APIVersion(version_header.lower())
                except ValueError:
                    pass
            
            # Default version
            return self.default_version
            
        except Exception as e:
            self.logger.warning("API version detection failed", error=str(e))
            return self.default_version
    
    def validate_version_support(self, version: APIVersion) -> Dict[str, Any]:
        """Validate if API version is supported"""
        
        config = self.version_configs.get(version)
        if not config:
            return {
                "supported": False,
                "error": f"Unknown API version: {version.value}"
            }
        
        if not config["supported"]:
            return {
                "supported": False,
                "error": f"API version {version.value} is no longer supported"
            }
        
        result = {
            "supported": True,
            "version": version.value,
            "deprecated": config["deprecated"],
            "features": config["features"]
        }
        
        if config["deprecated"]:
            result["deprecation_notice"] = f"API version {version.value} is deprecated"
            if config["sunset_date"]:
                result["sunset_date"] = config["sunset_date"]
            if config["migration_guide"]:
                result["migration_guide"] = config["migration_guide"]
        
        return result
    
    def get_version_info(self, version: APIVersion) -> Dict[str, Any]:
        """Get detailed version information"""
        
        config = self.version_configs.get(version, {})
        
        return {
            "version": version.value,
            "supported": config.get("supported", False),
            "deprecated": config.get("deprecated", False),
            "sunset_date": config.get("sunset_date"),
            "features": config.get("features", []),
            "breaking_changes": config.get("breaking_changes", []),
            "migration_guide": config.get("migration_guide")
        }
    
    def get_all_versions(self) -> List[Dict[str, Any]]:
        """Get information about all API versions"""
        
        return [
            self.get_version_info(version)
            for version in APIVersion
        ]
    
    def add_version_headers(self, response: Response, version: APIVersion):
        """Add version-related headers to response"""
        
        config = self.version_configs.get(version, {})
        
        response.headers["API-Version"] = version.value
        response.headers["API-Supported-Versions"] = ",".join([v.value for v in APIVersion])
        
        if config.get("deprecated"):
            response.headers["API-Deprecated"] = "true"
            if config.get("sunset_date"):
                response.headers["API-Sunset"] = config["sunset_date"]
            if config.get("migration_guide"):
                response.headers["API-Migration-Guide"] = config["migration_guide"]

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI
    """
    
    def __init__(self, app, rate_limiter: AdvancedRateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.logger = structlog.get_logger("rate_limit_middleware")
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        
        try:
            # Skip rate limiting for certain paths
            skip_paths = ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
            if any(request.url.path.startswith(path) for path in skip_paths):
                return await call_next(request)
            
            # Get identifier (IP address or user ID)
            identifier = self._get_identifier(request)
            
            # Get user tier
            user_tier = self._get_user_tier(request)
            
            # Check rate limit
            rate_limit_result = await self.rate_limiter.check_rate_limit(
                identifier=identifier,
                endpoint=request.url.path,
                user_tier=user_tier
            )
            
            if not rate_limit_result["allowed"]:
                # Rate limit exceeded
                retry_after = rate_limit_result.get("retry_after", 60)
                
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "Too many requests",
                        "retry_after": retry_after,
                        "limits": rate_limit_result.get("limits", {}),
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(rate_limit_result.get("limit", "unknown")),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + retry_after)
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            if "limits" in rate_limit_result:
                for limit_type, limit_info in rate_limit_result["limits"].items():
                    if limit_info.get("allowed"):
                        response.headers[f"X-RateLimit-{limit_type.title()}-Limit"] = str(limit_info.get("limit", "unknown"))
                        response.headers[f"X-RateLimit-{limit_type.title()}-Remaining"] = str(limit_info.get("remaining", "unknown"))
            
            return response
            
        except Exception as e:
            self.logger.error("Rate limiting middleware error", error=str(e))
            # Fail open - allow request if middleware fails
            return await call_next(request)
    
    def _get_identifier(self, request: Request) -> str:
        """Get unique identifier for rate limiting"""
        
        # Try to get user ID from request state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def _get_user_tier(self, request: Request) -> str:
        """Get user tier for rate limiting"""
        
        # Check if user is authenticated
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            return "anonymous"
        
        # Check for admin or premium users (simplified)
        user_email = getattr(request.state, "user_email", "")
        if user_email.endswith("@admin.insurance-ai.com"):
            return "admin"
        elif "premium" in user_email:
            return "premium"
        else:
            return "authenticated"

class APIVersionMiddleware(BaseHTTPMiddleware):
    """
    API versioning middleware for FastAPI
    """
    
    def __init__(self, app, version_manager: APIVersionManager):
        super().__init__(app)
        self.version_manager = version_manager
        self.logger = structlog.get_logger("api_version_middleware")
    
    async def dispatch(self, request: Request, call_next):
        """Process request with API versioning"""
        
        try:
            # Detect API version
            api_version = self.version_manager.detect_api_version(request)
            
            # Validate version support
            version_validation = self.version_manager.validate_version_support(api_version)
            
            if not version_validation["supported"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Unsupported API version",
                        "message": version_validation["error"],
                        "supported_versions": [v.value for v in APIVersion],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Add version to request state
            request.state.api_version = api_version
            
            # Process request
            response = await call_next(request)
            
            # Add version headers
            self.version_manager.add_version_headers(response, api_version)
            
            return response
            
        except Exception as e:
            self.logger.error("API versioning middleware error", error=str(e))
            return await call_next(request)

# Factory functions
async def create_rate_limiter(redis_client: redis.Redis) -> AdvancedRateLimiter:
    """Create rate limiter instance"""
    return AdvancedRateLimiter(redis_client)

def create_version_manager() -> APIVersionManager:
    """Create API version manager instance"""
    return APIVersionManager()

def create_rate_limit_middleware(app, rate_limiter: AdvancedRateLimiter) -> RateLimitMiddleware:
    """Create rate limiting middleware"""
    return RateLimitMiddleware(app, rate_limiter)

def create_version_middleware(app, version_manager: APIVersionManager) -> APIVersionMiddleware:
    """Create API versioning middleware"""
    return APIVersionMiddleware(app, version_manager)

