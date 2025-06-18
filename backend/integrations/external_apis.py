"""
Insurance AI Agent System - External API Integration
Production-ready external API integration system
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import hashlib
import hmac
import base64
from urllib.parse import urlencode, urlparse
import ssl
import certifi

# HTTP client
import aiohttp
import httpx
from aiohttp import ClientSession, ClientTimeout, ClientError
from aiohttp.client_exceptions import ClientConnectorError, ClientResponseError

# Database and Redis
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text

# Internal imports
from backend.shared.models import ExternalAPI, APICredential, APICall, APIResponse
from backend.shared.schemas import APIRequest, APIConfig, IntegrationStatus
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

import structlog
logger = structlog.get_logger(__name__)

class APIProvider(Enum):
    """Supported API providers"""
    CREDIT_BUREAU = "credit_bureau"
    FRAUD_DETECTION = "fraud_detection"
    IDENTITY_VERIFICATION = "identity_verification"
    DOCUMENT_VERIFICATION = "document_verification"
    PAYMENT_GATEWAY = "payment_gateway"
    EMAIL_SERVICE = "email_service"
    SMS_SERVICE = "sms_service"
    WEATHER_SERVICE = "weather_service"
    GEOLOCATION = "geolocation"
    VEHICLE_DATA = "vehicle_data"
    PROPERTY_DATA = "property_data"
    MEDICAL_RECORDS = "medical_records"
    REGULATORY_DATA = "regulatory_data"
    MARKET_DATA = "market_data"
    SOCIAL_MEDIA = "social_media"
    NEWS_FEEDS = "news_feeds"
    ANALYTICS = "analytics"
    MACHINE_LEARNING = "machine_learning"
    BLOCKCHAIN = "blockchain"
    IOT_DEVICES = "iot_devices"

class AuthType(Enum):
    """API authentication types"""
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    HMAC = "hmac"
    CUSTOM = "custom"

class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    name: str
    url: str
    method: HTTPMethod
    auth_required: bool = True
    rate_limit: Optional[int] = None  # requests per minute
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1
    cache_ttl: Optional[int] = None  # seconds
    required_params: List[str] = None
    optional_params: List[str] = None
    response_format: str = "json"
    
    def __post_init__(self):
        if self.required_params is None:
            self.required_params = []
        if self.optional_params is None:
            self.optional_params = []

@dataclass
class APICredentials:
    """API credentials"""
    provider: APIProvider
    auth_type: AuthType
    credentials: Dict[str, str]
    environment: str = "production"  # production, staging, sandbox
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = structlog.get_logger("rate_limiter")
    
    async def check_rate_limit(self, provider: APIProvider, endpoint: str, limit: int) -> bool:
        """
        Check if API call is within rate limits
        
        Args:
            provider: API provider
            endpoint: Endpoint name
            limit: Rate limit (requests per minute)
            
        Returns:
            True if within limits
        """
        
        try:
            key = f"rate_limit:{provider.value}:{endpoint}"
            current_time = int(time.time())
            window_start = current_time - (current_time % 60)  # 1-minute window
            
            # Count requests in current window
            count = await self.redis_client.get(f"{key}:{window_start}")
            current_count = int(count) if count else 0
            
            if current_count >= limit:
                return False
            
            # Increment counter
            await self.redis_client.incr(f"{key}:{window_start}")
            await self.redis_client.expire(f"{key}:{window_start}", 120)  # Keep for 2 minutes
            
            return True
            
        except Exception as e:
            self.logger.error("Rate limit check failed", error=str(e))
            return True  # Fail open
    
    async def get_rate_limit_status(self, provider: APIProvider, endpoint: str, limit: int) -> Dict[str, Any]:
        """Get current rate limit status"""
        try:
            key = f"rate_limit:{provider.value}:{endpoint}"
            current_time = int(time.time())
            window_start = current_time - (current_time % 60)
            
            count = await self.redis_client.get(f"{key}:{window_start}")
            current_count = int(count) if count else 0
            
            return {
                "limit": limit,
                "used": current_count,
                "remaining": max(0, limit - current_count),
                "reset_at": window_start + 60
            }
            
        except Exception as e:
            self.logger.error("Failed to get rate limit status", error=str(e))
            return {"limit": limit, "used": 0, "remaining": limit, "reset_at": 0}

class APICache:
    """Cache for API responses"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = structlog.get_logger("api_cache")
    
    def _generate_cache_key(self, provider: APIProvider, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        params_str = urlencode(sorted_params)
        
        # Create hash of parameters
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"api_cache:{provider.value}:{endpoint}:{params_hash}"
    
    async def get(self, provider: APIProvider, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        try:
            cache_key = self._generate_cache_key(provider, endpoint, params)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            self.logger.error("Cache get failed", error=str(e))
            return None
    
    async def set(
        self,
        provider: APIProvider,
        endpoint: str,
        params: Dict[str, Any],
        response: Dict[str, Any],
        ttl: int
    ):
        """Cache response"""
        try:
            cache_key = self._generate_cache_key(provider, endpoint, params)
            cached_data = json.dumps(response)
            
            await self.redis_client.setex(cache_key, ttl, cached_data)
            
        except Exception as e:
            self.logger.error("Cache set failed", error=str(e))
    
    async def invalidate(self, provider: APIProvider, endpoint: str, params: Dict[str, Any]):
        """Invalidate cached response"""
        try:
            cache_key = self._generate_cache_key(provider, endpoint, params)
            await self.redis_client.delete(cache_key)
            
        except Exception as e:
            self.logger.error("Cache invalidation failed", error=str(e))

class ExternalAPIManager:
    """
    Comprehensive external API integration manager
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = structlog.get_logger("external_api_manager")
        
        # Components
        self.rate_limiter = RateLimiter(redis_client)
        self.cache = APICache(redis_client)
        
        # API configurations
        self.api_configs: Dict[APIProvider, Dict[str, APIEndpoint]] = {}
        self.credentials: Dict[APIProvider, APICredentials] = {}
        
        # HTTP session
        self.session: Optional[ClientSession] = None
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "rate_limited_requests": 0,
            "average_response_time": 0.0
        }
        
        # Initialize API configurations
        self._initialize_api_configs()
    
    async def start(self):
        """Start the API manager"""
        try:
            self.logger.info("Starting external API manager")
            
            # Create HTTP session with SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context, limit=100, limit_per_host=20)
            timeout = ClientTimeout(total=60, connect=10)
            
            self.session = ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Insurance-AI-System/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            self.logger.info("External API manager started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start external API manager", error=str(e))
            raise
    
    async def stop(self):
        """Stop the API manager"""
        try:
            self.logger.info("Stopping external API manager")
            
            if self.session:
                await self.session.close()
            
            self.logger.info("External API manager stopped")
            
        except Exception as e:
            self.logger.error("Error stopping external API manager", error=str(e))
    
    def register_api_config(self, provider: APIProvider, endpoints: Dict[str, APIEndpoint]):
        """Register API configuration for a provider"""
        self.api_configs[provider] = endpoints
        self.logger.info("API configuration registered", provider=provider.value, endpoints=len(endpoints))
    
    def register_credentials(self, credentials: APICredentials):
        """Register API credentials"""
        self.credentials[credentials.provider] = credentials
        self.logger.info("API credentials registered", provider=credentials.provider.value)
    
    async def call_api(
        self,
        provider: APIProvider,
        endpoint_name: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
        bypass_rate_limit: bool = False
    ) -> Dict[str, Any]:
        """
        Make an API call
        
        Args:
            provider: API provider
            endpoint_name: Endpoint name
            params: Query parameters
            data: Request body data
            headers: Additional headers
            use_cache: Whether to use cached responses
            bypass_rate_limit: Whether to bypass rate limiting
            
        Returns:
            API response
        """
        
        try:
            # Get endpoint configuration
            if provider not in self.api_configs or endpoint_name not in self.api_configs[provider]:
                raise ServiceException(f"Unknown endpoint: {provider.value}.{endpoint_name}")
            
            endpoint = self.api_configs[provider][endpoint_name]
            params = params or {}
            data = data or {}
            headers = headers or {}
            
            # Validate required parameters
            missing_params = [p for p in endpoint.required_params if p not in params]
            if missing_params:
                raise ServiceException(f"Missing required parameters: {missing_params}")
            
            # Check cache first
            if use_cache and endpoint.cache_ttl and endpoint.method == HTTPMethod.GET:
                cached_response = await self.cache.get(provider, endpoint_name, params)
                if cached_response:
                    self.stats["cached_responses"] += 1
                    return cached_response
            
            # Check rate limits
            if not bypass_rate_limit and endpoint.rate_limit:
                if not await self.rate_limiter.check_rate_limit(provider, endpoint_name, endpoint.rate_limit):
                    self.stats["rate_limited_requests"] += 1
                    raise ServiceException("Rate limit exceeded")
            
            # Prepare request
            url = endpoint.url
            request_headers = await self._prepare_headers(provider, endpoint, headers)
            
            # Make request with retries
            response_data = None
            last_error = None
            
            for attempt in range(endpoint.retry_count + 1):
                try:
                    start_time = time.time()
                    
                    async with self.session.request(
                        endpoint.method.value,
                        url,
                        params=params if endpoint.method == HTTPMethod.GET else None,
                        json=data if endpoint.method != HTTPMethod.GET and data else None,
                        headers=request_headers,
                        timeout=ClientTimeout(total=endpoint.timeout)
                    ) as response:
                        
                        response_time = time.time() - start_time
                        
                        # Update response time statistics
                        self._update_response_time_stats(response_time)
                        
                        # Check response status
                        if response.status >= 400:
                            error_text = await response.text()
                            raise ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=error_text
                            )
                        
                        # Parse response
                        if endpoint.response_format == "json":
                            response_data = await response.json()
                        else:
                            response_data = {"content": await response.text()}
                        
                        # Add metadata
                        response_data["_metadata"] = {
                            "provider": provider.value,
                            "endpoint": endpoint_name,
                            "status_code": response.status,
                            "response_time": response_time,
                            "timestamp": datetime.utcnow().isoformat(),
                            "cached": False
                        }
                        
                        break
                        
                except (ClientError, asyncio.TimeoutError) as e:
                    last_error = e
                    if attempt < endpoint.retry_count:
                        await asyncio.sleep(endpoint.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        raise ServiceException(f"API call failed after {endpoint.retry_count + 1} attempts: {str(e)}")
            
            # Cache successful response
            if response_data and use_cache and endpoint.cache_ttl and endpoint.method == HTTPMethod.GET:
                await self.cache.set(provider, endpoint_name, params, response_data, endpoint.cache_ttl)
            
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            
            # Log API call
            await self._log_api_call(provider, endpoint_name, params, response_data, True)
            
            return response_data
            
        except Exception as e:
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            
            # Log failed API call
            await self._log_api_call(provider, endpoint_name, params, None, False, str(e))
            
            self.logger.error(
                "API call failed",
                provider=provider.value,
                endpoint=endpoint_name,
                error=str(e)
            )
            
            raise ServiceException(f"API call failed: {str(e)}")
    
    async def batch_call_api(
        self,
        calls: List[Tuple[APIProvider, str, Dict[str, Any]]],
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Make multiple API calls concurrently
        
        Args:
            calls: List of (provider, endpoint_name, params) tuples
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of responses in the same order as calls
        """
        
        try:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def make_call(provider, endpoint_name, params):
                async with semaphore:
                    try:
                        return await self.call_api(provider, endpoint_name, params)
                    except Exception as e:
                        return {"error": str(e), "provider": provider.value, "endpoint": endpoint_name}
            
            tasks = [make_call(provider, endpoint, params) for provider, endpoint, params in calls]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error dictionaries
            results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    provider, endpoint, _ = calls[i]
                    results.append({
                        "error": str(response),
                        "provider": provider.value,
                        "endpoint": endpoint
                    })
                else:
                    results.append(response)
            
            return results
            
        except Exception as e:
            self.logger.error("Batch API call failed", error=str(e))
            raise ServiceException(f"Batch API call failed: {str(e)}")
    
    async def get_api_status(self, provider: APIProvider) -> Dict[str, Any]:
        """Get API status and health information"""
        try:
            if provider not in self.api_configs:
                return {"status": "unknown", "error": "Provider not configured"}
            
            # Try to make a health check call if available
            if "health" in self.api_configs[provider]:
                try:
                    response = await self.call_api(provider, "health", use_cache=False)
                    return {"status": "healthy", "response": response}
                except Exception as e:
                    return {"status": "unhealthy", "error": str(e)}
            
            # Check credentials expiration
            if provider in self.credentials:
                creds = self.credentials[provider]
                if creds.expires_at and datetime.utcnow() > creds.expires_at:
                    return {"status": "credentials_expired"}
            
            return {"status": "configured"}
            
        except Exception as e:
            self.logger.error("Failed to get API status", provider=provider.value, error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def get_rate_limit_status(self, provider: APIProvider, endpoint_name: str) -> Dict[str, Any]:
        """Get rate limit status for an endpoint"""
        try:
            if provider not in self.api_configs or endpoint_name not in self.api_configs[provider]:
                return {"error": "Unknown endpoint"}
            
            endpoint = self.api_configs[provider][endpoint_name]
            if not endpoint.rate_limit:
                return {"rate_limit": None}
            
            return await self.rate_limiter.get_rate_limit_status(provider, endpoint_name, endpoint.rate_limit)
            
        except Exception as e:
            self.logger.error("Failed to get rate limit status", error=str(e))
            return {"error": str(e)}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return self.stats.copy()
    
    async def _prepare_headers(
        self,
        provider: APIProvider,
        endpoint: APIEndpoint,
        additional_headers: Dict[str, str]
    ) -> Dict[str, str]:
        """Prepare request headers with authentication"""
        
        headers = additional_headers.copy()
        
        if not endpoint.auth_required or provider not in self.credentials:
            return headers
        
        creds = self.credentials[provider]
        
        if creds.auth_type == AuthType.API_KEY:
            # API key authentication
            key_name = creds.credentials.get("key_name", "X-API-Key")
            api_key = creds.credentials.get("api_key")
            if api_key:
                headers[key_name] = api_key
        
        elif creds.auth_type == AuthType.BEARER_TOKEN:
            # Bearer token authentication
            token = creds.credentials.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif creds.auth_type == AuthType.BASIC_AUTH:
            # Basic authentication
            username = creds.credentials.get("username")
            password = creds.credentials.get("password")
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        elif creds.auth_type == AuthType.HMAC:
            # HMAC authentication
            secret = creds.credentials.get("secret")
            if secret:
                timestamp = str(int(time.time()))
                message = f"{endpoint.method.value}{endpoint.url}{timestamp}"
                signature = hmac.new(
                    secret.encode(),
                    message.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                headers["X-Timestamp"] = timestamp
                headers["X-Signature"] = signature
        
        return headers
    
    def _update_response_time_stats(self, response_time: float):
        """Update response time statistics"""
        if self.stats["average_response_time"] == 0:
            self.stats["average_response_time"] = response_time
        else:
            # Simple moving average
            self.stats["average_response_time"] = (
                self.stats["average_response_time"] * 0.9 + response_time * 0.1
            )
    
    async def _log_api_call(
        self,
        provider: APIProvider,
        endpoint_name: str,
        params: Dict[str, Any],
        response: Optional[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None
    ):
        """Log API call for audit purposes"""
        try:
            log_data = {
                "provider": provider.value,
                "endpoint": endpoint_name,
                "params": params,
                "success": success,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if error:
                log_data["error"] = error
            
            if response and "_metadata" in response:
                log_data["response_time"] = response["_metadata"]["response_time"]
                log_data["status_code"] = response["_metadata"]["status_code"]
            
            # Store in Redis for recent history
            log_key = f"api_log:{provider.value}:{endpoint_name}"
            await self.redis_client.lpush(log_key, json.dumps(log_data))
            await self.redis_client.ltrim(log_key, 0, 999)  # Keep last 1000 entries
            await self.redis_client.expire(log_key, 86400)  # Expire after 24 hours
            
        except Exception as e:
            self.logger.warning("Failed to log API call", error=str(e))
    
    def _initialize_api_configs(self):
        """Initialize default API configurations"""
        
        # Credit Bureau API
        self.api_configs[APIProvider.CREDIT_BUREAU] = {
            "credit_score": APIEndpoint(
                name="credit_score",
                url="https://api.creditbureau.com/v1/credit-score",
                method=HTTPMethod.POST,
                rate_limit=100,
                cache_ttl=3600,
                required_params=["ssn", "first_name", "last_name"],
                optional_params=["date_of_birth", "address"]
            ),
            "credit_report": APIEndpoint(
                name="credit_report",
                url="https://api.creditbureau.com/v1/credit-report",
                method=HTTPMethod.POST,
                rate_limit=50,
                cache_ttl=1800,
                required_params=["ssn", "first_name", "last_name", "date_of_birth"]
            )
        }
        
        # Fraud Detection API
        self.api_configs[APIProvider.FRAUD_DETECTION] = {
            "analyze_transaction": APIEndpoint(
                name="analyze_transaction",
                url="https://api.frauddetection.com/v2/analyze",
                method=HTTPMethod.POST,
                rate_limit=200,
                required_params=["transaction_data"],
                timeout=45
            ),
            "risk_score": APIEndpoint(
                name="risk_score",
                url="https://api.frauddetection.com/v2/risk-score",
                method=HTTPMethod.POST,
                rate_limit=500,
                cache_ttl=300,
                required_params=["user_data", "transaction_data"]
            )
        }
        
        # Identity Verification API
        self.api_configs[APIProvider.IDENTITY_VERIFICATION] = {
            "verify_identity": APIEndpoint(
                name="verify_identity",
                url="https://api.idverify.com/v1/verify",
                method=HTTPMethod.POST,
                rate_limit=100,
                required_params=["document_type", "document_data"],
                timeout=60
            ),
            "liveness_check": APIEndpoint(
                name="liveness_check",
                url="https://api.idverify.com/v1/liveness",
                method=HTTPMethod.POST,
                rate_limit=50,
                required_params=["image_data"],
                timeout=30
            )
        }
        
        # Document Verification API
        self.api_configs[APIProvider.DOCUMENT_VERIFICATION] = {
            "extract_data": APIEndpoint(
                name="extract_data",
                url="https://api.docverify.com/v1/extract",
                method=HTTPMethod.POST,
                rate_limit=100,
                required_params=["document_image"],
                timeout=120
            ),
            "verify_authenticity": APIEndpoint(
                name="verify_authenticity",
                url="https://api.docverify.com/v1/verify",
                method=HTTPMethod.POST,
                rate_limit=50,
                required_params=["document_image", "document_type"],
                timeout=90
            )
        }
        
        # Weather Service API
        self.api_configs[APIProvider.WEATHER_SERVICE] = {
            "current_weather": APIEndpoint(
                name="current_weather",
                url="https://api.weather.com/v1/current",
                method=HTTPMethod.GET,
                rate_limit=1000,
                cache_ttl=600,
                required_params=["location"]
            ),
            "historical_weather": APIEndpoint(
                name="historical_weather",
                url="https://api.weather.com/v1/historical",
                method=HTTPMethod.GET,
                rate_limit=100,
                cache_ttl=3600,
                required_params=["location", "date"]
            )
        }
        
        # Vehicle Data API
        self.api_configs[APIProvider.VEHICLE_DATA] = {
            "vehicle_info": APIEndpoint(
                name="vehicle_info",
                url="https://api.vehicledata.com/v1/info",
                method=HTTPMethod.GET,
                rate_limit=200,
                cache_ttl=86400,
                required_params=["vin"]
            ),
            "accident_history": APIEndpoint(
                name="accident_history",
                url="https://api.vehicledata.com/v1/accidents",
                method=HTTPMethod.GET,
                rate_limit=100,
                cache_ttl=3600,
                required_params=["vin"]
            )
        }

# Factory function
async def create_external_api_manager(redis_client: redis.Redis) -> ExternalAPIManager:
    """Create and start external API manager"""
    manager = ExternalAPIManager(redis_client)
    await manager.start()
    return manager

