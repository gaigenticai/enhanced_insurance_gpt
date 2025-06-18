"""
API Connector - Production Ready Implementation
Generic API connection and communication handler
"""

import asyncio
import json
import logging
import aiohttp
import ssl
import certifi
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
import hmac
import base64
from urllib.parse import urlencode, urlparse, parse_qs
import xml.etree.ElementTree as ET
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Monitoring
from prometheus_client import Counter, Histogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration')
api_errors_total = Counter('api_errors_total', 'Total API errors', ['endpoint', 'error_type'])

Base = declarative_base()

class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class ContentType(Enum):
    JSON = "application/json"
    XML = "application/xml"
    FORM_DATA = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    TEXT = "text/plain"

class AuthenticationMethod(Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH1 = "oauth1"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"
    HMAC_SIGNATURE = "hmac_signature"

@dataclass
class APIEndpoint:
    name: str
    base_url: str
    path: str
    method: HTTPMethod
    authentication: AuthenticationMethod
    content_type: ContentType
    required_params: List[str]
    optional_params: List[str]
    headers: Dict[str, str]
    timeout: int
    retry_attempts: int
    rate_limit_per_minute: int

@dataclass
class APICredentials:
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None

@dataclass
class APIRequest:
    request_id: str
    endpoint_name: str
    method: HTTPMethod
    url: str
    headers: Dict[str, str]
    params: Dict[str, Any]
    data: Optional[Any]
    timeout: int
    timestamp: datetime

@dataclass
class APIResponse:
    request_id: str
    status_code: int
    headers: Dict[str, str]
    content: Any
    content_type: str
    response_time: float
    timestamp: datetime
    error_message: Optional[str] = None

class APIRequestRecord(Base):
    __tablename__ = 'api_requests'
    
    request_id = Column(String, primary_key=True)
    endpoint_name = Column(String, nullable=False, index=True)
    method = Column(String, nullable=False)
    url = Column(String, nullable=False)
    request_headers = Column(JSON)
    request_params = Column(JSON)
    request_data = Column(JSON)
    response_status = Column(Integer)
    response_headers = Column(JSON)
    response_content = Column(JSON)
    response_time = Column(Float)
    error_message = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False)

class APIConnector:
    """Production-ready API Connector for external service integration"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # HTTP session configuration
        self.session = None
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Registered endpoints
        self.endpoints = {}
        
        # Default credentials
        self.default_credentials = APICredentials()
        
        logger.info("APIConnector initialized successfully")

    async def initialize(self):
        """Initialize HTTP session and connections"""
        
        if not self.session:
            connector = aiohttp.TCPConnector(
                ssl=self.ssl_context,
                limit=100,
                limit_per_host=20,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=300, connect=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'Insurance-AI-System/1.0'}
            )

    async def close(self):
        """Close HTTP session and connections"""
        
        if self.session:
            await self.session.close()
            self.session = None

    def register_endpoint(self, endpoint: APIEndpoint):
        """Register an API endpoint"""
        
        self.endpoints[endpoint.name] = endpoint
        logger.info(f"Registered API endpoint: {endpoint.name}")

    def set_credentials(self, credentials: APICredentials):
        """Set default credentials for API requests"""
        
        self.default_credentials = credentials

    async def make_request(self, endpoint_name: str, params: Dict[str, Any] = None,
                          data: Any = None, credentials: APICredentials = None,
                          custom_headers: Dict[str, str] = None) -> APIResponse:
        """Make API request to registered endpoint"""
        
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not registered")
        
        endpoint = self.endpoints[endpoint_name]
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Check rate limiting
            await self._check_rate_limit(endpoint_name, endpoint.rate_limit_per_minute)
            
            # Build request
            request = await self._build_request(
                request_id, endpoint, params or {}, data, 
                credentials or self.default_credentials, custom_headers or {}
            )
            
            # Execute request with retries
            response = await self._execute_request_with_retries(request, endpoint.retry_attempts)
            
            # Store request/response
            await self._store_api_request(request, response)
            
            # Update metrics
            api_requests_total.labels(
                endpoint=endpoint_name,
                method=endpoint.method.value,
                status=str(response.status_code)
            ).inc()
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            
            # Create error response
            response = APIResponse(
                request_id=request_id,
                status_code=0,
                headers={},
                content=None,
                content_type="",
                response_time=0.0,
                timestamp=datetime.utcnow(),
                error_message=error_msg
            )
            
            # Store error
            request = APIRequest(
                request_id=request_id,
                endpoint_name=endpoint_name,
                method=endpoint.method,
                url="",
                headers={},
                params=params or {},
                data=data,
                timeout=endpoint.timeout,
                timestamp=start_time
            )
            
            await self._store_api_request(request, response)
            
            api_errors_total.labels(
                endpoint=endpoint_name,
                error_type="request_failed"
            ).inc()
            
            logger.error(f"API request failed for {endpoint_name}: {error_msg}")
            raise

    async def make_custom_request(self, method: HTTPMethod, url: str,
                                 params: Dict[str, Any] = None, data: Any = None,
                                 headers: Dict[str, str] = None,
                                 timeout: int = 30) -> APIResponse:
        """Make custom API request to any URL"""
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            await self.initialize()
            
            # Prepare request parameters
            request_params = params or {}
            request_headers = headers or {}
            request_data = data
            
            # Execute request
            with api_request_duration.time():
                async with self.session.request(
                    method.value,
                    url,
                    params=request_params,
                    json=request_data if isinstance(request_data, (dict, list)) else None,
                    data=request_data if not isinstance(request_data, (dict, list)) else None,
                    headers=request_headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Parse response content
                    content_type = resp.headers.get('Content-Type', '').lower()
                    
                    if 'application/json' in content_type:
                        content = await resp.json()
                    elif 'application/xml' in content_type or 'text/xml' in content_type:
                        text_content = await resp.text()
                        content = self._parse_xml_response(text_content)
                    else:
                        content = await resp.text()
                    
                    response = APIResponse(
                        request_id=request_id,
                        status_code=resp.status,
                        headers=dict(resp.headers),
                        content=content,
                        content_type=content_type,
                        response_time=response_time,
                        timestamp=datetime.utcnow()
                    )
                    
                    return response
                    
        except Exception as e:
            error_msg = str(e)
            
            response = APIResponse(
                request_id=request_id,
                status_code=0,
                headers={},
                content=None,
                content_type="",
                response_time=0.0,
                timestamp=datetime.utcnow(),
                error_message=error_msg
            )
            
            logger.error(f"Custom API request failed: {error_msg}")
            raise

    async def upload_file(self, endpoint_name: str, file_path: str,
                         file_field_name: str = "file",
                         additional_data: Dict[str, Any] = None) -> APIResponse:
        """Upload file to API endpoint"""
        
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not registered")
        
        endpoint = self.endpoints[endpoint_name]
        request_id = str(uuid.uuid4())
        
        try:
            await self.initialize()
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            
            # Add file
            with open(file_path, 'rb') as f:
                data.add_field(file_field_name, f, filename=file_path.split('/')[-1])
            
            # Add additional data
            if additional_data:
                for key, value in additional_data.items():
                    data.add_field(key, str(value))
            
            # Build URL
            url = f"{endpoint.base_url.rstrip('/')}/{endpoint.path.lstrip('/')}"
            
            # Prepare headers
            headers = await self._prepare_headers(endpoint, self.default_credentials, {})
            
            # Execute request
            start_time = datetime.utcnow()
            
            async with self.session.request(
                endpoint.method.value,
                url,
                data=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
            ) as resp:
                
                response_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Parse response
                content_type = resp.headers.get('Content-Type', '').lower()
                
                if 'application/json' in content_type:
                    content = await resp.json()
                else:
                    content = await resp.text()
                
                response = APIResponse(
                    request_id=request_id,
                    status_code=resp.status,
                    headers=dict(resp.headers),
                    content=content,
                    content_type=content_type,
                    response_time=response_time,
                    timestamp=datetime.utcnow()
                )
                
                return response
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"File upload failed: {error_msg}")
            raise

    async def stream_request(self, endpoint_name: str, params: Dict[str, Any] = None,
                           chunk_handler: Callable[[bytes], None] = None) -> APIResponse:
        """Make streaming API request"""
        
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not registered")
        
        endpoint = self.endpoints[endpoint_name]
        request_id = str(uuid.uuid4())
        
        try:
            await self.initialize()
            
            # Build request
            request = await self._build_request(
                request_id, endpoint, params or {}, None, self.default_credentials, {}
            )
            
            start_time = datetime.utcnow()
            
            # Execute streaming request
            async with self.session.request(
                request.method.value,
                request.url,
                params=request.params,
                headers=request.headers,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as resp:
                
                content_chunks = []
                
                async for chunk in resp.content.iter_chunked(8192):
                    if chunk_handler:
                        chunk_handler(chunk)
                    content_chunks.append(chunk)
                
                response_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Combine chunks
                full_content = b''.join(content_chunks)
                
                # Parse content based on type
                content_type = resp.headers.get('Content-Type', '').lower()
                
                if 'application/json' in content_type:
                    content = json.loads(full_content.decode('utf-8'))
                else:
                    content = full_content.decode('utf-8')
                
                response = APIResponse(
                    request_id=request_id,
                    status_code=resp.status,
                    headers=dict(resp.headers),
                    content=content,
                    content_type=content_type,
                    response_time=response_time,
                    timestamp=datetime.utcnow()
                )
                
                return response
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Streaming request failed: {error_msg}")
            raise

    async def batch_requests(self, requests: List[Dict[str, Any]],
                           max_concurrent: int = 10) -> List[APIResponse]:
        """Execute multiple API requests concurrently"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_request(request_config):
            async with semaphore:
                endpoint_name = request_config['endpoint_name']
                params = request_config.get('params', {})
                data = request_config.get('data')
                
                return await self.make_request(endpoint_name, params, data)
        
        tasks = [execute_single_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = APIResponse(
                    request_id=str(uuid.uuid4()),
                    status_code=0,
                    headers={},
                    content=None,
                    content_type="",
                    response_time=0.0,
                    timestamp=datetime.utcnow(),
                    error_message=str(response)
                )
                processed_responses.append(error_response)
            else:
                processed_responses.append(response)
        
        return processed_responses

    # Helper methods
    
    async def _build_request(self, request_id: str, endpoint: APIEndpoint,
                           params: Dict[str, Any], data: Any,
                           credentials: APICredentials,
                           custom_headers: Dict[str, str]) -> APIRequest:
        """Build API request object"""
        
        # Validate required parameters
        missing_params = [param for param in endpoint.required_params if param not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Build URL
        url = f"{endpoint.base_url.rstrip('/')}/{endpoint.path.lstrip('/')}"
        
        # Prepare headers
        headers = await self._prepare_headers(endpoint, credentials, custom_headers)
        
        return APIRequest(
            request_id=request_id,
            endpoint_name=endpoint.name,
            method=endpoint.method,
            url=url,
            headers=headers,
            params=params,
            data=data,
            timeout=endpoint.timeout,
            timestamp=datetime.utcnow()
        )

    async def _prepare_headers(self, endpoint: APIEndpoint, credentials: APICredentials,
                             custom_headers: Dict[str, str]) -> Dict[str, str]:
        """Prepare request headers with authentication"""
        
        headers = {}
        
        # Add endpoint default headers
        headers.update(endpoint.headers)
        
        # Add content type
        headers['Content-Type'] = endpoint.content_type.value
        
        # Add authentication headers
        if endpoint.authentication == AuthenticationMethod.API_KEY:
            if credentials.api_key:
                headers['X-API-Key'] = credentials.api_key
        
        elif endpoint.authentication == AuthenticationMethod.BEARER_TOKEN:
            if credentials.access_token:
                headers['Authorization'] = f"Bearer {credentials.access_token}"
        
        elif endpoint.authentication == AuthenticationMethod.BASIC_AUTH:
            if credentials.username and credentials.password:
                auth_string = f"{credentials.username}:{credentials.password}"
                auth_bytes = auth_string.encode('utf-8')
                auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
                headers['Authorization'] = f"Basic {auth_b64}"
        
        elif endpoint.authentication == AuthenticationMethod.CUSTOM_HEADER:
            if credentials.custom_headers:
                headers.update(credentials.custom_headers)
        
        elif endpoint.authentication == AuthenticationMethod.HMAC_SIGNATURE:
            if credentials.api_key and credentials.api_secret:
                timestamp = str(int(datetime.utcnow().timestamp()))
                signature = self._generate_hmac_signature(
                    credentials.api_secret, endpoint.method.value, endpoint.path, timestamp
                )
                headers['X-API-Key'] = credentials.api_key
                headers['X-Timestamp'] = timestamp
                headers['X-Signature'] = signature
        
        # Add custom headers (override defaults)
        headers.update(custom_headers)
        
        return headers

    def _generate_hmac_signature(self, secret: str, method: str, path: str, timestamp: str) -> str:
        """Generate HMAC signature for request authentication"""
        
        message = f"{method.upper()}{path}{timestamp}"
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature

    async def _execute_request_with_retries(self, request: APIRequest, max_retries: int) -> APIResponse:
        """Execute API request with retry logic"""
        
        await self.initialize()
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                start_time = datetime.utcnow()
                
                with api_request_duration.time():
                    async with self.session.request(
                        request.method.value,
                        request.url,
                        params=request.params,
                        json=request.data if isinstance(request.data, (dict, list)) else None,
                        data=request.data if not isinstance(request.data, (dict, list)) else None,
                        headers=request.headers,
                        timeout=aiohttp.ClientTimeout(total=request.timeout)
                    ) as resp:
                        
                        response_time = (datetime.utcnow() - start_time).total_seconds()
                        
                        # Parse response content
                        content_type = resp.headers.get('Content-Type', '').lower()
                        
                        if 'application/json' in content_type:
                            content = await resp.json()
                        elif 'application/xml' in content_type or 'text/xml' in content_type:
                            text_content = await resp.text()
                            content = self._parse_xml_response(text_content)
                        else:
                            content = await resp.text()
                        
                        response = APIResponse(
                            request_id=request.request_id,
                            status_code=resp.status,
                            headers=dict(resp.headers),
                            content=content,
                            content_type=content_type,
                            response_time=response_time,
                            timestamp=datetime.utcnow()
                        )
                        
                        # Check if response indicates success
                        if 200 <= resp.status < 300:
                            return response
                        elif resp.status >= 500:  # Server error - retry
                            if attempt < max_retries:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                        
                        # Client error or final attempt - return response
                        return response
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise e
        
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception

    def _parse_xml_response(self, xml_content: str) -> Dict[str, Any]:
        """Parse XML response content"""
        
        try:
            root = ET.fromstring(xml_content)
            return self._xml_to_dict(root)
        except ET.ParseError as e:
            logger.error(f"XML parsing failed: {e}")
            return {"xml_content": xml_content, "parse_error": str(e)}

    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        
        result = {}
        
        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # Leaf node
                return element.text.strip()
            else:
                result['#text'] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            
            if child.tag in result:
                # Multiple elements with same tag - convert to list
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result

    async def _check_rate_limit(self, endpoint_name: str, rate_limit_per_minute: int):
        """Check and enforce rate limiting"""
        
        if rate_limit_per_minute <= 0:
            return
        
        current_minute = datetime.utcnow().strftime('%Y%m%d%H%M')
        rate_limit_key = f"rate_limit:{endpoint_name}:{current_minute}"
        
        current_count = self.redis_client.get(rate_limit_key)
        current_count = int(current_count) if current_count else 0
        
        if current_count >= rate_limit_per_minute:
            raise Exception(f"Rate limit exceeded for endpoint {endpoint_name}: {current_count}/{rate_limit_per_minute} per minute")
        
        # Increment counter
        self.redis_client.incr(rate_limit_key)
        self.redis_client.expire(rate_limit_key, 60)

    async def _store_api_request(self, request: APIRequest, response: APIResponse):
        """Store API request and response"""
        
        try:
            with self.Session() as session:
                record = APIRequestRecord(
                    request_id=request.request_id,
                    endpoint_name=request.endpoint_name,
                    method=request.method.value,
                    url=request.url,
                    request_headers=request.headers,
                    request_params=request.params,
                    request_data=request.data,
                    response_status=response.status_code,
                    response_headers=response.headers,
                    response_content=response.content,
                    response_time=response.response_time,
                    error_message=response.error_message,
                    timestamp=request.timestamp,
                    created_at=datetime.utcnow()
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing API request: {e}")

def create_api_connector(db_url: str = None, redis_url: str = None) -> APIConnector:
    """Create and configure APIConnector instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return APIConnector(db_url=db_url, redis_url=redis_url)

