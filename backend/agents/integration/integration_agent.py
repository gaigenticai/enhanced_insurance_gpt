"""
Integration Agent - Production Ready Implementation
External system integration and data synchronization for insurance operations
"""

import asyncio
import json
import logging
import os
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, urlparse
import hashlib
import hmac
import base64

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

integration_requests_total = Counter('integration_requests_total', 'Total integration requests', ['service', 'operation'])
integration_request_duration = Histogram('integration_request_duration_seconds', 'Integration request duration')
integration_errors_total = Counter('integration_errors_total', 'Total integration errors', ['service', 'error_type'])
active_integrations = Gauge('active_integrations', 'Number of active integrations')

Base = declarative_base()

class IntegrationType(Enum):
    REST_API = "rest_api"
    SOAP_API = "soap_api"
    WEBHOOK = "webhook"
    DATABASE = "database"
    FILE_TRANSFER = "file_transfer"
    MESSAGE_QUEUE = "message_queue"

class AuthenticationType(Enum):
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    JWT = "jwt"
    CERTIFICATE = "certificate"
    HMAC = "hmac"

class ServiceProvider(Enum):
    # Insurance Industry APIs
    ISO_CLAIMSEARCH = "iso_claimsearch"
    CARFAX = "carfax"
    AUTOCHECK = "autocheck"
    MITCHELL = "mitchell"
    CCC_ONE = "ccc_one"
    AUDATEX = "audatex"
    
    # Government APIs
    DMV_RECORDS = "dmv_records"
    NHTSA = "nhtsa"
    POLICE_REPORTS = "police_reports"
    COURT_RECORDS = "court_records"
    
    # Credit and Financial
    EXPERIAN = "experian"
    EQUIFAX = "equifax"
    TRANSUNION = "transunion"
    LexisNexis = "lexisnexis"
    
    # Verification Services
    IDENTITY_VERIFICATION = "identity_verification"
    ADDRESS_VERIFICATION = "address_verification"
    EMPLOYMENT_VERIFICATION = "employment_verification"
    
    # Weather and Environmental
    WEATHER_API = "weather_api"
    NOAA = "noaa"
    
    # Medical and Healthcare
    MEDICAL_RECORDS = "medical_records"
    PHARMACY_RECORDS = "pharmacy_records"

@dataclass
class IntegrationConfig:
    service_provider: ServiceProvider
    integration_type: IntegrationType
    authentication_type: AuthenticationType
    base_url: str
    api_version: str
    credentials: Dict[str, str]
    rate_limits: Dict[str, int]
    timeout_seconds: int
    retry_attempts: int
    webhook_endpoints: List[str]
    custom_headers: Dict[str, str]
    data_mapping: Dict[str, str]

@dataclass
class IntegrationRequest:
    request_id: str
    service_provider: ServiceProvider
    operation: str
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]]
    status: str
    error_message: Optional[str]
    processing_time: float
    timestamp: datetime

@dataclass
class SyncResult:
    sync_id: str
    service_provider: ServiceProvider
    records_processed: int
    records_successful: int
    records_failed: int
    sync_duration: float
    last_sync_timestamp: datetime
    next_sync_timestamp: Optional[datetime]
    errors: List[str]

class IntegrationRequestRecord(Base):
    __tablename__ = 'integration_requests'
    
    request_id = Column(String, primary_key=True)
    service_provider = Column(String, nullable=False, index=True)
    operation = Column(String, nullable=False)
    request_data = Column(JSON)
    response_data = Column(JSON)
    status = Column(String, nullable=False)
    error_message = Column(Text)
    processing_time = Column(Float)
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False)

class IntegrationSyncRecord(Base):
    __tablename__ = 'integration_syncs'
    
    sync_id = Column(String, primary_key=True)
    service_provider = Column(String, nullable=False, index=True)
    records_processed = Column(Integer)
    records_successful = Column(Integer)
    records_failed = Column(Integer)
    sync_duration = Column(Float)
    last_sync_timestamp = Column(DateTime, nullable=False)
    next_sync_timestamp = Column(DateTime)
    errors = Column(JSON)
    created_at = Column(DateTime, nullable=False)

class IntegrationAgent:
    """Production-ready Integration Agent for external system integration"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # Integration configurations
        self.integrations = {}
        self._load_integration_configs()
        
        # Session management
        self.http_session = None
        
        logger.info("IntegrationAgent initialized successfully")

    def _load_integration_configs(self):
        """Load integration configurations"""
        
        # ISO ClaimsSearch Configuration
        self.integrations[ServiceProvider.ISO_CLAIMSEARCH] = IntegrationConfig(
            service_provider=ServiceProvider.ISO_CLAIMSEARCH,
            integration_type=IntegrationType.REST_API,
            authentication_type=AuthenticationType.API_KEY,
            base_url="https://api.iso.com/claimsearch/v2",
            api_version="v2",
            credentials={"api_key": os.getenv("ISO_API_KEY", "")},
            rate_limits={"requests_per_minute": 100, "requests_per_day": 10000},
            timeout_seconds=30,
            retry_attempts=3,
            webhook_endpoints=[],
            custom_headers={"Accept": "application/json"},
            data_mapping={"policy_number": "policyNumber", "claim_number": "claimNumber"}
        )
        
        # CARFAX Configuration
        self.integrations[ServiceProvider.CARFAX] = IntegrationConfig(
            service_provider=ServiceProvider.CARFAX,
            integration_type=IntegrationType.REST_API,
            authentication_type=AuthenticationType.API_KEY,
            base_url="https://api.carfax.com/v1",
            api_version="v1",
            credentials={"api_key": os.getenv("CARFAX_API_KEY", "")},
            rate_limits={"requests_per_minute": 60, "requests_per_day": 5000},
            timeout_seconds=20,
            retry_attempts=2,
            webhook_endpoints=[],
            custom_headers={"Accept": "application/json"},
            data_mapping={"vin": "vin", "license_plate": "licensePlate"}
        )
        
        # NHTSA Configuration
        self.integrations[ServiceProvider.NHTSA] = IntegrationConfig(
            service_provider=ServiceProvider.NHTSA,
            integration_type=IntegrationType.REST_API,
            authentication_type=AuthenticationType.API_KEY,
            base_url="https://api.nhtsa.gov/vehicles/v1",
            api_version="v1",
            credentials={},  # Public API
            rate_limits={"requests_per_minute": 120, "requests_per_day": 50000},
            timeout_seconds=15,
            retry_attempts=2,
            webhook_endpoints=[],
            custom_headers={"Accept": "application/json"},
            data_mapping={"vin": "vin", "year": "modelYear", "make": "make", "model": "model"}
        )
        
        # Weather API Configuration
        self.integrations[ServiceProvider.WEATHER_API] = IntegrationConfig(
            service_provider=ServiceProvider.WEATHER_API,
            integration_type=IntegrationType.REST_API,
            authentication_type=AuthenticationType.API_KEY,
            base_url="https://api.weatherapi.com/v1",
            api_version="v1",
            credentials={"api_key": os.getenv("WEATHER_API_KEY", "")},
            rate_limits={"requests_per_minute": 100, "requests_per_day": 10000},
            timeout_seconds=10,
            retry_attempts=2,
            webhook_endpoints=[],
            custom_headers={"Accept": "application/json"},
            data_mapping={"location": "q", "date": "dt"}
        )
        
        # Credit Bureau Configuration (Experian)
        self.integrations[ServiceProvider.EXPERIAN] = IntegrationConfig(
            service_provider=ServiceProvider.EXPERIAN,
            integration_type=IntegrationType.REST_API,
            authentication_type=AuthenticationType.OAUTH2,
            base_url="https://api.experian.com/consumerservices/v1",
            api_version="v1",
            credentials={
                "client_id": os.getenv("EXPERIAN_CLIENT_ID", ""),
                "client_secret": os.getenv("EXPERIAN_CLIENT_SECRET", ""),
                "username": os.getenv("EXPERIAN_USERNAME", ""),
                "password": os.getenv("EXPERIAN_PASSWORD", "")
            },
            rate_limits={"requests_per_minute": 50, "requests_per_day": 5000},
            timeout_seconds=45,
            retry_attempts=3,
            webhook_endpoints=[],
            custom_headers={"Accept": "application/json"},
            data_mapping={"ssn": "ssn", "first_name": "firstName", "last_name": "lastName"}
        )

    async def initialize_session(self):
        """Initialize HTTP session"""
        if not self.http_session:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            timeout = aiohttp.ClientTimeout(total=60)
            self.http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def close_session(self):
        """Close HTTP session"""
        if self.http_session:
            await self.http_session.close()
            self.http_session = None

    async def search_claims_history(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Search claims history using ISO ClaimsSearch"""
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            config = self.integrations[ServiceProvider.ISO_CLAIMSEARCH]
            
            # Map request data
            mapped_data = self._map_request_data(policy_data, config.data_mapping)
            
            # Prepare request
            url = f"{config.base_url}/claims/search"
            headers = {
                "Authorization": f"Bearer {config.credentials['api_key']}",
                **config.custom_headers
            }
            
            # Execute request with rate limiting
            await self._check_rate_limit(ServiceProvider.ISO_CLAIMSEARCH)
            
            await self.initialize_session()
            
            async with self.http_session.post(url, json=mapped_data, headers=headers) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    # Process successful response
                    claims_data = self._process_claims_response(response_data)
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Store request record
                    await self._store_integration_request(IntegrationRequest(
                        request_id=request_id,
                        service_provider=ServiceProvider.ISO_CLAIMSEARCH,
                        operation="search_claims_history",
                        request_data=mapped_data,
                        response_data=response_data,
                        status="success",
                        error_message=None,
                        processing_time=processing_time,
                        timestamp=start_time
                    ))
                    
                    integration_requests_total.labels(
                        service="iso_claimsearch",
                        operation="search_claims_history"
                    ).inc()
                    
                    return claims_data
                else:
                    raise Exception(f"API request failed with status {response.status}: {response_data}")
                    
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            
            await self._store_integration_request(IntegrationRequest(
                request_id=request_id,
                service_provider=ServiceProvider.ISO_CLAIMSEARCH,
                operation="search_claims_history",
                request_data=policy_data,
                response_data=None,
                status="error",
                error_message=error_msg,
                processing_time=processing_time,
                timestamp=start_time
            ))
            
            integration_errors_total.labels(
                service="iso_claimsearch",
                error_type="api_error"
            ).inc()
            
            logger.error(f"Claims history search failed: {error_msg}")
            raise

    async def get_vehicle_history(self, vin: str) -> Dict[str, Any]:
        """Get vehicle history from CARFAX"""
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            config = self.integrations[ServiceProvider.CARFAX]
            
            # Prepare request
            url = f"{config.base_url}/vehicles/{vin}/history"
            headers = {
                "X-API-Key": config.credentials['api_key'],
                **config.custom_headers
            }
            
            # Execute request with rate limiting
            await self._check_rate_limit(ServiceProvider.CARFAX)
            
            await self.initialize_session()
            
            async with self.http_session.get(url, headers=headers) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    # Process vehicle history data
                    vehicle_data = self._process_vehicle_history(response_data)
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    await self._store_integration_request(IntegrationRequest(
                        request_id=request_id,
                        service_provider=ServiceProvider.CARFAX,
                        operation="get_vehicle_history",
                        request_data={"vin": vin},
                        response_data=response_data,
                        status="success",
                        error_message=None,
                        processing_time=processing_time,
                        timestamp=start_time
                    ))
                    
                    integration_requests_total.labels(
                        service="carfax",
                        operation="get_vehicle_history"
                    ).inc()
                    
                    return vehicle_data
                else:
                    raise Exception(f"CARFAX API request failed with status {response.status}")
                    
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            
            await self._store_integration_request(IntegrationRequest(
                request_id=request_id,
                service_provider=ServiceProvider.CARFAX,
                operation="get_vehicle_history",
                request_data={"vin": vin},
                response_data=None,
                status="error",
                error_message=error_msg,
                processing_time=processing_time,
                timestamp=start_time
            ))
            
            integration_errors_total.labels(
                service="carfax",
                error_type="api_error"
            ).inc()
            
            logger.error(f"Vehicle history lookup failed: {error_msg}")
            raise

    async def get_weather_data(self, location: str, date: datetime) -> Dict[str, Any]:
        """Get historical weather data"""
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            config = self.integrations[ServiceProvider.WEATHER_API]
            
            # Prepare request parameters
            params = {
                "key": config.credentials['api_key'],
                "q": location,
                "dt": date.strftime("%Y-%m-%d"),
                "hour": date.hour
            }
            
            url = f"{config.base_url}/history.json"
            
            # Execute request with rate limiting
            await self._check_rate_limit(ServiceProvider.WEATHER_API)
            
            await self.initialize_session()
            
            async with self.http_session.get(url, params=params) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    # Process weather data
                    weather_data = self._process_weather_data(response_data)
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    await self._store_integration_request(IntegrationRequest(
                        request_id=request_id,
                        service_provider=ServiceProvider.WEATHER_API,
                        operation="get_weather_data",
                        request_data={"location": location, "date": date.isoformat()},
                        response_data=response_data,
                        status="success",
                        error_message=None,
                        processing_time=processing_time,
                        timestamp=start_time
                    ))
                    
                    integration_requests_total.labels(
                        service="weather_api",
                        operation="get_weather_data"
                    ).inc()
                    
                    return weather_data
                else:
                    raise Exception(f"Weather API request failed with status {response.status}")
                    
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            
            await self._store_integration_request(IntegrationRequest(
                request_id=request_id,
                service_provider=ServiceProvider.WEATHER_API,
                operation="get_weather_data",
                request_data={"location": location, "date": date.isoformat()},
                response_data=None,
                status="error",
                error_message=error_msg,
                processing_time=processing_time,
                timestamp=start_time
            ))
            
            integration_errors_total.labels(
                service="weather_api",
                error_type="api_error"
            ).inc()
            
            logger.error(f"Weather data lookup failed: {error_msg}")
            raise

    async def verify_identity(self, identity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify identity using credit bureau data"""
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            config = self.integrations[ServiceProvider.EXPERIAN]
            
            # Get OAuth2 token
            access_token = await self._get_oauth2_token(config)
            
            # Map request data
            mapped_data = self._map_request_data(identity_data, config.data_mapping)
            
            # Prepare request
            url = f"{config.base_url}/identity/verify"
            headers = {
                "Authorization": f"Bearer {access_token}",
                **config.custom_headers
            }
            
            # Execute request with rate limiting
            await self._check_rate_limit(ServiceProvider.EXPERIAN)
            
            await self.initialize_session()
            
            async with self.http_session.post(url, json=mapped_data, headers=headers) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    # Process identity verification response
                    verification_data = self._process_identity_verification(response_data)
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    await self._store_integration_request(IntegrationRequest(
                        request_id=request_id,
                        service_provider=ServiceProvider.EXPERIAN,
                        operation="verify_identity",
                        request_data=mapped_data,
                        response_data=response_data,
                        status="success",
                        error_message=None,
                        processing_time=processing_time,
                        timestamp=start_time
                    ))
                    
                    integration_requests_total.labels(
                        service="experian",
                        operation="verify_identity"
                    ).inc()
                    
                    return verification_data
                else:
                    raise Exception(f"Identity verification failed with status {response.status}")
                    
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            
            await self._store_integration_request(IntegrationRequest(
                request_id=request_id,
                service_provider=ServiceProvider.EXPERIAN,
                operation="verify_identity",
                request_data=identity_data,
                response_data=None,
                status="error",
                error_message=error_msg,
                processing_time=processing_time,
                timestamp=start_time
            ))
            
            integration_errors_total.labels(
                service="experian",
                error_type="api_error"
            ).inc()
            
            logger.error(f"Identity verification failed: {error_msg}")
            raise

    async def get_vehicle_specifications(self, vin: str) -> Dict[str, Any]:
        """Get vehicle specifications from NHTSA"""
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            config = self.integrations[ServiceProvider.NHTSA]
            
            # Prepare request
            url = f"{config.base_url}/DecodeVin/{vin}"
            params = {"format": "json"}
            
            # Execute request with rate limiting
            await self._check_rate_limit(ServiceProvider.NHTSA)
            
            await self.initialize_session()
            
            async with self.http_session.get(url, params=params) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    # Process vehicle specifications
                    vehicle_specs = self._process_vehicle_specifications(response_data)
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    await self._store_integration_request(IntegrationRequest(
                        request_id=request_id,
                        service_provider=ServiceProvider.NHTSA,
                        operation="get_vehicle_specifications",
                        request_data={"vin": vin},
                        response_data=response_data,
                        status="success",
                        error_message=None,
                        processing_time=processing_time,
                        timestamp=start_time
                    ))
                    
                    integration_requests_total.labels(
                        service="nhtsa",
                        operation="get_vehicle_specifications"
                    ).inc()
                    
                    return vehicle_specs
                else:
                    raise Exception(f"NHTSA API request failed with status {response.status}")
                    
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            
            await self._store_integration_request(IntegrationRequest(
                request_id=request_id,
                service_provider=ServiceProvider.NHTSA,
                operation="get_vehicle_specifications",
                request_data={"vin": vin},
                response_data=None,
                status="error",
                error_message=error_msg,
                processing_time=processing_time,
                timestamp=start_time
            ))
            
            integration_errors_total.labels(
                service="nhtsa",
                error_type="api_error"
            ).inc()
            
            logger.error(f"Vehicle specifications lookup failed: {error_msg}")
            raise

    async def sync_external_data(self, service_provider: ServiceProvider, 
                               sync_config: Dict[str, Any]) -> SyncResult:
        """Synchronize data from external service"""
        
        sync_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            records_processed = 0
            records_successful = 0
            records_failed = 0
            errors = []
            
            # Get sync configuration
            batch_size = sync_config.get('batch_size', 100)
            max_records = sync_config.get('max_records', 1000)
            
            # Process data in batches
            for batch_start in range(0, max_records, batch_size):
                try:
                    batch_data = await self._fetch_batch_data(
                        service_provider, batch_start, batch_size, sync_config
                    )
                    
                    for record in batch_data:
                        try:
                            await self._process_sync_record(service_provider, record)
                            records_successful += 1
                        except Exception as e:
                            records_failed += 1
                            errors.append(f"Record processing failed: {str(e)}")
                        
                        records_processed += 1
                    
                    # Rate limiting between batches
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    errors.append(f"Batch processing failed: {str(e)}")
                    break
            
            sync_duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Calculate next sync time
            sync_interval = sync_config.get('sync_interval_hours', 24)
            next_sync = start_time + timedelta(hours=sync_interval)
            
            result = SyncResult(
                sync_id=sync_id,
                service_provider=service_provider,
                records_processed=records_processed,
                records_successful=records_successful,
                records_failed=records_failed,
                sync_duration=sync_duration,
                last_sync_timestamp=start_time,
                next_sync_timestamp=next_sync,
                errors=errors
            )
            
            await self._store_sync_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Data synchronization failed: {e}")
            raise

    # Helper methods
    
    def _map_request_data(self, source_data: Dict[str, Any], 
                         mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map request data according to API requirements"""
        
        mapped_data = {}
        
        for source_key, target_key in mapping.items():
            if source_key in source_data:
                mapped_data[target_key] = source_data[source_key]
        
        # Include unmapped data
        for key, value in source_data.items():
            if key not in mapping:
                mapped_data[key] = value
        
        return mapped_data

    async def _check_rate_limit(self, service_provider: ServiceProvider):
        """Check and enforce rate limits"""
        
        try:
            config = self.integrations[service_provider]
            rate_limits = config.rate_limits
            
            # Check requests per minute
            minute_key = f"rate_limit:{service_provider.value}:minute:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
            minute_count = self.redis_client.get(minute_key)
            
            if minute_count and int(minute_count) >= rate_limits.get('requests_per_minute', 100):
                raise Exception(f"Rate limit exceeded for {service_provider.value}: requests per minute")
            
            # Check requests per day
            day_key = f"rate_limit:{service_provider.value}:day:{datetime.utcnow().strftime('%Y%m%d')}"
            day_count = self.redis_client.get(day_key)
            
            if day_count and int(day_count) >= rate_limits.get('requests_per_day', 10000):
                raise Exception(f"Rate limit exceeded for {service_provider.value}: requests per day")
            
            # Increment counters
            self.redis_client.incr(minute_key)
            self.redis_client.expire(minute_key, 60)
            self.redis_client.incr(day_key)
            self.redis_client.expire(day_key, 86400)
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            raise

    async def _get_oauth2_token(self, config: IntegrationConfig) -> str:
        """Get OAuth2 access token"""
        
        try:
            # Check if token is cached
            cache_key = f"oauth2_token:{config.service_provider.value}"
            cached_token = self.redis_client.get(cache_key)
            
            if cached_token:
                return cached_token.decode('utf-8')
            
            # Request new token
            token_url = f"{config.base_url}/oauth/token"
            
            data = {
                "grant_type": "client_credentials",
                "client_id": config.credentials['client_id'],
                "client_secret": config.credentials['client_secret']
            }
            
            await self.initialize_session()
            
            async with self.http_session.post(token_url, data=data) as response:
                token_data = await response.json()
                
                if response.status == 200:
                    access_token = token_data['access_token']
                    expires_in = token_data.get('expires_in', 3600)
                    
                    # Cache token
                    self.redis_client.setex(cache_key, expires_in - 60, access_token)
                    
                    return access_token
                else:
                    raise Exception(f"OAuth2 token request failed: {token_data}")
                    
        except Exception as e:
            logger.error(f"OAuth2 token acquisition failed: {e}")
            raise

    def _process_claims_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process claims search response"""
        
        try:
            claims = response_data.get('claims', [])
            processed_claims = []
            
            for claim in claims:
                processed_claim = {
                    'claim_number': claim.get('claimNumber'),
                    'policy_number': claim.get('policyNumber'),
                    'loss_date': claim.get('lossDate'),
                    'loss_type': claim.get('lossType'),
                    'loss_amount': claim.get('lossAmount'),
                    'status': claim.get('status'),
                    'description': claim.get('description'),
                    'adjuster': claim.get('adjuster'),
                    'settlement_amount': claim.get('settlementAmount'),
                    'settlement_date': claim.get('settlementDate')
                }
                processed_claims.append(processed_claim)
            
            return {
                'total_claims': len(processed_claims),
                'claims': processed_claims,
                'search_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Claims response processing failed: {e}")
            return {'total_claims': 0, 'claims': [], 'error': str(e)}

    def _process_vehicle_history(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process vehicle history response"""
        
        try:
            vehicle_info = response_data.get('vehicle', {})
            history_records = response_data.get('historyRecords', [])
            
            processed_history = []
            for record in history_records:
                processed_record = {
                    'date': record.get('date'),
                    'event_type': record.get('eventType'),
                    'description': record.get('description'),
                    'mileage': record.get('mileage'),
                    'location': record.get('location'),
                    'source': record.get('source')
                }
                processed_history.append(processed_record)
            
            return {
                'vin': vehicle_info.get('vin'),
                'year': vehicle_info.get('year'),
                'make': vehicle_info.get('make'),
                'model': vehicle_info.get('model'),
                'trim': vehicle_info.get('trim'),
                'history_count': len(processed_history),
                'history_records': processed_history,
                'report_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Vehicle history processing failed: {e}")
            return {'history_count': 0, 'history_records': [], 'error': str(e)}

    def _process_weather_data(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process weather data response"""
        
        try:
            forecast = response_data.get('forecast', {})
            forecastday = forecast.get('forecastday', [])
            
            if forecastday:
                day_data = forecastday[0]
                day_info = day_data.get('day', {})
                
                return {
                    'date': day_data.get('date'),
                    'max_temp_f': day_info.get('maxtemp_f'),
                    'min_temp_f': day_info.get('mintemp_f'),
                    'avg_temp_f': day_info.get('avgtemp_f'),
                    'max_wind_mph': day_info.get('maxwind_mph'),
                    'total_precip_in': day_info.get('totalprecip_in'),
                    'avg_humidity': day_info.get('avghumidity'),
                    'condition': day_info.get('condition', {}).get('text'),
                    'uv_index': day_info.get('uv')
                }
            
            return {'error': 'No weather data available'}
            
        except Exception as e:
            logger.error(f"Weather data processing failed: {e}")
            return {'error': str(e)}

    def _process_identity_verification(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process identity verification response"""
        
        try:
            verification_result = response_data.get('verificationResult', {})
            
            return {
                'identity_verified': verification_result.get('identityVerified', False),
                'confidence_score': verification_result.get('confidenceScore', 0),
                'verification_status': verification_result.get('status'),
                'matched_elements': verification_result.get('matchedElements', []),
                'risk_indicators': verification_result.get('riskIndicators', []),
                'verification_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Identity verification processing failed: {e}")
            return {'identity_verified': False, 'error': str(e)}

    def _process_vehicle_specifications(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process vehicle specifications response"""
        
        try:
            results = response_data.get('Results', [])
            
            specifications = {}
            for result in results:
                variable = result.get('Variable')
                value = result.get('Value')
                
                if variable and value:
                    specifications[variable] = value
            
            return {
                'vin': specifications.get('VIN'),
                'make': specifications.get('Make'),
                'model': specifications.get('Model'),
                'model_year': specifications.get('Model Year'),
                'vehicle_type': specifications.get('Vehicle Type'),
                'body_class': specifications.get('Body Class'),
                'engine_info': specifications.get('Engine Configuration'),
                'fuel_type': specifications.get('Fuel Type - Primary'),
                'transmission': specifications.get('Transmission Style'),
                'drive_type': specifications.get('Drive Type'),
                'safety_rating': specifications.get('Overall Front Rating'),
                'specifications': specifications
            }
            
        except Exception as e:
            logger.error(f"Vehicle specifications processing failed: {e}")
            return {'specifications': {}, 'error': str(e)}

    async def _fetch_batch_data(self, service_provider: ServiceProvider, 
                              offset: int, limit: int, 
                              sync_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch batch data for synchronization"""
        
        # This would be implemented based on specific service provider APIs
        # For now, return empty list
        return []

    async def _process_sync_record(self, service_provider: ServiceProvider, 
                                 record: Dict[str, Any]):
        """Process individual sync record"""
        
        # This would implement record-specific processing logic
        # Store in local database, transform data, etc.
        pass

    async def _store_integration_request(self, request: IntegrationRequest):
        """Store integration request record"""
        
        try:
            with self.Session() as session:
                record = IntegrationRequestRecord(
                    request_id=request.request_id,
                    service_provider=request.service_provider.value,
                    operation=request.operation,
                    request_data=request.request_data,
                    response_data=request.response_data,
                    status=request.status,
                    error_message=request.error_message,
                    processing_time=request.processing_time,
                    timestamp=request.timestamp,
                    created_at=datetime.utcnow()
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing integration request: {e}")

    async def _store_sync_result(self, result: SyncResult):
        """Store synchronization result"""
        
        try:
            with self.Session() as session:
                record = IntegrationSyncRecord(
                    sync_id=result.sync_id,
                    service_provider=result.service_provider.value,
                    records_processed=result.records_processed,
                    records_successful=result.records_successful,
                    records_failed=result.records_failed,
                    sync_duration=result.sync_duration,
                    last_sync_timestamp=result.last_sync_timestamp,
                    next_sync_timestamp=result.next_sync_timestamp,
                    errors=result.errors,
                    created_at=datetime.utcnow()
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing sync result: {e}")

def create_integration_agent(db_url: str = None, redis_url: str = None) -> IntegrationAgent:
    """Create and configure IntegrationAgent instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return IntegrationAgent(db_url=db_url, redis_url=redis_url)

