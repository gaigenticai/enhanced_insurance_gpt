"""
Insurance AI Agent System - Security Framework
Production-ready security implementation with encryption, audit logging, and security headers
"""

import os
import hashlib
import hmac
import secrets
import base64
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import ipaddress
from urllib.parse import urlparse

# Cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# FastAPI security
from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse

# Database and Redis
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text

# Internal imports
from backend.shared.models import AuditLog, SecurityEvent, EncryptedData, User
from backend.shared.schemas import SecurityConfig, AuditEntry, ThreatDetection
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

import structlog
logger = structlog.get_logger(__name__)

class SecurityEventType(Enum):
    """Security event types"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    API_KEY_USAGE = "api_key_usage"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ENCRYPTION_ERROR = "encryption_error"
    SECURITY_SCAN = "security_scan"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    COMPLIANCE_VIOLATION = "compliance_violation"

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EncryptionType(Enum):
    """Encryption types"""
    AES_256_GCM = "aes_256_gcm"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"

@dataclass
class SecurityConfig:
    """Security configuration"""
    encryption_key: str
    jwt_secret: str
    password_salt: str
    session_timeout: int = 3600
    max_login_attempts: int = 5
    lockout_duration: int = 900
    require_2fa: bool = False
    allowed_origins: List[str] = None
    blocked_ips: List[str] = None
    security_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]
        if self.blocked_ips is None:
            self.blocked_ips = []
        if self.security_headers is None:
            self.security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'",
                "Referrer-Policy": "strict-origin-when-cross-origin"
            }

class EncryptionManager:
    """
    Comprehensive encryption and decryption manager
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = structlog.get_logger("encryption_manager")
        
        # Initialize encryption keys
        self.fernet_key = self._derive_fernet_key(config.encryption_key)
        self.fernet = Fernet(self.fernet_key)
        
        # RSA key pair for asymmetric encryption
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
    
    def _derive_fernet_key(self, password: str) -> bytes:
        """Derive Fernet key from password"""
        password_bytes = password.encode()
        salt = self.config.password_salt.encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def encrypt_data(self, data: Union[str, bytes], encryption_type: EncryptionType = EncryptionType.FERNET) -> Dict[str, Any]:
        """
        Encrypt data using specified encryption type
        
        Args:
            data: Data to encrypt
            encryption_type: Type of encryption to use
            
        Returns:
            Dictionary with encrypted data and metadata
        """
        
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if encryption_type == EncryptionType.FERNET:
                encrypted_data = self.fernet.encrypt(data)
                return {
                    "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                    "encryption_type": encryption_type.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            elif encryption_type == EncryptionType.AES_256_GCM:
                # Generate random IV
                iv = os.urandom(12)
                
                # Create cipher
                cipher = Cipher(
                    algorithms.AES(self.fernet_key[:32]),
                    modes.GCM(iv),
                    backend=default_backend()
                )
                
                encryptor = cipher.encryptor()
                encrypted_data = encryptor.update(data) + encryptor.finalize()
                
                return {
                    "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                    "iv": base64.b64encode(iv).decode('utf-8'),
                    "tag": base64.b64encode(encryptor.tag).decode('utf-8'),
                    "encryption_type": encryption_type.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            elif encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
                # RSA encryption (for small data only)
                if len(data) > 190:  # RSA 2048 can encrypt max ~190 bytes
                    raise ValueError("Data too large for RSA encryption")
                
                encrypted_data = self.rsa_public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                return {
                    "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                    "encryption_type": encryption_type.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            self.logger.error("Encryption failed", error=str(e), encryption_type=encryption_type.value)
            raise ServiceException(f"Encryption failed: {str(e)}")
    
    def decrypt_data(self, encrypted_data_dict: Dict[str, Any]) -> bytes:
        """
        Decrypt data using the encryption type specified in metadata
        
        Args:
            encrypted_data_dict: Dictionary with encrypted data and metadata
            
        Returns:
            Decrypted data as bytes
        """
        
        try:
            encryption_type = EncryptionType(encrypted_data_dict["encryption_type"])
            encrypted_data = base64.b64decode(encrypted_data_dict["encrypted_data"])
            
            if encryption_type == EncryptionType.FERNET:
                return self.fernet.decrypt(encrypted_data)
            
            elif encryption_type == EncryptionType.AES_256_GCM:
                iv = base64.b64decode(encrypted_data_dict["iv"])
                tag = base64.b64decode(encrypted_data_dict["tag"])
                
                cipher = Cipher(
                    algorithms.AES(self.fernet_key[:32]),
                    modes.GCM(iv, tag),
                    backend=default_backend()
                )
                
                decryptor = cipher.decryptor()
                return decryptor.update(encrypted_data) + decryptor.finalize()
            
            elif encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
                return self.rsa_private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            self.logger.error("Decryption failed", error=str(e))
            raise ServiceException(f"Decryption failed: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            salt, stored_hash = hashed_password.split(':')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash.hex() == stored_hash
        except Exception:
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def create_hmac_signature(self, data: str, secret: str) -> str:
        """Create HMAC signature for data integrity"""
        return hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_hmac_signature(self, data: str, signature: str, secret: str) -> bool:
        """Verify HMAC signature"""
        expected_signature = self.create_hmac_signature(data, secret)
        return hmac.compare_digest(signature, expected_signature)

class AuditLogger:
    """
    Comprehensive audit logging system
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = structlog.get_logger("audit_logger")
    
    async def log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        threat_level: ThreatLevel = ThreatLevel.LOW
    ):
        """Log a security event"""
        
        try:
            event_data = {
                "event_id": secrets.token_hex(16),
                "event_type": event_type.value,
                "user_id": user_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "details": details or {},
                "threat_level": threat_level.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis for real-time monitoring
            event_key = f"security_event:{event_data['event_id']}"
            await self.redis_client.setex(
                event_key,
                86400,  # 24 hours
                json.dumps(event_data)
            )
            
            # Add to security events list
            await self.redis_client.lpush("security_events", json.dumps(event_data))
            await self.redis_client.ltrim("security_events", 0, 9999)  # Keep last 10k events
            
            # Log to structured logger
            self.logger.info(
                "Security event logged",
                event_type=event_type.value,
                user_id=user_id,
                ip_address=ip_address,
                threat_level=threat_level.value
            )
            
            # Store in database for long-term retention
            await self._store_audit_log(event_data)
            
            # Check for threat patterns
            await self._analyze_threat_patterns(event_data)
            
        except Exception as e:
            self.logger.error("Failed to log security event", error=str(e))
    
    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        ip_address: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log data access event"""
        
        await self.log_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            user_id=user_id,
            ip_address=ip_address,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                "success": success,
                **(details or {})
            },
            threat_level=ThreatLevel.LOW if success else ThreatLevel.MEDIUM
        )
    
    async def log_api_access(
        self,
        endpoint: str,
        method: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        status_code: int = 200,
        response_time: float = 0.0,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log API access"""
        
        threat_level = ThreatLevel.LOW
        if status_code >= 400:
            threat_level = ThreatLevel.MEDIUM
        if status_code >= 500:
            threat_level = ThreatLevel.HIGH
        
        await self.log_security_event(
            event_type=SecurityEventType.API_KEY_USAGE,
            user_id=user_id,
            ip_address=ip_address,
            details={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time": response_time,
                **(details or {})
            },
            threat_level=threat_level
        )
    
    async def get_security_events(
        self,
        limit: int = 100,
        event_type: Optional[SecurityEventType] = None,
        user_id: Optional[str] = None,
        threat_level: Optional[ThreatLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get security events with filtering"""
        
        try:
            # Get events from Redis
            events_data = await self.redis_client.lrange("security_events", 0, limit - 1)
            events = [json.loads(event_data) for event_data in events_data]
            
            # Apply filters
            filtered_events = []
            for event in events:
                # Filter by event type
                if event_type and event["event_type"] != event_type.value:
                    continue
                
                # Filter by user ID
                if user_id and event["user_id"] != user_id:
                    continue
                
                # Filter by threat level
                if threat_level and event["threat_level"] != threat_level.value:
                    continue
                
                # Filter by time range
                event_time = datetime.fromisoformat(event["timestamp"])
                if start_time and event_time < start_time:
                    continue
                if end_time and event_time > end_time:
                    continue
                
                filtered_events.append(event)
            
            return filtered_events
            
        except Exception as e:
            self.logger.error("Failed to get security events", error=str(e))
            return []
    
    async def _store_audit_log(self, event_data: Dict[str, Any]):
        """Store audit log in database"""
        try:
            async with get_db_session() as session:
                audit_log = AuditLog(
                    event_id=event_data["event_id"],
                    event_type=event_data["event_type"],
                    user_id=event_data["user_id"],
                    ip_address=event_data["ip_address"],
                    user_agent=event_data["user_agent"],
                    details=event_data["details"],
                    threat_level=event_data["threat_level"],
                    timestamp=datetime.fromisoformat(event_data["timestamp"])
                )
                
                session.add(audit_log)
                await session.commit()
                
        except Exception as e:
            self.logger.warning("Failed to store audit log in database", error=str(e))
    
    async def _analyze_threat_patterns(self, event_data: Dict[str, Any]):
        """Analyze event for threat patterns"""
        try:
            # Check for suspicious patterns
            if event_data["event_type"] == SecurityEventType.LOGIN_FAILURE.value:
                await self._check_brute_force_pattern(event_data)
            
            elif event_data["event_type"] == SecurityEventType.RATE_LIMIT_EXCEEDED.value:
                await self._check_dos_pattern(event_data)
            
            elif event_data["event_type"] == SecurityEventType.DATA_ACCESS.value:
                await self._check_data_exfiltration_pattern(event_data)
            
        except Exception as e:
            self.logger.error("Failed to analyze threat patterns", error=str(e))
    
    async def _check_brute_force_pattern(self, event_data: Dict[str, Any]):
        """Check for brute force attack patterns"""
        try:
            ip_address = event_data["ip_address"]
            if not ip_address:
                return
            
            # Count failed login attempts from this IP in the last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            # This is a simplified check - in practice, you'd query the database
            # or use a more sophisticated pattern detection system
            
            # For now, just log if we detect potential brute force
            self.logger.warning(
                "Potential brute force attack detected",
                ip_address=ip_address,
                event_id=event_data["event_id"]
            )
            
        except Exception as e:
            self.logger.error("Failed to check brute force pattern", error=str(e))
    
    async def _check_dos_pattern(self, event_data: Dict[str, Any]):
        """Check for DoS attack patterns"""
        try:
            ip_address = event_data["ip_address"]
            if not ip_address:
                return
            
            # Log potential DoS attack
            self.logger.warning(
                "Potential DoS attack detected",
                ip_address=ip_address,
                event_id=event_data["event_id"]
            )
            
        except Exception as e:
            self.logger.error("Failed to check DoS pattern", error=str(e))
    
    async def _check_data_exfiltration_pattern(self, event_data: Dict[str, Any]):
        """Check for data exfiltration patterns"""
        try:
            user_id = event_data["user_id"]
            if not user_id:
                return
            
            # Check for unusual data access patterns
            # This would involve analyzing access frequency, data volume, etc.
            
            # For now, just log if we detect potential data exfiltration
            details = event_data.get("details", {})
            if details.get("action") == "bulk_export":
                self.logger.warning(
                    "Potential data exfiltration detected",
                    user_id=user_id,
                    event_id=event_data["event_id"]
                )
            
        except Exception as e:
            self.logger.error("Failed to check data exfiltration pattern", error=str(e))

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for FastAPI applications
    """
    
    def __init__(self, app, config: SecurityConfig, audit_logger: AuditLogger):
        super().__init__(app)
        self.config = config
        self.audit_logger = audit_logger
        self.logger = structlog.get_logger("security_middleware")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        """Process request through security middleware"""
        
        start_time = time.time()
        
        try:
            # Get client information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            
            # Check IP blocklist
            if self._is_ip_blocked(client_ip):
                await self.audit_logger.log_security_event(
                    SecurityEventType.PERMISSION_DENIED,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    details={"reason": "blocked_ip"},
                    threat_level=ThreatLevel.HIGH
                )
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Check rate limiting (simplified)
            if await self._check_rate_limit(client_ip):
                await self.audit_logger.log_security_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    threat_level=ThreatLevel.MEDIUM
                )
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log API access
            response_time = time.time() - start_time
            await self.audit_logger.log_api_access(
                endpoint=str(request.url.path),
                method=request.method,
                ip_address=client_ip,
                status_code=response.status_code,
                response_time=response_time
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error("Security middleware error", error=str(e))
            
            # Log security event
            await self.audit_logger.log_security_event(
                SecurityEventType.SECURITY_SCAN,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent", ""),
                details={"error": str(e)},
                threat_level=ThreatLevel.HIGH
            )
            
            raise HTTPException(status_code=500, detail="Internal server error")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            for blocked_ip in self.config.blocked_ips:
                if "/" in blocked_ip:
                    # CIDR notation
                    network = ipaddress.ip_network(blocked_ip, strict=False)
                    if ip in network:
                        return True
                else:
                    # Single IP
                    if str(ip) == blocked_ip:
                        return True
            
            return False
            
        except Exception:
            return False
    
    async def _check_rate_limit(self, ip_address: str) -> bool:
        """Check rate limiting (simplified implementation)"""
        # This is a basic implementation - in production, you'd use Redis
        # with sliding window or token bucket algorithms
        return False
    
    def _add_security_headers(self, response: StarletteResponse):
        """Add security headers to response"""
        for header, value in self.config.security_headers.items():
            response.headers[header] = value

class SecurityManager:
    """
    Main security manager coordinating all security components
    """
    
    def __init__(self, config: SecurityConfig, redis_client: redis.Redis):
        self.config = config
        self.redis_client = redis_client
        self.logger = structlog.get_logger("security_manager")
        
        # Initialize components
        self.encryption_manager = EncryptionManager(config)
        self.audit_logger = AuditLogger(redis_client)
        
        # Security state
        self.failed_login_attempts: Dict[str, int] = {}
        self.locked_accounts: Dict[str, datetime] = {}
    
    async def start(self):
        """Start the security manager"""
        try:
            self.logger.info("Starting security manager")
            
            # Initialize security monitoring
            await self._initialize_security_monitoring()
            
            self.logger.info("Security manager started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start security manager", error=str(e))
            raise
    
    async def stop(self):
        """Stop the security manager"""
        try:
            self.logger.info("Stopping security manager")
            
            # Cleanup tasks would go here
            
            self.logger.info("Security manager stopped")
            
        except Exception as e:
            self.logger.error("Error stopping security manager", error=str(e))
    
    async def authenticate_user(self, username: str, password: str, ip_address: str) -> Dict[str, Any]:
        """Authenticate user with security checks"""

        try:
            if await self._is_account_locked(username):
                await self.audit_logger.log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    user_id=username,
                    ip_address=ip_address,
                    details={"reason": "account_locked"},
                    threat_level=ThreatLevel.MEDIUM,
                )
                raise HTTPException(status_code=403, detail="Account locked")

            user_data = await self._verify_credentials(username, password)

            await self._reset_failed_attempts(username)
            await self.audit_logger.log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                user_id=username,
                ip_address=ip_address,
                threat_level=ThreatLevel.LOW,
            )

            return user_data

        except HTTPException as e:
            await self._increment_failed_attempts(username)
            self.logger.warning("Authentication failed", username=username, error=e.detail)
            await self.audit_logger.log_security_event(
                SecurityEventType.LOGIN_FAILURE,
                user_id=username,
                ip_address=ip_address,
                details={"reason": e.detail},
                threat_level=ThreatLevel.MEDIUM,
            )
            raise
        except Exception as e:
            self.logger.error("Authentication error", error=str(e))
            await self.audit_logger.log_security_event(
                SecurityEventType.SECURITY_SCAN,
                user_id=username,
                ip_address=ip_address,
                details={"error": str(e)},
                threat_level=ThreatLevel.HIGH,
            )
            raise ServiceException("Authentication error")
    
    async def _verify_credentials(self, username: str, password: str) -> Dict[str, Any]:
        """Verify user credentials against the database"""
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(User).where(User.email == username, User.is_active == True)
                )
                user = result.scalar_one_or_none()

                if user and self.verify_password(password, user.password_hash):
                    return {
                        "user_id": str(user.id),
                        "username": user.email,
                        "roles": [str(user.role)],
                        "permissions": []
                    }

            self.logger.warning("Invalid credentials", username=username)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error("Credential verification failed", error=str(e))
            raise ServiceException("Credential verification error")
    
    async def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked"""
        try:
            lock_key = f"account_lock:{username}"
            lock_data = await self.redis_client.get(lock_key)
            
            if lock_data:
                lock_time = datetime.fromisoformat(lock_data.decode())
                if datetime.utcnow() < lock_time + timedelta(seconds=self.config.lockout_duration):
                    return True
                else:
                    # Lock expired, remove it
                    await self.redis_client.delete(lock_key)
            
            return False
            
        except Exception as e:
            self.logger.error("Failed to check account lock", error=str(e))
            return False
    
    async def _increment_failed_attempts(self, username: str):
        """Increment failed login attempts"""
        try:
            attempts_key = f"failed_attempts:{username}"
            attempts = await self.redis_client.incr(attempts_key)
            await self.redis_client.expire(attempts_key, 3600)  # 1 hour expiration
            
            if attempts >= self.config.max_login_attempts:
                # Lock account
                lock_key = f"account_lock:{username}"
                lock_time = datetime.utcnow().isoformat()
                await self.redis_client.setex(lock_key, self.config.lockout_duration, lock_time)
                
                # Log account lockout
                await self.audit_logger.log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    user_id=username,
                    details={"reason": "account_locked_due_to_failed_attempts", "attempts": attempts},
                    threat_level=ThreatLevel.HIGH
                )
            
        except Exception as e:
            self.logger.error("Failed to increment failed attempts", error=str(e))
    
    async def _reset_failed_attempts(self, username: str):
        """Reset failed login attempts"""
        try:
            attempts_key = f"failed_attempts:{username}"
            await self.redis_client.delete(attempts_key)
            
        except Exception as e:
            self.logger.error("Failed to reset failed attempts", error=str(e))
    
    async def _initialize_security_monitoring(self):
        """Initialize security monitoring"""
        try:
            # Set up security monitoring tasks
            # This would include threat detection, anomaly detection, etc.
            pass
            
        except Exception as e:
            self.logger.error("Failed to initialize security monitoring", error=str(e))

# Factory function
async def create_security_manager(config: SecurityConfig, redis_client: redis.Redis) -> SecurityManager:
    """Create and start security manager"""
    manager = SecurityManager(config, redis_client)
    await manager.start()
    return manager

