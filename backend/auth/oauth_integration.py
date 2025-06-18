"""
Insurance AI Agent System - OAuth2 Integration
Production-ready OAuth2 integration with multiple providers
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import secrets
import hashlib

# OAuth2 and authentication
from authlib.integrations.starlette_client import OAuth, OAuthError
from authlib.integrations.base_client import OAuthError as BaseOAuthError
from starlette.requests import Request
from starlette.responses import RedirectResponse, JSONResponse
import httpx

# Database and utilities
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
import structlog
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator

# Internal imports
from backend.shared.models import User, Organization, OAuthProvider, OAuthToken
from backend.shared.schemas import UserCreate, UserResponse, OAuthProviderConfig
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

logger = structlog.get_logger(__name__)

class OAuthProviderType:
    """Supported OAuth providers"""
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    GITHUB = "github"
    LINKEDIN = "linkedin"
    OKTA = "okta"
    AUTH0 = "auth0"

class OAuthIntegration:
    """
    Comprehensive OAuth2 integration supporting multiple providers
    """
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
        self.logger = structlog.get_logger("oauth_integration")
        
        # OAuth client
        self.oauth = OAuth()
        
        # Provider configurations
        self.provider_configs = {}
        
        # Initialize providers
        self._initialize_providers()
        
        # OAuth state storage
        self.state_storage = {}
    
    def _initialize_providers(self):
        """Initialize OAuth providers"""
        
        # Google OAuth
        self.provider_configs[OAuthProviderType.GOOGLE] = {
            "client_id": "your-google-client-id",  # From environment
            "client_secret": "your-google-client-secret",  # From environment
            "server_metadata_url": "https://accounts.google.com/.well-known/openid_configuration",
            "client_kwargs": {
                "scope": "openid email profile"
            },
            "user_info_endpoint": "https://www.googleapis.com/oauth2/v2/userinfo",
            "user_mapping": {
                "id": "id",
                "email": "email",
                "first_name": "given_name",
                "last_name": "family_name",
                "picture": "picture"
            }
        }
        
        # Microsoft OAuth
        self.provider_configs[OAuthProviderType.MICROSOFT] = {
            "client_id": "your-microsoft-client-id",  # From environment
            "client_secret": "your-microsoft-client-secret",  # From environment
            "server_metadata_url": "https://login.microsoftonline.com/common/v2.0/.well-known/openid_configuration",
            "client_kwargs": {
                "scope": "openid email profile"
            },
            "user_info_endpoint": "https://graph.microsoft.com/v1.0/me",
            "user_mapping": {
                "id": "id",
                "email": "mail",
                "first_name": "givenName",
                "last_name": "surname",
                "picture": "photo"
            }
        }
        
        # GitHub OAuth
        self.provider_configs[OAuthProviderType.GITHUB] = {
            "client_id": "your-github-client-id",  # From environment
            "client_secret": "your-github-client-secret",  # From environment
            "access_token_url": "https://github.com/login/oauth/access_token",
            "authorize_url": "https://github.com/login/oauth/authorize",
            "api_base_url": "https://api.github.com/",
            "client_kwargs": {
                "scope": "user:email"
            },
            "user_info_endpoint": "https://api.github.com/user",
            "user_mapping": {
                "id": "id",
                "email": "email",
                "first_name": "name",
                "last_name": "",
                "picture": "avatar_url"
            }
        }
        
        # LinkedIn OAuth
        self.provider_configs[OAuthProviderType.LINKEDIN] = {
            "client_id": "your-linkedin-client-id",  # From environment
            "client_secret": "your-linkedin-client-secret",  # From environment
            "access_token_url": "https://www.linkedin.com/oauth/v2/accessToken",
            "authorize_url": "https://www.linkedin.com/oauth/v2/authorization",
            "api_base_url": "https://api.linkedin.com/v2/",
            "client_kwargs": {
                "scope": "r_liteprofile r_emailaddress"
            },
            "user_info_endpoint": "https://api.linkedin.com/v2/people/~",
            "user_mapping": {
                "id": "id",
                "email": "emailAddress",
                "first_name": "firstName.localized.en_US",
                "last_name": "lastName.localized.en_US",
                "picture": "profilePicture.displayImage"
            }
        }
        
        # Register OAuth clients
        for provider_name, config in self.provider_configs.items():
            try:
                self.oauth.register(
                    name=provider_name,
                    client_id=config["client_id"],
                    client_secret=config["client_secret"],
                    server_metadata_url=config.get("server_metadata_url"),
                    access_token_url=config.get("access_token_url"),
                    authorize_url=config.get("authorize_url"),
                    api_base_url=config.get("api_base_url"),
                    client_kwargs=config.get("client_kwargs", {})
                )
                
                self.logger.info(f"OAuth provider {provider_name} registered successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to register OAuth provider {provider_name}", error=str(e))
    
    async def initiate_oauth_flow(
        self, 
        provider: str, 
        redirect_uri: str,
        state_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initiate OAuth flow for specified provider
        
        Args:
            provider: OAuth provider name
            redirect_uri: Callback URI after OAuth completion
            state_data: Additional state data to preserve
            
        Returns:
            Dictionary containing authorization URL and state
        """
        
        try:
            if provider not in self.provider_configs:
                raise ServiceException(f"Unsupported OAuth provider: {provider}")
            
            # Generate state parameter
            state = secrets.token_urlsafe(32)
            
            # Store state data
            state_info = {
                "provider": provider,
                "redirect_uri": redirect_uri,
                "created_at": datetime.utcnow().isoformat(),
                "data": state_data or {}
            }
            
            # Store state in Redis with expiration
            await self.redis_client.setex(
                f"oauth_state:{state}",
                timedelta(minutes=10),
                json.dumps(state_info)
            )
            
            # Get OAuth client
            oauth_client = self.oauth.create_client(provider)
            
            # Generate authorization URL
            authorization_url = await oauth_client.create_authorization_url(
                redirect_uri,
                state=state
            )
            
            self.logger.info(
                "OAuth flow initiated",
                provider=provider,
                state=state,
                redirect_uri=redirect_uri
            )
            
            return {
                "authorization_url": authorization_url["url"],
                "state": state,
                "provider": provider,
                "expires_at": (datetime.utcnow() + timedelta(minutes=10)).isoformat()
            }
            
        except Exception as e:
            self.logger.error("OAuth flow initiation failed", provider=provider, error=str(e))
            raise ServiceException(f"OAuth flow initiation failed: {str(e)}")
    
    async def handle_oauth_callback(
        self, 
        provider: str, 
        code: str, 
        state: str,
        request: Request
    ) -> Dict[str, Any]:
        """
        Handle OAuth callback and complete authentication
        
        Args:
            provider: OAuth provider name
            code: Authorization code from provider
            state: State parameter for validation
            request: Starlette request object
            
        Returns:
            Dictionary containing user info and tokens
        """
        
        try:
            # Validate state
            state_data = await self._validate_oauth_state(state, provider)
            
            # Get OAuth client
            oauth_client = self.oauth.create_client(provider)
            
            # Exchange code for token
            token = await oauth_client.fetch_token(
                state_data["redirect_uri"],
                code=code
            )
            
            # Get user info from provider
            user_info = await self._get_user_info_from_provider(provider, token)
            
            # Find or create user
            user = await self._find_or_create_oauth_user(provider, user_info, token)
            
            # Generate application tokens
            from .api_gateway import create_access_token
            
            access_token_expires = timedelta(minutes=30)
            access_token = create_access_token(
                data={"sub": str(user.id), "email": user.email},
                expires_delta=access_token_expires
            )
            
            refresh_token_expires = timedelta(days=7)
            refresh_token = create_access_token(
                data={"sub": str(user.id), "type": "refresh"},
                expires_delta=refresh_token_expires
            )
            
            # Update last login
            user.last_login = datetime.utcnow()
            user_service = BaseService(User, self.db_session)
            await user_service.update(user.id, user)
            
            # Log OAuth login
            audit_logger.log_user_action(
                user_id=str(user.id),
                action="oauth_login",
                resource_type="user",
                resource_id=str(user.id),
                details={
                    "provider": provider,
                    "provider_user_id": user_info.get("id"),
                    "ip_address": request.client.host if request.client else None
                }
            )
            
            # Clean up state
            await self.redis_client.delete(f"oauth_state:{state}")
            
            self.logger.info(
                "OAuth callback completed successfully",
                provider=provider,
                user_id=str(user.id),
                user_email=user.email
            )
            
            return {
                "user": {
                    "id": str(user.id),
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "is_active": user.is_active
                },
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "expires_in": 30 * 60
                },
                "provider_info": {
                    "provider": provider,
                    "provider_user_id": user_info.get("id"),
                    "connected_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error("OAuth callback handling failed", provider=provider, error=str(e))
            raise ServiceException(f"OAuth callback handling failed: {str(e)}")
    
    async def _validate_oauth_state(self, state: str, provider: str) -> Dict[str, Any]:
        """Validate OAuth state parameter"""
        
        try:
            # Get state data from Redis
            state_json = await self.redis_client.get(f"oauth_state:{state}")
            
            if not state_json:
                raise ServiceException("Invalid or expired OAuth state")
            
            state_data = json.loads(state_json)
            
            # Validate provider
            if state_data.get("provider") != provider:
                raise ServiceException("OAuth state provider mismatch")
            
            # Check expiration
            created_at = datetime.fromisoformat(state_data["created_at"])
            if datetime.utcnow() - created_at > timedelta(minutes=10):
                raise ServiceException("OAuth state expired")
            
            return state_data
            
        except json.JSONDecodeError:
            raise ServiceException("Invalid OAuth state format")
        except Exception as e:
            self.logger.error("OAuth state validation failed", error=str(e))
            raise ServiceException(f"OAuth state validation failed: {str(e)}")
    
    async def _get_user_info_from_provider(
        self, 
        provider: str, 
        token: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get user information from OAuth provider"""
        
        try:
            config = self.provider_configs[provider]
            user_info_endpoint = config["user_info_endpoint"]
            user_mapping = config["user_mapping"]
            
            # Make request to user info endpoint
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {token['access_token']}",
                    "Accept": "application/json"
                }
                
                response = await client.get(user_info_endpoint, headers=headers)
                response.raise_for_status()
                
                provider_data = response.json()
            
            # Map provider data to standard format
            user_info = {}
            for standard_field, provider_field in user_mapping.items():
                if "." in provider_field:
                    # Handle nested fields
                    value = provider_data
                    for field_part in provider_field.split("."):
                        value = value.get(field_part, {}) if isinstance(value, dict) else None
                        if value is None:
                            break
                    user_info[standard_field] = value
                else:
                    user_info[standard_field] = provider_data.get(provider_field)
            
            # Handle special cases
            if provider == OAuthProviderType.GITHUB and not user_info.get("email"):
                # GitHub might not return email in user endpoint
                email_response = await client.get(
                    "https://api.github.com/user/emails",
                    headers=headers
                )
                if email_response.status_code == 200:
                    emails = email_response.json()
                    primary_email = next(
                        (email["email"] for email in emails if email.get("primary")),
                        emails[0]["email"] if emails else None
                    )
                    user_info["email"] = primary_email
            
            # Split full name if needed
            if provider == OAuthProviderType.GITHUB and user_info.get("first_name"):
                full_name = user_info["first_name"]
                name_parts = full_name.split(" ", 1)
                user_info["first_name"] = name_parts[0]
                user_info["last_name"] = name_parts[1] if len(name_parts) > 1 else ""
            
            self.logger.info(
                "User info retrieved from provider",
                provider=provider,
                user_id=user_info.get("id"),
                email=user_info.get("email")
            )
            
            return user_info
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error getting user info from {provider}", status_code=e.response.status_code)
            raise ServiceException(f"Failed to get user info from {provider}")
        except Exception as e:
            self.logger.error(f"Error getting user info from {provider}", error=str(e))
            raise ServiceException(f"Failed to get user info from {provider}: {str(e)}")
    
    async def _find_or_create_oauth_user(
        self, 
        provider: str, 
        user_info: Dict[str, Any], 
        token: Dict[str, Any]
    ) -> User:
        """Find existing user or create new one from OAuth data"""
        
        try:
            user_service = BaseService(User, self.db_session)
            oauth_provider_service = BaseService(OAuthProvider, self.db_session)
            
            provider_user_id = str(user_info.get("id"))
            email = user_info.get("email")
            
            if not email:
                raise ServiceException("Email not provided by OAuth provider")
            
            # Check if user exists with this OAuth provider
            existing_oauth = await oauth_provider_service.get_by_fields({
                "provider_name": provider,
                "provider_user_id": provider_user_id
            })
            
            if existing_oauth:
                # Update existing OAuth record
                existing_oauth.access_token = token.get("access_token")
                existing_oauth.refresh_token = token.get("refresh_token")
                existing_oauth.token_expires_at = datetime.utcnow() + timedelta(
                    seconds=token.get("expires_in", 3600)
                )
                existing_oauth.last_used_at = datetime.utcnow()
                
                await oauth_provider_service.update(existing_oauth.id, existing_oauth)
                
                # Get associated user
                user = await user_service.get(existing_oauth.user_id)
                
                if not user:
                    raise ServiceException("Associated user not found")
                
                return user
            
            # Check if user exists with this email
            existing_user = await user_service.get_by_field("email", email)
            
            if existing_user:
                # Link OAuth provider to existing user
                oauth_provider = OAuthProvider(
                    user_id=existing_user.id,
                    provider_name=provider,
                    provider_user_id=provider_user_id,
                    access_token=token.get("access_token"),
                    refresh_token=token.get("refresh_token"),
                    token_expires_at=datetime.utcnow() + timedelta(
                        seconds=token.get("expires_in", 3600)
                    ),
                    provider_data=user_info,
                    last_used_at=datetime.utcnow()
                )
                
                await oauth_provider_service.create(oauth_provider)
                
                self.logger.info(
                    "OAuth provider linked to existing user",
                    provider=provider,
                    user_id=str(existing_user.id),
                    email=email
                )
                
                return existing_user
            
            # Create new user
            new_user = User(
                email=email,
                first_name=user_info.get("first_name", ""),
                last_name=user_info.get("last_name", ""),
                is_active=True,
                email_verified=True,  # Assume verified if from OAuth provider
                created_via_oauth=True,
                oauth_provider=provider
            )
            
            created_user = await user_service.create(new_user)
            
            # Create OAuth provider record
            oauth_provider = OAuthProvider(
                user_id=created_user.id,
                provider_name=provider,
                provider_user_id=provider_user_id,
                access_token=token.get("access_token"),
                refresh_token=token.get("refresh_token"),
                token_expires_at=datetime.utcnow() + timedelta(
                    seconds=token.get("expires_in", 3600)
                ),
                provider_data=user_info,
                last_used_at=datetime.utcnow()
            )
            
            await oauth_provider_service.create(oauth_provider)
            
            self.logger.info(
                "New user created from OAuth",
                provider=provider,
                user_id=str(created_user.id),
                email=email
            )
            
            return created_user
            
        except Exception as e:
            self.logger.error("OAuth user creation/lookup failed", error=str(e))
            raise ServiceException(f"OAuth user creation/lookup failed: {str(e)}")
    
    async def refresh_oauth_token(
        self, 
        user_id: uuid.UUID, 
        provider: str
    ) -> Optional[Dict[str, Any]]:
        """Refresh OAuth token for user"""
        
        try:
            oauth_provider_service = BaseService(OAuthProvider, self.db_session)
            
            # Get OAuth provider record
            oauth_record = await oauth_provider_service.get_by_fields({
                "user_id": user_id,
                "provider_name": provider
            })
            
            if not oauth_record or not oauth_record.refresh_token:
                return None
            
            # Get OAuth client
            oauth_client = self.oauth.create_client(provider)
            
            # Refresh token
            new_token = await oauth_client.fetch_token(
                refresh_token=oauth_record.refresh_token
            )
            
            # Update OAuth record
            oauth_record.access_token = new_token.get("access_token")
            if new_token.get("refresh_token"):
                oauth_record.refresh_token = new_token.get("refresh_token")
            
            oauth_record.token_expires_at = datetime.utcnow() + timedelta(
                seconds=new_token.get("expires_in", 3600)
            )
            oauth_record.last_used_at = datetime.utcnow()
            
            await oauth_provider_service.update(oauth_record.id, oauth_record)
            
            self.logger.info(
                "OAuth token refreshed",
                provider=provider,
                user_id=str(user_id)
            )
            
            return new_token
            
        except Exception as e:
            self.logger.error("OAuth token refresh failed", provider=provider, user_id=str(user_id), error=str(e))
            return None
    
    async def disconnect_oauth_provider(
        self, 
        user_id: uuid.UUID, 
        provider: str
    ) -> bool:
        """Disconnect OAuth provider from user account"""
        
        try:
            oauth_provider_service = BaseService(OAuthProvider, self.db_session)
            
            # Get OAuth provider record
            oauth_record = await oauth_provider_service.get_by_fields({
                "user_id": user_id,
                "provider_name": provider
            })
            
            if not oauth_record:
                return False
            
            # Revoke token with provider if possible
            try:
                await self._revoke_oauth_token(provider, oauth_record.access_token)
            except Exception as e:
                self.logger.warning("Failed to revoke OAuth token with provider", error=str(e))
            
            # Delete OAuth record
            await oauth_provider_service.delete(oauth_record.id)
            
            # Log disconnection
            audit_logger.log_user_action(
                user_id=str(user_id),
                action="oauth_provider_disconnected",
                resource_type="oauth_provider",
                resource_id=str(oauth_record.id),
                details={"provider": provider}
            )
            
            self.logger.info(
                "OAuth provider disconnected",
                provider=provider,
                user_id=str(user_id)
            )
            
            return True
            
        except Exception as e:
            self.logger.error("OAuth provider disconnection failed", provider=provider, user_id=str(user_id), error=str(e))
            return False
    
    async def _revoke_oauth_token(self, provider: str, access_token: str):
        """Revoke OAuth token with provider"""
        
        try:
            revoke_urls = {
                OAuthProviderType.GOOGLE: "https://oauth2.googleapis.com/revoke",
                OAuthProviderType.MICROSOFT: "https://login.microsoftonline.com/common/oauth2/v2.0/logout",
                OAuthProviderType.GITHUB: "https://github.com/settings/connections/applications/{client_id}",
            }
            
            revoke_url = revoke_urls.get(provider)
            if not revoke_url:
                return  # Provider doesn't support token revocation
            
            async with httpx.AsyncClient() as client:
                if provider == OAuthProviderType.GOOGLE:
                    response = await client.post(
                        revoke_url,
                        data={"token": access_token}
                    )
                elif provider == OAuthProviderType.MICROSOFT:
                    response = await client.get(revoke_url)
                
                response.raise_for_status()
                
        except Exception as e:
            self.logger.warning(f"Failed to revoke token with {provider}", error=str(e))
    
    async def get_user_oauth_providers(self, user_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Get OAuth providers connected to user"""
        
        try:
            oauth_provider_service = BaseService(OAuthProvider, self.db_session)
            
            providers = await oauth_provider_service.get_by_field("user_id", user_id)
            
            return [
                {
                    "id": str(provider.id),
                    "provider_name": provider.provider_name,
                    "provider_user_id": provider.provider_user_id,
                    "connected_at": provider.created_at.isoformat(),
                    "last_used_at": provider.last_used_at.isoformat() if provider.last_used_at else None,
                    "token_expires_at": provider.token_expires_at.isoformat() if provider.token_expires_at else None
                }
                for provider in providers
            ]
            
        except Exception as e:
            self.logger.error("Failed to get user OAuth providers", user_id=str(user_id), error=str(e))
            return []

# Factory function
async def create_oauth_integration(db_session: AsyncSession, redis_client: redis.Redis) -> OAuthIntegration:
    """Create OAuth integration instance"""
    return OAuthIntegration(db_session, redis_client)

