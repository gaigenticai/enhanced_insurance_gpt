"""
Insurance AI Agent System - Simplified Authentication Service
Production-ready FastAPI authentication with user ID/password and registration
"""

import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import create_engine, MetaData, select, insert, update, and_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext
import jwt
from jwt.exceptions import InvalidTokenError
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

app = FastAPI(
    title="Insurance AI Authentication Service",
    description="Production-ready authentication service with user ID/password",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-production-jwt-secret-change-this")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))

# Database configuration
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_NAME = os.getenv("POSTGRES_DB", "insurance_ai")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
metadata = MetaData()

try:
    metadata.reflect(bind=engine, schema="public")
    users_table = metadata.tables["public.users"]
    organizations_table = metadata.tables["public.organizations"]
except Exception as e:
    logger.error("Failed to reflect database tables", error=str(e))
    raise

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Pydantic models
class UserRegistration(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=50, description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password")
    first_name: str = Field(..., min_length=1, max_length=100, description="First name")
    last_name: str = Field(..., min_length=1, max_length=100, description="Last name")
    phone: Optional[str] = Field(None, max_length=20, description="Phone number")
    organization_name: Optional[str] = Field(None, max_length=255, description="Organization name")

    @validator('user_id')
    def validate_user_id(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('User ID must contain only alphanumeric characters, hyphens, and underscores')
        return v.lower()

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserLogin(BaseModel):
    user_id: str = Field(..., description="User ID or email")
    password: str = Field(..., description="User password")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    email: str
    role: str
    organization_id: Optional[str] = None

class UserProfile(BaseModel):
    id: str
    user_id: str
    email: str
    first_name: str
    last_name: str
    phone: Optional[str]
    role: str
    organization_id: Optional[str]
    organization_name: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]

class UserUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[EmailStr] = None

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)

    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

# Utility functions
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error("Password verification failed", error=str(e))
        return False

def create_access_token(user_data: Dict[str, Any]) -> str:
    """Create a JWT access token."""
    payload = {
        "sub": user_data["id"],
        "user_id": user_data.get("user_id", user_data["email"]),
        "email": user_data["email"],
        "role": user_data["role"],
        "organization_id": user_data.get("organization_id"),
        "exp": datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS),
        "iat": datetime.utcnow(),
        "type": "access"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except InvalidTokenError as e:
        logger.error("Token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependencies
def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get current authenticated user."""
    token_data = verify_token(credentials.credentials)
    
    # Fetch fresh user data from database
    stmt = select(users_table).where(users_table.c.id == token_data["sub"])
    result = db.execute(stmt).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    user = dict(result._mapping)
    
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated"
        )
    
    return user

# API Endpoints
@app.post("/api/v1/auth/register", response_model=TokenResponse)
async def register_user(
    user_data: UserRegistration,
    db: Session = Depends(get_db)
):
    """Register a new user with default admin role."""
    try:
        # Check if user_id or email already exists
        existing_user_stmt = select(users_table).where(
            or_(
                users_table.c.email == user_data.email,
                users_table.c.user_id == user_data.user_id
            )
        )
        existing_user = db.execute(existing_user_stmt).first()
        
        if existing_user:
            existing_data = dict(existing_user._mapping)
            if existing_data["email"] == user_data.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User ID already taken"
                )
        
        # Create or get organization
        organization_id = None
        if user_data.organization_name:
            # Check if organization exists
            org_stmt = select(organizations_table).where(
                organizations_table.c.name == user_data.organization_name
            )
            org_result = db.execute(org_stmt).first()
            
            if org_result:
                organization_id = str(org_result._mapping["id"])
            else:
                # Create new organization
                org_insert = insert(organizations_table).values(
                    id=str(uuid.uuid4()),
                    name=user_data.organization_name,
                    is_active=True,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                org_result = db.execute(org_insert)
                organization_id = str(org_insert.compile().params["id"])
        
        # Hash password
        hashed_password = hash_password(user_data.password)
        
        # Create user with admin role by default
        user_id = str(uuid.uuid4())
        user_insert = insert(users_table).values(
            id=user_id,
            user_id=user_data.user_id,
            email=user_data.email,
            password_hash=hashed_password,
            role="admin",  # Default admin role as requested
            organization_id=organization_id,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            phone=user_data.phone,
            status="active",
            is_active=True,
            is_verified=True,  # Auto-verify for simplicity
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.execute(user_insert)
        db.commit()
        
        # Fetch the created user
        user_stmt = select(users_table).where(users_table.c.id == user_id)
        user_result = db.execute(user_stmt).first()
        user = dict(user_result._mapping)
        
        # Create access token
        access_token = create_access_token(user)
        
        logger.info("User registered successfully", user_id=user_data.user_id, email=user_data.email)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
            user_id=user["user_id"],
            email=user["email"],
            role=user["role"],
            organization_id=user.get("organization_id")
        )
        
    except IntegrityError as e:
        db.rollback()
        logger.error("Database integrity error during registration", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or user ID already exists"
        )
    except Exception as e:
        db.rollback()
        logger.error("Registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login_user(
    login_data: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """Authenticate user with user ID/email and password."""
    try:
        # Find user by user_id or email
        user_stmt = select(users_table).where(
            or_(
                users_table.c.user_id == login_data.user_id.lower(),
                users_table.c.email == login_data.user_id.lower()
            )
        )
        user_result = db.execute(user_stmt).first()
        
        if not user_result:
            logger.warning("Login attempt with non-existent user", user_id=login_data.user_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        user = dict(user_result._mapping)
        
        # Check if user is active and verified
        if not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated"
            )
        
        if not user["is_verified"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is not verified"
            )
        
        # Verify password
        if not verify_password(login_data.password, user["password_hash"]):
            logger.warning("Failed login attempt", user_id=login_data.user_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Update last login
        update_stmt = update(users_table).where(
            users_table.c.id == user["id"]
        ).values(
            last_login=datetime.utcnow(),
            failed_login_attempts=0
        )
        db.execute(update_stmt)
        db.commit()
        
        # Create access token
        access_token = create_access_token(user)
        
        logger.info("User logged in successfully", user_id=user["user_id"], email=user["email"])
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
            user_id=user["user_id"],
            email=user["email"],
            role=user["role"],
            organization_id=user.get("organization_id")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post("/api/v1/auth/token", response_model=TokenResponse)
async def login_oauth_compatible(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """OAuth2 compatible login endpoint for backward compatibility."""
    login_data = UserLogin(user_id=form_data.username, password=form_data.password)
    return await login_user(login_data, None, db)

@app.get("/api/v1/auth/profile", response_model=UserProfile)
async def get_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user profile."""
    try:
        # Get organization name if user has one
        organization_name = None
        if current_user.get("organization_id"):
            org_stmt = select(organizations_table).where(
                organizations_table.c.id == current_user["organization_id"]
            )
            org_result = db.execute(org_stmt).first()
            if org_result:
                organization_name = org_result._mapping["name"]
        
        return UserProfile(
            id=str(current_user["id"]),
            user_id=current_user["user_id"],
            email=current_user["email"],
            first_name=current_user["first_name"],
            last_name=current_user["last_name"],
            phone=current_user.get("phone"),
            role=current_user["role"],
            organization_id=str(current_user["organization_id"]) if current_user.get("organization_id") else None,
            organization_name=organization_name,
            is_active=current_user["is_active"],
            is_verified=current_user["is_verified"],
            created_at=current_user["created_at"],
            last_login=current_user.get("last_login")
        )
    except Exception as e:
        logger.error("Failed to get user profile", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )

@app.put("/api/v1/auth/profile", response_model=UserProfile)
async def update_user_profile(
    update_data: UserUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user profile."""
    try:
        update_values = {}
        
        if update_data.first_name is not None:
            update_values["first_name"] = update_data.first_name
        if update_data.last_name is not None:
            update_values["last_name"] = update_data.last_name
        if update_data.phone is not None:
            update_values["phone"] = update_data.phone
        if update_data.email is not None:
            # Check if email is already taken
            email_check = select(users_table).where(
                and_(
                    users_table.c.email == update_data.email,
                    users_table.c.id != current_user["id"]
                )
            )
            if db.execute(email_check).first():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use"
                )
            update_values["email"] = update_data.email
        
        if update_values:
            update_values["updated_at"] = datetime.utcnow()
            
            update_stmt = update(users_table).where(
                users_table.c.id == current_user["id"]
            ).values(**update_values)
            
            db.execute(update_stmt)
            db.commit()
        
        # Return updated profile
        return await get_user_profile(current_user, db)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to update user profile", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@app.post("/api/v1/auth/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password."""
    try:
        # Verify current password
        if not verify_password(password_data.current_password, current_user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        new_password_hash = hash_password(password_data.new_password)
        
        # Update password
        update_stmt = update(users_table).where(
            users_table.c.id == current_user["id"]
        ).values(
            password_hash=new_password_hash,
            updated_at=datetime.utcnow()
        )
        
        db.execute(update_stmt)
        db.commit()
        
        logger.info("Password changed successfully", user_id=current_user["user_id"])
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Failed to change password", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

@app.post("/api/v1/auth/verify-token")
async def verify_user_token(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Verify if token is valid and return user info."""
    return {
        "valid": True,
        "user_id": current_user["user_id"],
        "email": current_user["email"],
        "role": current_user["role"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )

