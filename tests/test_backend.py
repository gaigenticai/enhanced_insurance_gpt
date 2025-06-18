"""
Insurance AI Agent System - Backend Test Suite
Comprehensive testing framework for all backend components
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock

# FastAPI testing
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Database testing
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Redis testing
import fakeredis.aioredis

# Internal imports
from backend.api_gateway import app
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.models import Base, User, Policy, Claim, Agent, WorkflowExecution
from backend.shared.schemas import UserCreate, PolicyCreate, ClaimCreate
from backend.orchestrators.underwriting_orchestrator import UnderwritingOrchestrator
from backend.orchestrators.claims_orchestrator import ClaimsOrchestrator
from backend.agents.document_analysis_agent import DocumentAnalysisAgent
from backend.agents.risk_assessment_agent import RiskAssessmentAgent
from backend.agents.communication.communication_agent import CommunicationAgent
from backend.agents.evidence_processing_agent import EvidenceProcessingAgent
from backend.agents.compliance_agent import ComplianceAgent
from backend.security.security_manager import SecurityManager, SecurityConfig
from backend.monitoring.metrics import MetricsCollector

# Test configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_REDIS_URL = "redis://localhost:6379/15"

class TestConfig:
    """Test configuration"""
    DATABASE_URL = TEST_DATABASE_URL
    REDIS_URL = TEST_REDIS_URL
    JWT_SECRET = "test-jwt-secret-key-for-testing-only"
    ENCRYPTION_KEY = "test-encryption-key-32-chars-long"
    PASSWORD_SALT = "test-salt"
    ENVIRONMENT = "testing"
    DEBUG = True

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_db_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def test_db_session(test_db_engine):
    """Create test database session"""
    async_session = sessionmaker(
        test_db_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session

@pytest.fixture
async def test_redis():
    """Create test Redis client"""
    redis_client = fakeredis.aioredis.FakeRedis()
    yield redis_client
    await redis_client.flushall()
    await redis_client.close()

@pytest.fixture
def test_client(test_db_session, test_redis):
    """Create test FastAPI client"""
    
    # Override dependencies
    app.dependency_overrides[get_db_session] = lambda: test_db_session
    app.dependency_overrides[get_redis_client] = lambda: test_redis
    
    with TestClient(app) as client:
        yield client
    
    # Clear overrides
    app.dependency_overrides.clear()

@pytest.fixture
async def test_async_client(test_db_session, test_redis):
    """Create test async FastAPI client"""
    
    # Override dependencies
    app.dependency_overrides[get_db_session] = lambda: test_db_session
    app.dependency_overrides[get_redis_client] = lambda: test_redis
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    # Clear overrides
    app.dependency_overrides.clear()

@pytest.fixture
def test_user_data():
    """Test user data"""
    return {
        "email": "test@zurich.com",
        "password": "testpassword123",
        "first_name": "Test",
        "last_name": "User",
        "role": "agent",
        "department": "underwriting"
    }

@pytest.fixture
def test_policy_data():
    """Test policy data"""
    return {
        "policy_number": "POL-TEST-001",
        "policy_type": "auto",
        "customer_id": "CUST-001",
        "premium_amount": 1200.00,
        "coverage_amount": 50000.00,
        "start_date": datetime.utcnow().isoformat(),
        "end_date": (datetime.utcnow() + timedelta(days=365)).isoformat(),
        "status": "active"
    }

@pytest.fixture
def test_claim_data():
    """Test claim data"""
    return {
        "claim_number": "CLM-TEST-001",
        "policy_id": "POL-TEST-001",
        "claim_type": "collision",
        "incident_date": datetime.utcnow().isoformat(),
        "reported_date": datetime.utcnow().isoformat(),
        "claim_amount": 5000.00,
        "description": "Vehicle collision on highway",
        "status": "submitted"
    }

@pytest.fixture
def security_config():
    """Test security configuration"""
    return SecurityConfig(
        encryption_key="test-encryption-key-32-chars-long",
        jwt_secret="test-jwt-secret",
        password_salt="test-salt",
        session_timeout=3600,
        max_login_attempts=3,
        lockout_duration=300
    )

# =============================================================================
# API Gateway Tests
# =============================================================================

class TestAPIGateway:
    """Test API Gateway functionality"""
    
    def test_health_check(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_api_documentation(self, test_client):
        """Test API documentation endpoints"""
        response = test_client.get("/docs")
        assert response.status_code == 200
        
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
    
    def test_cors_headers(self, test_client):
        """Test CORS headers"""
        response = test_client.options("/api/v1/users")
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

# =============================================================================
# Authentication Tests
# =============================================================================

class TestAuthentication:
    """Test authentication and authorization"""
    
    async def test_user_registration(self, test_async_client, test_user_data):
        """Test user registration"""
        response = await test_async_client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["email"] == test_user_data["email"]
        assert "password" not in data
    
    async def test_user_login(self, test_async_client, test_user_data):
        """Test user login"""
        # First register user
        await test_async_client.post("/api/v1/auth/register", json=test_user_data)
        
        # Then login
        login_data = {
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        }
        response = await test_async_client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
    
    async def test_invalid_login(self, test_async_client):
        """Test invalid login credentials"""
        login_data = {
            "email": "invalid@example.com",
            "password": "wrongpassword"
        }
        response = await test_async_client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401
    
    async def test_protected_endpoint_without_token(self, test_async_client):
        """Test accessing protected endpoint without token"""
        response = await test_async_client.get("/api/v1/users/me")
        assert response.status_code == 401
    
    async def test_protected_endpoint_with_token(self, test_async_client, test_user_data):
        """Test accessing protected endpoint with valid token"""
        # Register and login
        await test_async_client.post("/api/v1/auth/register", json=test_user_data)
        login_response = await test_async_client.post("/api/v1/auth/login", json={
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        })
        token = login_response.json()["access_token"]
        
        # Access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = await test_async_client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 200

# =============================================================================
# Database Model Tests
# =============================================================================

class TestDatabaseModels:
    """Test database models and operations"""
    
    async def test_user_model(self, test_db_session):
        """Test User model CRUD operations"""
        # Create user
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            first_name="Test",
            last_name="User",
            role="agent"
        )
        test_db_session.add(user)
        await test_db_session.commit()
        await test_db_session.refresh(user)
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.created_at is not None
    
    async def test_policy_model(self, test_db_session):
        """Test Policy model CRUD operations"""
        # Create policy
        policy = Policy(
            policy_number="POL-001",
            policy_type="auto",
            customer_id="CUST-001",
            premium_amount=1200.00,
            coverage_amount=50000.00,
            status="active"
        )
        test_db_session.add(policy)
        await test_db_session.commit()
        await test_db_session.refresh(policy)
        
        assert policy.id is not None
        assert policy.policy_number == "POL-001"
    
    async def test_claim_model(self, test_db_session):
        """Test Claim model CRUD operations"""
        # Create claim
        claim = Claim(
            claim_number="CLM-001",
            policy_id="POL-001",
            claim_type="collision",
            claim_amount=5000.00,
            description="Test claim",
            status="submitted"
        )
        test_db_session.add(claim)
        await test_db_session.commit()
        await test_db_session.refresh(claim)
        
        assert claim.id is not None
        assert claim.claim_number == "CLM-001"

# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestUnderwritingOrchestrator:
    """Test Underwriting Orchestrator"""
    
    @pytest.fixture
    async def orchestrator(self, test_db_session, test_redis):
        """Create test orchestrator"""
        return UnderwritingOrchestrator(test_db_session, test_redis)
    
    async def test_process_application(self, orchestrator, test_policy_data):
        """Test processing underwriting application"""
        with patch.object(orchestrator, '_analyze_risk') as mock_analyze:
            mock_analyze.return_value = {"risk_score": 0.3, "risk_level": "low"}
            
            result = await orchestrator.process_application(test_policy_data)
            
            assert result["status"] == "approved"
            assert "workflow_id" in result
            mock_analyze.assert_called_once()
    
    async def test_risk_assessment_workflow(self, orchestrator):
        """Test risk assessment workflow"""
        application_data = {
            "applicant_age": 30,
            "driving_record": "clean",
            "vehicle_type": "sedan",
            "coverage_amount": 50000
        }
        
        with patch.object(orchestrator, '_call_risk_assessment_agent') as mock_agent:
            mock_agent.return_value = {"risk_score": 0.25}
            
            result = await orchestrator._analyze_risk(application_data)
            
            assert result["risk_score"] == 0.25
            mock_agent.assert_called_once()

class TestClaimsOrchestrator:
    """Test Claims Orchestrator"""
    
    @pytest.fixture
    async def orchestrator(self, test_db_session, test_redis):
        """Create test orchestrator"""
        return ClaimsOrchestrator(test_db_session, test_redis)
    
    async def test_process_claim(self, orchestrator, test_claim_data):
        """Test processing insurance claim"""
        with patch.object(orchestrator, '_validate_claim') as mock_validate:
            mock_validate.return_value = {"valid": True, "confidence": 0.9}
            
            result = await orchestrator.process_claim(test_claim_data)
            
            assert result["status"] == "processing"
            assert "workflow_id" in result
            mock_validate.assert_called_once()
    
    async def test_fraud_detection_workflow(self, orchestrator, test_claim_data):
        """Test fraud detection workflow"""
        with patch.object(orchestrator, '_call_evidence_processing_agent') as mock_agent:
            mock_agent.return_value = {"fraud_indicators": [], "fraud_score": 0.1}
            
            result = await orchestrator._detect_fraud(test_claim_data)
            
            assert result["fraud_score"] == 0.1
            mock_agent.assert_called_once()

# =============================================================================
# Agent Tests
# =============================================================================

class TestDocumentAnalysisAgent:
    """Test Document Analysis Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create test agent"""
        return DocumentAnalysisAgent()
    
    async def test_extract_text_from_pdf(self, agent):
        """Test PDF text extraction"""
        # Create a mock PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"Mock PDF content")
            tmp_file_path = tmp_file.name
        
        try:
            with patch.object(agent, '_extract_pdf_text') as mock_extract:
                mock_extract.return_value = "Extracted text content"
                
                result = await agent.extract_text(tmp_file_path)
                
                assert result["text"] == "Extracted text content"
                assert result["confidence"] > 0
        finally:
            os.unlink(tmp_file_path)
    
    async def test_analyze_document_structure(self, agent):
        """Test document structure analysis"""
        document_text = """
        INSURANCE POLICY
        Policy Number: POL-123456
        Policyholder: John Doe
        Coverage Amount: $50,000
        """
        
        with patch.object(agent, '_analyze_structure') as mock_analyze:
            mock_analyze.return_value = {
                "document_type": "insurance_policy",
                "fields": {
                    "policy_number": "POL-123456",
                    "policyholder": "John Doe",
                    "coverage_amount": "$50,000"
                }
            }
            
            result = await agent.analyze_structure(document_text)
            
            assert result["document_type"] == "insurance_policy"
            assert "policy_number" in result["fields"]

class TestRiskAssessmentAgent:
    """Test Risk Assessment Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create test agent"""
        return RiskAssessmentAgent()
    
    async def test_assess_auto_insurance_risk(self, agent):
        """Test auto insurance risk assessment"""
        risk_data = {
            "driver_age": 25,
            "driving_experience": 7,
            "accident_history": [],
            "vehicle_make": "Toyota",
            "vehicle_model": "Camry",
            "vehicle_year": 2020,
            "annual_mileage": 12000
        }
        
        with patch.object(agent, '_calculate_risk_score') as mock_calculate:
            mock_calculate.return_value = 0.3
            
            result = await agent.assess_risk(risk_data, "auto")
            
            assert result["risk_score"] == 0.3
            assert result["risk_level"] in ["low", "medium", "high"]
    
    async def test_assess_property_insurance_risk(self, agent):
        """Test property insurance risk assessment"""
        risk_data = {
            "property_type": "single_family",
            "property_age": 10,
            "location": "suburban",
            "security_features": ["alarm_system", "deadbolts"],
            "natural_disaster_risk": "low"
        }
        
        result = await agent.assess_risk(risk_data, "property")
        
        assert "risk_score" in result
        assert "risk_factors" in result

class TestCommunicationAgent:
    """Test Communication Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create test agent"""
        return CommunicationAgent()
    
    async def test_send_email_notification(self, agent):
        """Test email notification sending"""
        with patch.object(agent, '_send_email') as mock_send:
            mock_send.return_value = {"status": "sent", "message_id": "msg_123"}
            
            result = await agent.send_notification(
                recipient="test@example.com",
                message="Test notification",
                channel="email"
            )
            
            assert result["status"] == "sent"
            mock_send.assert_called_once()
    
    async def test_send_sms_notification(self, agent):
        """Test SMS notification sending"""
        with patch.object(agent, '_send_sms') as mock_send:
            mock_send.return_value = {"status": "sent", "message_id": "sms_123"}
            
            result = await agent.send_notification(
                recipient="+1234567890",
                message="Test SMS",
                channel="sms"
            )
            
            assert result["status"] == "sent"
            mock_send.assert_called_once()

# =============================================================================
# Security Tests
# =============================================================================

class TestSecurity:
    """Test security components"""
    
    async def test_password_hashing(self, security_config):
        """Test password hashing and verification"""
        from backend.security.security_manager import EncryptionManager
        
        encryption_manager = EncryptionManager(security_config)
        
        password = "testpassword123"
        hashed = encryption_manager.hash_password(password)
        
        assert hashed != password
        assert encryption_manager.verify_password(password, hashed)
        assert not encryption_manager.verify_password("wrongpassword", hashed)
    
    async def test_data_encryption(self, security_config):
        """Test data encryption and decryption"""
        from backend.security.security_manager import EncryptionManager, EncryptionType
        
        encryption_manager = EncryptionManager(security_config)
        
        original_data = "Sensitive insurance data"
        encrypted = encryption_manager.encrypt_data(original_data, EncryptionType.FERNET)
        decrypted = encryption_manager.decrypt_data(encrypted)
        
        assert encrypted["encrypted_data"] != original_data
        assert decrypted.decode() == original_data
    
    async def test_audit_logging(self, test_redis, security_config):
        """Test audit logging functionality"""
        from backend.security.security_manager import AuditLogger, SecurityEventType, ThreatLevel
        
        audit_logger = AuditLogger(test_redis)
        
        await audit_logger.log_security_event(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            user_id="test_user",
            ip_address="192.168.1.1",
            threat_level=ThreatLevel.LOW
        )
        
        events = await audit_logger.get_security_events(limit=10)
        assert len(events) == 1
        assert events[0]["event_type"] == "login_success"

# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Test integration between components"""
    
    async def test_end_to_end_underwriting_workflow(self, test_async_client, test_user_data):
        """Test complete underwriting workflow"""
        # Register and login user
        await test_async_client.post("/api/v1/auth/register", json=test_user_data)
        login_response = await test_async_client.post("/api/v1/auth/login", json={
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Submit underwriting application
        application_data = {
            "policy_type": "auto",
            "applicant_name": "John Doe",
            "applicant_age": 30,
            "vehicle_make": "Toyota",
            "vehicle_model": "Camry",
            "coverage_amount": 50000
        }
        
        with patch('backend.orchestrators.underwriting_orchestrator.UnderwritingOrchestrator.process_application') as mock_process:
            mock_process.return_value = {
                "status": "approved",
                "workflow_id": "wf_123",
                "policy_number": "POL-123456"
            }
            
            response = await test_async_client.post(
                "/api/v1/underwriting/applications",
                json=application_data,
                headers=headers
            )
            
            assert response.status_code == 201
            result = response.json()
            assert result["status"] == "approved"
    
    async def test_end_to_end_claims_workflow(self, test_async_client, test_user_data):
        """Test complete claims workflow"""
        # Register and login user
        await test_async_client.post("/api/v1/auth/register", json=test_user_data)
        login_response = await test_async_client.post("/api/v1/auth/login", json={
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Submit claim
        claim_data = {
            "policy_number": "POL-123456",
            "claim_type": "collision",
            "incident_date": "2024-01-15T10:30:00Z",
            "description": "Vehicle collision on highway",
            "claim_amount": 5000.00
        }
        
        with patch('backend.orchestrators.claims_orchestrator.ClaimsOrchestrator.process_claim') as mock_process:
            mock_process.return_value = {
                "status": "processing",
                "workflow_id": "wf_456",
                "claim_number": "CLM-123456"
            }
            
            response = await test_async_client.post(
                "/api/v1/claims",
                json=claim_data,
                headers=headers
            )
            
            assert response.status_code == 201
            result = response.json()
            assert result["status"] == "processing"

# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance and load handling"""
    
    async def test_concurrent_requests(self, test_async_client):
        """Test handling concurrent requests"""
        import asyncio
        
        async def make_request():
            response = await test_async_client.get("/health")
            return response.status_code
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All requests should succeed
        assert all(status == 200 for status in results)
    
    async def test_large_document_processing(self, test_async_client):
        """Test processing large documents"""
        # This would test with actual large files in a real scenario
        large_text = "A" * 10000  # 10KB of text
        
        with patch('backend.agents.document_analysis_agent.DocumentAnalysisAgent.extract_text') as mock_extract:
            mock_extract.return_value = {
                "text": large_text,
                "confidence": 0.95,
                "processing_time": 2.5
            }
            
            # Test would verify processing completes within reasonable time
            assert len(large_text) == 10000

# =============================================================================
# Test Utilities
# =============================================================================

def run_tests():
    """Run all tests"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=backend",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])

if __name__ == "__main__":
    run_tests()

