#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/ubuntu/enhanced_insurance_gpt')

def test_imports():
    """Test all the critical imports"""
    try:
        print("Testing imports...")
        
        # Test the fixed communication agent import
        print("Testing communication agent import...")
        from backend.agents.communication.communication_agent import CommunicationAgent
        print("✓ CommunicationAgent import successful")
        
        # Test other agent imports
        print("Testing other agent imports...")
        from backend.agents.document_analysis_agent import DocumentAnalysisAgent
        print("✓ DocumentAnalysisAgent import successful")
        
        from backend.agents.evidence_processing_agent import EvidenceProcessingAgent
        print("✓ EvidenceProcessingAgent import successful")
        
        from backend.agents.risk_assessment_agent import RiskAssessmentAgent
        print("✓ RiskAssessmentAgent import successful")
        
        from backend.agents.compliance_agent import ComplianceAgent
        print("✓ ComplianceAgent import successful")
        
        # Test orchestrator imports
        print("Testing orchestrator imports...")
        from backend.orchestrators.claims_orchestrator import ClaimsOrchestrator
        print("✓ ClaimsOrchestrator import successful")
        
        from backend.orchestrators.underwriting_orchestrator import UnderwritingOrchestrator
        print("✓ UnderwritingOrchestrator import successful")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

