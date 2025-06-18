"""
Evidence Processing Agent
Advanced forensic analysis and evidence processing for insurance claims
"""

from importlib import import_module

EvidenceProcessor = import_module('backend.agents.evidence-processing.evidence_processor').EvidenceProcessor
PhotoAnalyzer = import_module('backend.agents.evidence-processing.photo_analyzer').PhotoAnalyzer
DamageAssessor = import_module('backend.agents.evidence-processing.damage_assessor').DamageAssessor
ForensicAnalyzer = import_module('backend.agents.evidence-processing.forensic_analyzer').ForensicAnalyzer
MetadataExtractor = import_module('backend.agents.evidence-processing.metadata_extractor').MetadataExtractor

__all__ = [
    'EvidenceProcessor',
    'PhotoAnalyzer', 
    'DamageAssessor',
    'ForensicAnalyzer',
    'MetadataExtractor'
]

