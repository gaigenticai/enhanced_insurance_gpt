"""
Document Analysis Agent
Advanced document processing and analysis for insurance operations
"""

from importlib import import_module

DocumentProcessor = import_module('backend.agents.document-analysis.document_processor').DocumentProcessor
OCREngine = import_module('backend.agents.document-analysis.ocr_engine').OCREngine
NLPAnalyzer = import_module('backend.agents.document-analysis.nlp_analyzer').NLPAnalyzer
DocumentClassifier = import_module('backend.agents.document-analysis.document_classifier').DocumentClassifier

__all__ = [
    'DocumentProcessor',
    'OCREngine',
    'NLPAnalyzer',
    'DocumentClassifier'
]

