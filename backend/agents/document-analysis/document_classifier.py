"""
Document Classifier - Production Ready Implementation
Advanced machine learning-based document classification for insurance operations
"""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import pandas as pd
import redis
import re
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Machine Learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# Deep Learning
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Text processing
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
classification_operations_total = Counter('classification_operations_total', 'Total classification operations', ['model_type', 'status'])
classification_duration = Histogram('classification_duration_seconds', 'Time to classify documents')
classification_accuracy_gauge = Gauge('classification_accuracy', 'Classification accuracy', ['model_name'])

Base = declarative_base()

class DocumentType(Enum):
    POLICY_APPLICATION = "policy_application"
    POLICY_RENEWAL = "policy_renewal"
    POLICY_AMENDMENT = "policy_amendment"
    CLAIM_FORM = "claim_form"
    CLAIM_SUPPLEMENT = "claim_supplement"
    PROOF_OF_LOSS = "proof_of_loss"
    MEDICAL_RECORD = "medical_record"
    MEDICAL_BILL = "medical_bill"
    PRESCRIPTION = "prescription"
    FINANCIAL_STATEMENT = "financial_statement"
    BANK_STATEMENT = "bank_statement"
    TAX_RETURN = "tax_return"
    IDENTITY_DOCUMENT = "identity_document"
    DRIVERS_LICENSE = "drivers_license"
    PASSPORT = "passport"
    PROPERTY_DEED = "property_deed"
    VEHICLE_TITLE = "vehicle_title"
    VEHICLE_REGISTRATION = "vehicle_registration"
    POLICE_REPORT = "police_report"
    ACCIDENT_REPORT = "accident_report"
    REPAIR_ESTIMATE = "repair_estimate"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CORRESPONDENCE = "correspondence"
    LEGAL_DOCUMENT = "legal_document"
    CONTRACT = "contract"
    POWER_OF_ATTORNEY = "power_of_attorney"
    COURT_DOCUMENT = "court_document"
    INSPECTION_REPORT = "inspection_report"
    APPRAISAL = "appraisal"
    PHOTO_EVIDENCE = "photo_evidence"
    OTHER = "other"

class ClassificationMethod(Enum):
    RULE_BASED = "rule_based"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ClassificationFeatures:
    """Features extracted for document classification"""
    text_features: Dict[str, Any]
    metadata_features: Dict[str, Any]
    structural_features: Dict[str, Any]
    linguistic_features: Dict[str, Any]

@dataclass
class ClassificationResult:
    """Document classification result"""
    classification_id: str
    document_type: DocumentType
    confidence: float
    confidence_level: ConfidenceLevel
    method_used: ClassificationMethod
    all_predictions: Dict[str, float]
    features_used: List[str]
    processing_time: float
    model_version: str
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class ClassificationRecord(Base):
    """SQLAlchemy model for classification records"""
    __tablename__ = 'document_classifications'
    
    classification_id = Column(String, primary_key=True)
    document_id = Column(String)
    document_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    confidence_level = Column(String)
    method_used = Column(String)
    all_predictions = Column(JSON)
    features_used = Column(JSON)
    processing_time = Column(Float)
    model_version = Column(String)
    error_message = Column(Text)
    created_at = Column(DateTime, nullable=False)
    metadata = Column(JSON)

class TrainingRecord(Base):
    """SQLAlchemy model for training records"""
    __tablename__ = 'model_training_records'
    
    training_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    training_data_size = Column(Integer)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_time = Column(Float)
    hyperparameters = Column(JSON)
    feature_importance = Column(JSON)
    created_at = Column(DateTime, nullable=False)
    model_path = Column(String)

class DocumentClassifier:
    """
    Production-ready Document Classifier
    Advanced ML-based document classification for insurance operations
    """
    
    def __init__(self, db_url: str, redis_url: str, models_path: str = "/tmp/classification_models"):
        self.db_url = db_url
        self.redis_url = redis_url
        self.models_path = models_path
        
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis setup
        self.redis_client = redis.from_url(redis_url)
        
        # Models directory
        os.makedirs(models_path, exist_ok=True)
        
        # Initialize components
        self._initialize_nlp_components()
        self._initialize_classification_rules()
        self._load_trained_models()
        
        # Feature extractors
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.label_encoder = LabelEncoder()
        
        logger.info("DocumentClassifier initialized successfully")

    def _initialize_nlp_components(self):
        """Initialize NLP components for feature extraction"""
        
        try:
            # Download required NLTK data
            nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for item in nltk_downloads:
                try:
                    nltk.download(item, quiet=True)
                except:
                    pass
            
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found")
                self.nlp = None
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")

    def _initialize_classification_rules(self):
        """Initialize rule-based classification patterns"""
        
        # Document type patterns based on keywords and structure
        self.classification_rules = {
            DocumentType.POLICY_APPLICATION: {
                'keywords': [
                    'application', 'applicant', 'policy', 'coverage', 'premium',
                    'beneficiary', 'insured', 'underwriting', 'effective date'
                ],
                'required_fields': ['name', 'address', 'coverage'],
                'exclusion_keywords': ['claim', 'accident', 'loss'],
                'weight': 1.0
            },
            DocumentType.CLAIM_FORM: {
                'keywords': [
                    'claim', 'claimant', 'loss', 'damage', 'accident', 'incident',
                    'adjuster', 'settlement', 'deductible', 'occurred'
                ],
                'required_fields': ['date', 'description', 'amount'],
                'exclusion_keywords': ['application', 'renewal'],
                'weight': 1.0
            },
            DocumentType.MEDICAL_RECORD: {
                'keywords': [
                    'medical', 'doctor', 'physician', 'hospital', 'diagnosis',
                    'treatment', 'patient', 'symptoms', 'medication', 'therapy'
                ],
                'required_fields': ['patient', 'date', 'diagnosis'],
                'exclusion_keywords': ['policy', 'premium'],
                'weight': 1.0
            },
            DocumentType.FINANCIAL_STATEMENT: {
                'keywords': [
                    'financial', 'income', 'expense', 'balance', 'statement',
                    'assets', 'liabilities', 'revenue', 'profit', 'loss'
                ],
                'required_fields': ['amount', 'date', 'account'],
                'exclusion_keywords': ['medical', 'accident'],
                'weight': 1.0
            },
            DocumentType.IDENTITY_DOCUMENT: {
                'keywords': [
                    'identification', 'license', 'passport', 'id', 'social security',
                    'birth certificate', 'driver', 'state issued'
                ],
                'required_fields': ['name', 'number', 'date'],
                'exclusion_keywords': ['claim', 'medical'],
                'weight': 1.0
            },
            DocumentType.VEHICLE_TITLE: {
                'keywords': [
                    'title', 'vehicle', 'car', 'auto', 'vin', 'make', 'model',
                    'year', 'owner', 'registration', 'certificate'
                ],
                'required_fields': ['vin', 'make', 'model'],
                'exclusion_keywords': ['medical', 'policy'],
                'weight': 1.0
            },
            DocumentType.POLICE_REPORT: {
                'keywords': [
                    'police', 'officer', 'report', 'incident', 'accident',
                    'violation', 'citation', 'department', 'badge'
                ],
                'required_fields': ['officer', 'date', 'incident'],
                'exclusion_keywords': ['medical', 'financial'],
                'weight': 1.0
            },
            DocumentType.REPAIR_ESTIMATE: {
                'keywords': [
                    'estimate', 'repair', 'parts', 'labor', 'cost', 'damage',
                    'shop', 'mechanic', 'service', 'quote'
                ],
                'required_fields': ['cost', 'description', 'date'],
                'exclusion_keywords': ['medical', 'policy'],
                'weight': 1.0
            },
            DocumentType.CORRESPONDENCE: {
                'keywords': [
                    'letter', 'email', 'correspondence', 'communication',
                    'dear', 'sincerely', 'regards', 'message'
                ],
                'required_fields': ['date', 'recipient'],
                'exclusion_keywords': [],
                'weight': 0.8
            }
        }

    def _load_trained_models(self):
        """Load pre-trained classification models"""
        
        self.trained_models = {}
        
        try:
            # Load models from disk if they exist
            model_files = [
                'random_forest_classifier.pkl',
                'gradient_boosting_classifier.pkl',
                'svm_classifier.pkl',
                'logistic_regression_classifier.pkl',
                'neural_network_classifier.pkl'
            ]
            
            for model_file in model_files:
                model_path = os.path.join(self.models_path, model_file)
                if os.path.exists(model_path):
                    try:
                        model_name = model_file.replace('.pkl', '')
                        self.trained_models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded model: {model_name}")
                    except Exception as e:
                        logger.warning(f"Could not load model {model_file}: {e}")
            
            # Load feature extractors
            tfidf_path = os.path.join(self.models_path, 'tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                logger.info("Loaded TF-IDF vectorizer")
            
            label_encoder_path = os.path.join(self.models_path, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                logger.info("Loaded label encoder")
            
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")

    async def classify_document(self, 
                              text: str, 
                              metadata: Dict[str, Any] = None,
                              method: ClassificationMethod = ClassificationMethod.ENSEMBLE) -> ClassificationResult:
        """Classify document using specified method"""
        
        start_time = datetime.utcnow()
        classification_id = str(uuid.uuid4())
        
        with classification_duration.time():
            try:
                logger.info(f"Starting document classification using {method.value}")
                
                # Extract features
                features = await self._extract_features(text, metadata or {})
                
                # Perform classification based on method
                if method == ClassificationMethod.RULE_BASED:
                    result = await self._classify_rule_based(text, features)
                elif method == ClassificationMethod.MACHINE_LEARNING:
                    result = await self._classify_ml(text, features)
                elif method == ClassificationMethod.DEEP_LEARNING:
                    result = await self._classify_deep_learning(text, features)
                elif method == ClassificationMethod.ENSEMBLE:
                    result = await self._classify_ensemble(text, features)
                else:
                    raise ValueError(f"Unknown classification method: {method}")
                
                # Determine confidence level
                confidence_level = self._determine_confidence_level(result['confidence'])
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Create classification result
                classification_result = ClassificationResult(
                    classification_id=classification_id,
                    document_type=DocumentType(result['document_type']),
                    confidence=result['confidence'],
                    confidence_level=confidence_level,
                    method_used=method,
                    all_predictions=result.get('all_predictions', {}),
                    features_used=result.get('features_used', []),
                    processing_time=processing_time,
                    model_version=result.get('model_version', '1.0'),
                    metadata=metadata or {}
                )
                
                # Store result
                await self._store_classification_result(classification_result)
                
                # Update metrics
                classification_operations_total.labels(
                    model_type=method.value,
                    status='success'
                ).inc()
                
                logger.info(f"Classification completed: {result['document_type']} ({result['confidence']:.2f})")
                
                return classification_result
                
            except Exception as e:
                logger.error(f"Error in document classification: {e}")
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                error_result = ClassificationResult(
                    classification_id=classification_id,
                    document_type=DocumentType.OTHER,
                    confidence=0.0,
                    confidence_level=ConfidenceLevel.LOW,
                    method_used=method,
                    all_predictions={},
                    features_used=[],
                    processing_time=processing_time,
                    model_version='1.0',
                    metadata=metadata or {},
                    error_message=str(e)
                )
                
                classification_operations_total.labels(
                    model_type=method.value,
                    status='failed'
                ).inc()
                
                return error_result

    async def _extract_features(self, text: str, metadata: Dict[str, Any]) -> ClassificationFeatures:
        """Extract comprehensive features for classification"""
        
        try:
            # Text features
            text_features = await self._extract_text_features(text)
            
            # Metadata features
            metadata_features = await self._extract_metadata_features(metadata)
            
            # Structural features
            structural_features = await self._extract_structural_features(text)
            
            # Linguistic features
            linguistic_features = await self._extract_linguistic_features(text)
            
            return ClassificationFeatures(
                text_features=text_features,
                metadata_features=metadata_features,
                structural_features=structural_features,
                linguistic_features=linguistic_features
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return ClassificationFeatures({}, {}, {}, {})

    async def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract text-based features"""
        
        features = {}
        
        try:
            text_lower = text.lower()
            
            # Basic text statistics
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
            features['avg_word_length'] = sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
            
            # Keyword presence for each document type
            for doc_type, rules in self.classification_rules.items():
                keyword_count = sum(1 for keyword in rules['keywords'] if keyword in text_lower)
                features[f'{doc_type.value}_keyword_count'] = keyword_count
                features[f'{doc_type.value}_keyword_ratio'] = keyword_count / len(rules['keywords']) if rules['keywords'] else 0
            
            # Common insurance terms
            insurance_terms = [
                'policy', 'claim', 'premium', 'deductible', 'coverage', 'beneficiary',
                'insured', 'claimant', 'adjuster', 'underwriting', 'liability'
            ]
            
            insurance_term_count = sum(1 for term in insurance_terms if term in text_lower)
            features['insurance_term_count'] = insurance_term_count
            features['insurance_term_ratio'] = insurance_term_count / len(insurance_terms)
            
            # Specific pattern matches
            import re
            
            # Numbers and amounts
            features['has_currency'] = bool(re.search(r'\$[0-9,]+\.?[0-9]*', text))
            features['has_percentage'] = bool(re.search(r'\d+\.?\d*%', text))
            features['has_phone'] = bool(re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text))
            features['has_email'] = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
            features['has_date'] = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text))
            features['has_ssn'] = bool(re.search(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', text))
            features['has_vin'] = bool(re.search(r'\b[A-HJ-NPR-Z0-9]{17}\b', text))
            
            # Document structure indicators
            features['has_signature_line'] = bool(re.search(r'signature|signed|_____', text_lower))
            features['has_date_line'] = bool(re.search(r'date:|dated:|_____', text_lower))
            features['has_form_fields'] = bool(re.search(r'\[\s*\]|\(\s*\)|____', text))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {}

    async def _extract_metadata_features(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata-based features"""
        
        features = {}
        
        try:
            # File information
            features['filename'] = metadata.get('filename', '')
            features['file_size'] = metadata.get('file_size', 0)
            features['page_count'] = metadata.get('page_count', 1)
            features['file_format'] = metadata.get('format', '')
            
            # Filename analysis
            filename = features['filename'].lower()
            
            # Document type hints in filename
            filename_indicators = {
                'policy': ['policy', 'pol', 'coverage'],
                'claim': ['claim', 'clm', 'loss'],
                'medical': ['medical', 'med', 'doctor', 'hospital'],
                'financial': ['financial', 'bank', 'statement'],
                'identity': ['id', 'license', 'passport'],
                'vehicle': ['vehicle', 'car', 'auto', 'title'],
                'legal': ['legal', 'court', 'contract']
            }
            
            for category, indicators in filename_indicators.items():
                features[f'filename_indicates_{category}'] = any(indicator in filename for indicator in indicators)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting metadata features: {e}")
            return {}

    async def _extract_structural_features(self, text: str) -> Dict[str, Any]:
        """Extract document structure features"""
        
        features = {}
        
        try:
            lines = text.split('\n')
            
            # Line statistics
            features['line_count'] = len(lines)
            features['empty_line_count'] = sum(1 for line in lines if not line.strip())
            features['avg_line_length'] = sum(len(line) for line in lines) / len(lines) if lines else 0
            
            # Indentation and formatting
            indented_lines = sum(1 for line in lines if line.startswith(' ') or line.startswith('\t'))
            features['indented_line_ratio'] = indented_lines / len(lines) if lines else 0
            
            # Capitalization patterns
            all_caps_lines = sum(1 for line in lines if line.isupper() and len(line.strip()) > 5)
            features['all_caps_line_ratio'] = all_caps_lines / len(lines) if lines else 0
            
            # List indicators
            features['has_bullet_points'] = bool(re.search(r'^\s*[•\-\*]\s', text, re.MULTILINE))
            features['has_numbered_list'] = bool(re.search(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
            
            # Table indicators
            features['has_table_structure'] = bool(re.search(r'\|.*\|', text))
            features['has_tab_separated'] = '\t' in text
            
            # Header/footer indicators
            features['has_header_footer'] = bool(re.search(r'page \d+ of \d+|confidential|proprietary', text.lower()))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting structural features: {e}")
            return {}

    async def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features using NLP"""
        
        features = {}
        
        try:
            if self.nlp:
                doc = self.nlp(text[:100000])  # Limit text length for performance
                
                # POS tag distribution
                pos_counts = {}
                for token in doc:
                    if not token.is_space:
                        pos = token.pos_
                        pos_counts[pos] = pos_counts.get(pos, 0) + 1
                
                total_tokens = sum(pos_counts.values())
                if total_tokens > 0:
                    for pos, count in pos_counts.items():
                        features[f'pos_{pos.lower()}_ratio'] = count / total_tokens
                
                # Named entity types
                entity_types = {}
                for ent in doc.ents:
                    ent_type = ent.label_
                    entity_types[ent_type] = entity_types.get(ent_type, 0) + 1
                
                for ent_type, count in entity_types.items():
                    features[f'entity_{ent_type.lower()}_count'] = count
                
                # Sentence complexity
                sentences = list(doc.sents)
                if sentences:
                    avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
                    features['avg_sentence_length'] = avg_sentence_length
                    features['sentence_count'] = len(sentences)
            
            # TextBlob features
            try:
                blob = TextBlob(text)
                features['sentiment_polarity'] = blob.sentiment.polarity
                features['sentiment_subjectivity'] = blob.sentiment.subjectivity
            except:
                features['sentiment_polarity'] = 0.0
                features['sentiment_subjectivity'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting linguistic features: {e}")
            return {}

    async def _classify_rule_based(self, text: str, features: ClassificationFeatures) -> Dict[str, Any]:
        """Rule-based classification using keyword matching and patterns"""
        
        try:
            text_lower = text.lower()
            scores = {}
            
            for doc_type, rules in self.classification_rules.items():
                score = 0.0
                
                # Keyword matching
                keyword_matches = sum(1 for keyword in rules['keywords'] if keyword in text_lower)
                keyword_score = keyword_matches / len(rules['keywords']) if rules['keywords'] else 0
                
                # Required field presence (simplified check)
                field_matches = 0
                for field in rules['required_fields']:
                    if field.lower() in text_lower:
                        field_matches += 1
                
                field_score = field_matches / len(rules['required_fields']) if rules['required_fields'] else 0
                
                # Exclusion keywords penalty
                exclusion_penalty = 0
                for exclusion in rules['exclusion_keywords']:
                    if exclusion in text_lower:
                        exclusion_penalty += 0.2
                
                # Calculate final score
                score = (keyword_score * 0.7 + field_score * 0.3) * rules['weight'] - exclusion_penalty
                score = max(0, score)  # Ensure non-negative
                
                scores[doc_type.value] = score
            
            # Find best match
            if scores:
                best_type = max(scores, key=scores.get)
                best_score = scores[best_type]
            else:
                best_type = DocumentType.OTHER.value
                best_score = 0.0
            
            return {
                'document_type': best_type,
                'confidence': min(best_score, 1.0),
                'all_predictions': scores,
                'features_used': ['keywords', 'required_fields', 'exclusions'],
                'model_version': 'rule_based_v1.0'
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based classification: {e}")
            return {
                'document_type': DocumentType.OTHER.value,
                'confidence': 0.0,
                'all_predictions': {},
                'features_used': [],
                'model_version': 'rule_based_v1.0'
            }

    async def _classify_ml(self, text: str, features: ClassificationFeatures) -> Dict[str, Any]:
        """Machine learning-based classification"""
        
        try:
            if not self.trained_models:
                logger.warning("No trained ML models available, falling back to rule-based")
                return await self._classify_rule_based(text, features)
            
            # Prepare features for ML models
            feature_vector = self._prepare_feature_vector(text, features)
            
            # Use ensemble of available models
            predictions = {}
            confidences = {}
            
            for model_name, model in self.trained_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba([feature_vector])[0]
                        prediction = model.classes_[np.argmax(proba)]
                        confidence = np.max(proba)
                    else:
                        prediction = model.predict([feature_vector])[0]
                        confidence = 0.8  # Default confidence for models without probability
                    
                    predictions[model_name] = prediction
                    confidences[model_name] = confidence
                    
                except Exception as e:
                    logger.warning(f"Error with model {model_name}: {e}")
            
            if not predictions:
                return await self._classify_rule_based(text, features)
            
            # Ensemble prediction (majority vote with confidence weighting)
            weighted_votes = {}
            for model_name, prediction in predictions.items():
                confidence = confidences[model_name]
                if prediction not in weighted_votes:
                    weighted_votes[prediction] = 0
                weighted_votes[prediction] += confidence
            
            best_prediction = max(weighted_votes, key=weighted_votes.get)
            avg_confidence = weighted_votes[best_prediction] / len(predictions)
            
            # Get all class probabilities (simplified)
            all_predictions = {doc_type.value: 0.0 for doc_type in DocumentType}
            for prediction, weight in weighted_votes.items():
                if prediction in all_predictions:
                    all_predictions[prediction] = weight / len(predictions)
            
            return {
                'document_type': best_prediction,
                'confidence': min(avg_confidence, 1.0),
                'all_predictions': all_predictions,
                'features_used': ['text_features', 'structural_features', 'linguistic_features'],
                'model_version': 'ml_ensemble_v1.0'
            }
            
        except Exception as e:
            logger.error(f"Error in ML classification: {e}")
            return await self._classify_rule_based(text, features)

    async def _classify_deep_learning(self, text: str, features: ClassificationFeatures) -> Dict[str, Any]:
        """Deep learning-based classification using transformers"""
        
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers not available, falling back to ML")
                return await self._classify_ml(text, features)
            
            # This would use a fine-tuned transformer model
            # For now, fall back to ML approach
            logger.info("Deep learning classification not yet implemented, using ML")
            return await self._classify_ml(text, features)
            
        except Exception as e:
            logger.error(f"Error in deep learning classification: {e}")
            return await self._classify_ml(text, features)

    async def _classify_ensemble(self, text: str, features: ClassificationFeatures) -> Dict[str, Any]:
        """Ensemble classification combining multiple methods"""
        
        try:
            # Get predictions from different methods
            rule_result = await self._classify_rule_based(text, features)
            ml_result = await self._classify_ml(text, features)
            
            # Combine predictions with weights
            rule_weight = 0.3
            ml_weight = 0.7
            
            # Combine confidence scores
            combined_predictions = {}
            
            # Add rule-based predictions
            for doc_type, score in rule_result['all_predictions'].items():
                combined_predictions[doc_type] = score * rule_weight
            
            # Add ML predictions
            for doc_type, score in ml_result['all_predictions'].items():
                if doc_type in combined_predictions:
                    combined_predictions[doc_type] += score * ml_weight
                else:
                    combined_predictions[doc_type] = score * ml_weight
            
            # Find best prediction
            if combined_predictions:
                best_type = max(combined_predictions, key=combined_predictions.get)
                best_confidence = combined_predictions[best_type]
            else:
                best_type = DocumentType.OTHER.value
                best_confidence = 0.0
            
            return {
                'document_type': best_type,
                'confidence': min(best_confidence, 1.0),
                'all_predictions': combined_predictions,
                'features_used': ['rule_based', 'machine_learning'],
                'model_version': 'ensemble_v1.0'
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble classification: {e}")
            return await self._classify_rule_based(text, features)

    def _prepare_feature_vector(self, text: str, features: ClassificationFeatures) -> np.ndarray:
        """Prepare feature vector for ML models"""
        
        try:
            # Combine all features into a single vector
            feature_list = []
            
            # Text features
            for key, value in features.text_features.items():
                if isinstance(value, (int, float)):
                    feature_list.append(value)
                elif isinstance(value, bool):
                    feature_list.append(1.0 if value else 0.0)
            
            # Structural features
            for key, value in features.structural_features.items():
                if isinstance(value, (int, float)):
                    feature_list.append(value)
                elif isinstance(value, bool):
                    feature_list.append(1.0 if value else 0.0)
            
            # Linguistic features
            for key, value in features.linguistic_features.items():
                if isinstance(value, (int, float)):
                    feature_list.append(value)
                elif isinstance(value, bool):
                    feature_list.append(1.0 if value else 0.0)
            
            # TF-IDF features if available
            if self.tfidf_vectorizer:
                try:
                    tfidf_features = self.tfidf_vectorizer.transform([text]).toarray()[0]
                    feature_list.extend(tfidf_features)
                except:
                    pass
            
            return np.array(feature_list)
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return np.array([0.0] * 100)  # Default feature vector

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level based on score"""
        
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    async def _store_classification_result(self, result: ClassificationResult):
        """Store classification result in database"""
        
        try:
            with self.Session() as session:
                record = ClassificationRecord(
                    classification_id=result.classification_id,
                    document_id=result.metadata.get('document_id'),
                    document_type=result.document_type.value,
                    confidence=result.confidence,
                    confidence_level=result.confidence_level.value,
                    method_used=result.method_used.value,
                    all_predictions=result.all_predictions,
                    features_used=result.features_used,
                    processing_time=result.processing_time,
                    model_version=result.model_version,
                    error_message=result.error_message,
                    created_at=datetime.utcnow(),
                    metadata=result.metadata
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing classification result: {e}")

    async def train_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train classification models on provided data"""
        
        try:
            if not training_data:
                raise ValueError("No training data provided")
            
            logger.info(f"Training models on {len(training_data)} samples")
            
            # Prepare training data
            texts = [item['text'] for item in training_data]
            labels = [item['label'] for item in training_data]
            
            # Fit label encoder
            self.label_encoder.fit(labels)
            encoded_labels = self.label_encoder.transform(labels)
            
            # Extract features for all samples
            feature_vectors = []
            for item in training_data:
                features = await self._extract_features(item['text'], item.get('metadata', {}))
                feature_vector = self._prepare_feature_vector(item['text'], features)
                feature_vectors.append(feature_vector)
            
            X = np.array(feature_vectors)
            y = encoded_labels
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.tfidf_vectorizer.fit(texts)
            
            # Train multiple models
            models_to_train = {
                'random_forest_classifier': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting_classifier': GradientBoostingClassifier(random_state=42),
                'svm_classifier': SVC(probability=True, random_state=42),
                'logistic_regression_classifier': LogisticRegression(random_state=42, max_iter=1000),
                'neural_network_classifier': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            }
            
            training_results = {}
            
            for model_name, model in models_to_train.items():
                try:
                    start_time = datetime.utcnow()
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Store trained model
                    self.trained_models[model_name] = model
                    
                    # Save model to disk
                    model_path = os.path.join(self.models_path, f'{model_name}.pkl')
                    joblib.dump(model, model_path)
                    
                    training_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    training_results[model_name] = {
                        'accuracy': accuracy,
                        'training_time': training_time,
                        'model_path': model_path
                    }
                    
                    # Store training record
                    await self._store_training_record(
                        model_name, 'sklearn', len(training_data),
                        accuracy, training_time, model_path
                    )
                    
                    logger.info(f"Trained {model_name}: accuracy={accuracy:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
            
            # Save feature extractors
            joblib.dump(self.tfidf_vectorizer, os.path.join(self.models_path, 'tfidf_vectorizer.pkl'))
            joblib.dump(self.label_encoder, os.path.join(self.models_path, 'label_encoder.pkl'))
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}

    async def _store_training_record(self, model_name: str, model_type: str, data_size: int,
                                   accuracy: float, training_time: float, model_path: str):
        """Store model training record"""
        
        try:
            with self.Session() as session:
                record = TrainingRecord(
                    training_id=str(uuid.uuid4()),
                    model_name=model_name,
                    model_type=model_type,
                    training_data_size=data_size,
                    accuracy=accuracy,
                    precision=0.0,  # Would calculate from classification report
                    recall=0.0,     # Would calculate from classification report
                    f1_score=0.0,   # Would calculate from classification report
                    training_time=training_time,
                    hyperparameters={},
                    feature_importance={},
                    created_at=datetime.utcnow(),
                    model_path=model_path
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing training record: {e}")

    async def get_classification_statistics(self) -> Dict[str, Any]:
        """Get classification statistics"""
        
        try:
            with self.Session() as session:
                total_classifications = session.query(ClassificationRecord).count()
                
                # Classifications by document type
                type_stats = {}
                for doc_type in DocumentType:
                    count = session.query(ClassificationRecord).filter(
                        ClassificationRecord.document_type == doc_type.value
                    ).count()
                    type_stats[doc_type.value] = count
                
                # Classifications by confidence level
                confidence_stats = {}
                for conf_level in ConfidenceLevel:
                    count = session.query(ClassificationRecord).filter(
                        ClassificationRecord.confidence_level == conf_level.value
                    ).count()
                    confidence_stats[conf_level.value] = count
                
                # Average confidence and processing time
                records = session.query(
                    ClassificationRecord.confidence,
                    ClassificationRecord.processing_time
                ).all()
                
                avg_confidence = sum(r[0] for r in records) / len(records) if records else 0
                avg_processing_time = sum(r[1] for r in records) / len(records) if records else 0
                
                return {
                    "total_classifications": total_classifications,
                    "classifications_by_type": type_stats,
                    "classifications_by_confidence": confidence_stats,
                    "average_confidence": round(avg_confidence, 3),
                    "average_processing_time": round(avg_processing_time, 3),
                    "available_models": list(self.trained_models.keys())
                }
                
        except Exception as e:
            logger.error(f"Error getting classification statistics: {e}")
            return {}

# Factory function
def create_document_classifier(db_url: str = None, redis_url: str = None, models_path: str = None) -> DocumentClassifier:
    """Create and configure DocumentClassifier instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    if not models_path:
        models_path = "/tmp/insurance_classification_models"
    
    return DocumentClassifier(db_url=db_url, redis_url=redis_url, models_path=models_path)

# Example usage
if __name__ == "__main__":
    async def test_document_classifier():
        """Test document classifier functionality"""
        
        classifier = create_document_classifier()
        
        # Test documents
        test_documents = [
            {
                'text': """
                INSURANCE POLICY APPLICATION
                
                Applicant Name: John Smith
                Date of Birth: 01/15/1980
                Policy Number: POL-2024-001234
                Coverage Amount: $500,000
                Premium: $2,400 annually
                Effective Date: 01/01/2024
                
                I hereby apply for insurance coverage as specified above.
                
                Signature: _________________ Date: _________
                """,
                'expected': DocumentType.POLICY_APPLICATION
            },
            {
                'text': """
                CLAIM FORM
                
                Claim Number: CLM-2024-005678
                Claimant: Jane Doe
                Date of Loss: 03/15/2024
                Description of Loss: Vehicle accident at Main St intersection
                Estimated Damage: $8,500
                
                I certify that the above information is true and correct.
                
                Claimant Signature: _________________ Date: _________
                """,
                'expected': DocumentType.CLAIM_FORM
            }
        ]
        
        try:
            for i, doc in enumerate(test_documents):
                print(f"\n--- Testing Document {i+1} ---")
                
                # Test different classification methods
                methods = [
                    ClassificationMethod.RULE_BASED,
                    ClassificationMethod.MACHINE_LEARNING,
                    ClassificationMethod.ENSEMBLE
                ]
                
                for method in methods:
                    result = await classifier.classify_document(
                        doc['text'],
                        metadata={'document_id': f'test_{i+1}'},
                        method=method
                    )
                    
                    print(f"{method.value}:")
                    print(f"  Predicted: {result.document_type.value}")
                    print(f"  Expected: {doc['expected'].value}")
                    print(f"  Confidence: {result.confidence:.3f}")
                    print(f"  Confidence Level: {result.confidence_level.value}")
                    print(f"  Processing Time: {result.processing_time:.3f}s")
                    
                    # Check if prediction is correct
                    correct = result.document_type == doc['expected']
                    print(f"  Correct: {correct}")
            
            # Get statistics
            stats = await classifier.get_classification_statistics()
            print(f"\nClassification Statistics: {stats}")
            
        except Exception as e:
            print(f"Error in test: {e}")
    
    # Run test
    # asyncio.run(test_document_classifier())

