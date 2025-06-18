"""
NLP Analyzer - Production Ready Implementation
Advanced Natural Language Processing for insurance document analysis
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import pandas as pd
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# NLP libraries
import spacy
from spacy import displacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span, Token
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer

# Transformers for advanced NLP
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    pipeline, BertTokenizer, BertModel
)
import torch

# Text processing
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
from textblob import TextBlob
import langdetect
from fuzzywuzzy import fuzz, process

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
nlp_operations_total = Counter('nlp_operations_total', 'Total NLP operations', ['operation_type', 'status'])
nlp_processing_duration = Histogram('nlp_processing_duration_seconds', 'Time to process NLP operations')
entity_extraction_accuracy = Gauge('entity_extraction_accuracy', 'Entity extraction accuracy', ['entity_type'])

Base = declarative_base()

class AnalysisType(Enum):
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_CLASSIFICATION = "text_classification"
    KEYWORD_EXTRACTION = "keyword_extraction"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    LANGUAGE_DETECTION = "language_detection"
    READABILITY_ANALYSIS = "readability_analysis"
    SIMILARITY_ANALYSIS = "similarity_analysis"
    INTENT_CLASSIFICATION = "intent_classification"

class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    POLICY_NUMBER = "POLICY_NUMBER"
    CLAIM_NUMBER = "CLAIM_NUMBER"
    VEHICLE_ID = "VEHICLE_ID"
    MEDICAL_CONDITION = "MEDICAL_CONDITION"
    INSURANCE_TERM = "INSURANCE_TERM"

class SentimentLabel(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class DocumentCategory(Enum):
    POLICY = "policy"
    CLAIM = "claim"
    CORRESPONDENCE = "correspondence"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    OTHER = "other"

@dataclass
class Entity:
    """Extracted entity information"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    metadata: Dict[str, Any] = None

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    label: SentimentLabel
    confidence: float
    scores: Dict[str, float]
    
@dataclass
class ClassificationResult:
    """Text classification result"""
    category: str
    confidence: float
    all_scores: Dict[str, float]

@dataclass
class KeywordResult:
    """Keyword extraction result"""
    keywords: List[Dict[str, Any]]
    phrases: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]

@dataclass
class NLPAnalysisResult:
    """Complete NLP analysis result"""
    analysis_id: str
    text: str
    language: str
    entities: List[Entity]
    sentiment: SentimentResult
    classification: ClassificationResult
    keywords: KeywordResult
    summary: str
    readability_scores: Dict[str, float]
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class NLPAnalysisRecord(Base):
    """SQLAlchemy model for NLP analysis records"""
    __tablename__ = 'nlp_analyses'
    
    analysis_id = Column(String, primary_key=True)
    input_text = Column(Text, nullable=False)
    language = Column(String)
    entities = Column(JSON)
    sentiment = Column(JSON)
    classification = Column(JSON)
    keywords = Column(JSON)
    summary = Column(Text)
    readability_scores = Column(JSON)
    processing_time = Column(Float)
    error_message = Column(Text)
    created_at = Column(DateTime, nullable=False)
    metadata = Column(JSON)

class NLPAnalyzer:
    """
    Production-ready NLP Analyzer
    Advanced natural language processing for insurance documents
    """
    
    def __init__(self, db_url: str, redis_url: str, models_path: str = "/tmp/nlp_models"):
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
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        # Insurance-specific patterns and rules
        self._initialize_insurance_patterns()
        
        logger.info("NLPAnalyzer initialized successfully")

    def _initialize_nlp_components(self):
        """Initialize NLP models and components"""
        
        try:
            # Download required NLTK data
            nltk_downloads = [
                'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'maxent_ne_chunker', 'words', 'vader_lexicon'
            ]
            
            for item in nltk_downloads:
                try:
                    nltk.download(item, quiet=True)
                except:
                    pass
            
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found, using basic NLP processing")
                self.nlp = None
            
            # Initialize transformers models
            self._initialize_transformer_models()
            
            # Setup custom matchers
            if self.nlp:
                self.matcher = Matcher(self.nlp.vocab)
                self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
                self._setup_custom_matchers()
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")

    def _initialize_transformer_models(self):
        """Initialize transformer models for advanced NLP tasks"""
        
        try:
            # Sentiment analysis model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Text classification model
            self.classification_pipeline = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
            
            # Named Entity Recognition model
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            # Question Answering model
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2"
            )
            
            # Summarization model
            self.summarization_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            
            logger.info("Transformer models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Some transformer models could not be loaded: {e}")
            # Initialize fallback models
            self.sentiment_pipeline = None
            self.classification_pipeline = None
            self.ner_pipeline = None
            self.qa_pipeline = None
            self.summarization_pipeline = None

    def _initialize_insurance_patterns(self):
        """Initialize insurance-specific patterns and vocabularies"""
        
        # Insurance terminology
        self.insurance_terms = {
            'policy_terms': [
                'premium', 'deductible', 'coverage', 'beneficiary', 'policyholder',
                'underwriting', 'actuarial', 'risk assessment', 'liability',
                'comprehensive', 'collision', 'uninsured motorist'
            ],
            'claim_terms': [
                'claim', 'claimant', 'adjuster', 'settlement', 'damages',
                'loss', 'incident', 'accident', 'injury', 'repair',
                'replacement cost', 'actual cash value'
            ],
            'medical_terms': [
                'diagnosis', 'treatment', 'physician', 'hospital', 'surgery',
                'medication', 'therapy', 'rehabilitation', 'disability',
                'pre-existing condition', 'medical necessity'
            ],
            'legal_terms': [
                'liability', 'negligence', 'fault', 'damages', 'settlement',
                'litigation', 'subrogation', 'indemnity', 'tort',
                'statute of limitations', 'burden of proof'
            ]
        }
        
        # Regex patterns for specific entities
        self.entity_patterns = {
            'policy_number': [
                r'\b[A-Z]{2,4}[-]?\d{6,12}\b',
                r'\bPOL[-]?\d{6,12}\b',
                r'\bPolicy\s+(?:Number|#)?\s*:?\s*([A-Z0-9-]+)\b'
            ],
            'claim_number': [
                r'\b[A-Z]{2,4}[-]?\d{6,12}\b',
                r'\bCLM[-]?\d{6,12}\b',
                r'\bClaim\s+(?:Number|#)?\s*:?\s*([A-Z0-9-]+)\b'
            ],
            'phone_number': [
                r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'ssn': [
                r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'
            ],
            'vin': [
                r'\b[A-HJ-NPR-Z0-9]{17}\b'
            ],
            'license_plate': [
                r'\b[A-Z0-9]{2,8}\b'
            ],
            'currency': [
                r'\$[0-9,]+\.?[0-9]*',
                r'\b\d+\.\d{2}\s*(?:dollars?|USD)\b'
            ],
            'date': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'
            ]
        }
        
        # Intent classification patterns
        self.intent_patterns = {
            'file_claim': [
                'file a claim', 'submit claim', 'report accident', 'report incident',
                'claim submission', 'new claim', 'accident report'
            ],
            'policy_inquiry': [
                'policy information', 'coverage details', 'policy status',
                'premium amount', 'payment due', 'policy renewal'
            ],
            'complaint': [
                'complaint', 'dissatisfied', 'unhappy', 'poor service',
                'escalate', 'manager', 'supervisor'
            ],
            'payment': [
                'make payment', 'pay premium', 'payment method', 'billing',
                'invoice', 'payment due', 'overdue'
            ],
            'coverage_change': [
                'change coverage', 'add coverage', 'remove coverage',
                'update policy', 'modify policy', 'coverage options'
            ]
        }

    def _setup_custom_matchers(self):
        """Setup custom spaCy matchers for insurance entities"""
        
        if not self.nlp:
            return
        
        try:
            # Policy number patterns
            policy_patterns = [
                [{"TEXT": {"REGEX": r"[A-Z]{2,4}"}}, {"TEXT": "-", "OP": "?"}, {"TEXT": {"REGEX": r"\d{6,12}"}}],
                [{"LOWER": "policy"}, {"LOWER": {"IN": ["number", "#", "no"]}}, {"TEXT": {"REGEX": r"[A-Z0-9-]+"}}]
            ]
            self.matcher.add("POLICY_NUMBER", policy_patterns)
            
            # Claim number patterns
            claim_patterns = [
                [{"TEXT": {"REGEX": r"[A-Z]{2,4}"}}, {"TEXT": "-", "OP": "?"}, {"TEXT": {"REGEX": r"\d{6,12}"}}],
                [{"LOWER": "claim"}, {"LOWER": {"IN": ["number", "#", "no"]}}, {"TEXT": {"REGEX": r"[A-Z0-9-]+"}}]
            ]
            self.matcher.add("CLAIM_NUMBER", claim_patterns)
            
            # Insurance terms phrase matcher
            insurance_phrases = []
            for category, terms in self.insurance_terms.items():
                for term in terms:
                    doc = self.nlp(term)
                    insurance_phrases.append(doc)
            
            self.phrase_matcher.add("INSURANCE_TERM", insurance_phrases)
            
            logger.info("Custom matchers setup successfully")
            
        except Exception as e:
            logger.error(f"Error setting up custom matchers: {e}")

    async def analyze_text(self, 
                         text: str, 
                         analysis_types: List[AnalysisType] = None,
                         metadata: Dict[str, Any] = None) -> NLPAnalysisResult:
        """Perform comprehensive NLP analysis on text"""
        
        if analysis_types is None:
            analysis_types = [
                AnalysisType.ENTITY_EXTRACTION,
                AnalysisType.SENTIMENT_ANALYSIS,
                AnalysisType.TEXT_CLASSIFICATION,
                AnalysisType.KEYWORD_EXTRACTION,
                AnalysisType.LANGUAGE_DETECTION,
                AnalysisType.READABILITY_ANALYSIS
            ]
        
        start_time = datetime.utcnow()
        analysis_id = str(uuid.uuid4())
        
        with nlp_processing_duration.time():
            try:
                logger.info(f"Starting NLP analysis for text length: {len(text)}")
                
                # Detect language
                language = await self._detect_language(text)
                
                # Initialize result components
                entities = []
                sentiment = SentimentResult(SentimentLabel.NEUTRAL, 0.0, {})
                classification = ClassificationResult("other", 0.0, {})
                keywords = KeywordResult([], [], [])
                summary = ""
                readability_scores = {}
                
                # Perform requested analyses
                if AnalysisType.ENTITY_EXTRACTION in analysis_types:
                    entities = await self._extract_entities(text)
                
                if AnalysisType.SENTIMENT_ANALYSIS in analysis_types:
                    sentiment = await self._analyze_sentiment(text)
                
                if AnalysisType.TEXT_CLASSIFICATION in analysis_types:
                    classification = await self._classify_text(text)
                
                if AnalysisType.KEYWORD_EXTRACTION in analysis_types:
                    keywords = await self._extract_keywords(text)
                
                if AnalysisType.SUMMARIZATION in analysis_types:
                    summary = await self._summarize_text(text)
                
                if AnalysisType.READABILITY_ANALYSIS in analysis_types:
                    readability_scores = await self._analyze_readability(text)
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Create result
                result = NLPAnalysisResult(
                    analysis_id=analysis_id,
                    text=text,
                    language=language,
                    entities=entities,
                    sentiment=sentiment,
                    classification=classification,
                    keywords=keywords,
                    summary=summary,
                    readability_scores=readability_scores,
                    processing_time=processing_time,
                    metadata=metadata or {}
                )
                
                # Store result
                await self._store_analysis_result(result)
                
                # Update metrics
                nlp_operations_total.labels(
                    operation_type='full_analysis',
                    status='success'
                ).inc()
                
                logger.info(f"NLP analysis completed for {analysis_id} in {processing_time:.2f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in NLP analysis: {e}")
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                error_result = NLPAnalysisResult(
                    analysis_id=analysis_id,
                    text=text,
                    language="unknown",
                    entities=[],
                    sentiment=SentimentResult(SentimentLabel.NEUTRAL, 0.0, {}),
                    classification=ClassificationResult("other", 0.0, {}),
                    keywords=KeywordResult([], [], []),
                    summary="",
                    readability_scores={},
                    processing_time=processing_time,
                    metadata=metadata or {},
                    error_message=str(e)
                )
                
                nlp_operations_total.labels(
                    operation_type='full_analysis',
                    status='failed'
                ).inc()
                
                return error_result

    async def _detect_language(self, text: str) -> str:
        """Detect text language"""
        
        try:
            if len(text.strip()) < 10:
                return "en"  # Default to English for short texts
            
            detected = langdetect.detect(text)
            return detected
            
        except Exception as e:
            logger.warning(f"Error detecting language: {e}")
            return "en"

    async def _extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        
        entities = []
        
        try:
            # Use spaCy for entity extraction
            if self.nlp:
                doc = self.nlp(text)
                
                # Standard spaCy entities
                for ent in doc.ents:
                    entities.append(Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8,  # spaCy doesn't provide confidence scores
                        metadata={"source": "spacy"}
                    ))
                
                # Custom matcher entities
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    span = doc[start:end]
                    label = self.nlp.vocab.strings[match_id]
                    
                    entities.append(Entity(
                        text=span.text,
                        label=label,
                        start=span.start_char,
                        end=span.end_char,
                        confidence=0.9,
                        metadata={"source": "custom_matcher"}
                    ))
                
                # Phrase matcher for insurance terms
                phrase_matches = self.phrase_matcher(doc)
                for match_id, start, end in phrase_matches:
                    span = doc[start:end]
                    
                    entities.append(Entity(
                        text=span.text,
                        label="INSURANCE_TERM",
                        start=span.start_char,
                        end=span.end_char,
                        confidence=0.7,
                        metadata={"source": "phrase_matcher"}
                    ))
            
            # Use transformer model for additional entity extraction
            if self.ner_pipeline:
                try:
                    ner_results = self.ner_pipeline(text)
                    for result in ner_results:
                        entities.append(Entity(
                            text=result['word'],
                            label=result['entity_group'],
                            start=result['start'],
                            end=result['end'],
                            confidence=result['score'],
                            metadata={"source": "transformer"}
                        ))
                except Exception as e:
                    logger.warning(f"Error with transformer NER: {e}")
            
            # Regex-based entity extraction
            regex_entities = await self._extract_regex_entities(text)
            entities.extend(regex_entities)
            
            # Remove duplicates and sort by position
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x.start)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    async def _extract_regex_entities(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        
        entities = []
        
        try:
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entities.append(Entity(
                            text=match.group(),
                            label=entity_type.upper(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8,
                            metadata={"source": "regex", "pattern": pattern}
                        ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in regex entity extraction: {e}")
            return []

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities"""
        
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create a key based on text and position
            key = (entity.text.lower(), entity.start, entity.end)
            
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # Keep the entity with higher confidence
                for i, existing in enumerate(unique_entities):
                    if (existing.text.lower(), existing.start, existing.end) == key:
                        if entity.confidence > existing.confidence:
                            unique_entities[i] = entity
                        break
        
        return unique_entities

    async def _analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze text sentiment"""
        
        try:
            # Use VADER sentiment analyzer (NLTK)
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Determine primary sentiment
            if vader_scores['compound'] >= 0.05:
                primary_sentiment = SentimentLabel.POSITIVE
                confidence = vader_scores['pos']
            elif vader_scores['compound'] <= -0.05:
                primary_sentiment = SentimentLabel.NEGATIVE
                confidence = vader_scores['neg']
            else:
                primary_sentiment = SentimentLabel.NEUTRAL
                confidence = vader_scores['neu']
            
            scores = {
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'compound': vader_scores['compound']
            }
            
            # Use transformer model if available
            if self.sentiment_pipeline:
                try:
                    transformer_results = self.sentiment_pipeline(text[:512])  # Limit text length
                    
                    # Combine results (simple averaging)
                    for result in transformer_results:
                        label = result['label'].lower()
                        if label in scores:
                            scores[label] = (scores[label] + result['score']) / 2
                    
                    # Recalculate primary sentiment
                    max_score = max(scores['positive'], scores['negative'], scores['neutral'])
                    if scores['positive'] == max_score:
                        primary_sentiment = SentimentLabel.POSITIVE
                        confidence = scores['positive']
                    elif scores['negative'] == max_score:
                        primary_sentiment = SentimentLabel.NEGATIVE
                        confidence = scores['negative']
                    else:
                        primary_sentiment = SentimentLabel.NEUTRAL
                        confidence = scores['neutral']
                        
                except Exception as e:
                    logger.warning(f"Error with transformer sentiment: {e}")
            
            return SentimentResult(
                label=primary_sentiment,
                confidence=confidence,
                scores=scores
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                scores={}
            )

    async def _classify_text(self, text: str) -> ClassificationResult:
        """Classify text into document categories"""
        
        try:
            # Rule-based classification using keywords
            category_scores = {}
            
            text_lower = text.lower()
            
            # Score each category based on keyword presence
            for category in DocumentCategory:
                score = 0
                
                if category == DocumentCategory.POLICY:
                    keywords = self.insurance_terms['policy_terms']
                elif category == DocumentCategory.CLAIM:
                    keywords = self.insurance_terms['claim_terms']
                elif category == DocumentCategory.MEDICAL:
                    keywords = self.insurance_terms['medical_terms']
                elif category == DocumentCategory.LEGAL:
                    keywords = self.insurance_terms['legal_terms']
                else:
                    keywords = []
                
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        score += 1
                
                # Normalize score
                if keywords:
                    score = score / len(keywords)
                
                category_scores[category.value] = score
            
            # Find best category
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                best_score = category_scores[best_category]
            else:
                best_category = DocumentCategory.OTHER.value
                best_score = 0.0
            
            # Use transformer model if available
            if self.classification_pipeline:
                try:
                    transformer_results = self.classification_pipeline(text[:512])
                    
                    # This would need a model trained on insurance document categories
                    # For now, we'll use the rule-based approach
                    pass
                    
                except Exception as e:
                    logger.warning(f"Error with transformer classification: {e}")
            
            return ClassificationResult(
                category=best_category,
                confidence=best_score,
                all_scores=category_scores
            )
            
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            return ClassificationResult(
                category=DocumentCategory.OTHER.value,
                confidence=0.0,
                all_scores={}
            )

    async def _extract_keywords(self, text: str) -> KeywordResult:
        """Extract keywords and key phrases from text"""
        
        try:
            keywords = []
            phrases = []
            topics = []
            
            # Use spaCy for keyword extraction
            if self.nlp:
                doc = self.nlp(text)
                
                # Extract important tokens
                keyword_candidates = []
                for token in doc:
                    if (not token.is_stop and 
                        not token.is_punct and 
                        not token.is_space and
                        len(token.text) > 2 and
                        token.pos_ in ['NOUN', 'ADJ', 'VERB']):
                        keyword_candidates.append(token.lemma_.lower())
                
                # Count frequency
                from collections import Counter
                word_freq = Counter(keyword_candidates)
                
                # Get top keywords
                for word, freq in word_freq.most_common(20):
                    keywords.append({
                        'word': word,
                        'frequency': freq,
                        'score': freq / len(keyword_candidates) if keyword_candidates else 0
                    })
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 1:  # Multi-word phrases
                        phrases.append({
                            'phrase': chunk.text,
                            'root': chunk.root.text,
                            'score': 0.8  # Default score
                        })
            
            # Use TextBlob for additional keyword extraction
            try:
                blob = TextBlob(text)
                
                # Extract noun phrases
                for phrase in blob.noun_phrases:
                    if len(phrase.split()) > 1:
                        phrases.append({
                            'phrase': phrase,
                            'root': phrase.split()[0],
                            'score': 0.7
                        })
                        
            except Exception as e:
                logger.warning(f"Error with TextBlob keyword extraction: {e}")
            
            # Identify insurance-specific topics
            text_lower = text.lower()
            for category, terms in self.insurance_terms.items():
                topic_score = 0
                matched_terms = []
                
                for term in terms:
                    if term.lower() in text_lower:
                        topic_score += 1
                        matched_terms.append(term)
                
                if topic_score > 0:
                    topics.append({
                        'topic': category,
                        'score': topic_score / len(terms),
                        'matched_terms': matched_terms
                    })
            
            # Remove duplicates and sort
            keywords = sorted(keywords, key=lambda x: x['score'], reverse=True)[:10]
            phrases = sorted(phrases, key=lambda x: x['score'], reverse=True)[:10]
            topics = sorted(topics, key=lambda x: x['score'], reverse=True)
            
            return KeywordResult(
                keywords=keywords,
                phrases=phrases,
                topics=topics
            )
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return KeywordResult([], [], [])

    async def _summarize_text(self, text: str) -> str:
        """Generate text summary"""
        
        try:
            # Use transformer model if available
            if self.summarization_pipeline:
                try:
                    # Split text into chunks if too long
                    max_length = 1024
                    if len(text) > max_length:
                        # Advanced text chunking with sentence boundary preservation
                        sentences = sent_tokenize(text)
                        chunks = []
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk + sentence) <= max_length:
                                current_chunk += sentence + " "
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sentence + " "
                        
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        summaries = []
                        
                        for chunk in chunks[:5]:  # Process up to 5 chunks for comprehensive coverage
                            if len(chunk.strip()) > 50:
                                summary = self.summarization_pipeline(
                                    chunk,
                                    max_length=150,
                                    min_length=30,
                                    do_sample=False
                                )
                                summaries.append(summary[0]['summary_text'])
                        
                        return ' '.join(summaries)
                    else:
                        summary = self.summarization_pipeline(
                            text,
                            max_length=150,
                            min_length=30,
                            do_sample=False
                        )
                        return summary[0]['summary_text']
                        
                except Exception as e:
                    logger.warning(f"Error with transformer summarization: {e}")
            
            # Fallback: extractive summarization using sentence scoring
            sentences = sent_tokenize(text)
            if len(sentences) <= 3:
                return text
            
            # Score sentences based on word frequency
            word_freq = {}
            words = word_tokenize(text.lower())
            
            for word in words:
                if word not in self.stop_words and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score sentences
            sentence_scores = {}
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                score = 0
                word_count = 0
                
                for word in words:
                    if word in word_freq:
                        score += word_freq[word]
                        word_count += 1
                
                if word_count > 0:
                    sentence_scores[sentence] = score / word_count
            
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            summary_sentences = [sent[0] for sent in top_sentences[:3]]
            
            return ' '.join(summary_sentences)
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return ""

    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Analyze text readability"""
        
        try:
            scores = {}
            
            # Flesch Reading Ease
            scores['flesch_reading_ease'] = flesch_reading_ease(text)
            
            # Flesch-Kincaid Grade Level
            scores['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
            
            # Automated Readability Index
            scores['automated_readability_index'] = automated_readability_index(text)
            
            # Additional metrics
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            scores['sentence_count'] = len(sentences)
            scores['word_count'] = len(words)
            scores['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
            
            # Character count
            scores['character_count'] = len(text)
            scores['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
            
            # Complexity indicators
            complex_words = [word for word in words if len(word) > 6]
            scores['complex_word_ratio'] = len(complex_words) / len(words) if words else 0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error analyzing readability: {e}")
            return {}

    async def _store_analysis_result(self, result: NLPAnalysisResult):
        """Store NLP analysis result in database"""
        
        try:
            with self.Session() as session:
                record = NLPAnalysisRecord(
                    analysis_id=result.analysis_id,
                    input_text=result.text,
                    language=result.language,
                    entities=[asdict(entity) for entity in result.entities],
                    sentiment=asdict(result.sentiment),
                    classification=asdict(result.classification),
                    keywords=asdict(result.keywords),
                    summary=result.summary,
                    readability_scores=result.readability_scores,
                    processing_time=result.processing_time,
                    error_message=result.error_message,
                    created_at=datetime.utcnow(),
                    metadata=result.metadata
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing NLP analysis result: {e}")

    async def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer questions based on context using QA model"""
        
        try:
            if not self.qa_pipeline:
                return {"answer": "QA model not available", "confidence": 0.0}
            
            result = self.qa_pipeline(question=question, context=context)
            
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "start": result['start'],
                "end": result['end']
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {"answer": "Error processing question", "confidence": 0.0}

    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify user intent from text"""
        
        try:
            text_lower = text.lower()
            intent_scores = {}
            
            for intent, patterns in self.intent_patterns.items():
                score = 0
                for pattern in patterns:
                    if pattern.lower() in text_lower:
                        score += 1
                
                if patterns:
                    intent_scores[intent] = score / len(patterns)
            
            if intent_scores:
                best_intent = max(intent_scores, key=intent_scores.get)
                confidence = intent_scores[best_intent]
            else:
                best_intent = "unknown"
                confidence = 0.0
            
            return {
                "intent": best_intent,
                "confidence": confidence,
                "all_scores": intent_scores
            }
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return {"intent": "unknown", "confidence": 0.0, "all_scores": {}}

    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        
        try:
            # Use fuzzy string matching as fallback
            similarity = fuzz.ratio(text1, text2) / 100.0
            
            # Use spaCy for semantic similarity if available
            if self.nlp:
                doc1 = self.nlp(text1)
                doc2 = self.nlp(text2)
                
                if doc1.vector_norm and doc2.vector_norm:
                    semantic_similarity = doc1.similarity(doc2)
                    # Combine with fuzzy matching
                    similarity = (similarity + semantic_similarity) / 2
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get NLP analysis statistics"""
        
        try:
            with self.Session() as session:
                total_analyses = session.query(NLPAnalysisRecord).count()
                
                # Language distribution
                language_stats = {}
                languages = session.query(NLPAnalysisRecord.language).distinct().all()
                for (lang,) in languages:
                    count = session.query(NLPAnalysisRecord).filter(
                        NLPAnalysisRecord.language == lang
                    ).count()
                    language_stats[lang] = count
                
                # Average processing time
                avg_time_result = session.query(NLPAnalysisRecord.processing_time).all()
                avg_processing_time = sum(t[0] for t in avg_time_result if t[0]) / len(avg_time_result) if avg_time_result else 0
                
                return {
                    "total_analyses": total_analyses,
                    "language_distribution": language_stats,
                    "average_processing_time": round(avg_processing_time, 3),
                    "available_models": {
                        "spacy": self.nlp is not None,
                        "sentiment_pipeline": self.sentiment_pipeline is not None,
                        "ner_pipeline": self.ner_pipeline is not None,
                        "qa_pipeline": self.qa_pipeline is not None,
                        "summarization_pipeline": self.summarization_pipeline is not None
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting NLP statistics: {e}")
            return {}

# Factory function
def create_nlp_analyzer(db_url: str = None, redis_url: str = None, models_path: str = None) -> NLPAnalyzer:
    """Create and configure NLPAnalyzer instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    if not models_path:
        models_path = "/tmp/insurance_nlp_models"
    
    return NLPAnalyzer(db_url=db_url, redis_url=redis_url, models_path=models_path)

# Example usage
if __name__ == "__main__":
    async def test_nlp_analyzer():
        """Test NLP analyzer functionality"""
        
        analyzer = create_nlp_analyzer()
        
        # Test text
        test_text = """
        Dear Insurance Company,
        
        I am writing to file a claim for the car accident that occurred on March 15, 2024.
        My policy number is POL-2024-001234. The accident happened at the intersection
        of Main Street and Oak Avenue. The other driver ran a red light and hit my vehicle.
        
        I have attached the police report and photos of the damage. The estimated
        repair cost is $3,500. Please process this claim as soon as possible.
        
        Thank you for your assistance.
        
        John Smith
        Phone: (555) 123-4567
        Email: john.smith@email.com
        """
        
        try:
            # Perform full analysis
            result = await analyzer.analyze_text(test_text)
            
            print(f"NLP Analysis Result:")
            print(f"Analysis ID: {result.analysis_id}")
            print(f"Language: {result.language}")
            print(f"Processing Time: {result.processing_time:.3f}s")
            
            print(f"\nEntities ({len(result.entities)}):")
            for entity in result.entities[:10]:  # Show first 10
                print(f"  {entity.text} ({entity.label}) - {entity.confidence:.2f}")
            
            print(f"\nSentiment: {result.sentiment.label.value} ({result.sentiment.confidence:.2f})")
            print(f"Classification: {result.classification.category} ({result.classification.confidence:.2f})")
            
            print(f"\nKeywords:")
            for kw in result.keywords.keywords[:5]:
                print(f"  {kw['word']} (score: {kw['score']:.2f})")
            
            print(f"\nSummary: {result.summary}")
            
            # Test question answering
            qa_result = await analyzer.answer_question(
                "What is the policy number?", 
                test_text
            )
            print(f"\nQ&A Result: {qa_result}")
            
            # Test intent classification
            intent_result = await analyzer.classify_intent(test_text)
            print(f"Intent: {intent_result}")
            
            # Get statistics
            stats = await analyzer.get_analysis_statistics()
            print(f"Statistics: {stats}")
            
        except Exception as e:
            print(f"Error in test: {e}")
    
    # Run test
    # asyncio.run(test_nlp_analyzer())

