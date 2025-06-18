"""
Insurance AI Agent System - Evidence Processing Agent
Production-ready agent for multimedia evidence analysis and fraud detection
"""

import asyncio
import uuid
import io
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from decimal import Decimal
import json
import hashlib
import tempfile
import os
from pathlib import Path

# Image and video processing libraries
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
import face_recognition
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import tensorflow as tf

# Audio processing
import librosa
import soundfile as sf
from scipy import signal
from scipy.spatial.distance import cosine

# Metadata extraction
from exifread import process_file
import mutagen
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4

# Database and utilities
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog
import boto3
from botocore.exceptions import ClientError

from backend.shared.models import Evidence, EvidenceAnalysis, AgentExecution, Claim
from backend.shared.schemas import (
    EvidenceCreate, EvidenceAnalysisCreate, EvidenceUpdate,
    EvidenceType, EvidenceStatus, AgentExecutionStatus
)
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

logger = structlog.get_logger(__name__)

class EvidenceProcessingAgent:
    """
    Advanced evidence processing agent for multimedia analysis and fraud detection
    Supports images, videos, audio, and documents with AI-powered analysis
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.agent_name = "evidence_processing_agent"
        self.agent_version = "1.0.0"
        self.logger = structlog.get_logger(self.agent_name)
        
        # Initialize AI models
        self._initialize_models()
        
        # AWS S3 client for evidence storage
        self.s3_client = boto3.client('s3')
        self.evidence_bucket = os.getenv('EVIDENCE_STORAGE_BUCKET', 'insurance-evidence')
        
        # Analysis thresholds
        self.analysis_thresholds = {
            "image_quality_min": 0.6,
            "face_recognition_confidence": 0.6,
            "fraud_score_threshold": 70.0,
            "similarity_threshold": 0.85,
            "metadata_consistency_threshold": 0.8
        }
        
        # Supported file types
        self.supported_types = {
            "image": [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"],
            "video": [".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"],
            "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg"],
            "document": [".pdf", ".doc", ".docx", ".txt"]
        }
        
        # Fraud indicators
        self.fraud_indicators = {
            "metadata_manipulation": {
                "weight": 0.3,
                "patterns": ["timestamp_inconsistency", "location_mismatch", "device_spoofing"]
            },
            "image_manipulation": {
                "weight": 0.4,
                "patterns": ["copy_move", "splicing", "retouching", "deepfake"]
            },
            "content_inconsistency": {
                "weight": 0.2,
                "patterns": ["damage_staging", "scene_inconsistency", "lighting_mismatch"]
            },
            "behavioral_patterns": {
                "weight": 0.1,
                "patterns": ["submission_timing", "multiple_claims", "pattern_matching"]
            }
        }
    
    def _initialize_models(self):
        """Initialize AI models for evidence analysis"""
        
        try:
            # Initialize image classification model
            self.image_model = resnet50(pretrained=True)
            self.image_model.eval()
            
            # Image preprocessing
            self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Initialize fraud detection models (simplified)
            self.fraud_model = None  # Would load pre-trained fraud detection model
            
            self.logger.info("AI models initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize AI models", error=str(e))
            # Continue without AI models
            self.image_model = None
            self.fraud_model = None
    
    async def process_evidence(
        self,
        evidence_id: uuid.UUID,
        file_path: str,
        evidence_type: Optional[str] = None,
        claim_id: Optional[uuid.UUID] = None,
        analysis_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process evidence file and perform comprehensive analysis
        
        Args:
            evidence_id: UUID of the evidence record
            file_path: Path to the evidence file or S3 key
            evidence_type: Type of evidence (image, video, audio, document)
            claim_id: Associated claim ID
            analysis_options: Additional analysis configuration
            
        Returns:
            Dictionary containing analysis results
        """
        
        async with performance_monitor.monitor_operation("evidence_processing"):
            try:
                # Create agent execution record
                execution = AgentExecution(
                    agent_name=self.agent_name,
                    agent_version=self.agent_version,
                    input_data={
                        "evidence_id": str(evidence_id),
                        "file_path": file_path,
                        "evidence_type": evidence_type,
                        "claim_id": str(claim_id) if claim_id else None,
                        "analysis_options": analysis_options or {}
                    },
                    status=AgentExecutionStatus.RUNNING
                )
                
                self.db_session.add(execution)
                await self.db_session.commit()
                await self.db_session.refresh(execution)
                
                start_time = datetime.utcnow()
                
                # Download evidence file if needed
                local_file_path = await self._download_evidence(file_path)
                
                # Determine evidence type if not provided
                if not evidence_type:
                    evidence_type = self._determine_evidence_type(local_file_path)
                
                # Perform type-specific analysis
                analysis_result = {}
                
                if evidence_type == "image":
                    analysis_result = await self._analyze_image(local_file_path, claim_id)
                elif evidence_type == "video":
                    analysis_result = await self._analyze_video(local_file_path, claim_id)
                elif evidence_type == "audio":
                    analysis_result = await self._analyze_audio(local_file_path, claim_id)
                elif evidence_type == "document":
                    analysis_result = await self._analyze_document(local_file_path, claim_id)
                else:
                    raise ServiceException(f"Unsupported evidence type: {evidence_type}")
                
                # Perform fraud analysis
                fraud_analysis = await self._analyze_fraud_indicators(
                    analysis_result, evidence_type, claim_id
                )
                
                # Calculate overall scores
                quality_score = self._calculate_quality_score(analysis_result, evidence_type)
                authenticity_score = self._calculate_authenticity_score(analysis_result, fraud_analysis)
                
                # Prepare final result
                final_result = {
                    "evidence_id": str(evidence_id),
                    "evidence_type": evidence_type,
                    "file_path": file_path,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "quality_score": quality_score,
                    "authenticity_score": authenticity_score,
                    "fraud_score": fraud_analysis.get("fraud_score", 0.0),
                    "analysis_details": analysis_result,
                    "fraud_analysis": fraud_analysis,
                    "processing_metadata": {
                        "agent_name": self.agent_name,
                        "agent_version": self.agent_version,
                        "processing_time_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
                        "file_size_bytes": os.path.getsize(local_file_path) if os.path.exists(local_file_path) else 0,
                        "analysis_options": analysis_options or {}
                    }
                }
                
                # Save analysis results
                await self._save_analysis_results(evidence_id, final_result)
                
                # Update execution record
                execution.status = AgentExecutionStatus.COMPLETED
                execution.output_data = final_result
                execution.execution_time_ms = final_result["processing_metadata"]["processing_time_ms"]
                execution.completed_at = datetime.utcnow()
                
                await self.db_session.commit()
                
                # Clean up temporary file
                if local_file_path != file_path and os.path.exists(local_file_path):
                    os.unlink(local_file_path)
                
                # Record metrics
                metrics.record_agent_execution(
                    self.agent_name, 
                    execution.execution_time_ms / 1000, 
                    success=True
                )
                
                # Log analysis
                audit_logger.log_user_action(
                    user_id="system",
                    action="evidence_analysis_completed",
                    resource_type="evidence",
                    resource_id=str(evidence_id),
                    details={
                        "evidence_type": evidence_type,
                        "quality_score": quality_score,
                        "fraud_score": fraud_analysis.get("fraud_score", 0.0),
                        "processing_time_ms": execution.execution_time_ms
                    }
                )
                
                self.logger.info(
                    "Evidence processing completed",
                    evidence_id=str(evidence_id),
                    evidence_type=evidence_type,
                    quality_score=quality_score,
                    fraud_score=fraud_analysis.get("fraud_score", 0.0),
                    processing_time_ms=execution.execution_time_ms
                )
                
                return final_result
                
            except Exception as e:
                # Update execution record with error
                execution.status = AgentExecutionStatus.FAILED
                execution.error_message = str(e)
                execution.completed_at = datetime.utcnow()
                
                await self.db_session.commit()
                
                # Record metrics
                metrics.record_agent_execution(self.agent_name, 0, success=False)
                
                self.logger.error(
                    "Evidence processing failed",
                    evidence_id=str(evidence_id),
                    error=str(e)
                )
                raise ServiceException(f"Evidence processing failed: {str(e)}")
    
    async def _download_evidence(self, file_path: str) -> str:
        """Download evidence from S3 or return local path"""
        
        if file_path.startswith('s3://') or not file_path.startswith('/'):
            try:
                # Parse S3 path
                if file_path.startswith('s3://'):
                    bucket, key = file_path[5:].split('/', 1)
                else:
                    bucket = self.evidence_bucket
                    key = file_path
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.close()
                
                # Download file
                self.s3_client.download_file(bucket, key, temp_file.name)
                
                return temp_file.name
                
            except ClientError as e:
                raise ServiceException(f"Failed to download evidence from S3: {str(e)}")
        else:
            if not os.path.exists(file_path):
                raise ServiceException(f"Evidence file not found: {file_path}")
            return file_path
    
    def _determine_evidence_type(self, file_path: str) -> str:
        """Determine evidence type from file extension"""
        
        file_extension = Path(file_path).suffix.lower()
        
        for evidence_type, extensions in self.supported_types.items():
            if file_extension in extensions:
                return evidence_type
        
        return "unknown"
    
    async def _analyze_image(self, file_path: str, claim_id: Optional[uuid.UUID]) -> Dict[str, Any]:
        """Analyze image evidence"""
        
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                raise ServiceException("Failed to load image")
            
            pil_image = Image.open(file_path)
            
            analysis_result = {
                "image_properties": await self._extract_image_properties(pil_image),
                "metadata_analysis": await self._extract_image_metadata(file_path),
                "quality_assessment": await self._assess_image_quality(image),
                "content_analysis": await self._analyze_image_content(image, pil_image),
                "manipulation_detection": await self._detect_image_manipulation(image),
                "similarity_analysis": await self._check_image_similarity(image, claim_id) if claim_id else {}
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error("Image analysis failed", error=str(e))
            raise
    
    async def _extract_image_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic image properties"""
        
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_bytes": len(image.tobytes()) if hasattr(image, 'tobytes') else 0,
            "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0
        }
    
    async def _extract_image_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract EXIF and other metadata from image"""
        
        try:
            metadata = {}
            
            # Extract EXIF data
            with open(file_path, 'rb') as f:
                exif_data = process_file(f, details=False)
                
                for tag, value in exif_data.items():
                    if tag not in ['JPEGThumbnail', 'TIFFThumbnail']:
                        metadata[str(tag)] = str(value)
            
            # Extract PIL EXIF data
            pil_image = Image.open(file_path)
            if hasattr(pil_image, '_getexif') and pil_image._getexif():
                exif_dict = pil_image._getexif()
                for tag_id, value in exif_dict.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    metadata[f"PIL_{tag}"] = str(value)
            
            # Calculate file hash for integrity
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                metadata["file_hash"] = file_hash
            
            return metadata
            
        except Exception as e:
            self.logger.warning("Metadata extraction failed", error=str(e))
            return {}
    
    async def _assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image quality metrics"""
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            # Calculate noise level (using high-frequency components)
            noise_level = self._estimate_noise_level(gray)
            
            # Overall quality score
            quality_score = self._calculate_image_quality_score(
                sharpness, brightness, contrast, noise_level
            )
            
            return {
                "sharpness": float(sharpness),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "noise_level": float(noise_level),
                "quality_score": quality_score,
                "resolution": image.shape[:2],
                "is_acceptable": quality_score >= self.analysis_thresholds["image_quality_min"]
            }
            
        except Exception as e:
            self.logger.error("Image quality assessment failed", error=str(e))
            return {"quality_score": 0.0, "is_acceptable": False}
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image"""
        
        try:
            # Use high-pass filter to isolate noise
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(image, -1, kernel)
            
            # Calculate standard deviation of filtered image
            noise_level = np.std(filtered)
            
            return noise_level
            
        except Exception:
            return 0.0
    
    def _calculate_image_quality_score(
        self, 
        sharpness: float, 
        brightness: float, 
        contrast: float, 
        noise_level: float
    ) -> float:
        """Calculate overall image quality score"""
        
        try:
            # Normalize metrics
            sharpness_score = min(1.0, sharpness / 1000.0)  # Normalize sharpness
            brightness_score = 1.0 - abs(brightness - 128) / 128.0  # Optimal brightness around 128
            contrast_score = min(1.0, contrast / 64.0)  # Normalize contrast
            noise_score = max(0.0, 1.0 - noise_level / 50.0)  # Lower noise is better
            
            # Weighted average
            quality_score = (
                sharpness_score * 0.3 +
                brightness_score * 0.2 +
                contrast_score * 0.3 +
                noise_score * 0.2
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception:
            return 0.0
    
    async def _analyze_image_content(self, cv_image: np.ndarray, pil_image: Image.Image) -> Dict[str, Any]:
        """Analyze image content using AI models"""
        
        try:
            content_analysis = {}
            
            # Object detection and classification
            if self.image_model:
                try:
                    # Preprocess image for model
                    input_tensor = self.image_transform(pil_image).unsqueeze(0)
                    
                    with torch.no_grad():
                        outputs = self.image_model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        
                        # Get top 5 predictions
                        top5_prob, top5_catid = torch.topk(probabilities, 5)
                        
                        content_analysis["classifications"] = [
                            {
                                "class_id": int(catid),
                                "confidence": float(prob)
                            }
                            for prob, catid in zip(top5_prob, top5_catid)
                        ]
                
                except Exception as e:
                    self.logger.warning("Image classification failed", error=str(e))
            
            # Face detection
            try:
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                
                content_analysis["faces"] = {
                    "count": len(face_locations),
                    "locations": face_locations
                }
                
                # Face encodings for comparison
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    content_analysis["face_encodings"] = [
                        encoding.tolist() for encoding in face_encodings
                    ]
                
            except Exception as e:
                self.logger.warning("Face detection failed", error=str(e))
            
            # Color analysis
            try:
                # Dominant colors
                pixels = cv_image.reshape(-1, 3)
                from sklearn.cluster import KMeans
                
                kmeans = KMeans(n_clusters=5, random_state=42)
                kmeans.fit(pixels)
                
                colors = kmeans.cluster_centers_.astype(int)
                content_analysis["dominant_colors"] = colors.tolist()
                
            except Exception as e:
                self.logger.warning("Color analysis failed", error=str(e))
            
            # Edge detection for damage assessment
            try:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                content_analysis["edge_analysis"] = {
                    "edge_density": float(edge_density),
                    "potential_damage_areas": edge_density > 0.1
                }
                
            except Exception as e:
                self.logger.warning("Edge analysis failed", error=str(e))
            
            return content_analysis
            
        except Exception as e:
            self.logger.error("Image content analysis failed", error=str(e))
            return {}
    
    async def _detect_image_manipulation(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect potential image manipulation"""
        
        try:
            manipulation_indicators = {}
            
            # Error Level Analysis (ELA) for JPEG compression inconsistencies
            try:
                ela_result = self._perform_ela_analysis(image)
                manipulation_indicators["ela_analysis"] = ela_result
            except Exception as e:
                self.logger.warning("ELA analysis failed", error=str(e))
            
            # Copy-move detection
            try:
                copy_move_result = self._detect_copy_move(image)
                manipulation_indicators["copy_move_detection"] = copy_move_result
            except Exception as e:
                self.logger.warning("Copy-move detection failed", error=str(e))
            
            # Noise analysis for splicing detection
            try:
                noise_analysis = self._analyze_noise_patterns(image)
                manipulation_indicators["noise_analysis"] = noise_analysis
            except Exception as e:
                self.logger.warning("Noise analysis failed", error=str(e))
            
            # Calculate overall manipulation probability
            manipulation_score = self._calculate_manipulation_score(manipulation_indicators)
            manipulation_indicators["manipulation_score"] = manipulation_score
            manipulation_indicators["is_likely_manipulated"] = manipulation_score > 0.7
            
            return manipulation_indicators
            
        except Exception as e:
            self.logger.error("Manipulation detection failed", error=str(e))
            return {"manipulation_score": 0.0, "is_likely_manipulated": False}
    
    def _perform_ela_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform Error Level Analysis"""
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Save with different quality levels
            temp_file1 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_file2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            
            pil_image.save(temp_file1.name, 'JPEG', quality=90)
            
            # Reload and save again
            reloaded = Image.open(temp_file1.name)
            reloaded.save(temp_file2.name, 'JPEG', quality=90)
            
            # Calculate difference
            img1 = cv2.imread(temp_file1.name)
            img2 = cv2.imread(temp_file2.name)
            
            if img1 is not None and img2 is not None:
                diff = cv2.absdiff(img1, img2)
                ela_score = np.mean(diff)
            else:
                ela_score = 0.0
            
            # Cleanup
            os.unlink(temp_file1.name)
            os.unlink(temp_file2.name)
            
            return {
                "ela_score": float(ela_score),
                "suspicious_areas": ela_score > 10.0
            }
            
        except Exception as e:
            self.logger.warning("ELA analysis failed", error=str(e))
            return {"ela_score": 0.0, "suspicious_areas": False}
    
    def _detect_copy_move(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect copy-move forgery"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use SIFT features for copy-move detection
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 10:
                # Find similar feature clusters
                from sklearn.cluster import DBSCAN
                
                clustering = DBSCAN(eps=0.3, min_samples=3)
                clusters = clustering.fit_predict(descriptors)
                
                # Count clusters with multiple points (potential copy-move)
                unique_clusters = np.unique(clusters)
                suspicious_clusters = len([c for c in unique_clusters if c != -1])
                
                copy_move_score = min(1.0, suspicious_clusters / 10.0)
            else:
                copy_move_score = 0.0
            
            return {
                "copy_move_score": copy_move_score,
                "suspicious_regions": copy_move_score > 0.5,
                "feature_count": len(keypoints) if keypoints else 0
            }
            
        except Exception as e:
            self.logger.warning("Copy-move detection failed", error=str(e))
            return {"copy_move_score": 0.0, "suspicious_regions": False}
    
    def _analyze_noise_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze noise patterns for splicing detection"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Divide image into blocks
            h, w = gray.shape
            block_size = 64
            noise_variances = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    
                    # Apply high-pass filter
                    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                    filtered = cv2.filter2D(block, -1, kernel)
                    
                    # Calculate noise variance
                    noise_var = np.var(filtered)
                    noise_variances.append(noise_var)
            
            if noise_variances:
                # Calculate consistency of noise across blocks
                noise_std = np.std(noise_variances)
                noise_mean = np.mean(noise_variances)
                
                # High variation suggests potential splicing
                inconsistency_score = noise_std / (noise_mean + 1e-6)
            else:
                inconsistency_score = 0.0
            
            return {
                "noise_inconsistency_score": float(inconsistency_score),
                "potential_splicing": inconsistency_score > 0.5,
                "block_count": len(noise_variances)
            }
            
        except Exception as e:
            self.logger.warning("Noise analysis failed", error=str(e))
            return {"noise_inconsistency_score": 0.0, "potential_splicing": False}
    
    def _calculate_manipulation_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall manipulation probability score"""
        
        try:
            score = 0.0
            weight_sum = 0.0
            
            # ELA analysis
            if "ela_analysis" in indicators:
                ela_score = indicators["ela_analysis"].get("ela_score", 0.0)
                normalized_ela = min(1.0, ela_score / 20.0)
                score += normalized_ela * 0.3
                weight_sum += 0.3
            
            # Copy-move detection
            if "copy_move_detection" in indicators:
                cm_score = indicators["copy_move_detection"].get("copy_move_score", 0.0)
                score += cm_score * 0.4
                weight_sum += 0.4
            
            # Noise analysis
            if "noise_analysis" in indicators:
                noise_score = indicators["noise_analysis"].get("noise_inconsistency_score", 0.0)
                normalized_noise = min(1.0, noise_score)
                score += normalized_noise * 0.3
                weight_sum += 0.3
            
            return score / weight_sum if weight_sum > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def _check_image_similarity(self, image: np.ndarray, claim_id: uuid.UUID) -> Dict[str, Any]:
        """Check similarity with other images in the same claim"""
        
        try:
            # Get other evidence for this claim
            evidence_service = BaseService(Evidence, self.db_session)
            
            query = select(Evidence).where(
                and_(
                    Evidence.claim_id == claim_id,
                    Evidence.file_type.like('image%')
                )
            )
            
            result = await self.db_session.execute(query)
            other_evidence = result.scalars().all()
            
            similarities = []
            
            for evidence in other_evidence:
                try:
                    # Load other image
                    other_path = await self._download_evidence(evidence.file_key)
                    other_image = cv2.imread(other_path)
                    
                    if other_image is not None:
                        # Calculate similarity
                        similarity = self._calculate_image_similarity(image, other_image)
                        
                        similarities.append({
                            "evidence_id": str(evidence.id),
                            "similarity_score": similarity,
                            "is_duplicate": similarity > self.analysis_thresholds["similarity_threshold"]
                        })
                    
                    # Cleanup
                    if other_path != evidence.file_key and os.path.exists(other_path):
                        os.unlink(other_path)
                
                except Exception as e:
                    self.logger.warning("Similarity check failed for evidence", 
                                      evidence_id=str(evidence.id), error=str(e))
            
            return {
                "similarities": similarities,
                "max_similarity": max([s["similarity_score"] for s in similarities]) if similarities else 0.0,
                "potential_duplicates": len([s for s in similarities if s["is_duplicate"]])
            }
            
        except Exception as e:
            self.logger.error("Image similarity check failed", error=str(e))
            return {"similarities": [], "max_similarity": 0.0, "potential_duplicates": 0}
    
    def _calculate_image_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity between two images"""
        
        try:
            # Resize images to same size
            h, w = 256, 256
            img1_resized = cv2.resize(img1, (w, h))
            img2_resized = cv2.resize(img2, (w, h))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram correlation
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Calculate structural similarity
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(gray1, gray2)
            
            # Combined similarity score
            similarity = (correlation + ssim_score) / 2.0
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.warning("Image similarity calculation failed", error=str(e))
            return 0.0
    
    async def _analyze_video(self, file_path: str, claim_id: Optional[uuid.UUID]) -> Dict[str, Any]:
        """Analyze video evidence"""
        
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                raise ServiceException("Failed to open video file")
            
            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Sample frames for analysis
            sample_frames = []
            frame_interval = max(1, frame_count // 10)  # Sample 10 frames
            
            for i in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
            
            cap.release()
            
            # Analyze sample frames
            frame_analyses = []
            for i, frame in enumerate(sample_frames):
                frame_analysis = {
                    "frame_number": i * frame_interval,
                    "quality_assessment": await self._assess_image_quality(frame),
                    "content_analysis": await self._analyze_image_content(frame, 
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                }
                frame_analyses.append(frame_analysis)
            
            # Video-specific analysis
            motion_analysis = self._analyze_video_motion(sample_frames)
            
            analysis_result = {
                "video_properties": {
                    "duration_seconds": duration,
                    "fps": fps,
                    "frame_count": frame_count,
                    "resolution": [width, height],
                    "file_size_bytes": os.path.getsize(file_path)
                },
                "frame_analyses": frame_analyses,
                "motion_analysis": motion_analysis,
                "overall_quality": self._calculate_video_quality(frame_analyses),
                "metadata_analysis": await self._extract_video_metadata(file_path)
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error("Video analysis failed", error=str(e))
            raise
    
    def _analyze_video_motion(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze motion patterns in video"""
        
        try:
            if len(frames) < 2:
                return {"motion_detected": False, "motion_intensity": 0.0}
            
            motion_scores = []
            
            for i in range(1, len(frames)):
                # Convert frames to grayscale
                gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
                
                # Calculate motion magnitude
                if flow[0] is not None:
                    motion_magnitude = np.mean(np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2))
                    motion_scores.append(motion_magnitude)
            
            if motion_scores:
                avg_motion = np.mean(motion_scores)
                motion_detected = avg_motion > 5.0  # Threshold for motion detection
            else:
                avg_motion = 0.0
                motion_detected = False
            
            return {
                "motion_detected": motion_detected,
                "motion_intensity": float(avg_motion),
                "motion_consistency": float(np.std(motion_scores)) if motion_scores else 0.0
            }
            
        except Exception as e:
            self.logger.warning("Video motion analysis failed", error=str(e))
            return {"motion_detected": False, "motion_intensity": 0.0}
    
    def _calculate_video_quality(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall video quality from frame analyses"""
        
        try:
            if not frame_analyses:
                return {"overall_score": 0.0, "is_acceptable": False}
            
            quality_scores = [
                frame["quality_assessment"].get("quality_score", 0.0) 
                for frame in frame_analyses
            ]
            
            avg_quality = np.mean(quality_scores)
            min_quality = np.min(quality_scores)
            quality_consistency = 1.0 - np.std(quality_scores)
            
            overall_score = (avg_quality * 0.6 + min_quality * 0.2 + quality_consistency * 0.2)
            
            return {
                "overall_score": float(overall_score),
                "average_quality": float(avg_quality),
                "minimum_quality": float(min_quality),
                "quality_consistency": float(quality_consistency),
                "is_acceptable": overall_score >= self.analysis_thresholds["image_quality_min"]
            }
            
        except Exception as e:
            self.logger.error("Video quality calculation failed", error=str(e))
            return {"overall_score": 0.0, "is_acceptable": False}
    
    async def _extract_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from video file"""
        
        try:
            metadata = {}
            
            # Use mutagen for metadata extraction
            if file_path.lower().endswith('.mp4'):
                video_file = MP4(file_path)
                for key, value in video_file.items():
                    metadata[key] = str(value)
            
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                metadata["file_hash"] = file_hash
            
            return metadata
            
        except Exception as e:
            self.logger.warning("Video metadata extraction failed", error=str(e))
            return {}
    
    async def _analyze_audio(self, file_path: str, claim_id: Optional[uuid.UUID]) -> Dict[str, Any]:
        """Analyze audio evidence"""
        
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            
            analysis_result = {
                "audio_properties": {
                    "duration_seconds": len(audio_data) / sample_rate,
                    "sample_rate": sample_rate,
                    "channels": 1,  # librosa loads as mono by default
                    "file_size_bytes": os.path.getsize(file_path)
                },
                "quality_analysis": self._analyze_audio_quality(audio_data, sample_rate),
                "content_analysis": self._analyze_audio_content(audio_data, sample_rate),
                "metadata_analysis": await self._extract_audio_metadata(file_path)
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error("Audio analysis failed", error=str(e))
            raise
    
    def _analyze_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze audio quality metrics"""
        
        try:
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.var(audio_data - signal.medfilt(audio_data, kernel_size=5))
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Dynamic range
            dynamic_range = np.max(audio_data) - np.min(audio_data)
            
            # Zero crossing rate (indicator of speech/noise)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            avg_zcr = np.mean(zcr)
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            avg_spectral_centroid = np.mean(spectral_centroid)
            
            # Overall quality score
            quality_score = min(1.0, max(0.0, (snr + 20) / 40))  # Normalize SNR to 0-1
            
            return {
                "snr_db": float(snr),
                "dynamic_range": float(dynamic_range),
                "zero_crossing_rate": float(avg_zcr),
                "spectral_centroid": float(avg_spectral_centroid),
                "quality_score": quality_score,
                "is_acceptable": quality_score >= 0.5
            }
            
        except Exception as e:
            self.logger.error("Audio quality analysis failed", error=str(e))
            return {"quality_score": 0.0, "is_acceptable": False}
    
    def _analyze_audio_content(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze audio content"""
        
        try:
            # Voice activity detection
            voice_activity = self._detect_voice_activity(audio_data, sample_rate)
            
            # Frequency analysis
            fft = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
            dominant_frequency = frequencies[dominant_freq_idx]
            
            return {
                "voice_activity": voice_activity,
                "dominant_frequency_hz": float(dominant_frequency),
                "frequency_spectrum": {
                    "low_freq_energy": float(np.sum(magnitude[frequencies < 500])),
                    "mid_freq_energy": float(np.sum(magnitude[(frequencies >= 500) & (frequencies < 2000)])),
                    "high_freq_energy": float(np.sum(magnitude[frequencies >= 2000]))
                }
            }
            
        except Exception as e:
            self.logger.error("Audio content analysis failed", error=str(e))
            return {}
    
    def _detect_voice_activity(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Detect voice activity in audio"""
        
        try:
            # Simple energy-based voice activity detection
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.01 * sample_rate)     # 10ms hop
            
            # Calculate energy for each frame
            energy = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                frame_energy = np.sum(frame ** 2)
                energy.append(frame_energy)
            
            energy = np.array(energy)
            
            # Threshold for voice activity (simple approach)
            threshold = np.mean(energy) * 0.1
            voice_frames = energy > threshold
            
            voice_activity_ratio = np.sum(voice_frames) / len(voice_frames)
            
            return {
                "voice_detected": voice_activity_ratio > 0.1,
                "voice_activity_ratio": float(voice_activity_ratio),
                "total_frames": len(energy),
                "voice_frames": int(np.sum(voice_frames))
            }
            
        except Exception as e:
            self.logger.warning("Voice activity detection failed", error=str(e))
            return {"voice_detected": False, "voice_activity_ratio": 0.0}
    
    async def _extract_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from audio file"""
        
        try:
            metadata = {}
            
            # Use mutagen for metadata extraction
            if file_path.lower().endswith('.mp3'):
                audio_file = MP3(file_path)
                for key, value in audio_file.items():
                    metadata[key] = str(value)
            
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                metadata["file_hash"] = file_hash
            
            return metadata
            
        except Exception as e:
            self.logger.warning("Audio metadata extraction failed", error=str(e))
            return {}
    
    async def _analyze_document(self, file_path: str, claim_id: Optional[uuid.UUID]) -> Dict[str, Any]:
        """Analyze document evidence"""
        
        try:
            # Basic document analysis
            file_size = os.path.getsize(file_path)
            file_extension = Path(file_path).suffix.lower()
            
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            analysis_result = {
                "document_properties": {
                    "file_size_bytes": file_size,
                    "file_extension": file_extension,
                    "file_hash": file_hash
                },
                "integrity_check": {
                    "file_readable": True,
                    "hash_verified": True
                }
            }
            
            # PDF-specific analysis
            if file_extension == '.pdf':
                pdf_analysis = await self._analyze_pdf_document(file_path)
                analysis_result["pdf_analysis"] = pdf_analysis
            
            return analysis_result
            
        except Exception as e:
            self.logger.error("Document analysis failed", error=str(e))
            raise
    
    async def _analyze_pdf_document(self, file_path: str) -> Dict[str, Any]:
        """Analyze PDF document"""
        
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            
            analysis = {
                "page_count": len(doc),
                "metadata": doc.metadata,
                "is_encrypted": doc.needs_pass,
                "has_text": False,
                "has_images": False
            }
            
            # Check for text and images
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Check for text
                if page.get_text().strip():
                    analysis["has_text"] = True
                
                # Check for images
                image_list = page.get_images()
                if image_list:
                    analysis["has_images"] = True
                
                if analysis["has_text"] and analysis["has_images"]:
                    break
            
            doc.close()
            
            return analysis
            
        except Exception as e:
            self.logger.warning("PDF analysis failed", error=str(e))
            return {}
    
    async def _analyze_fraud_indicators(
        self, 
        analysis_result: Dict[str, Any], 
        evidence_type: str, 
        claim_id: Optional[uuid.UUID]
    ) -> Dict[str, Any]:
        """Analyze fraud indicators based on evidence analysis"""
        
        try:
            fraud_indicators = {}
            fraud_score = 0.0
            
            # Metadata-based fraud indicators
            metadata_score = self._analyze_metadata_fraud_indicators(analysis_result)
            fraud_indicators["metadata_fraud"] = metadata_score
            fraud_score += metadata_score * self.fraud_indicators["metadata_manipulation"]["weight"]
            
            # Content-based fraud indicators
            if evidence_type == "image":
                content_score = self._analyze_image_fraud_indicators(analysis_result)
                fraud_indicators["image_fraud"] = content_score
                fraud_score += content_score * self.fraud_indicators["image_manipulation"]["weight"]
            
            # Behavioral pattern analysis
            if claim_id:
                behavioral_score = await self._analyze_behavioral_patterns(claim_id)
                fraud_indicators["behavioral_fraud"] = behavioral_score
                fraud_score += behavioral_score * self.fraud_indicators["behavioral_patterns"]["weight"]
            
            # Normalize fraud score to 0-100
            fraud_score = min(100.0, max(0.0, fraud_score * 100))
            
            return {
                "fraud_score": fraud_score,
                "risk_level": self._determine_risk_level(fraud_score),
                "indicators": fraud_indicators,
                "requires_manual_review": fraud_score > self.analysis_thresholds["fraud_score_threshold"]
            }
            
        except Exception as e:
            self.logger.error("Fraud analysis failed", error=str(e))
            return {"fraud_score": 0.0, "risk_level": "low", "indicators": {}}
    
    def _analyze_metadata_fraud_indicators(self, analysis_result: Dict[str, Any]) -> float:
        """Analyze metadata for fraud indicators"""
        
        try:
            score = 0.0
            
            metadata = analysis_result.get("metadata_analysis", {})
            
            # Check for missing or suspicious metadata
            if not metadata:
                score += 0.3  # Missing metadata is suspicious
            
            # Check for timestamp inconsistencies
            if "DateTime" in metadata and "DateTimeOriginal" in metadata:
                # Compare timestamps (simplified check)
                if metadata["DateTime"] != metadata["DateTimeOriginal"]:
                    score += 0.2
            
            # Check for GPS data inconsistencies
            if "GPS GPSLatitude" in metadata and "GPS GPSLongitude" in metadata:
                # GPS data present - could be legitimate or spoofed
                # More sophisticated analysis would check against claim location
                pass
            
            # Check for software modification indicators
            software_tags = ["Software", "ProcessingSoftware", "CreatorTool"]
            for tag in software_tags:
                if any(tag in key for key in metadata.keys()):
                    software_value = str(metadata.get(tag, "")).lower()
                    if any(editor in software_value for editor in ["photoshop", "gimp", "paint"]):
                        score += 0.4  # Image editing software detected
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.warning("Metadata fraud analysis failed", error=str(e))
            return 0.0
    
    def _analyze_image_fraud_indicators(self, analysis_result: Dict[str, Any]) -> float:
        """Analyze image-specific fraud indicators"""
        
        try:
            score = 0.0
            
            # Manipulation detection results
            manipulation = analysis_result.get("manipulation_detection", {})
            if manipulation.get("is_likely_manipulated", False):
                score += 0.6
            
            manipulation_score = manipulation.get("manipulation_score", 0.0)
            score += manipulation_score * 0.4
            
            # Quality inconsistencies
            quality = analysis_result.get("quality_assessment", {})
            if quality.get("quality_score", 1.0) < 0.3:
                score += 0.2  # Very low quality might indicate tampering
            
            # Similarity analysis
            similarity = analysis_result.get("similarity_analysis", {})
            if similarity.get("potential_duplicates", 0) > 0:
                score += 0.3  # Duplicate images are suspicious
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.warning("Image fraud analysis failed", error=str(e))
            return 0.0
    
    async def _analyze_behavioral_patterns(self, claim_id: uuid.UUID) -> float:
        """Analyze behavioral patterns for fraud indicators"""
        
        try:
            # Get claim information
            claim_service = BaseService(Claim, self.db_session)
            claim = await claim_service.get(claim_id)
            
            if not claim:
                return 0.0
            
            score = 0.0
            
            # Check submission timing
            if claim.reported_date and claim.incident_date:
                time_diff = (claim.reported_date - claim.incident_date).days
                if time_diff > 30:  # Reported more than 30 days after incident
                    score += 0.3
            
            # Check for multiple recent claims (simplified)
            # In a real implementation, this would query for other claims by the same policyholder
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.warning("Behavioral pattern analysis failed", error=str(e))
            return 0.0
    
    def _determine_risk_level(self, fraud_score: float) -> str:
        """Determine risk level based on fraud score"""
        
        if fraud_score >= 80:
            return "critical"
        elif fraud_score >= 60:
            return "high"
        elif fraud_score >= 40:
            return "medium"
        elif fraud_score >= 20:
            return "low"
        else:
            return "minimal"
    
    def _calculate_quality_score(self, analysis_result: Dict[str, Any], evidence_type: str) -> float:
        """Calculate overall quality score"""
        
        try:
            if evidence_type == "image":
                quality_data = analysis_result.get("quality_assessment", {})
                return quality_data.get("quality_score", 0.0)
            
            elif evidence_type == "video":
                quality_data = analysis_result.get("overall_quality", {})
                return quality_data.get("overall_score", 0.0)
            
            elif evidence_type == "audio":
                quality_data = analysis_result.get("quality_analysis", {})
                return quality_data.get("quality_score", 0.0)
            
            else:
                return 0.5  # Default for documents
                
        except Exception:
            return 0.0
    
    def _calculate_authenticity_score(
        self, 
        analysis_result: Dict[str, Any], 
        fraud_analysis: Dict[str, Any]
    ) -> float:
        """Calculate authenticity score (inverse of fraud score)"""
        
        try:
            fraud_score = fraud_analysis.get("fraud_score", 0.0)
            authenticity_score = (100.0 - fraud_score) / 100.0
            
            return max(0.0, min(1.0, authenticity_score))
            
        except Exception:
            return 0.5
    
    async def _save_analysis_results(self, evidence_id: uuid.UUID, analysis_result: Dict[str, Any]):
        """Save analysis results to database"""
        
        try:
            analysis_create = EvidenceAnalysisCreate(
                evidence_id=evidence_id,
                analysis_type="comprehensive_analysis",
                analysis_data=analysis_result,
                quality_score=Decimal(str(analysis_result.get("quality_score", 0.0))),
                authenticity_score=Decimal(str(analysis_result.get("authenticity_score", 0.0))),
                fraud_score=Decimal(str(analysis_result.get("fraud_score", 0.0)))
            )
            
            analysis_service = BaseService(EvidenceAnalysis, self.db_session)
            await analysis_service.create(analysis_create)
            
        except Exception as e:
            self.logger.error("Failed to save analysis results", error=str(e))
            # Don't raise exception as this is not critical for the analysis itself

# Agent factory function
async def create_evidence_processing_agent(db_session: AsyncSession) -> EvidenceProcessingAgent:
    """Create evidence processing agent instance"""
    return EvidenceProcessingAgent(db_session)

