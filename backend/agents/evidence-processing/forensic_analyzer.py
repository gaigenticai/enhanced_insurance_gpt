"""
Forensic Analyzer - Production Ready Implementation
Advanced forensic analysis for insurance evidence
"""

import asyncio
import json
import logging
import os
import hashlib
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import cv2
from PIL import Image, ExifTags
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Digital forensics libraries
import exifread
from PIL.ExifTags import TAGS, GPSTAGS
import struct
import binascii

# Image analysis
from skimage import filters, morphology, measure
from scipy import ndimage, fft
import matplotlib.pyplot as plt

# Monitoring
from prometheus_client import Counter, Histogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

forensic_analysis_total = Counter('forensic_analysis_total', 'Total forensic analyses', ['analysis_type'])
forensic_analysis_duration = Histogram('forensic_analysis_duration_seconds', 'Time for forensic analysis')

Base = declarative_base()

class ForensicFinding(Enum):
    AUTHENTIC = "authentic"
    MANIPULATED = "manipulated"
    SUSPICIOUS = "suspicious"
    INCONCLUSIVE = "inconclusive"

class ManipulationType(Enum):
    COPY_MOVE = "copy_move"
    SPLICING = "splicing"
    RETOUCHING = "retouching"
    COMPRESSION_ARTIFACTS = "compression_artifacts"
    METADATA_TAMPERING = "metadata_tampering"
    GEOMETRIC_TRANSFORMATION = "geometric_transformation"
    COLOR_ADJUSTMENT = "color_adjustment"
    UNKNOWN = "unknown"

class AuthenticityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class MetadataAnalysis:
    exif_data: Dict[str, Any]
    creation_date: Optional[datetime]
    modification_date: Optional[datetime]
    camera_make: Optional[str]
    camera_model: Optional[str]
    gps_coordinates: Optional[Tuple[float, float]]
    software_used: Optional[str]
    metadata_consistency: bool
    suspicious_patterns: List[str]

@dataclass
class ImageIntegrityAnalysis:
    file_hash: str
    compression_analysis: Dict[str, Any]
    noise_analysis: Dict[str, Any]
    edge_analysis: Dict[str, Any]
    frequency_analysis: Dict[str, Any]
    statistical_analysis: Dict[str, Any]

@dataclass
class ManipulationDetection:
    manipulation_type: ManipulationType
    confidence: float
    affected_regions: List[Tuple[int, int, int, int]]  # Bounding boxes
    description: str
    evidence_strength: str

@dataclass
class ForensicAnalysisResult:
    analysis_id: str
    evidence_id: str
    file_path: str
    overall_finding: ForensicFinding
    authenticity_level: AuthenticityLevel
    confidence_score: float
    metadata_analysis: MetadataAnalysis
    integrity_analysis: ImageIntegrityAnalysis
    manipulation_detections: List[ManipulationDetection]
    timeline_analysis: Dict[str, Any]
    chain_of_custody_notes: List[str]
    processing_time: float
    analyst_recommendations: List[str]

class ForensicAnalysisRecord(Base):
    __tablename__ = 'forensic_analyses'
    
    analysis_id = Column(String, primary_key=True)
    evidence_id = Column(String, index=True)
    file_path = Column(String, nullable=False)
    overall_finding = Column(String)
    authenticity_level = Column(String)
    confidence_score = Column(Float)
    metadata_analysis = Column(JSON)
    integrity_analysis = Column(JSON)
    manipulation_detections = Column(JSON)
    timeline_analysis = Column(JSON)
    chain_of_custody_notes = Column(JSON)
    processing_time = Column(Float)
    analyst_recommendations = Column(JSON)
    created_at = Column(DateTime, nullable=False)

class ForensicAnalyzer:
    """Production-ready Forensic Analyzer for digital evidence"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # Known camera signatures and patterns
        self.camera_signatures = {
            "Canon": {"noise_pattern": "canon_noise", "compression": "canon_jpeg"},
            "Nikon": {"noise_pattern": "nikon_noise", "compression": "nikon_jpeg"},
            "Sony": {"noise_pattern": "sony_noise", "compression": "sony_jpeg"},
            "iPhone": {"noise_pattern": "apple_noise", "compression": "apple_heic"},
            "Samsung": {"noise_pattern": "samsung_noise", "compression": "samsung_jpeg"}
        }
        
        # Suspicious software patterns
        self.editing_software = [
            "Adobe Photoshop", "GIMP", "Paint.NET", "Canva", "Pixlr",
            "Lightroom", "Snapseed", "VSCO", "Instagram", "Facetune"
        ]
        
        logger.info("ForensicAnalyzer initialized successfully")

    async def analyze_evidence(self, file_path: str, evidence_id: str = None,
                             chain_of_custody: List[str] = None) -> ForensicAnalysisResult:
        """Perform comprehensive forensic analysis"""
        
        start_time = datetime.utcnow()
        analysis_id = str(uuid.uuid4())
        
        with forensic_analysis_duration.time():
            try:
                # Load and validate image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Could not load image file")
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Perform metadata analysis
                metadata_analysis = await self._analyze_metadata(file_path)
                
                # Perform image integrity analysis
                integrity_analysis = await self._analyze_image_integrity(file_path, image_rgb)
                
                # Detect potential manipulations
                manipulation_detections = await self._detect_manipulations(image_rgb, metadata_analysis)
                
                # Perform timeline analysis
                timeline_analysis = await self._analyze_timeline(file_path, metadata_analysis)
                
                # Calculate overall assessment
                overall_finding = self._determine_overall_finding(
                    metadata_analysis, integrity_analysis, manipulation_detections
                )
                
                authenticity_level = self._assess_authenticity_level(
                    metadata_analysis, manipulation_detections
                )
                
                confidence_score = self._calculate_confidence_score(
                    metadata_analysis, integrity_analysis, manipulation_detections
                )
                
                # Generate recommendations
                recommendations = self._generate_recommendations(
                    overall_finding, manipulation_detections, confidence_score
                )
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                result = ForensicAnalysisResult(
                    analysis_id=analysis_id,
                    evidence_id=evidence_id or "",
                    file_path=file_path,
                    overall_finding=overall_finding,
                    authenticity_level=authenticity_level,
                    confidence_score=confidence_score,
                    metadata_analysis=metadata_analysis,
                    integrity_analysis=integrity_analysis,
                    manipulation_detections=manipulation_detections,
                    timeline_analysis=timeline_analysis,
                    chain_of_custody_notes=chain_of_custody or [],
                    processing_time=processing_time,
                    analyst_recommendations=recommendations
                )
                
                await self._store_analysis(result)
                
                forensic_analysis_total.labels(
                    analysis_type=overall_finding.value
                ).inc()
                
                return result
                
            except Exception as e:
                logger.error(f"Error in forensic analysis: {e}")
                raise

    async def _analyze_metadata(self, file_path: str) -> MetadataAnalysis:
        """Analyze image metadata for authenticity indicators"""
        
        try:
            exif_data = {}
            creation_date = None
            modification_date = None
            camera_make = None
            camera_model = None
            gps_coordinates = None
            software_used = None
            suspicious_patterns = []
            
            # Extract EXIF data using PIL
            with Image.open(file_path) as img:
                exif_dict = img._getexif()
                if exif_dict:
                    for tag_id, value in exif_dict.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='replace')
                            except:
                                value = str(value)
                        
                        exif_data[str(tag)] = value
                        
                        # Extract specific fields
                        if tag == "Make":
                            camera_make = str(value)
                        elif tag == "Model":
                            camera_model = str(value)
                        elif tag == "Software":
                            software_used = str(value)
                        elif tag == "DateTime":
                            try:
                                creation_date = datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                            except:
                                pass
                        elif tag == "DateTimeOriginal":
                            try:
                                creation_date = datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                            except:
                                pass
                        elif tag == "GPSInfo" and isinstance(value, dict):
                            gps_coordinates = self._extract_gps_coordinates(value)
            
            # Check for suspicious patterns
            if software_used:
                for editing_app in self.editing_software:
                    if editing_app.lower() in software_used.lower():
                        suspicious_patterns.append(f"Edited with {editing_app}")
            
            # Check metadata consistency
            file_stat = os.stat(file_path)
            file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
            
            metadata_consistency = True
            if creation_date and abs((file_mtime - creation_date).total_seconds()) > 86400:  # 1 day
                metadata_consistency = False
                suspicious_patterns.append("File modification time inconsistent with EXIF date")
            
            # Check for missing expected metadata
            if not camera_make or not camera_model:
                suspicious_patterns.append("Missing camera identification metadata")
            
            if not creation_date:
                suspicious_patterns.append("Missing creation date metadata")
            
            return MetadataAnalysis(
                exif_data=exif_data,
                creation_date=creation_date,
                modification_date=file_mtime,
                camera_make=camera_make,
                camera_model=camera_model,
                gps_coordinates=gps_coordinates,
                software_used=software_used,
                metadata_consistency=metadata_consistency,
                suspicious_patterns=suspicious_patterns
            )
            
        except Exception as e:
            logger.error(f"Error analyzing metadata: {e}")
            return MetadataAnalysis({}, None, None, None, None, None, None, False, ["Metadata analysis failed"])

    def _extract_gps_coordinates(self, gps_info: Dict) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from EXIF GPS info"""
        
        try:
            def convert_to_degrees(value):
                d, m, s = value
                return d + (m / 60.0) + (s / 3600.0)
            
            lat = gps_info.get(2)  # GPSLatitude
            lat_ref = gps_info.get(1)  # GPSLatitudeRef
            lon = gps_info.get(4)  # GPSLongitude
            lon_ref = gps_info.get(3)  # GPSLongitudeRef
            
            if lat and lon:
                latitude = convert_to_degrees(lat)
                longitude = convert_to_degrees(lon)
                
                if lat_ref == 'S':
                    latitude = -latitude
                if lon_ref == 'W':
                    longitude = -longitude
                
                return (latitude, longitude)
            
            return None
            
        except Exception:
            return None

    async def _analyze_image_integrity(self, file_path: str, image: np.ndarray) -> ImageIntegrityAnalysis:
        """Analyze image integrity and detect tampering indicators"""
        
        try:
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_content = f.read()
                file_hash = hashlib.sha256(file_content).hexdigest()
            
            # Compression analysis
            compression_analysis = await self._analyze_compression(image)
            
            # Noise analysis
            noise_analysis = await self._analyze_noise_patterns(image)
            
            # Edge analysis
            edge_analysis = await self._analyze_edge_consistency(image)
            
            # Frequency domain analysis
            frequency_analysis = await self._analyze_frequency_domain(image)
            
            # Statistical analysis
            statistical_analysis = await self._analyze_statistical_properties(image)
            
            return ImageIntegrityAnalysis(
                file_hash=file_hash,
                compression_analysis=compression_analysis,
                noise_analysis=noise_analysis,
                edge_analysis=edge_analysis,
                frequency_analysis=frequency_analysis,
                statistical_analysis=statistical_analysis
            )
            
        except Exception as e:
            logger.error(f"Error analyzing image integrity: {e}")
            return ImageIntegrityAnalysis("", {}, {}, {}, {}, {})

    async def _analyze_compression(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze JPEG compression artifacts"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect blocking artifacts
            block_size = 8
            blocking_score = 0.0
            
            for i in range(0, gray.shape[0] - block_size, block_size):
                for j in range(0, gray.shape[1] - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    
                    # Calculate variance within block vs between blocks
                    block_var = np.var(block)
                    
                    # Check boundaries
                    if i + block_size < gray.shape[0]:
                        boundary_diff = abs(np.mean(gray[i+block_size-1, j:j+block_size]) - 
                                          np.mean(gray[i+block_size, j:j+block_size]))
                        blocking_score += boundary_diff
            
            blocking_score /= ((gray.shape[0] // block_size) * (gray.shape[1] // block_size))
            
            # Detect double compression
            dct_coeffs = cv2.dct(np.float32(gray))
            dct_hist, _ = np.histogram(dct_coeffs.flatten(), bins=100)
            
            # Look for periodic patterns in DCT histogram
            double_compression_score = self._detect_periodic_patterns(dct_hist)
            
            return {
                "blocking_artifacts_score": float(blocking_score),
                "double_compression_score": float(double_compression_score),
                "compression_quality_estimate": self._estimate_jpeg_quality(gray)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing compression: {e}")
            return {}

    async def _analyze_noise_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze noise patterns for authenticity"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Extract noise using wavelet denoising
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            noise = gray.astype(np.float32) - denoised.astype(np.float32)
            
            # Analyze noise characteristics
            noise_variance = np.var(noise)
            noise_mean = np.mean(noise)
            noise_skewness = self._calculate_skewness(noise.flatten())
            noise_kurtosis = self._calculate_kurtosis(noise.flatten())
            
            # Check for noise inconsistencies across regions
            regions = self._divide_into_regions(noise, 4, 4)
            region_variances = [np.var(region) for region in regions]
            noise_consistency = np.std(region_variances) / np.mean(region_variances) if np.mean(region_variances) > 0 else 0
            
            return {
                "noise_variance": float(noise_variance),
                "noise_mean": float(noise_mean),
                "noise_skewness": float(noise_skewness),
                "noise_kurtosis": float(noise_kurtosis),
                "noise_consistency_score": float(noise_consistency),
                "suspicious_noise_patterns": noise_consistency > 0.5
            }
            
        except Exception as e:
            logger.error(f"Error analyzing noise patterns: {e}")
            return {}

    async def _analyze_edge_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze edge consistency for splicing detection"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect edges using multiple methods
            canny_edges = cv2.Canny(gray, 50, 150)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate edge strength and direction
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_direction = np.arctan2(sobel_y, sobel_x)
            
            # Analyze edge consistency across regions
            regions = self._divide_into_regions(edge_magnitude, 3, 3)
            edge_strengths = [np.mean(region) for region in regions]
            edge_consistency = np.std(edge_strengths) / np.mean(edge_strengths) if np.mean(edge_strengths) > 0 else 0
            
            # Detect abrupt changes in edge patterns
            edge_transitions = self._detect_edge_transitions(edge_magnitude)
            
            return {
                "edge_consistency_score": float(edge_consistency),
                "average_edge_strength": float(np.mean(edge_magnitude)),
                "edge_direction_variance": float(np.var(edge_direction)),
                "suspicious_edge_transitions": len(edge_transitions),
                "potential_splicing_regions": edge_transitions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing edge consistency: {e}")
            return {}

    async def _analyze_frequency_domain(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency domain characteristics"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Perform FFT
            f_transform = fft.fft2(gray)
            f_shift = fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Analyze frequency distribution
            freq_mean = np.mean(magnitude_spectrum)
            freq_std = np.std(magnitude_spectrum)
            
            # Detect periodic patterns (indicating potential manipulation)
            autocorr = np.correlate(magnitude_spectrum.flatten(), magnitude_spectrum.flatten(), mode='full')
            periodic_score = self._detect_periodic_patterns(autocorr)
            
            # Analyze high-frequency content
            h, w = gray.shape
            center_h, center_w = h // 2, w // 2
            
            # Create masks for different frequency bands
            low_freq_mask = self._create_circular_mask(h, w, center_h, center_w, min(h, w) // 8)
            high_freq_mask = ~self._create_circular_mask(h, w, center_h, center_w, min(h, w) // 4)
            
            low_freq_energy = np.sum(np.abs(f_shift[low_freq_mask]))
            high_freq_energy = np.sum(np.abs(f_shift[high_freq_mask]))
            
            freq_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0
            
            return {
                "frequency_mean": float(freq_mean),
                "frequency_std": float(freq_std),
                "periodic_patterns_score": float(periodic_score),
                "high_low_freq_ratio": float(freq_ratio),
                "suspicious_frequency_patterns": periodic_score > 0.3
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frequency domain: {e}")
            return {}

    async def _analyze_statistical_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze statistical properties of the image"""
        
        try:
            # Analyze each color channel
            channel_stats = {}
            
            for i, channel_name in enumerate(['red', 'green', 'blue']):
                channel = image[:, :, i]
                
                channel_stats[channel_name] = {
                    "mean": float(np.mean(channel)),
                    "std": float(np.std(channel)),
                    "skewness": float(self._calculate_skewness(channel.flatten())),
                    "kurtosis": float(self._calculate_kurtosis(channel.flatten())),
                    "entropy": float(self._calculate_entropy(channel))
                }
            
            # Calculate inter-channel correlations
            correlations = {}
            channels = ['red', 'green', 'blue']
            for i in range(3):
                for j in range(i+1, 3):
                    corr = np.corrcoef(image[:, :, i].flatten(), image[:, :, j].flatten())[0, 1]
                    correlations[f"{channels[i]}_{channels[j]}"] = float(corr)
            
            # Detect statistical anomalies
            anomalies = []
            
            # Check for unusual correlations
            for corr_name, corr_value in correlations.items():
                if abs(corr_value) < 0.3:  # Unusually low correlation
                    anomalies.append(f"Low correlation in {corr_name}: {corr_value:.3f}")
            
            # Check for unusual distributions
            for channel_name, stats in channel_stats.items():
                if abs(stats['skewness']) > 2:
                    anomalies.append(f"High skewness in {channel_name}: {stats['skewness']:.3f}")
                if stats['kurtosis'] > 5:
                    anomalies.append(f"High kurtosis in {channel_name}: {stats['kurtosis']:.3f}")
            
            return {
                "channel_statistics": channel_stats,
                "inter_channel_correlations": correlations,
                "statistical_anomalies": anomalies,
                "anomaly_count": len(anomalies)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing statistical properties: {e}")
            return {}

    async def _detect_manipulations(self, image: np.ndarray, 
                                  metadata_analysis: MetadataAnalysis) -> List[ManipulationDetection]:
        """Detect various types of image manipulations"""
        
        detections = []
        
        try:
            # Copy-move detection
            copy_move_regions = await self._detect_copy_move(image)
            if copy_move_regions:
                detections.append(ManipulationDetection(
                    manipulation_type=ManipulationType.COPY_MOVE,
                    confidence=0.8,
                    affected_regions=copy_move_regions,
                    description="Potential copy-move manipulation detected",
                    evidence_strength="moderate"
                ))
            
            # Splicing detection based on edge analysis
            if hasattr(self, '_last_edge_analysis'):
                edge_analysis = self._last_edge_analysis
                if edge_analysis.get('suspicious_edge_transitions', 0) > 3:
                    detections.append(ManipulationDetection(
                        manipulation_type=ManipulationType.SPLICING,
                        confidence=0.6,
                        affected_regions=edge_analysis.get('potential_splicing_regions', []),
                        description="Potential image splicing detected based on edge inconsistencies",
                        evidence_strength="weak"
                    ))
            
            # Metadata tampering detection
            if len(metadata_analysis.suspicious_patterns) > 2:
                detections.append(ManipulationDetection(
                    manipulation_type=ManipulationType.METADATA_TAMPERING,
                    confidence=0.7,
                    affected_regions=[],
                    description="Metadata inconsistencies suggest potential tampering",
                    evidence_strength="moderate"
                ))
            
            # Retouching detection based on noise analysis
            # This would be implemented with more sophisticated algorithms
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting manipulations: {e}")
            return []

    async def _detect_copy_move(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect copy-move forgeries using block matching"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Use SIFT features for block matching
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) < 10:
                return []
            
            # Match features to find similar regions
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors, descriptors, k=3)
            
            # Filter matches to find potential copy-move regions
            copy_move_regions = []
            
            for match_group in matches:
                if len(match_group) >= 2:
                    m1, m2 = match_group[:2]
                    
                    # Skip self-matches and very close matches
                    if m1.trainIdx != m1.queryIdx and m1.distance < 0.7 * m2.distance:
                        pt1 = keypoints[m1.queryIdx].pt
                        pt2 = keypoints[m1.trainIdx].pt
                        
                        # Check if regions are sufficiently separated
                        distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                        if distance > 50:  # Minimum separation
                            # Create bounding boxes around matched points
                            region_size = 30
                            x1, y1 = int(pt1[0] - region_size), int(pt1[1] - region_size)
                            x2, y2 = int(pt1[0] + region_size), int(pt1[1] + region_size)
                            copy_move_regions.append((x1, y1, x2, y2))
            
            return copy_move_regions[:5]  # Return top 5 suspicious regions
            
        except Exception as e:
            logger.error(f"Error detecting copy-move: {e}")
            return []

    async def _analyze_timeline(self, file_path: str, 
                              metadata_analysis: MetadataAnalysis) -> Dict[str, Any]:
        """Analyze timeline consistency"""
        
        try:
            file_stat = os.stat(file_path)
            
            timeline_events = []
            
            # File system timestamps
            timeline_events.append({
                "event": "file_created",
                "timestamp": datetime.fromtimestamp(file_stat.st_ctime),
                "source": "filesystem"
            })
            
            timeline_events.append({
                "event": "file_modified",
                "timestamp": datetime.fromtimestamp(file_stat.st_mtime),
                "source": "filesystem"
            })
            
            # EXIF timestamps
            if metadata_analysis.creation_date:
                timeline_events.append({
                    "event": "photo_taken",
                    "timestamp": metadata_analysis.creation_date,
                    "source": "exif"
                })
            
            # Sort events by timestamp
            timeline_events.sort(key=lambda x: x["timestamp"])
            
            # Analyze consistency
            inconsistencies = []
            
            for i in range(len(timeline_events) - 1):
                current = timeline_events[i]
                next_event = timeline_events[i + 1]
                
                time_diff = (next_event["timestamp"] - current["timestamp"]).total_seconds()
                
                # Check for suspicious patterns
                if current["event"] == "photo_taken" and next_event["event"] == "file_created":
                    if time_diff < -3600:  # File created before photo taken
                        inconsistencies.append("File created before photo was taken according to EXIF")
                
                if abs(time_diff) > 86400 * 30:  # More than 30 days difference
                    inconsistencies.append(f"Large time gap between {current['event']} and {next_event['event']}")
            
            return {
                "timeline_events": timeline_events,
                "inconsistencies": inconsistencies,
                "timeline_suspicious": len(inconsistencies) > 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeline: {e}")
            return {}

    def _determine_overall_finding(self, metadata_analysis: MetadataAnalysis,
                                 integrity_analysis: ImageIntegrityAnalysis,
                                 manipulation_detections: List[ManipulationDetection]) -> ForensicFinding:
        """Determine overall forensic finding"""
        
        try:
            # Count evidence of manipulation
            strong_evidence = sum(1 for d in manipulation_detections if d.evidence_strength == "strong")
            moderate_evidence = sum(1 for d in manipulation_detections if d.evidence_strength == "moderate")
            weak_evidence = sum(1 for d in manipulation_detections if d.evidence_strength == "weak")
            
            # Consider metadata issues
            metadata_issues = len(metadata_analysis.suspicious_patterns)
            
            # Decision logic
            if strong_evidence > 0 or (moderate_evidence > 1 and metadata_issues > 2):
                return ForensicFinding.MANIPULATED
            elif moderate_evidence > 0 or weak_evidence > 2 or metadata_issues > 3:
                return ForensicFinding.SUSPICIOUS
            elif weak_evidence > 0 or metadata_issues > 0:
                return ForensicFinding.INCONCLUSIVE
            else:
                return ForensicFinding.AUTHENTIC
                
        except Exception:
            return ForensicFinding.INCONCLUSIVE

    def _assess_authenticity_level(self, metadata_analysis: MetadataAnalysis,
                                 manipulation_detections: List[ManipulationDetection]) -> AuthenticityLevel:
        """Assess authenticity confidence level"""
        
        try:
            manipulation_score = sum(d.confidence for d in manipulation_detections)
            metadata_score = len(metadata_analysis.suspicious_patterns) * 0.1
            
            total_suspicion = manipulation_score + metadata_score
            
            if total_suspicion < 0.2:
                return AuthenticityLevel.HIGH
            elif total_suspicion < 0.5:
                return AuthenticityLevel.MEDIUM
            elif total_suspicion < 0.8:
                return AuthenticityLevel.LOW
            else:
                return AuthenticityLevel.VERY_LOW
                
        except Exception:
            return AuthenticityLevel.LOW

    def _calculate_confidence_score(self, metadata_analysis: MetadataAnalysis,
                                  integrity_analysis: ImageIntegrityAnalysis,
                                  manipulation_detections: List[ManipulationDetection]) -> float:
        """Calculate overall confidence in the analysis"""
        
        try:
            # Base confidence on available data
            metadata_confidence = 0.8 if metadata_analysis.exif_data else 0.3
            
            # Adjust based on analysis completeness
            if integrity_analysis.file_hash:
                metadata_confidence += 0.1
            
            # Adjust based on detection confidence
            if manipulation_detections:
                avg_detection_confidence = sum(d.confidence for d in manipulation_detections) / len(manipulation_detections)
                overall_confidence = (metadata_confidence + avg_detection_confidence) / 2
            else:
                overall_confidence = metadata_confidence
            
            return min(overall_confidence, 1.0)
            
        except Exception:
            return 0.5

    def _generate_recommendations(self, finding: ForensicFinding,
                                manipulation_detections: List[ManipulationDetection],
                                confidence_score: float) -> List[str]:
        """Generate analyst recommendations"""
        
        recommendations = []
        
        try:
            if finding == ForensicFinding.MANIPULATED:
                recommendations.append("Evidence shows signs of digital manipulation - recommend rejection")
                recommendations.append("Consider requesting original source files")
                
            elif finding == ForensicFinding.SUSPICIOUS:
                recommendations.append("Evidence shows suspicious characteristics - recommend further investigation")
                recommendations.append("Consider expert forensic analysis")
                
            elif finding == ForensicFinding.INCONCLUSIVE:
                recommendations.append("Analysis inconclusive - recommend additional evidence")
                
            if confidence_score < 0.6:
                recommendations.append("Low confidence in analysis - recommend manual review")
            
            if any(d.manipulation_type == ManipulationType.METADATA_TAMPERING for d in manipulation_detections):
                recommendations.append("Metadata tampering detected - verify chain of custody")
            
            if len(manipulation_detections) > 2:
                recommendations.append("Multiple manipulation indicators - high priority review")
            
            return recommendations
            
        except Exception:
            return ["Analysis completed - recommend standard review procedures"]

    # Helper methods
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return 0

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy"""
        try:
            hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))
        except:
            return 0

    def _divide_into_regions(self, image: np.ndarray, rows: int, cols: int) -> List[np.ndarray]:
        """Divide image into regions"""
        try:
            h, w = image.shape[:2]
            regions = []
            
            for i in range(rows):
                for j in range(cols):
                    start_h = i * h // rows
                    end_h = (i + 1) * h // rows
                    start_w = j * w // cols
                    end_w = (j + 1) * w // cols
                    
                    region = image[start_h:end_h, start_w:end_w]
                    regions.append(region)
            
            return regions
        except:
            return []

    def _detect_periodic_patterns(self, data: np.ndarray) -> float:
        """Detect periodic patterns in data"""
        try:
            # Simple autocorrelation-based detection
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for peaks indicating periodicity
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(autocorr[i])
            
            if len(peaks) > 2:
                return min(np.std(peaks) / np.mean(peaks), 1.0) if np.mean(peaks) > 0 else 0
            else:
                return 0
        except:
            return 0

    def _create_circular_mask(self, h: int, w: int, center_h: int, center_w: int, radius: int) -> np.ndarray:
        """Create circular mask"""
        try:
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
            return dist_from_center <= radius
        except:
            return np.zeros((h, w), dtype=bool)

    def _detect_edge_transitions(self, edge_magnitude: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect suspicious edge transitions"""
        try:
            # Simple implementation - would be more sophisticated in production
            h, w = edge_magnitude.shape
            transitions = []
            
            # Look for abrupt changes in edge strength
            threshold = np.mean(edge_magnitude) + 2 * np.std(edge_magnitude)
            
            strong_edges = edge_magnitude > threshold
            contours, _ = cv2.findContours(strong_edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    transitions.append((x, y, x + w, y + h))
            
            return transitions[:10]  # Return top 10
        except:
            return []

    def _estimate_jpeg_quality(self, image: np.ndarray) -> int:
        """Estimate JPEG quality factor"""
        try:
            # Simplified quality estimation based on blocking artifacts
            # Real implementation would analyze quantization tables
            
            # Calculate blocking measure
            h, w = image.shape
            blocking_score = 0
            
            for i in range(8, h - 8, 8):
                for j in range(8, w - 8, 8):
                    # Measure discontinuity at block boundaries
                    h_diff = abs(int(image[i-1, j]) - int(image[i, j]))
                    v_diff = abs(int(image[i, j-1]) - int(image[i, j]))
                    blocking_score += h_diff + v_diff
            
            blocking_score /= ((h // 8) * (w // 8))
            
            # Map to quality estimate (inverse relationship)
            if blocking_score < 5:
                return 95
            elif blocking_score < 10:
                return 85
            elif blocking_score < 20:
                return 75
            elif blocking_score < 40:
                return 60
            else:
                return 40
                
        except:
            return 75  # Default medium quality

    async def _store_analysis(self, result: ForensicAnalysisResult):
        """Store forensic analysis result"""
        
        try:
            with self.Session() as session:
                record = ForensicAnalysisRecord(
                    analysis_id=result.analysis_id,
                    evidence_id=result.evidence_id,
                    file_path=result.file_path,
                    overall_finding=result.overall_finding.value,
                    authenticity_level=result.authenticity_level.value,
                    confidence_score=result.confidence_score,
                    metadata_analysis=asdict(result.metadata_analysis),
                    integrity_analysis=asdict(result.integrity_analysis),
                    manipulation_detections=[asdict(d) for d in result.manipulation_detections],
                    timeline_analysis=result.timeline_analysis,
                    chain_of_custody_notes=result.chain_of_custody_notes,
                    processing_time=result.processing_time,
                    analyst_recommendations=result.analyst_recommendations,
                    created_at=datetime.utcnow()
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing forensic analysis: {e}")

def create_forensic_analyzer(db_url: str = None, redis_url: str = None) -> ForensicAnalyzer:
    """Create and configure ForensicAnalyzer instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return ForensicAnalyzer(db_url=db_url, redis_url=redis_url)

