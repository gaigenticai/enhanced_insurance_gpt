"""
OCR Engine - Production Ready Implementation
Advanced Optical Character Recognition for insurance documents
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import pytesseract
import pdf2image
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Advanced image processing
from skimage import filters, morphology, segmentation, measure
from skimage.transform import rotate
from skimage.color import rgb2gray
import scipy.ndimage as ndi

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
ocr_operations_total = Counter('ocr_operations_total', 'Total OCR operations', ['operation_type', 'status'])
ocr_processing_duration = Histogram('ocr_processing_duration_seconds', 'Time to process OCR operations')
ocr_accuracy_gauge = Gauge('ocr_accuracy', 'OCR accuracy scores', ['document_type'])

Base = declarative_base()

class OCRMethod(Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    COMBINED = "combined"

class ImageQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class PreprocessingStep(Enum):
    DESKEW = "deskew"
    DENOISE = "denoise"
    ENHANCE_CONTRAST = "enhance_contrast"
    BINARIZE = "binarize"
    RESIZE = "resize"
    ROTATE = "rotate"
    CROP = "crop"

@dataclass
class OCRConfiguration:
    """OCR processing configuration"""
    method: OCRMethod = OCRMethod.TESSERACT
    language: str = "eng"
    page_segmentation_mode: int = 6
    ocr_engine_mode: int = 3
    preprocessing_steps: List[PreprocessingStep] = None
    confidence_threshold: float = 0.6
    dpi: int = 300
    enhance_image: bool = True
    
    def __post_init__(self):
        if self.preprocessing_steps is None:
            self.preprocessing_steps = [
                PreprocessingStep.DESKEW,
                PreprocessingStep.DENOISE,
                PreprocessingStep.ENHANCE_CONTRAST,
                PreprocessingStep.BINARIZE
            ]

@dataclass
class OCRResult:
    """OCR processing result"""
    operation_id: str
    text: str
    confidence: float
    word_confidences: List[Dict[str, Any]]
    bounding_boxes: List[Dict[str, Any]]
    image_quality: ImageQuality
    preprocessing_applied: List[str]
    processing_time: float
    method_used: OCRMethod
    language_detected: str
    error_message: Optional[str] = None

class OCROperationRecord(Base):
    """SQLAlchemy model for OCR operations"""
    __tablename__ = 'ocr_operations'
    
    operation_id = Column(String, primary_key=True)
    input_path = Column(String, nullable=False)
    output_text = Column(Text)
    confidence = Column(Float)
    word_confidences = Column(JSON)
    bounding_boxes = Column(JSON)
    image_quality = Column(String)
    preprocessing_applied = Column(JSON)
    processing_time = Column(Float)
    method_used = Column(String)
    language_detected = Column(String)
    error_message = Column(Text)
    created_at = Column(DateTime, nullable=False)
    metadata = Column(JSON)

class OCREngine:
    """
    Production-ready OCR Engine
    Advanced optical character recognition with multiple methods and preprocessing
    """
    
    def __init__(self, db_url: str, redis_url: str, temp_path: str = "/tmp/ocr"):
        self.db_url = db_url
        self.redis_url = redis_url
        self.temp_path = temp_path
        
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis setup
        self.redis_client = redis.from_url(redis_url)
        
        # Temp directory setup
        os.makedirs(temp_path, exist_ok=True)
        
        # Initialize OCR engines
        self._initialize_ocr_engines()
        
        # Tesseract configuration
        self.tesseract_config = {
            'preserve_interword_spaces': 1,
            'tessedit_char_whitelist': '',
            'tessedit_char_blacklist': '',
        }
        
        logger.info("OCREngine initialized successfully")

    def _initialize_ocr_engines(self):
        """Initialize available OCR engines"""
        
        self.available_engines = [OCRMethod.TESSERACT]
        
        # Check for EasyOCR
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(['en'])
            self.available_engines.append(OCRMethod.EASYOCR)
            logger.info("EasyOCR initialized successfully")
        except ImportError:
            logger.info("EasyOCR not available")
            self.easyocr_reader = None
        
        # Check for PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.available_engines.append(OCRMethod.PADDLEOCR)
            logger.info("PaddleOCR initialized successfully")
        except ImportError:
            logger.info("PaddleOCR not available")
            self.paddleocr_reader = None
        
        logger.info(f"Available OCR engines: {[engine.value for engine in self.available_engines]}")

    async def process_image(self, 
                          image_path: str, 
                          config: OCRConfiguration = None) -> OCRResult:
        """Process image with OCR"""
        
        if config is None:
            config = OCRConfiguration()
        
        start_time = datetime.utcnow()
        operation_id = str(uuid.uuid4())
        
        with ocr_processing_duration.time():
            try:
                logger.info(f"Starting OCR processing for {image_path}")
                
                # Load and preprocess image
                image = self._load_image(image_path)
                image_quality = self._assess_image_quality(image)
                
                preprocessed_image, preprocessing_applied = await self._preprocess_image(
                    image, config.preprocessing_steps
                )
                
                # Perform OCR based on method
                if config.method == OCRMethod.TESSERACT:
                    result = await self._ocr_tesseract(preprocessed_image, config)
                elif config.method == OCRMethod.EASYOCR and self.easyocr_reader:
                    result = await self._ocr_easyocr(preprocessed_image, config)
                elif config.method == OCRMethod.PADDLEOCR and self.paddleocr_reader:
                    result = await self._ocr_paddleocr(preprocessed_image, config)
                elif config.method == OCRMethod.COMBINED:
                    result = await self._ocr_combined(preprocessed_image, config)
                else:
                    # Fallback to Tesseract
                    result = await self._ocr_tesseract(preprocessed_image, config)
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Create OCR result
                ocr_result = OCRResult(
                    operation_id=operation_id,
                    text=result["text"],
                    confidence=result["confidence"],
                    word_confidences=result.get("word_confidences", []),
                    bounding_boxes=result.get("bounding_boxes", []),
                    image_quality=image_quality,
                    preprocessing_applied=preprocessing_applied,
                    processing_time=processing_time,
                    method_used=config.method,
                    language_detected=result.get("language", config.language)
                )
                
                # Store result
                await self._store_ocr_result(ocr_result, image_path)
                
                # Update metrics
                ocr_operations_total.labels(
                    operation_type=config.method.value,
                    status='success'
                ).inc()
                
                ocr_accuracy_gauge.labels(
                    document_type='general'
                ).set(ocr_result.confidence)
                
                logger.info(f"OCR completed for {operation_id} in {processing_time:.2f}s with confidence {ocr_result.confidence:.2f}")
                
                return ocr_result
                
            except Exception as e:
                logger.error(f"Error processing OCR for {image_path}: {e}")
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                error_result = OCRResult(
                    operation_id=operation_id,
                    text="",
                    confidence=0.0,
                    word_confidences=[],
                    bounding_boxes=[],
                    image_quality=ImageQuality.POOR,
                    preprocessing_applied=[],
                    processing_time=processing_time,
                    method_used=config.method,
                    language_detected=config.language,
                    error_message=str(e)
                )
                
                ocr_operations_total.labels(
                    operation_type=config.method.value,
                    status='failed'
                ).inc()
                
                return error_result

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file"""
        
        try:
            # Handle PDF files
            if image_path.lower().endswith('.pdf'):
                images = pdf2image.convert_from_path(image_path, dpi=300)
                if images:
                    # Convert first page to numpy array
                    return np.array(images[0])
                else:
                    raise ValueError("No pages found in PDF")
            else:
                # Load regular image
                image = Image.open(image_path)
                return np.array(image)
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def _assess_image_quality(self, image: np.ndarray) -> ImageQuality:
        """Assess image quality for OCR"""
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Calculate various quality metrics
            
            # 1. Laplacian variance (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Image contrast
            contrast = gray.std()
            
            # 3. Image brightness
            brightness = gray.mean()
            
            # 4. Resolution check
            height, width = gray.shape
            resolution_score = min(height, width)
            
            # Combine metrics to assess quality
            quality_score = 0
            
            # Sharpness score (0-40)
            if laplacian_var > 1000:
                quality_score += 40
            elif laplacian_var > 500:
                quality_score += 30
            elif laplacian_var > 100:
                quality_score += 20
            else:
                quality_score += 10
            
            # Contrast score (0-30)
            if contrast > 60:
                quality_score += 30
            elif contrast > 40:
                quality_score += 20
            elif contrast > 20:
                quality_score += 10
            else:
                quality_score += 5
            
            # Brightness score (0-20)
            if 80 <= brightness <= 180:
                quality_score += 20
            elif 60 <= brightness <= 200:
                quality_score += 15
            elif 40 <= brightness <= 220:
                quality_score += 10
            else:
                quality_score += 5
            
            # Resolution score (0-10)
            if resolution_score >= 1000:
                quality_score += 10
            elif resolution_score >= 500:
                quality_score += 7
            elif resolution_score >= 300:
                quality_score += 5
            else:
                quality_score += 2
            
            # Determine quality level
            if quality_score >= 85:
                return ImageQuality.EXCELLENT
            elif quality_score >= 70:
                return ImageQuality.GOOD
            elif quality_score >= 50:
                return ImageQuality.FAIR
            else:
                return ImageQuality.POOR
                
        except Exception as e:
            logger.warning(f"Error assessing image quality: {e}")
            return ImageQuality.FAIR

    async def _preprocess_image(self, 
                              image: np.ndarray, 
                              steps: List[PreprocessingStep]) -> Tuple[np.ndarray, List[str]]:
        """Apply preprocessing steps to image"""
        
        processed_image = image.copy()
        applied_steps = []
        
        try:
            for step in steps:
                if step == PreprocessingStep.DESKEW:
                    processed_image = self._deskew_image(processed_image)
                    applied_steps.append("deskew")
                
                elif step == PreprocessingStep.DENOISE:
                    processed_image = self._denoise_image(processed_image)
                    applied_steps.append("denoise")
                
                elif step == PreprocessingStep.ENHANCE_CONTRAST:
                    processed_image = self._enhance_contrast(processed_image)
                    applied_steps.append("enhance_contrast")
                
                elif step == PreprocessingStep.BINARIZE:
                    processed_image = self._binarize_image(processed_image)
                    applied_steps.append("binarize")
                
                elif step == PreprocessingStep.RESIZE:
                    processed_image = self._resize_image(processed_image)
                    applied_steps.append("resize")
                
                elif step == PreprocessingStep.ROTATE:
                    processed_image = self._auto_rotate_image(processed_image)
                    applied_steps.append("rotate")
                
                elif step == PreprocessingStep.CROP:
                    processed_image = self._auto_crop_image(processed_image)
                    applied_steps.append("crop")
            
            return processed_image, applied_steps
            
        except Exception as e:
            logger.warning(f"Error in image preprocessing: {e}")
            return image, applied_steps

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in image"""
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (assumed to be the document)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                
                # Correct angle
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                # Rotate image if significant skew detected
                if abs(angle) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated
            
            return image
            
        except Exception as e:
            logger.warning(f"Error in deskewing: {e}")
            return image

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        
        try:
            if len(image.shape) == 3:
                # Color image
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                # Grayscale image
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            logger.warning(f"Error in denoising: {e}")
            return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge channels and convert back
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Error in contrast enhancement: {e}")
            return image

    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary (black and white)"""
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return binary
            
        except Exception as e:
            logger.warning(f"Error in binarization: {e}")
            return image

    def _resize_image(self, image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """Resize image to optimal resolution for OCR"""
        
        try:
            height, width = image.shape[:2]
            
            # Calculate scaling factor based on target DPI
            # Assume current image is at 72 DPI if not specified
            scale_factor = target_dpi / 72
            
            # Don't upscale too much
            if scale_factor > 3:
                scale_factor = 3
            elif scale_factor < 0.5:
                scale_factor = 0.5
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return resized
            
        except Exception as e:
            logger.warning(f"Error in resizing: {e}")
            return image

    def _auto_rotate_image(self, image: np.ndarray) -> np.ndarray:
        """Auto-rotate image based on text orientation"""
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Try different rotations and find the best one
            best_score = 0
            best_rotation = 0
            
            for angle in [0, 90, 180, 270]:
                if angle == 0:
                    rotated = gray
                else:
                    (h, w) = gray.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(gray, M, (w, h))
                
                # Score based on horizontal text lines
                score = self._score_text_orientation(rotated)
                
                if score > best_score:
                    best_score = score
                    best_rotation = angle
            
            # Apply best rotation
            if best_rotation != 0:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, best_rotation, 1.0)
                rotated_image = cv2.warpAffine(image, M, (w, h))
                return rotated_image
            
            return image
            
        except Exception as e:
            logger.warning(f"Error in auto-rotation: {e}")
            return image

    def _score_text_orientation(self, gray_image: np.ndarray) -> float:
        """Score image based on text orientation (higher score = better orientation)"""
        
        try:
            # Apply edge detection
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return 0
            
            # Count horizontal lines (angles close to 0 or 180 degrees)
            horizontal_count = 0
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                
                # Check if line is horizontal (within 10 degrees)
                if abs(angle) < 10 or abs(angle - 180) < 10:
                    horizontal_count += 1
            
            return horizontal_count
            
        except Exception as e:
            return 0

    def _auto_crop_image(self, image: np.ndarray) -> np.ndarray:
        """Auto-crop image to remove borders and focus on content"""
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply threshold to find content
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get bounding box of all contours
                x_coords = []
                y_coords = []
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_coords.extend([x, x + w])
                    y_coords.extend([y, y + h])
                
                # Calculate crop boundaries with some padding
                padding = 20
                x_min = max(0, min(x_coords) - padding)
                x_max = min(image.shape[1], max(x_coords) + padding)
                y_min = max(0, min(y_coords) - padding)
                y_max = min(image.shape[0], max(y_coords) + padding)
                
                # Crop image
                cropped = image[y_min:y_max, x_min:x_max]
                
                # Only return cropped image if it's significantly smaller
                original_area = image.shape[0] * image.shape[1]
                cropped_area = cropped.shape[0] * cropped.shape[1]
                
                if cropped_area < 0.8 * original_area and cropped_area > 0.1 * original_area:
                    return cropped
            
            return image
            
        except Exception as e:
            logger.warning(f"Error in auto-cropping: {e}")
            return image

    async def _ocr_tesseract(self, image: np.ndarray, config: OCRConfiguration) -> Dict[str, Any]:
        """Perform OCR using Tesseract"""
        
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image, mode='L')
            
            # Configure Tesseract
            custom_config = f'--oem {config.ocr_engine_mode} --psm {config.page_segmentation_mode} -l {config.language}'
            
            # Extract text
            text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            # Get detailed data with confidence scores
            data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Process word-level data
            word_confidences = []
            bounding_boxes = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Valid confidence
                    word_confidences.append({
                        'text': data['text'][i],
                        'confidence': float(data['conf'][i]) / 100.0,
                        'left': int(data['left'][i]),
                        'top': int(data['top'][i]),
                        'width': int(data['width'][i]),
                        'height': int(data['height'][i])
                    })
                    
                    bounding_boxes.append({
                        'text': data['text'][i],
                        'bbox': [
                            int(data['left'][i]),
                            int(data['top'][i]),
                            int(data['left'][i]) + int(data['width'][i]),
                            int(data['top'][i]) + int(data['height'][i])
                        ]
                    })
            
            # Calculate overall confidence
            valid_confidences = [wc['confidence'] for wc in word_confidences if wc['confidence'] > 0]
            overall_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
            
            return {
                'text': text.strip(),
                'confidence': overall_confidence,
                'word_confidences': word_confidences,
                'bounding_boxes': bounding_boxes,
                'language': config.language
            }
            
        except Exception as e:
            logger.error(f"Error in Tesseract OCR: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'word_confidences': [],
                'bounding_boxes': [],
                'language': config.language
            }

    async def _ocr_easyocr(self, image: np.ndarray, config: OCRConfiguration) -> Dict[str, Any]:
        """Perform OCR using EasyOCR"""
        
        try:
            if self.easyocr_reader is None:
                raise ValueError("EasyOCR not available")
            
            # EasyOCR expects image in RGB format
            if len(image.shape) == 3:
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Perform OCR
            results = self.easyocr_reader.readtext(rgb_image, detail=1, paragraph=False)
            
            # Process results
            text_parts = []
            word_confidences = []
            bounding_boxes = []
            
            for (bbox, text, confidence) in results:
                if confidence >= config.confidence_threshold:
                    text_parts.append(text)
                    
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    left = int(min(x_coords))
                    top = int(min(y_coords))
                    width = int(max(x_coords) - min(x_coords))
                    height = int(max(y_coords) - min(y_coords))
                    
                    word_confidences.append({
                        'text': text,
                        'confidence': float(confidence),
                        'left': left,
                        'top': top,
                        'width': width,
                        'height': height
                    })
                    
                    bounding_boxes.append({
                        'text': text,
                        'bbox': [left, top, left + width, top + height]
                    })
            
            # Combine text
            full_text = ' '.join(text_parts)
            
            # Calculate overall confidence
            confidences = [float(conf) for (_, _, conf) in results if conf >= config.confidence_threshold]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'text': full_text,
                'confidence': overall_confidence,
                'word_confidences': word_confidences,
                'bounding_boxes': bounding_boxes,
                'language': config.language
            }
            
        except Exception as e:
            logger.error(f"Error in EasyOCR: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'word_confidences': [],
                'bounding_boxes': [],
                'language': config.language
            }

    async def _ocr_paddleocr(self, image: np.ndarray, config: OCRConfiguration) -> Dict[str, Any]:
        """Perform OCR using PaddleOCR"""
        
        try:
            if self.paddleocr_reader is None:
                raise ValueError("PaddleOCR not available")
            
            # PaddleOCR expects image path or numpy array
            results = self.paddleocr_reader.ocr(image, cls=True)
            
            # Process results
            text_parts = []
            word_confidences = []
            bounding_boxes = []
            
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, confidence) = line
                    
                    if confidence >= config.confidence_threshold:
                        text_parts.append(text)
                        
                        # Convert bbox format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        left = int(min(x_coords))
                        top = int(min(y_coords))
                        width = int(max(x_coords) - min(x_coords))
                        height = int(max(y_coords) - min(y_coords))
                        
                        word_confidences.append({
                            'text': text,
                            'confidence': float(confidence),
                            'left': left,
                            'top': top,
                            'width': width,
                            'height': height
                        })
                        
                        bounding_boxes.append({
                            'text': text,
                            'bbox': [left, top, left + width, top + height]
                        })
            
            # Combine text
            full_text = '\n'.join(text_parts)
            
            # Calculate overall confidence
            confidences = [float(confidence) for line in (results[0] or []) for (_, (_, confidence)) in [line] if confidence >= config.confidence_threshold]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'text': full_text,
                'confidence': overall_confidence,
                'word_confidences': word_confidences,
                'bounding_boxes': bounding_boxes,
                'language': config.language
            }
            
        except Exception as e:
            logger.error(f"Error in PaddleOCR: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'word_confidences': [],
                'bounding_boxes': [],
                'language': config.language
            }

    async def _ocr_combined(self, image: np.ndarray, config: OCRConfiguration) -> Dict[str, Any]:
        """Perform OCR using multiple engines and combine results"""
        
        try:
            results = []
            
            # Try each available engine
            for engine in self.available_engines:
                if engine == OCRMethod.COMBINED:
                    continue
                
                engine_config = OCRConfiguration(
                    method=engine,
                    language=config.language,
                    confidence_threshold=config.confidence_threshold
                )
                
                if engine == OCRMethod.TESSERACT:
                    result = await self._ocr_tesseract(image, engine_config)
                elif engine == OCRMethod.EASYOCR and self.easyocr_reader:
                    result = await self._ocr_easyocr(image, engine_config)
                elif engine == OCRMethod.PADDLEOCR and self.paddleocr_reader:
                    result = await self._ocr_paddleocr(image, engine_config)
                else:
                    continue
                
                if result['confidence'] > 0:
                    results.append((engine, result))
            
            if not results:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'word_confidences': [],
                    'bounding_boxes': [],
                    'language': config.language
                }
            
            # Select best result based on confidence
            best_engine, best_result = max(results, key=lambda x: x[1]['confidence'])
            
            # Enhance with consensus from other engines
            if len(results) > 1:
                # Simple consensus: use the result with highest confidence
                # In a more sophisticated approach, you could merge results
                pass
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error in combined OCR: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'word_confidences': [],
                'bounding_boxes': [],
                'language': config.language
            }

    async def _store_ocr_result(self, result: OCRResult, input_path: str):
        """Store OCR result in database"""
        
        try:
            with self.Session() as session:
                record = OCROperationRecord(
                    operation_id=result.operation_id,
                    input_path=input_path,
                    output_text=result.text,
                    confidence=result.confidence,
                    word_confidences=result.word_confidences,
                    bounding_boxes=result.bounding_boxes,
                    image_quality=result.image_quality.value,
                    preprocessing_applied=result.preprocessing_applied,
                    processing_time=result.processing_time,
                    method_used=result.method_used.value,
                    language_detected=result.language_detected,
                    error_message=result.error_message,
                    created_at=datetime.utcnow(),
                    metadata={}
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing OCR result: {e}")

    async def process_pdf_pages(self, 
                              pdf_path: str, 
                              config: OCRConfiguration = None) -> List[OCRResult]:
        """Process all pages of a PDF document"""
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=config.dpi if config else 300)
            
            results = []
            for i, image in enumerate(images):
                # Save temporary image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    image.save(temp_file.name, 'PNG')
                    temp_path = temp_file.name
                
                try:
                    # Process page
                    page_result = await self.process_image(temp_path, config)
                    page_result.operation_id = f"{page_result.operation_id}_page_{i+1}"
                    results.append(page_result)
                    
                finally:
                    # Clean up temporary file
                    os.unlink(temp_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF pages: {e}")
            return []

    async def get_ocr_statistics(self) -> Dict[str, Any]:
        """Get OCR processing statistics"""
        
        try:
            with self.Session() as session:
                total_operations = session.query(OCROperationRecord).count()
                
                # Operations by method
                method_stats = {}
                for method in OCRMethod:
                    count = session.query(OCROperationRecord).filter(
                        OCROperationRecord.method_used == method.value
                    ).count()
                    method_stats[method.value] = count
                
                # Average confidence and processing time
                records = session.query(
                    OCROperationRecord.confidence,
                    OCROperationRecord.processing_time
                ).filter(
                    OCROperationRecord.confidence.isnot(None),
                    OCROperationRecord.processing_time.isnot(None)
                ).all()
                
                avg_confidence = sum(r[0] for r in records) / len(records) if records else 0
                avg_processing_time = sum(r[1] for r in records) / len(records) if records else 0
                
                return {
                    "total_operations": total_operations,
                    "operations_by_method": method_stats,
                    "average_confidence": round(avg_confidence, 3),
                    "average_processing_time": round(avg_processing_time, 3),
                    "available_engines": [engine.value for engine in self.available_engines]
                }
                
        except Exception as e:
            logger.error(f"Error getting OCR statistics: {e}")
            return {}

# Factory function
def create_ocr_engine(db_url: str = None, redis_url: str = None, temp_path: str = None) -> OCREngine:
    """Create and configure OCREngine instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    if not temp_path:
        temp_path = "/tmp/insurance_ocr"
    
    return OCREngine(db_url=db_url, redis_url=redis_url, temp_path=temp_path)

# Example usage
if __name__ == "__main__":
    async def test_ocr_engine():
        """Test OCR engine functionality"""
        
        engine = create_ocr_engine()
        
        # Create a test image with text
        from PIL import Image, ImageDraw, ImageFont
        
        # Create test image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add text
        text = "INSURANCE POLICY APPLICATION\nPolicy Number: POL-2024-001234\nApplicant: John Smith\nCoverage: $500,000"
        
        try:
            # Try to use a font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.multiline_text((50, 50), text, fill='black', font=font)
        
        # Save test image
        test_image_path = "/tmp/test_insurance_doc.png"
        img.save(test_image_path)
        
        try:
            # Test OCR
            config = OCRConfiguration(
                method=OCRMethod.TESSERACT,
                preprocessing_steps=[
                    PreprocessingStep.ENHANCE_CONTRAST,
                    PreprocessingStep.DENOISE
                ]
            )
            
            result = await engine.process_image(test_image_path, config)
            
            print(f"OCR Result:")
            print(f"Text: {result.text}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Image Quality: {result.image_quality.value}")
            print(f"Processing Time: {result.processing_time:.3f}s")
            print(f"Method Used: {result.method_used.value}")
            
            # Get statistics
            stats = await engine.get_ocr_statistics()
            print(f"Statistics: {stats}")
            
        finally:
            # Clean up
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
    
    # Run test
    # asyncio.run(test_ocr_engine())

