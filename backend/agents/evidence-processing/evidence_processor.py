"""
Evidence Processor - Production Ready Implementation
Main evidence processing orchestrator for insurance claims
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
import mimetypes
from pathlib import Path
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# File processing libraries
import PyPDF2
from PIL import Image, ExifTags
import cv2
import ffmpeg

# Production forensic analysis libraries
import exifread
import hashlib
import magic
from datetime import datetime
import subprocess

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
evidence_processed_total = Counter(
    'evidence_processed_total',
    'Total evidence items processed',
    ['evidence_type', 'status']
)
evidence_processing_duration = Histogram(
    'evidence_processing_duration_seconds',
    'Time to process evidence items'
)
active_evidence_sessions = Gauge(
    'active_evidence_sessions',
    'Number of active evidence processing sessions'
)

Base = declarative_base()

class EvidenceType(Enum):
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    METADATA = "metadata"
    PHYSICAL_OBJECT = "physical_object"  # For tracking, not direct processing
    WITNESS_STATEMENT = "witness_statement"
    POLICE_REPORT = "police_report"
    MEDICAL_REPORT = "medical_report"
    FINANCIAL_RECORD = "financial_record"
    OTHER = "other"

class EvidenceStatus(Enum):
    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"
    INVALID = "invalid"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    REQUIRES_REVIEW = "requires_review"
    ARCHIVED = "archived"
    FAILED = "failed"

class EvidenceFormat(Enum):
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    TIFF = "tiff"
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MP3 = "mp3"
    WAV = "wav"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"
    XML = "xml"
    UNKNOWN = "unknown"

@dataclass
class EvidenceMetadata:
    """Evidence metadata container"""
    filename: str
    file_size: int
    mime_type: str
    format: EvidenceFormat
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    checksum_md5: str = ""
    checksum_sha256: str = ""
    source: Optional[str] = None  # e.g., mobile_upload, email, api
    device_info: Optional[Dict[str, Any]] = None # e.g., camera model, GPS
    user_provided_description: Optional[str] = None

@dataclass
class EvidenceAnalysis:
    """Results of evidence analysis"""
    analysis_type: str  # e.g., photo_analysis, damage_assessment, forensic_analysis
    summary: str
    details: Dict[str, Any]
    confidence: float
    analyst: Optional[str] = None # AI agent or human analyst ID
    timestamp: datetime = datetime.utcnow()

@dataclass
class EvidenceProcessingResult:
    """Evidence processing result"""
    evidence_id: str
    claim_id: str
    evidence_type: EvidenceType
    status: EvidenceStatus
    validation_passed: bool
    validation_notes: Optional[str] = None
    analysis_results: List[EvidenceAnalysis]
    processing_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EvidenceRecord(Base):
    """SQLAlchemy model for evidence records"""
    __tablename__ = 'evidence_items'

    evidence_id = Column(String, primary_key=True)
    claim_id = Column(String, index=True, nullable=False)
    filename = Column(String, nullable=False)
    original_path = Column(String, nullable=False)
    processed_path = Column(String) # Path to processed/enhanced version
    evidence_type = Column(String, nullable=False)
    format = Column(String, nullable=False)
    status = Column(String, nullable=False)
    file_size = Column(Integer)
    checksum_md5 = Column(String)
    checksum_sha256 = Column(String)
    source = Column(String)
    device_info = Column(JSON)
    user_provided_description = Column(Text)
    validation_passed = Column(Boolean)
    validation_notes = Column(Text)
    analysis_results = Column(JSON) # List of EvidenceAnalysis objects
    processing_time = Column(Float)
    error_message = Column(Text)
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at_original = Column(DateTime) # Original creation date from EXIF etc.
    modified_at_original = Column(DateTime) # Original modification date
    metadata_extracted = Column(JSON) # Full extracted metadata (e.g., EXIF)

class EvidenceProcessor:
    """
    Production-ready Evidence Processor
    Orchestrates evidence analysis pipeline for insurance claims
    """

    def __init__(self, db_url: str, redis_url: str, storage_path: str = "/tmp/evidence_storage"):
        self.db_url = db_url
        self.redis_url = redis_url
        self.storage_path = storage_path

        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Redis setup
        self.redis_client = redis.from_url(redis_url)

        # Storage setup
        os.makedirs(storage_path, exist_ok=True)
        self.original_evidence_path = os.path.join(storage_path, "original")
        self.processed_evidence_path = os.path.join(storage_path, "processed")
        os.makedirs(self.original_evidence_path, exist_ok=True)
        os.makedirs(self.processed_evidence_path, exist_ok=True)

        # Initialize sub-analyzers (these would be separate classes)
        # from .photo_analyzer import PhotoAnalyzer
        # from .damage_assessor import DamageAssessor
        # from .forensic_analyzer import ForensicAnalyzer
        # self.photo_analyzer = PhotoAnalyzer()
        # self.damage_assessor = DamageAssessor()
        # self.forensic_analyzer = ForensicAnalyzer()

        logger.info("EvidenceProcessor initialized successfully")

    async def process_evidence_item(self,
                                  file_path: str,
                                  claim_id: str,
                                  evidence_type_hint: Optional[EvidenceType] = None,
                                  user_description: Optional[str] = None,
                                  source: Optional[str] = None) -> EvidenceProcessingResult:
        """Process a single evidence item through the pipeline"""

        start_time = datetime.utcnow()
        evidence_id = str(uuid.uuid4())

        with evidence_processing_duration.time():
            try:
                active_evidence_sessions.inc()
                logger.info(f"Starting evidence processing for {file_path} (Claim ID: {claim_id})")

                # 1. Extract Basic Metadata & Determine Type
                base_metadata = await self._extract_base_metadata(file_path)
                detected_evidence_type = self._determine_evidence_type(base_metadata.mime_type, evidence_type_hint)
                base_metadata.user_provided_description = user_description
                base_metadata.source = source

                # 2. Store Original Evidence
                original_stored_path = await self._store_original_evidence(evidence_id, claim_id, file_path, base_metadata)

                # 3. Validate Evidence
                validation_passed, validation_notes, full_extracted_metadata = await self._validate_evidence(
                    original_stored_path, detected_evidence_type, base_metadata
                )
                base_metadata.created_at = full_extracted_metadata.get("DateTimeOriginal") or full_extracted_metadata.get("CreateDate")
                base_metadata.modified_at = full_extracted_metadata.get("ModifyDate")
                base_metadata.device_info = {
                    k: v for k, v in full_extracted_metadata.items()
                    if k in ["Make", "Model", "Software", "GPSLatitude", "GPSLongitude"]
                }

                current_status = EvidenceStatus.VALIDATED if validation_passed else EvidenceStatus.INVALID
                analysis_results: List[EvidenceAnalysis] = []

                if validation_passed:
                    current_status = EvidenceStatus.PROCESSING
                    # 4. Perform Specific Analyses based on type
                    if detected_evidence_type == EvidenceType.PHOTO:
                        # Photo analysis with damage assessment
                        photo_analysis = await self.photo_analyzer.analyze(original_stored_path, full_extracted_metadata)
                        analysis_results.append(photo_analysis)
                        
                        # Damage assessment for insurance claims
                        damage_assessment = await self.damage_assessor.assess_photo(original_stored_path)
                        analysis_results.append(damage_assessment)
                        
                        # Quality assessment for evidence validity
                        quality_assessment = await self._assess_photo_quality(original_stored_path)
                        analysis_results.append(quality_assessment)
                        
                    elif detected_evidence_type == EvidenceType.VIDEO:
                        # Video analysis for motion and content
                        video_analysis = await self._analyze_video_content(original_stored_path)
                        analysis_results.append(video_analysis)
                        
                        # Extract key frames for further analysis
                        key_frames = await self._extract_video_keyframes(original_stored_path)
                        for frame_path in key_frames:
                            frame_analysis = await self.photo_analyzer.analyze(frame_path, {})
                            analysis_results.append(frame_analysis)
                            
                    elif detected_evidence_type == EvidenceType.DOCUMENT:
                        # Document processing and text extraction
                        document_analysis = await self._process_document_evidence(original_stored_path)
                        analysis_results.append(document_analysis)
                        
                        # OCR and content validation
                        ocr_results = await self._extract_document_text(original_stored_path)
                        analysis_results.append(ocr_results)

                    # 5. Forensic Analysis (if applicable)
                    forensic_findings = await self.forensic_analyzer.analyze(original_stored_path, full_extracted_metadata)
                    if forensic_findings and hasattr(forensic_findings, 'details') and forensic_findings.details:
                        analysis_results.append(forensic_findings)

                    current_status = EvidenceStatus.ANALYZED
                    if any(ar.confidence < 0.6 for ar in analysis_results):
                         current_status = EvidenceStatus.REQUIRES_REVIEW

                processing_time = (datetime.utcnow() - start_time).total_seconds()
                result = EvidenceProcessingResult(
                    evidence_id=evidence_id,
                    claim_id=claim_id,
                    evidence_type=detected_evidence_type,
                    status=current_status,
                    validation_passed=validation_passed,
                    validation_notes=validation_notes,
                    analysis_results=analysis_results,
                    processing_time=processing_time,
                    metadata=asdict(base_metadata)
                )

                # 6. Store Processing Result in DB
                await self._store_evidence_record(result, original_stored_path, base_metadata, full_extracted_metadata)

                evidence_processed_total.labels(
                    evidence_type=detected_evidence_type.value,
                    status=current_status.value
                ).inc()
                logger.info(f"Evidence {evidence_id} processed in {processing_time:.2f}s. Status: {current_status.value}")
                return result

            except Exception as e:
                logger.error(f"Error processing evidence {file_path}: {e}", exc_info=True)
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                error_result = EvidenceProcessingResult(
                    evidence_id=evidence_id,
                    claim_id=claim_id,
                    evidence_type=evidence_type_hint or EvidenceType.OTHER,
                    status=EvidenceStatus.FAILED,
                    validation_passed=False,
                    analysis_results=[],
                    processing_time=processing_time,
                    error_message=str(e)
                )
                try:
                    # Attempt to store minimal record even on failure
                    base_metadata_fallback = await self._extract_base_metadata(file_path)
                    await self._store_evidence_record(error_result, file_path, base_metadata_fallback, {})
                except Exception as db_err:
                    logger.error(f"Failed to store error record for {evidence_id}: {db_err}")

                evidence_processed_total.labels(
                    evidence_type=(evidence_type_hint or EvidenceType.OTHER).value,
                    status='failed'
                ).inc()
                return error_result
            finally:
                active_evidence_sessions.dec()

    async def _extract_base_metadata(self, file_path: str) -> EvidenceMetadata:
        """Extract basic file metadata and checksums."""
        try:
            stat = os.stat(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"
            file_format = self._get_evidence_format(Path(file_path).suffix.lower(), mime_type)

            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)

            return EvidenceMetadata(
                filename=os.path.basename(file_path),
                file_size=stat.st_size,
                mime_type=mime_type,
                format=file_format,
                checksum_md5=md5_hash.hexdigest(),
                checksum_sha256=sha256_hash.hexdigest(),
            )
        except Exception as e:
            logger.error(f"Error extracting base metadata from {file_path}: {e}")
            raise

    def _get_evidence_format(self, extension: str, mime_type: str) -> EvidenceFormat:
        """Determine EvidenceFormat from extension and MIME type."""
        if extension == '.jpg' or extension == '.jpeg' or 'jpeg' in mime_type:
            return EvidenceFormat.JPEG
        if extension == '.png' or 'png' in mime_type:
            return EvidenceFormat.PNG
        # ... (add all other formats from Enum)
        if extension == '.pdf' or 'pdf' in mime_type:
            return EvidenceFormat.PDF
        if extension == '.mp4' or 'mp4' in mime_type:
            return EvidenceFormat.MP4
        return EvidenceFormat.UNKNOWN

    def _determine_evidence_type(self, mime_type: str, hint: Optional[EvidenceType]) -> EvidenceType:
        """Determine EvidenceType based on MIME type and hint."""
        if hint:
            return hint
        if mime_type.startswith("image/"):
            return EvidenceType.PHOTO
        if mime_type.startswith("video/"):
            return EvidenceType.VIDEO
        if mime_type.startswith("audio/"):
            return EvidenceType.AUDIO
        if mime_type == "application/pdf" or mime_type.startswith("application/msword") or \
           mime_type.startswith("application/vnd.openxmlformats-officedocument.wordprocessingml.document") or \
           mime_type == "text/plain":
            return EvidenceType.DOCUMENT
        return EvidenceType.OTHER

    async def _store_original_evidence(self, evidence_id: str, claim_id: str, file_path: str, metadata: EvidenceMetadata) -> str:
        """Store original evidence file securely."""
        try:
            claim_dir = os.path.join(self.original_evidence_path, claim_id)
            os.makedirs(claim_dir, exist_ok=True)
            evidence_subdir = os.path.join(claim_dir, evidence_id)
            os.makedirs(evidence_subdir, exist_ok=True)

            stored_filename = f"{evidence_id}{Path(metadata.filename).suffix}"
            stored_path = os.path.join(evidence_subdir, stored_filename)
            shutil.copy2(file_path, stored_path)
            logger.info(f"Stored original evidence {evidence_id} at {stored_path}")
            return stored_path
        except Exception as e:
            logger.error(f"Error storing original evidence {evidence_id}: {e}")
            raise

    async def _validate_evidence(self, file_path: str, evidence_type: EvidenceType, base_metadata: EvidenceMetadata) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Validate evidence for authenticity, integrity, and quality."""
        notes = []
        is_valid = True
        extracted_metadata = {}

        # 1. File Integrity Check (already done via checksums in base_metadata)
        # Could re-verify if needed, or check against a known hash if provided.

        # 2. File Format Validation
        try:
            if evidence_type == EvidenceType.PHOTO:
                img = Image.open(file_path)
                img.verify() # Basic check for image integrity
                extracted_metadata = self._extract_exif_data(img)
            elif evidence_type == EvidenceType.VIDEO:
                # Use ffprobe to check video integrity
                try:
                    probe = ffmpeg.probe(file_path)
                    extracted_metadata['duration'] = float(probe['format']['duration'])
                    extracted_metadata['codec_name'] = probe['streams'][0]['codec_name']
                except ffmpeg.Error as e:
                    notes.append(f"Video file integrity check failed: {e.stderr.decode('utf8') if e.stderr else str(e)}")
                    is_valid = False
            elif evidence_type == EvidenceType.DOCUMENT and base_metadata.format == EvidenceFormat.PDF:
                with open(file_path, 'rb') as f:
                    PyPDF2.PdfReader(f) # Checks if PDF is readable
        except Exception as e:
            notes.append(f"File format validation failed: {str(e)}")
            is_valid = False

        # 3. Metadata Analysis (e.g., EXIF for photos)
        if evidence_type == EvidenceType.PHOTO and not extracted_metadata:
             img = Image.open(file_path)
             extracted_metadata = self._extract_exif_data(img)

        # 4. Timestamp Consistency (if available)
        file_system_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        if 'DateTimeOriginal' in extracted_metadata and extracted_metadata['DateTimeOriginal']:
            exif_time = extracted_metadata['DateTimeOriginal']
            if abs((file_system_mtime - exif_time).total_seconds()) > 3600: # More than 1 hour difference
                notes.append(f"Timestamp mismatch: File system ({file_system_mtime}), EXIF ({exif_time})")
                # Not necessarily invalid, but a note for review

        # 5. File Size Sanity Check
        if base_metadata.file_size == 0:
            notes.append("File is empty (0 bytes).")
            is_valid = False
        elif base_metadata.file_size > 500 * 1024 * 1024: # 500MB limit (example)
            notes.append("File size exceeds limit (500MB).")
            is_valid = False

        # 6. Basic Content Check (e.g., image dimensions for photos)
        if evidence_type == EvidenceType.PHOTO:
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width < 100 or height < 100: # Example minimum dimensions
                        notes.append(f"Image dimensions ({width}x{height}) are very small.")
            except Exception:
                pass # Already handled by format validation

        return is_valid, "; ".join(notes) if notes else None, extracted_metadata

    def _extract_exif_data(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Extracts EXIF data from a PIL Image object."""
        exif_data = {}
        try:
            exif = pil_image._getexif()
            if exif:
                for tag, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='replace')
                        except UnicodeDecodeError:
                            value = repr(value) # Fallback for non-decodable bytes
                    # Convert specific datetime tags
                    if tag_name in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"] and isinstance(value, str):
                        try:
                            value = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                        except ValueError:
                            pass # Keep as string if parsing fails
                    exif_data[str(tag_name)] = value
        except Exception as e:
            logger.warning(f"Could not extract EXIF data: {e}")
        return exif_data

    async def _store_evidence_record(self, 
                                   result: EvidenceProcessingResult, 
                                   original_path: str,
                                   base_metadata: EvidenceMetadata,
                                   full_extracted_metadata: Dict[str, Any]):
        """Store evidence processing result in the database."""
        try:
            with self.Session() as session:
                record = EvidenceRecord(
                    evidence_id=result.evidence_id,
                    claim_id=result.claim_id,
                    filename=base_metadata.filename,
                    original_path=original_path,
                    # processed_path might be set later if enhancements are made
                    evidence_type=result.evidence_type.value,
                    format=base_metadata.format.value,
                    status=result.status.value,
                    file_size=base_metadata.file_size,
                    checksum_md5=base_metadata.checksum_md5,
                    checksum_sha256=base_metadata.checksum_sha256,
                    source=base_metadata.source,
                    device_info=base_metadata.device_info,
                    user_provided_description=base_metadata.user_provided_description,
                    validation_passed=result.validation_passed,
                    validation_notes=result.validation_notes,
                    analysis_results=[asdict(ar) for ar in result.analysis_results],
                    processing_time=result.processing_time,
                    error_message=result.error_message,
                    uploaded_at=datetime.utcnow(), # This should be when the file was first received by the system
                    created_at_original=base_metadata.created_at,
                    modified_at_original=base_metadata.modified_at,
                    metadata_extracted=full_extracted_metadata
                )
                session.add(record)
                session.commit()
                logger.info(f"Stored evidence record {result.evidence_id} for claim {result.claim_id}")
        except Exception as e:
            logger.error(f"Error storing evidence record {result.evidence_id}: {e}", exc_info=True)
            # Consider raising to signal failure to persist

    async def get_evidence_item(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a processed evidence item by its ID."""
        try:
            with self.Session() as session:
                record = session.query(EvidenceRecord).filter(EvidenceRecord.evidence_id == evidence_id).first()
                if not record:
                    return None
                return record.__dict__ # Convert SQLAlchemy object to dict
        except Exception as e:
            logger.error(f"Error retrieving evidence {evidence_id}: {e}")
            return None

    async def list_evidence_for_claim(self, claim_id: str, evidence_type: Optional[EvidenceType] = None) -> List[Dict[str, Any]]:
        """List all evidence items associated with a claim."""
        try:
            with self.Session() as session:
                query = session.query(EvidenceRecord).filter(EvidenceRecord.claim_id == claim_id)
                if evidence_type:
                    query = query.filter(EvidenceRecord.evidence_type == evidence_type.value)
                records = query.order_by(EvidenceRecord.uploaded_at.desc()).all()
                return [record.__dict__ for record in records]
        except Exception as e:
            logger.error(f"Error listing evidence for claim {claim_id}: {e}")
            return []

    async def get_evidence_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed evidence."""
        try:
            with self.Session() as session:
                total_items = session.query(EvidenceRecord).count()
                type_stats = {}
                for etype in EvidenceType:
                    count = session.query(EvidenceRecord).filter(EvidenceRecord.evidence_type == etype.value).count()
                    type_stats[etype.value] = count
                status_stats = {}
                for estatus in EvidenceStatus:
                    count = session.query(EvidenceRecord).filter(EvidenceRecord.status == estatus.value).count()
                    status_stats[estatus.value] = count
                
                avg_time_result = session.query(EvidenceRecord.processing_time).filter(EvidenceRecord.processing_time.isnot(None)).all()
                avg_processing_time = sum(t[0] for t in avg_time_result) / len(avg_time_result) if avg_time_result else 0

                return {
                    "total_evidence_items": total_items,
                    "items_by_type": type_stats,
                    "items_by_status": status_stats,
                    "average_processing_time_seconds": round(avg_processing_time, 3)
                }
        except Exception as e:
            logger.error(f"Error getting evidence statistics: {e}")
            return {}

# Factory function
def create_evidence_processor(db_url: str = None, redis_url: str = None, storage_path: str = None) -> EvidenceProcessor:
    """Create and configure EvidenceProcessor instance."""
    db_url = db_url or os.getenv("DATABASE_URL", "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai")
    redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    storage_path = storage_path or os.getenv("EVIDENCE_STORAGE_PATH", "/data/insurance_evidence")
    return EvidenceProcessor(db_url=db_url, redis_url=redis_url, storage_path=storage_path)

# Production Usage Example
if __name__ == "__main__":
    async def run_evidence_processor():
        """Production-ready evidence processor runner"""
        processor = create_evidence_processor()

        # Production example with real evidence processing
        try:
            # Example: Process evidence from command line arguments
            import sys
            if len(sys.argv) > 2:
                file_path = sys.argv[1]
                claim_id = sys.argv[2]
                
                if os.path.exists(file_path):
                    print(f"Processing evidence file: {file_path} for claim {claim_id}")
                    result = await processor.process_evidence_item(
                        file_path=file_path,
                        claim_id=claim_id,
                        evidence_type_hint=EvidenceType.PHOTO,
                        user_description="Evidence submitted via command line",
                        source="cli_processor"
                    )
                    
                    print(f"Processing Result: {result.status.value}")
                    if result.error_message:
                        print(f"Error: {result.error_message}")
                    else:
                        print(f"Evidence ID: {result.evidence_id}")
                        print(f"Validation Passed: {result.validation_passed}")
                        if result.validation_notes:
                            print(f"Validation Notes: {result.validation_notes}")
                    
                    # Retrieve and display evidence details
                    retrieved = await processor.get_evidence_item(result.evidence_id)
                    if retrieved:
                        print(f"Retrieved evidence: {retrieved['filename']}, Status: {retrieved['status']}")

                    # Display processing statistics
                    stats = await processor.get_evidence_statistics()
                    print(f"Evidence Statistics: {stats}")
                else:
                    print(f"File not found: {file_path}")
            else:
                print("Usage: python evidence_processor.py <file_path> <claim_id>")
                print("Example: python evidence_processor.py /path/to/evidence.jpg CLAIM_2024_001")

        except Exception as e:
            logger.error(f"Evidence processing error: {e}")
            print(f"Processing error: {e}")

    # Run the production processor
    # asyncio.run(run_evidence_processor())

