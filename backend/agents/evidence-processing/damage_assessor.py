"""
Damage Assessor - Production Ready Implementation
Specialized damage assessment for insurance claims
"""

import asyncio
import json
import logging
import os
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Computer Vision and ML
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from skimage import measure, morphology, filters, segmentation
from scipy import ndimage
import matplotlib.pyplot as plt

# Monitoring
from prometheus_client import Counter, Histogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

damage_assessment_total = Counter('damage_assessment_total', 'Total damage assessments', ['damage_type', 'severity'])
damage_assessment_duration = Histogram('damage_assessment_duration_seconds', 'Time to assess damage')

Base = declarative_base()

class DamageCategory(Enum):
    STRUCTURAL = "structural"
    COSMETIC = "cosmetic"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    INTERIOR = "interior"
    GLASS = "glass"
    PAINT = "paint"
    BODY_PANEL = "body_panel"

class VehicleComponent(Enum):
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    HOOD = "hood"
    TRUNK = "trunk"
    DOOR_FRONT_LEFT = "door_front_left"
    DOOR_FRONT_RIGHT = "door_front_right"
    DOOR_REAR_LEFT = "door_rear_left"
    DOOR_REAR_RIGHT = "door_rear_right"
    FENDER_LEFT = "fender_left"
    FENDER_RIGHT = "fender_right"
    ROOF = "roof"
    WINDSHIELD = "windshield"
    REAR_WINDOW = "rear_window"
    SIDE_WINDOW = "side_window"
    HEADLIGHT = "headlight"
    TAILLIGHT = "taillight"
    MIRROR = "mirror"
    WHEEL = "wheel"
    TIRE = "tire"
    GRILLE = "grille"
    UNKNOWN = "unknown"

@dataclass
class DamageArea:
    component: VehicleComponent
    category: DamageCategory
    severity_score: float
    area_percentage: float
    repair_complexity: str
    estimated_hours: float
    parts_required: List[str]
    description: str
    confidence: float

@dataclass
class RepairEstimate:
    labor_hours: float
    labor_rate: float
    parts_cost: float
    paint_cost: float
    total_cost: float
    repair_time_days: int

@dataclass
class DamageAssessmentResult:
    assessment_id: str
    evidence_id: str
    vehicle_type: str
    damage_areas: List[DamageArea]
    overall_severity: str
    total_repair_estimate: RepairEstimate
    salvage_recommendation: bool
    processing_time: float
    confidence_score: float
    assessor_notes: List[str]
    requires_inspection: bool

class DamageAssessmentRecord(Base):
    __tablename__ = 'damage_assessments'
    
    assessment_id = Column(String, primary_key=True)
    evidence_id = Column(String, index=True)
    vehicle_type = Column(String)
    damage_areas = Column(JSON)
    overall_severity = Column(String)
    total_repair_estimate = Column(JSON)
    salvage_recommendation = Column(Boolean)
    processing_time = Column(Float)
    confidence_score = Column(Float)
    assessor_notes = Column(JSON)
    requires_inspection = Column(Boolean)
    created_at = Column(DateTime, nullable=False)

class DamageAssessor:
    """Production-ready Damage Assessor for insurance claims"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # Repair cost database
        self.labor_rates = {
            "body_shop": 85.0,
            "dealership": 125.0,
            "specialty": 150.0
        }
        
        self.parts_costs = {
            VehicleComponent.FRONT_BUMPER: {"economy": 400, "mid_range": 800, "luxury": 1500},
            VehicleComponent.REAR_BUMPER: {"economy": 350, "mid_range": 700, "luxury": 1300},
            VehicleComponent.HOOD: {"economy": 600, "mid_range": 1200, "luxury": 2500},
            VehicleComponent.DOOR_FRONT_LEFT: {"economy": 800, "mid_range": 1500, "luxury": 3000},
            VehicleComponent.DOOR_FRONT_RIGHT: {"economy": 800, "mid_range": 1500, "luxury": 3000},
            VehicleComponent.WINDSHIELD: {"economy": 300, "mid_range": 500, "luxury": 1200},
            VehicleComponent.HEADLIGHT: {"economy": 200, "mid_range": 600, "luxury": 1500},
            VehicleComponent.FENDER_LEFT: {"economy": 400, "mid_range": 800, "luxury": 1600},
            VehicleComponent.FENDER_RIGHT: {"economy": 400, "mid_range": 800, "luxury": 1600}
        }
        
        self.repair_hours = {
            DamageCategory.COSMETIC: {"minor": 2, "moderate": 6, "severe": 12},
            DamageCategory.STRUCTURAL: {"minor": 8, "moderate": 20, "severe": 40},
            DamageCategory.MECHANICAL: {"minor": 4, "moderate": 12, "severe": 24},
            DamageCategory.GLASS: {"minor": 1, "moderate": 2, "severe": 4},
            DamageCategory.PAINT: {"minor": 4, "moderate": 8, "severe": 16}
        }
        
        logger.info("DamageAssessor initialized successfully")

    async def assess_damage(self, image_path: str, evidence_id: str = None, 
                          vehicle_info: Dict[str, Any] = None) -> DamageAssessmentResult:
        """Perform comprehensive damage assessment"""
        
        start_time = datetime.utcnow()
        assessment_id = str(uuid.uuid4())
        
        with damage_assessment_duration.time():
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError("Could not load image")
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect and analyze damage areas
                damage_areas = await self._detect_damage_areas(image)
                
                # Classify vehicle type
                vehicle_type = self._classify_vehicle_type(image, vehicle_info)
                
                # Assess each damage area
                assessed_areas = []
                for area in damage_areas:
                    assessed_area = await self._assess_damage_area(area, image, vehicle_type)
                    assessed_areas.append(assessed_area)
                
                # Calculate overall assessment
                overall_severity = self._calculate_overall_severity(assessed_areas)
                total_estimate = self._calculate_total_estimate(assessed_areas, vehicle_type)
                salvage_recommendation = self._evaluate_salvage_status(total_estimate, vehicle_type)
                confidence_score = self._calculate_confidence(assessed_areas)
                assessor_notes = self._generate_assessor_notes(assessed_areas, total_estimate)
                requires_inspection = self._requires_physical_inspection(assessed_areas, total_estimate)
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                result = DamageAssessmentResult(
                    assessment_id=assessment_id,
                    evidence_id=evidence_id or "",
                    vehicle_type=vehicle_type,
                    damage_areas=assessed_areas,
                    overall_severity=overall_severity,
                    total_repair_estimate=total_estimate,
                    salvage_recommendation=salvage_recommendation,
                    processing_time=processing_time,
                    confidence_score=confidence_score,
                    assessor_notes=assessor_notes,
                    requires_inspection=requires_inspection
                )
                
                await self._store_assessment(result)
                
                damage_assessment_total.labels(
                    damage_type=overall_severity,
                    severity=overall_severity
                ).inc()
                
                return result
                
            except Exception as e:
                logger.error(f"Error assessing damage: {e}")
                raise

    async def _detect_damage_areas(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potential damage areas using computer vision"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Edge detection for structural damage
            edges = cv2.Canny(gray, 30, 100)
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            damage_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Filter small areas
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract region of interest
                    roi = image[y:y+h, x:x+w]
                    
                    # Analyze color characteristics
                    mean_color = np.mean(roi, axis=(0, 1))
                    std_color = np.std(roi, axis=(0, 1))
                    
                    # Calculate damage indicators
                    color_variance = np.sum(std_color)
                    brightness_anomaly = abs(np.mean(mean_color) - 128) / 128
                    
                    damage_areas.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'roi': roi,
                        'mean_color': mean_color,
                        'color_variance': color_variance,
                        'brightness_anomaly': brightness_anomaly
                    })
            
            return damage_areas
            
        except Exception as e:
            logger.error(f"Error detecting damage areas: {e}")
            return []

    def _classify_vehicle_type(self, image: np.ndarray, vehicle_info: Dict[str, Any] = None) -> str:
        """Classify vehicle type from image or provided info"""
        
        try:
            if vehicle_info and 'type' in vehicle_info:
                return vehicle_info['type'].lower()
            
            # Simple heuristic based on image dimensions and features
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            if aspect_ratio > 2.0:
                return "truck"
            elif aspect_ratio > 1.8:
                return "suv"
            elif aspect_ratio > 1.5:
                return "sedan"
            else:
                return "compact"
                
        except Exception:
            return "unknown"

    async def _assess_damage_area(self, damage_area: Dict[str, Any], 
                                image: np.ndarray, vehicle_type: str) -> DamageArea:
        """Assess individual damage area"""
        
        try:
            # Determine component based on location
            component = self._identify_component(damage_area['bbox'], image.shape)
            
            # Classify damage category
            category = self._classify_damage_category(damage_area)
            
            # Calculate severity
            severity_score = self._calculate_severity_score(damage_area)
            
            # Determine repair complexity
            repair_complexity = self._assess_repair_complexity(component, category, severity_score)
            
            # Estimate repair hours
            estimated_hours = self._estimate_repair_hours(category, repair_complexity)
            
            # Identify required parts
            parts_required = self._identify_required_parts(component, category, severity_score)
            
            # Generate description
            description = self._generate_damage_description(component, category, severity_score)
            
            # Calculate confidence
            confidence = self._calculate_area_confidence(damage_area)
            
            # Calculate area percentage
            total_area = image.shape[0] * image.shape[1]
            area_percentage = (damage_area['area'] / total_area) * 100
            
            return DamageArea(
                component=component,
                category=category,
                severity_score=severity_score,
                area_percentage=area_percentage,
                repair_complexity=repair_complexity,
                estimated_hours=estimated_hours,
                parts_required=parts_required,
                description=description,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error assessing damage area: {e}")
            return DamageArea(
                component=VehicleComponent.UNKNOWN,
                category=DamageCategory.COSMETIC,
                severity_score=0.5,
                area_percentage=0.0,
                repair_complexity="unknown",
                estimated_hours=0.0,
                parts_required=[],
                description="Assessment failed",
                confidence=0.0
            )

    def _identify_component(self, bbox: Tuple[int, int, int, int], 
                          image_shape: Tuple[int, int]) -> VehicleComponent:
        """Identify vehicle component based on damage location"""
        
        try:
            x, y, w, h = bbox
            img_height, img_width = image_shape[:2]
            
            # Normalize coordinates
            center_x = (x + w/2) / img_width
            center_y = (y + h/2) / img_height
            
            # Simple heuristic mapping based on typical vehicle photo composition
            if center_y < 0.3:  # Upper portion
                if center_x < 0.5:
                    return VehicleComponent.HOOD
                else:
                    return VehicleComponent.ROOF
            elif center_y < 0.7:  # Middle portion
                if center_x < 0.2:
                    return VehicleComponent.FENDER_LEFT
                elif center_x > 0.8:
                    return VehicleComponent.FENDER_RIGHT
                elif center_x < 0.4:
                    return VehicleComponent.DOOR_FRONT_LEFT
                elif center_x > 0.6:
                    return VehicleComponent.DOOR_FRONT_RIGHT
                else:
                    return VehicleComponent.FRONT_BUMPER
            else:  # Lower portion
                if center_x < 0.3:
                    return VehicleComponent.WHEEL
                elif center_x > 0.7:
                    return VehicleComponent.WHEEL
                else:
                    return VehicleComponent.REAR_BUMPER
                    
        except Exception:
            return VehicleComponent.UNKNOWN

    def _classify_damage_category(self, damage_area: Dict[str, Any]) -> DamageCategory:
        """Classify damage category based on visual characteristics"""
        
        try:
            color_variance = damage_area['color_variance']
            brightness_anomaly = damage_area['brightness_anomaly']
            area = damage_area['area']
            
            # Heuristic classification
            if color_variance > 50 and brightness_anomaly > 0.3:
                return DamageCategory.STRUCTURAL
            elif color_variance > 30:
                return DamageCategory.BODY_PANEL
            elif brightness_anomaly > 0.4:
                return DamageCategory.PAINT
            elif area < 5000:
                return DamageCategory.COSMETIC
            else:
                return DamageCategory.BODY_PANEL
                
        except Exception:
            return DamageCategory.COSMETIC

    def _calculate_severity_score(self, damage_area: Dict[str, Any]) -> float:
        """Calculate severity score (0.0 to 1.0)"""
        
        try:
            area_score = min(damage_area['area'] / 50000, 1.0)
            color_score = min(damage_area['color_variance'] / 100, 1.0)
            brightness_score = damage_area['brightness_anomaly']
            
            severity = (area_score * 0.4 + color_score * 0.3 + brightness_score * 0.3)
            return min(severity, 1.0)
            
        except Exception:
            return 0.5

    def _assess_repair_complexity(self, component: VehicleComponent, 
                                category: DamageCategory, severity_score: float) -> str:
        """Assess repair complexity"""
        
        try:
            complexity_matrix = {
                DamageCategory.COSMETIC: {"low": 0.3, "medium": 0.7},
                DamageCategory.PAINT: {"low": 0.4, "medium": 0.8},
                DamageCategory.BODY_PANEL: {"low": 0.2, "medium": 0.6},
                DamageCategory.STRUCTURAL: {"low": 0.1, "medium": 0.4},
                DamageCategory.GLASS: {"low": 0.5, "medium": 0.9}
            }
            
            thresholds = complexity_matrix.get(category, {"low": 0.3, "medium": 0.7})
            
            if severity_score <= thresholds["low"]:
                return "simple"
            elif severity_score <= thresholds["medium"]:
                return "moderate"
            else:
                return "complex"
                
        except Exception:
            return "moderate"

    def _estimate_repair_hours(self, category: DamageCategory, complexity: str) -> float:
        """Estimate repair hours"""
        
        try:
            base_hours = self.repair_hours.get(category, {"minor": 4, "moderate": 8, "severe": 16})
            
            complexity_multipliers = {
                "simple": 0.8,
                "moderate": 1.0,
                "complex": 1.5
            }
            
            severity_map = {
                "simple": "minor",
                "moderate": "moderate", 
                "complex": "severe"
            }
            
            base = base_hours[severity_map.get(complexity, "moderate")]
            multiplier = complexity_multipliers.get(complexity, 1.0)
            
            return base * multiplier
            
        except Exception:
            return 8.0  # Default 8 hours

    def _identify_required_parts(self, component: VehicleComponent, 
                               category: DamageCategory, severity_score: float) -> List[str]:
        """Identify required replacement parts"""
        
        try:
            parts = []
            
            if severity_score > 0.7:  # Severe damage likely requires replacement
                if component in [VehicleComponent.FRONT_BUMPER, VehicleComponent.REAR_BUMPER]:
                    parts.append(f"{component.value}_assembly")
                elif component in [VehicleComponent.DOOR_FRONT_LEFT, VehicleComponent.DOOR_FRONT_RIGHT]:
                    parts.append(f"{component.value}_shell")
                    if severity_score > 0.8:
                        parts.append(f"{component.value}_window")
                elif component == VehicleComponent.WINDSHIELD:
                    parts.append("windshield_glass")
                elif component in [VehicleComponent.HEADLIGHT, VehicleComponent.TAILLIGHT]:
                    parts.append(f"{component.value}_assembly")
            
            # Add consumables based on category
            if category == DamageCategory.PAINT:
                parts.extend(["primer", "base_coat", "clear_coat"])
            elif category == DamageCategory.BODY_PANEL:
                parts.extend(["body_filler", "primer", "paint"])
            
            return parts
            
        except Exception:
            return []

    def _generate_damage_description(self, component: VehicleComponent, 
                                   category: DamageCategory, severity_score: float) -> str:
        """Generate human-readable damage description"""
        
        try:
            severity_terms = {
                (0.0, 0.3): "minor",
                (0.3, 0.6): "moderate", 
                (0.6, 0.8): "significant",
                (0.8, 1.0): "severe"
            }
            
            severity_term = "moderate"
            for (low, high), term in severity_terms.items():
                if low <= severity_score < high:
                    severity_term = term
                    break
            
            component_name = component.value.replace("_", " ").title()
            category_name = category.value.replace("_", " ")
            
            return f"{severity_term.capitalize()} {category_name} damage to {component_name}"
            
        except Exception:
            return "Damage assessment incomplete"

    def _calculate_area_confidence(self, damage_area: Dict[str, Any]) -> float:
        """Calculate confidence in damage area assessment"""
        
        try:
            # Base confidence on area size and visual characteristics
            area_confidence = min(damage_area['area'] / 10000, 1.0) * 0.4
            color_confidence = min(damage_area['color_variance'] / 50, 1.0) * 0.3
            brightness_confidence = damage_area['brightness_anomaly'] * 0.3
            
            return min(area_confidence + color_confidence + brightness_confidence, 1.0)
            
        except Exception:
            return 0.5

    def _calculate_overall_severity(self, damage_areas: List[DamageArea]) -> str:
        """Calculate overall damage severity"""
        
        try:
            if not damage_areas:
                return "none"
            
            avg_severity = sum(area.severity_score for area in damage_areas) / len(damage_areas)
            total_area = sum(area.area_percentage for area in damage_areas)
            
            # Adjust for total affected area
            adjusted_severity = avg_severity * (1 + total_area / 100)
            
            if adjusted_severity >= 0.8:
                return "severe"
            elif adjusted_severity >= 0.6:
                return "moderate"
            elif adjusted_severity >= 0.3:
                return "minor"
            else:
                return "minimal"
                
        except Exception:
            return "unknown"

    def _calculate_total_estimate(self, damage_areas: List[DamageArea], vehicle_type: str) -> RepairEstimate:
        """Calculate total repair estimate"""
        
        try:
            total_hours = sum(area.estimated_hours for area in damage_areas)
            labor_rate = self.labor_rates["body_shop"]  # Default rate
            
            # Adjust labor rate based on vehicle type
            if vehicle_type in ["luxury", "exotic"]:
                labor_rate = self.labor_rates["specialty"]
            elif vehicle_type in ["truck", "suv"]:
                labor_rate = self.labor_rates["dealership"]
            
            # Calculate parts cost
            total_parts_cost = 0.0
            vehicle_class = "mid_range"  # Default
            if vehicle_type in ["luxury", "exotic"]:
                vehicle_class = "luxury"
            elif vehicle_type in ["compact", "economy"]:
                vehicle_class = "economy"
            
            for area in damage_areas:
                if area.component in self.parts_costs:
                    part_cost = self.parts_costs[area.component].get(vehicle_class, 500)
                    # Adjust based on severity
                    if area.severity_score > 0.7:
                        total_parts_cost += part_cost
                    elif area.severity_score > 0.4:
                        total_parts_cost += part_cost * 0.6  # Repair vs replace
                    else:
                        total_parts_cost += part_cost * 0.3  # Minor repair
            
            # Paint cost estimation
            paint_areas = [area for area in damage_areas if area.category == DamageCategory.PAINT]
            paint_cost = len(paint_areas) * 300  # Base paint cost per panel
            
            total_cost = (total_hours * labor_rate) + total_parts_cost + paint_cost
            
            # Estimate repair time
            repair_days = max(1, int(total_hours / 8))  # 8 hours per day
            if total_cost > 10000:
                repair_days += 3  # Additional time for complex repairs
            
            return RepairEstimate(
                labor_hours=total_hours,
                labor_rate=labor_rate,
                parts_cost=total_parts_cost,
                paint_cost=paint_cost,
                total_cost=total_cost,
                repair_time_days=repair_days
            )
            
        except Exception as e:
            logger.error(f"Error calculating repair estimate: {e}")
            return RepairEstimate(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    def _evaluate_salvage_status(self, estimate: RepairEstimate, vehicle_type: str) -> bool:
        """Evaluate if vehicle should be considered salvage"""
        
        try:
            # Estimated vehicle values by type
            vehicle_values = {
                "economy": 8000,
                "compact": 12000,
                "sedan": 18000,
                "suv": 25000,
                "truck": 30000,
                "luxury": 50000,
                "exotic": 100000
            }
            
            estimated_value = vehicle_values.get(vehicle_type, 15000)
            
            # Salvage if repair cost exceeds 75% of vehicle value
            return estimate.total_cost > (estimated_value * 0.75)
            
        except Exception:
            return False

    def _calculate_confidence(self, damage_areas: List[DamageArea]) -> float:
        """Calculate overall assessment confidence"""
        
        try:
            if not damage_areas:
                return 0.0
            
            avg_confidence = sum(area.confidence for area in damage_areas) / len(damage_areas)
            
            # Adjust based on number of damage areas
            area_count_factor = min(len(damage_areas) / 5, 1.0)
            
            return avg_confidence * (0.7 + area_count_factor * 0.3)
            
        except Exception:
            return 0.5

    def _generate_assessor_notes(self, damage_areas: List[DamageArea], 
                               estimate: RepairEstimate) -> List[str]:
        """Generate assessor notes"""
        
        notes = []
        
        try:
            if estimate.total_cost > 15000:
                notes.append("High repair cost - recommend detailed inspection")
            
            structural_damage = [area for area in damage_areas 
                               if area.category == DamageCategory.STRUCTURAL]
            if structural_damage:
                notes.append("Structural damage detected - safety inspection required")
            
            complex_repairs = [area for area in damage_areas 
                             if area.repair_complexity == "complex"]
            if len(complex_repairs) > 2:
                notes.append("Multiple complex repairs - consider specialist shop")
            
            if estimate.repair_time_days > 14:
                notes.append("Extended repair time - arrange alternative transportation")
            
            low_confidence_areas = [area for area in damage_areas if area.confidence < 0.6]
            if low_confidence_areas:
                notes.append("Some damage areas require manual verification")
            
            return notes
            
        except Exception:
            return ["Assessment completed with standard recommendations"]

    def _requires_physical_inspection(self, damage_areas: List[DamageArea], 
                                    estimate: RepairEstimate) -> bool:
        """Determine if physical inspection is required"""
        
        try:
            # Require inspection for high-cost repairs
            if estimate.total_cost > 10000:
                return True
            
            # Require inspection for structural damage
            structural_damage = any(area.category == DamageCategory.STRUCTURAL 
                                  for area in damage_areas)
            if structural_damage:
                return True
            
            # Require inspection for low confidence assessments
            low_confidence = any(area.confidence < 0.5 for area in damage_areas)
            if low_confidence:
                return True
            
            return False
            
        except Exception:
            return True  # Default to requiring inspection

    async def _store_assessment(self, result: DamageAssessmentResult):
        """Store damage assessment in database"""
        
        try:
            with self.Session() as session:
                record = DamageAssessmentRecord(
                    assessment_id=result.assessment_id,
                    evidence_id=result.evidence_id,
                    vehicle_type=result.vehicle_type,
                    damage_areas=[asdict(area) for area in result.damage_areas],
                    overall_severity=result.overall_severity,
                    total_repair_estimate=asdict(result.total_repair_estimate),
                    salvage_recommendation=result.salvage_recommendation,
                    processing_time=result.processing_time,
                    confidence_score=result.confidence_score,
                    assessor_notes=result.assessor_notes,
                    requires_inspection=result.requires_inspection,
                    created_at=datetime.utcnow()
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing damage assessment: {e}")

def create_damage_assessor(db_url: str = None, redis_url: str = None) -> DamageAssessor:
    """Create and configure DamageAssessor instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return DamageAssessor(db_url=db_url, redis_url=redis_url)

