"""
Utility Functions and Helpers - Production Ready Implementation
Comprehensive utility functions for the Insurance AI Agent System
"""

import asyncio
import json
import logging
import hashlib
import hmac
import secrets
import base64
import uuid
import os
import re
import phonenumbers
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import ssl
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
from functools import wraps
import time
import zipfile
import io
import csv
from PIL import Image
import requests
from urllib.parse import urlparse, parse_qs
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error"""
    pass

class SecurityError(Exception):
    """Custom security error"""
    pass

class APIError(Exception):
    """Custom API error"""
    pass

class DataProcessingError(Exception):
    """Custom data processing error"""
    pass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[Dict[str, Any]] = None

@dataclass
class EncryptionResult:
    encrypted_data: str
    salt: str
    iv: str

@dataclass
class APIResponse:
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    status_code: int
    response_time: float

class SecurityUtils:
    """Production-ready security utilities"""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate API key with specific format"""
        prefix = "iai_"
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}{random_part}"
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    @staticmethod
    def generate_jwt_token(payload: Dict[str, Any], secret_key: str, 
                          expires_in: int = 3600) -> str:
        """Generate JWT token"""
        payload['exp'] = datetime.utcnow() + timedelta(seconds=expires_in)
        payload['iat'] = datetime.utcnow()
        payload['jti'] = str(uuid.uuid4())
        
        return jwt.encode(payload, secret_key, algorithm='HS256')
    
    @staticmethod
    def verify_jwt_token(token: str, secret_key: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            return payload
        except ExpiredSignatureError:
            raise SecurityError("Token has expired")
        except JWTError:
            raise SecurityError("Invalid token")
    
    @staticmethod
    def encrypt_data(data: str, password: str) -> EncryptionResult:
        """Encrypt data using Fernet encryption"""
        # Generate salt
        salt = secrets.token_bytes(16)
        
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        # Encrypt data
        fernet = Fernet(key)
        encrypted = fernet.encrypt(data.encode())
        
        return EncryptionResult(
            encrypted_data=base64.urlsafe_b64encode(encrypted).decode(),
            salt=base64.urlsafe_b64encode(salt).decode(),
            iv=""  # Fernet handles IV internally
        )
    
    @staticmethod
    def decrypt_data(encrypted_result: EncryptionResult, password: str) -> str:
        """Decrypt data using Fernet encryption"""
        # Decode salt
        salt = base64.urlsafe_b64decode(encrypted_result.salt.encode())
        
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        # Decrypt data
        fernet = Fernet(key)
        encrypted_data = base64.urlsafe_b64decode(encrypted_result.encrypted_data.encode())
        decrypted = fernet.decrypt(encrypted_data)
        
        return decrypted.decode()
    
    @staticmethod
    def generate_hmac_signature(data: str, secret_key: str) -> str:
        """Generate HMAC signature for data integrity"""
        signature = hmac.new(
            secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    @staticmethod
    def verify_hmac_signature(data: str, signature: str, secret_key: str) -> bool:
        """Verify HMAC signature"""
        expected_signature = SecurityUtils.generate_hmac_signature(data, secret_key)
        return hmac.compare_digest(signature, expected_signature)

class ValidationUtils:
    """Production-ready validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        """Validate email address"""
        errors = []
        warnings = []
        
        if not email:
            errors.append("Email is required")
            return ValidationResult(False, errors, warnings)
        
        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            errors.append("Invalid email format")
        
        # Check for common typos
        common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        domain = email.split('@')[-1].lower()
        
        if domain not in common_domains and len(domain.split('.')) < 2:
            warnings.append("Unusual email domain detected")
        
        cleaned_email = email.lower().strip()
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={'email': cleaned_email}
        )
    
    @staticmethod
    def validate_phone(phone: str, country_code: str = 'US') -> ValidationResult:
        """Validate phone number using phonenumbers library"""
        errors = []
        warnings = []
        
        if not phone:
            errors.append("Phone number is required")
            return ValidationResult(False, errors, warnings)
        
        try:
            parsed_number = phonenumbers.parse(phone, country_code)
            
            if not phonenumbers.is_valid_number(parsed_number):
                errors.append("Invalid phone number")
            
            if not phonenumbers.is_possible_number(parsed_number):
                errors.append("Phone number is not possible")
            
            # Format phone number
            formatted_phone = phonenumbers.format_number(
                parsed_number, 
                phonenumbers.PhoneNumberFormat.E164
            )
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                cleaned_data={'phone': formatted_phone}
            )
            
        except phonenumbers.NumberParseException as e:
            errors.append(f"Phone number parsing error: {e}")
            return ValidationResult(False, errors, warnings)
    
    @staticmethod
    def validate_ssn(ssn: str) -> ValidationResult:
        """Validate Social Security Number"""
        errors = []
        warnings = []
        
        if not ssn:
            errors.append("SSN is required")
            return ValidationResult(False, errors, warnings)
        
        # Remove formatting
        cleaned_ssn = re.sub(r'[^\d]', '', ssn)
        
        if len(cleaned_ssn) != 9:
            errors.append("SSN must be 9 digits")
        
        # Check for invalid patterns
        invalid_patterns = [
            '000000000', '111111111', '222222222', '333333333',
            '444444444', '555555555', '666666666', '777777777',
            '888888888', '999999999', '123456789'
        ]
        
        if cleaned_ssn in invalid_patterns:
            errors.append("Invalid SSN pattern")
        
        # Check area number (first 3 digits)
        if cleaned_ssn.startswith('000') or cleaned_ssn.startswith('666'):
            errors.append("Invalid SSN area number")
        
        if cleaned_ssn.startswith('9'):
            errors.append("Invalid SSN area number")
        
        # Format SSN
        if len(cleaned_ssn) == 9:
            formatted_ssn = f"{cleaned_ssn[:3]}-{cleaned_ssn[3:5]}-{cleaned_ssn[5:]}"
        else:
            formatted_ssn = cleaned_ssn
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={'ssn': formatted_ssn, 'ssn_raw': cleaned_ssn}
        )
    
    @staticmethod
    def validate_vin(vin: str) -> ValidationResult:
        """Validate Vehicle Identification Number"""
        errors = []
        warnings = []
        
        if not vin:
            errors.append("VIN is required")
            return ValidationResult(False, errors, warnings)
        
        # Clean VIN
        cleaned_vin = vin.upper().strip()
        
        # Check length
        if len(cleaned_vin) != 17:
            errors.append("VIN must be 17 characters")
        
        # Check for invalid characters
        invalid_chars = ['I', 'O', 'Q']
        for char in invalid_chars:
            if char in cleaned_vin:
                errors.append(f"VIN cannot contain '{char}'")
        
        # Check character pattern
        vin_pattern = r'^[A-HJ-NPR-Z0-9]{17}$'
        if not re.match(vin_pattern, cleaned_vin):
            errors.append("VIN contains invalid characters")
        
        # Validate check digit (9th position)
        if len(cleaned_vin) == 17:
            check_digit = ValidationUtils._calculate_vin_check_digit(cleaned_vin)
            if cleaned_vin[8] != check_digit:
                warnings.append("VIN check digit validation failed")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={'vin': cleaned_vin}
        )
    
    @staticmethod
    def _calculate_vin_check_digit(vin: str) -> str:
        """Calculate VIN check digit"""
        # VIN weight factors
        weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
        
        # Character values
        char_values = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
            'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
            'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9
        }
        
        total = 0
        for i, char in enumerate(vin):
            if char.isdigit():
                value = int(char)
            else:
                value = char_values.get(char, 0)
            
            total += value * weights[i]
        
        remainder = total % 11
        return 'X' if remainder == 10 else str(remainder)
    
    @staticmethod
    def validate_policy_number(policy_number: str) -> ValidationResult:
        """Validate insurance policy number"""
        errors = []
        warnings = []
        
        if not policy_number:
            errors.append("Policy number is required")
            return ValidationResult(False, errors, warnings)
        
        # Clean policy number
        cleaned_policy = policy_number.upper().strip()
        
        # Check format (example: ABC-20231201-123456)
        policy_pattern = r'^[A-Z]{3}-\d{8}-\d{6}$'
        
        if not re.match(policy_pattern, cleaned_policy):
            errors.append("Invalid policy number format (expected: ABC-YYYYMMDD-NNNNNN)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={'policy_number': cleaned_policy}
        )
    
    @staticmethod
    def validate_claim_number(claim_number: str) -> ValidationResult:
        """Validate insurance claim number"""
        errors = []
        warnings = []
        
        if not claim_number:
            errors.append("Claim number is required")
            return ValidationResult(False, errors, warnings)
        
        # Clean claim number
        cleaned_claim = claim_number.upper().strip()
        
        # Check format (example: CLM-20231201-123456)
        claim_pattern = r'^CLM-\d{8}-\d{6}$'
        
        if not re.match(claim_pattern, cleaned_claim):
            errors.append("Invalid claim number format (expected: CLM-YYYYMMDD-NNNNNN)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={'claim_number': cleaned_claim}
        )
    
    @staticmethod
    def validate_currency_amount(amount: Union[str, float, Decimal], 
                                min_amount: float = 0.0,
                                max_amount: float = 10000000.0) -> ValidationResult:
        """Validate currency amount"""
        errors = []
        warnings = []
        
        if amount is None or amount == '':
            errors.append("Amount is required")
            return ValidationResult(False, errors, warnings)
        
        try:
            # Convert to Decimal for precise calculations
            if isinstance(amount, str):
                # Remove currency symbols and formatting
                cleaned_amount = re.sub(r'[^\d.-]', '', amount)
                decimal_amount = Decimal(cleaned_amount)
            else:
                decimal_amount = Decimal(str(amount))
            
            # Round to 2 decimal places
            rounded_amount = decimal_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            # Validate range
            if float(rounded_amount) < min_amount:
                errors.append(f"Amount must be at least ${min_amount:,.2f}")
            
            if float(rounded_amount) > max_amount:
                errors.append(f"Amount cannot exceed ${max_amount:,.2f}")
            
            # Check for reasonable values
            if float(rounded_amount) > 1000000:
                warnings.append("Large amount detected - please verify")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                cleaned_data={'amount': float(rounded_amount)}
            )
            
        except (ValueError, TypeError) as e:
            errors.append("Invalid amount format")
            return ValidationResult(False, errors, warnings)

class DateTimeUtils:
    """Production-ready date and time utilities"""
    
    @staticmethod
    def parse_date(date_str: str, formats: List[str] = None) -> Optional[datetime]:
        """Parse date string with multiple format support"""
        if not date_str:
            return None
        
        if formats is None:
            formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ'
            ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def format_date(date_obj: datetime, format_type: str = 'iso') -> str:
        """Format datetime object to string"""
        if not date_obj:
            return ''
        
        formats = {
            'iso': '%Y-%m-%d',
            'us': '%m/%d/%Y',
            'eu': '%d/%m/%Y',
            'iso_datetime': '%Y-%m-%dT%H:%M:%S',
            'readable': '%B %d, %Y',
            'short': '%b %d, %Y'
        }
        
        return date_obj.strftime(formats.get(format_type, '%Y-%m-%d'))
    
    @staticmethod
    def calculate_age(birth_date: datetime, reference_date: datetime = None) -> int:
        """Calculate age from birth date"""
        if not birth_date:
            return 0
        
        if reference_date is None:
            reference_date = datetime.utcnow()
        
        age = reference_date.year - birth_date.year
        
        # Adjust if birthday hasn't occurred this year
        if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
            age -= 1
        
        return max(0, age)
    
    @staticmethod
    def business_days_between(start_date: datetime, end_date: datetime) -> int:
        """Calculate business days between two dates"""
        if not start_date or not end_date:
            return 0
        
        # Use pandas for business day calculation
        return pd.bdate_range(start_date, end_date).size
    
    @staticmethod
    def add_business_days(start_date: datetime, days: int) -> datetime:
        """Add business days to a date"""
        if not start_date:
            return start_date
        
        # Use pandas for business day calculation
        business_dates = pd.bdate_range(start_date, periods=days + 1, freq='B')
        return business_dates[-1].to_pydatetime()
    
    @staticmethod
    def get_timezone_offset(timezone_name: str) -> timedelta:
        """Get timezone offset from UTC"""
        try:
            import pytz
            tz = pytz.timezone(timezone_name)
            offset = tz.utcoffset(datetime.utcnow())
            return offset
        except:
            return timedelta(0)
    
    @staticmethod
    def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
        """Convert datetime between timezones"""
        try:
            import pytz
            from_timezone = pytz.timezone(from_tz)
            to_timezone = pytz.timezone(to_tz)
            
            # Localize to source timezone
            localized_dt = from_timezone.localize(dt)
            
            # Convert to target timezone
            converted_dt = localized_dt.astimezone(to_timezone)
            
            return converted_dt.replace(tzinfo=None)
        except:
            return dt

class DataUtils:
    """Production-ready data processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ''
        
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        cleaned = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', '', cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize person/company name"""
        if not name:
            return ''
        
        # Clean text
        cleaned = DataUtils.clean_text(name)
        
        # Title case for names
        normalized = cleaned.title()
        
        # Handle common name prefixes/suffixes
        prefixes = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.']
        suffixes = ['Jr.', 'Sr.', 'II', 'III', 'IV']
        
        words = normalized.split()
        for i, word in enumerate(words):
            if word.upper() in [p.upper() for p in prefixes + suffixes]:
                words[i] = word.upper()
        
        return ' '.join(words)
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract all numbers from text"""
        if not text:
            return []
        
        # Pattern to match numbers (including decimals and negatives)
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    @staticmethod
    def extract_dates(text: str) -> List[datetime]:
        """Extract dates from text"""
        if not text:
            return []
        
        dates = []
        
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY or M/D/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\b\w+ \d{1,2}, \d{4}\b',  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                parsed_date = DateTimeUtils.parse_date(match)
                if parsed_date:
                    dates.append(parsed_date)
        
        return dates
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate text similarity using Jaccard similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    @staticmethod
    def generate_hash(data: Union[str, Dict, List]) -> str:
        """Generate consistent hash for data"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    @staticmethod
    def chunk_list(data_list: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split list into chunks"""
        chunks = []
        for i in range(0, len(data_list), chunk_size):
            chunks.append(data_list[i:i + chunk_size])
        return chunks
    
    @staticmethod
    def flatten_dict(nested_dict: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        def _flatten(obj, parent_key=''):
            items = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{parent_key}{separator}{key}" if parent_key else key
                    items.extend(_flatten(value, new_key).items())
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
                    items.extend(_flatten(value, new_key).items())
            else:
                return {parent_key: obj}
            return dict(items)
        
        return _flatten(nested_dict)
    
    @staticmethod
    def unflatten_dict(flat_dict: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
        """Unflatten dictionary"""
        result = {}
        
        for key, value in flat_dict.items():
            keys = key.split(separator)
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return result

class FileUtils:
    """Production-ready file processing utilities"""
    
    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """Calculate file hash"""
        hash_algorithms = {
            'md5': hashlib.md5(),
            'sha1': hashlib.sha1(),
            'sha256': hashlib.sha256(),
            'sha512': hashlib.sha512()
        }
        
        hasher = hash_algorithms.get(algorithm.lower(), hashlib.sha256())
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information"""
        import os
        import mimetypes
        
        try:
            stat = os.stat(file_path)
            mime_type, encoding = mimetypes.guess_type(file_path)
            
            return {
                'path': file_path,
                'name': os.path.basename(file_path),
                'size': stat.st_size,
                'size_human': FileUtils._format_file_size(stat.st_size),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'accessed': datetime.fromtimestamp(stat.st_atime),
                'mime_type': mime_type,
                'encoding': encoding,
                'extension': os.path.splitext(file_path)[1].lower(),
                'hash_sha256': FileUtils.get_file_hash(file_path, 'sha256')
            }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {}
    
    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    @staticmethod
    def create_zip_archive(files: List[str], output_path: str) -> bool:
        """Create ZIP archive from list of files"""
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files:
                    if os.path.exists(file_path):
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname)
            return True
        except Exception as e:
            logger.error(f"Error creating ZIP archive: {e}")
            return False
    
    @staticmethod
    def extract_zip_archive(zip_path: str, extract_to: str) -> bool:
        """Extract ZIP archive"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_to)
            return True
        except Exception as e:
            logger.error(f"Error extracting ZIP archive: {e}")
            return False
    
    @staticmethod
    def read_csv_file(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
        """Read CSV file and return list of dictionaries"""
        try:
            data = []
            with open(file_path, 'r', encoding=encoding, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(dict(row))
            return data
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return []
    
    @staticmethod
    def write_csv_file(data: List[Dict[str, Any]], file_path: str, 
                      encoding: str = 'utf-8') -> bool:
        """Write list of dictionaries to CSV file"""
        try:
            if not data:
                return False
            
            fieldnames = data[0].keys()
            
            with open(file_path, 'w', encoding=encoding, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            return True
        except Exception as e:
            logger.error(f"Error writing CSV file: {e}")
            return False

class APIUtils:
    """Production-ready API utilities"""
    
    @staticmethod
    async def make_http_request(url: str, method: str = 'GET', 
                               headers: Dict[str, str] = None,
                               data: Dict[str, Any] = None,
                               timeout: int = 30) -> APIResponse:
        """Make HTTP request with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Production SSL configuration
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            # Allow self-signed certificates in development environments only
            if os.getenv('ENVIRONMENT', 'production').lower() in ['development', 'dev', 'local']:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout),
                connector=aiohttp.TCPConnector(ssl=ssl_context)
            ) as session:
                kwargs = {
                    'headers': headers or {}
                }
                
                if data:
                    if method.upper() in ['POST', 'PUT', 'PATCH']:
                        kwargs['json'] = data
                    else:
                        kwargs['params'] = data
                
                async with session.request(method.upper(), url, **kwargs) as response:
                    response_time = time.time() - start_time
                    
                    try:
                        response_data = await response.json()
                    except:
                        response_data = {'text': await response.text()}
                    
                    return APIResponse(
                        success=response.status < 400,
                        data=response_data,
                        error=None if response.status < 400 else f"HTTP {response.status}",
                        status_code=response.status,
                        response_time=response_time
                    )
                    
        except asyncio.TimeoutError:
            return APIResponse(
                success=False,
                data=None,
                error="Request timeout",
                status_code=408,
                response_time=time.time() - start_time
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                status_code=500,
                response_time=time.time() - start_time
            )
    
    @staticmethod
    def validate_api_key(api_key: str, valid_keys: List[str]) -> bool:
        """Validate API key"""
        if not api_key or not valid_keys:
            return False
        
        return api_key in valid_keys
    
    @staticmethod
    def rate_limit_check(redis_client: redis.Redis, key: str, 
                        limit: int, window: int) -> Tuple[bool, int]:
        """Check rate limit using sliding window"""
        try:
            current_time = int(time.time())
            pipeline = redis_client.pipeline()
            
            # Remove old entries
            pipeline.zremrangebyscore(key, 0, current_time - window)
            
            # Count current requests
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(uuid.uuid4()): current_time})
            
            # Set expiration
            pipeline.expire(key, window)
            
            results = pipeline.execute()
            current_count = results[1]
            
            # Check if limit exceeded
            if current_count >= limit:
                return False, current_count
            
            return True, current_count
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True, 0  # Allow request if rate limiting fails
    
    @staticmethod
    def generate_api_documentation(endpoints: List[Dict[str, Any]]) -> str:
        """Generate API documentation in markdown format"""
        doc = "# API Documentation\n\n"
        
        for endpoint in endpoints:
            doc += f"## {endpoint.get('name', 'Unnamed Endpoint')}\n\n"
            doc += f"**Method:** {endpoint.get('method', 'GET')}\n\n"
            doc += f"**URL:** `{endpoint.get('url', '')}`\n\n"
            
            if endpoint.get('description'):
                doc += f"**Description:** {endpoint['description']}\n\n"
            
            if endpoint.get('parameters'):
                doc += "**Parameters:**\n\n"
                for param in endpoint['parameters']:
                    doc += f"- `{param.get('name', '')}` ({param.get('type', 'string')}): {param.get('description', '')}\n"
                doc += "\n"
            
            if endpoint.get('example_request'):
                doc += "**Example Request:**\n\n"
                doc += f"```json\n{json.dumps(endpoint['example_request'], indent=2)}\n```\n\n"
            
            if endpoint.get('example_response'):
                doc += "**Example Response:**\n\n"
                doc += f"```json\n{json.dumps(endpoint['example_response'], indent=2)}\n```\n\n"
            
            doc += "---\n\n"
        
        return doc

class EmailUtils:
    """Production-ready email utilities"""
    
    @staticmethod
    def send_email(smtp_server: str, smtp_port: int, username: str, password: str,
                  from_email: str, to_emails: List[str], subject: str, 
                  body: str, html_body: str = None, 
                  attachments: List[str] = None) -> bool:
        """Send email with attachments"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            
            # Add text body
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            # Add HTML body if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(file_path)}'
                        )
                        msg.attach(part)
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                server.login(username, password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    @staticmethod
    def validate_email_template(template: str, variables: List[str]) -> ValidationResult:
        """Validate email template"""
        errors = []
        warnings = []
        
        if not template:
            errors.append("Template is required")
            return ValidationResult(False, errors, warnings)
        
        # Check for required variables
        for var in variables:
            placeholder = f"{{{var}}}"
            if placeholder not in template:
                warnings.append(f"Variable '{var}' not found in template")
        
        # Check for undefined variables
        import re
        found_vars = re.findall(r'\{(\w+)\}', template)
        for var in found_vars:
            if var not in variables:
                warnings.append(f"Undefined variable '{var}' found in template")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={'template': template, 'variables': found_vars}
        )

class CacheUtils:
    """Production-ready caching utilities"""
    
    @staticmethod
    def get_cache_key(*args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_parts = []
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))
        
        # Add keyword arguments
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (dict, list)):
                key_parts.append(f"{key}:{json.dumps(value, sort_keys=True)}")
            else:
                key_parts.append(f"{key}:{value}")
        
        # Create hash of combined key
        combined_key = "|".join(key_parts)
        return hashlib.md5(combined_key.encode()).hexdigest()
    
    @staticmethod
    def cache_decorator(redis_client: redis.Redis, ttl: int = 3600, 
                       key_prefix: str = "cache"):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{key_prefix}:{CacheUtils.get_cache_key(*args, **kwargs)}"
                
                try:
                    # Try to get from cache
                    cached_result = redis_client.get(cache_key)
                    if cached_result:
                        return json.loads(cached_result)
                except Exception as e:
                    logger.warning(f"Cache read error: {e}")
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                try:
                    # Store in cache
                    redis_client.setex(
                        cache_key,
                        ttl,
                        json.dumps(result, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")
                
                return result
            
            return wrapper
        return decorator

class MonitoringUtils:
    """Production-ready monitoring utilities"""
    
    @staticmethod
    def log_performance(func_name: str, duration: float, success: bool, 
                       additional_data: Dict[str, Any] = None):
        """Log performance metrics"""
        log_data = {
            'function': func_name,
            'duration_ms': round(duration * 1000, 2),
            'success': success,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if additional_data:
            log_data.update(additional_data)
        
        if success:
            logger.info(f"Performance: {json.dumps(log_data)}")
        else:
            logger.error(f"Performance: {json.dumps(log_data)}")
    
    @staticmethod
    def performance_monitor(func: Callable):
        """Decorator for monitoring function performance"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error = None
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                MonitoringUtils.log_performance(
                    func.__name__,
                    duration,
                    success,
                    {'error': error} if error else None
                )
        
        return wrapper
    
    @staticmethod
    def health_check(components: Dict[str, Callable]) -> Dict[str, Any]:
        """Perform health check on system components"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        overall_healthy = True
        
        for component_name, check_func in components.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                response_time = time.time() - start_time
                
                health_status['components'][component_name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'response_time_ms': round(response_time * 1000, 2)
                }
                
                if not is_healthy:
                    overall_healthy = False
                    
            except Exception as e:
                health_status['components'][component_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                overall_healthy = False
        
        health_status['status'] = 'healthy' if overall_healthy else 'unhealthy'
        return health_status

# Export all utility classes
__all__ = [
    'SecurityUtils',
    'ValidationUtils', 
    'DateTimeUtils',
    'DataUtils',
    'FileUtils',
    'APIUtils',
    'EmailUtils',
    'CacheUtils',
    'MonitoringUtils',
    'ValidationResult',
    'EncryptionResult',
    'APIResponse',
    'ValidationError',
    'SecurityError',
    'APIError',
    'DataProcessingError'
]

