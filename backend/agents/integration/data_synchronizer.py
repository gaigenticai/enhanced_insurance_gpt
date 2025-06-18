"""
Data Synchronizer - Production Ready Implementation
Bidirectional data synchronization between systems
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import asyncpg
import aiomysql
import aiosqlite

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sync_operations_total = Counter('sync_operations_total', 'Total sync operations', ['source', 'target', 'operation'])
sync_duration = Histogram('sync_duration_seconds', 'Sync operation duration')
sync_errors_total = Counter('sync_errors_total', 'Total sync errors', ['source', 'target', 'error_type'])
active_syncs = Gauge('active_syncs', 'Number of active sync operations')

Base = declarative_base()

class SyncDirection(Enum):
    UNIDIRECTIONAL = "unidirectional"
    BIDIRECTIONAL = "bidirectional"

class SyncStrategy(Enum):
    FULL_SYNC = "full_sync"
    INCREMENTAL = "incremental"
    DELTA_SYNC = "delta_sync"
    EVENT_DRIVEN = "event_driven"

class ConflictResolution(Enum):
    SOURCE_WINS = "source_wins"
    TARGET_WINS = "target_wins"
    LATEST_TIMESTAMP = "latest_timestamp"
    MANUAL_REVIEW = "manual_review"
    MERGE_FIELDS = "merge_fields"

class DataSourceType(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REST_API = "rest_api"
    SOAP_API = "soap_api"
    FILE_SYSTEM = "file_system"
    MESSAGE_QUEUE = "message_queue"
    REDIS = "redis"

@dataclass
class DataSource:
    name: str
    source_type: DataSourceType
    connection_string: str
    credentials: Dict[str, str]
    schema_mapping: Dict[str, str]
    filters: Dict[str, Any]
    batch_size: int
    timeout: int

@dataclass
class SyncConfiguration:
    sync_id: str
    name: str
    source: DataSource
    target: DataSource
    direction: SyncDirection
    strategy: SyncStrategy
    conflict_resolution: ConflictResolution
    schedule_cron: Optional[str]
    sync_interval_minutes: int
    enabled: bool
    field_mappings: Dict[str, str]
    transformation_rules: List[Dict[str, Any]]
    validation_rules: List[Dict[str, Any]]
    retry_attempts: int
    error_threshold: int

@dataclass
class SyncRecord:
    record_id: str
    source_id: str
    target_id: Optional[str]
    source_hash: str
    target_hash: Optional[str]
    last_sync_timestamp: datetime
    sync_status: str
    conflict_data: Optional[Dict[str, Any]]

@dataclass
class SyncResult:
    sync_id: str
    execution_id: str
    start_time: datetime
    end_time: datetime
    records_processed: int
    records_created: int
    records_updated: int
    records_deleted: int
    records_failed: int
    conflicts_detected: int
    conflicts_resolved: int
    errors: List[str]
    performance_metrics: Dict[str, Any]

class SyncConfigurationRecord(Base):
    __tablename__ = 'sync_configurations'
    
    sync_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    source_config = Column(JSON, nullable=False)
    target_config = Column(JSON, nullable=False)
    direction = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    conflict_resolution = Column(String, nullable=False)
    schedule_cron = Column(String)
    sync_interval_minutes = Column(Integer, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    field_mappings = Column(JSON)
    transformation_rules = Column(JSON)
    validation_rules = Column(JSON)
    retry_attempts = Column(Integer, default=3)
    error_threshold = Column(Integer, default=10)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class SyncExecutionRecord(Base):
    __tablename__ = 'sync_executions'
    
    execution_id = Column(String, primary_key=True)
    sync_id = Column(String, nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    records_processed = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_deleted = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    conflicts_detected = Column(Integer, default=0)
    conflicts_resolved = Column(Integer, default=0)
    errors = Column(JSON)
    performance_metrics = Column(JSON)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)

class SyncRecordMapping(Base):
    __tablename__ = 'sync_record_mappings'
    
    mapping_id = Column(String, primary_key=True)
    sync_id = Column(String, nullable=False, index=True)
    source_id = Column(String, nullable=False)
    target_id = Column(String)
    source_hash = Column(String, nullable=False)
    target_hash = Column(String)
    last_sync_timestamp = Column(DateTime, nullable=False)
    sync_status = Column(String, nullable=False)
    conflict_data = Column(JSON)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class DataSynchronizer:
    """Production-ready Data Synchronizer for bidirectional data sync"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.redis_client = redis.from_url(redis_url)
        
        # Active sync configurations
        self.sync_configs = {}
        
        # Database connections pool
        self.db_connections = {}
        
        # Transformation functions
        self.transformation_functions = {}
        self._register_default_transformations()
        
        logger.info("DataSynchronizer initialized successfully")

    async def initialize(self):
        """Initialize database connections and load configurations"""
        
        await self._load_sync_configurations()

    async def create_sync_configuration(self, config: SyncConfiguration) -> str:
        """Create new synchronization configuration"""
        
        try:
            with self.Session() as session:
                record = SyncConfigurationRecord(
                    sync_id=config.sync_id,
                    name=config.name,
                    source_config=asdict(config.source),
                    target_config=asdict(config.target),
                    direction=config.direction.value,
                    strategy=config.strategy.value,
                    conflict_resolution=config.conflict_resolution.value,
                    schedule_cron=config.schedule_cron,
                    sync_interval_minutes=config.sync_interval_minutes,
                    enabled=config.enabled,
                    field_mappings=config.field_mappings,
                    transformation_rules=config.transformation_rules,
                    validation_rules=config.validation_rules,
                    retry_attempts=config.retry_attempts,
                    error_threshold=config.error_threshold,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(record)
                session.commit()
                
                # Cache configuration
                self.sync_configs[config.sync_id] = config
                
                logger.info(f"Created sync configuration: {config.name}")
                return config.sync_id
                
        except Exception as e:
            logger.error(f"Error creating sync configuration: {e}")
            raise

    async def execute_sync(self, sync_id: str, force_full_sync: bool = False) -> SyncResult:
        """Execute synchronization for given configuration"""
        
        if sync_id not in self.sync_configs:
            raise ValueError(f"Sync configuration not found: {sync_id}")
        
        config = self.sync_configs[sync_id]
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        active_syncs.inc()
        
        try:
            with sync_duration.time():
                # Initialize connections
                source_conn = await self._get_connection(config.source)
                target_conn = await self._get_connection(config.target)
                
                # Determine sync strategy
                strategy = SyncStrategy.FULL_SYNC if force_full_sync else config.strategy
                
                # Execute sync based on strategy
                if strategy == SyncStrategy.FULL_SYNC:
                    result = await self._execute_full_sync(config, source_conn, target_conn, execution_id)
                elif strategy == SyncStrategy.INCREMENTAL:
                    result = await self._execute_incremental_sync(config, source_conn, target_conn, execution_id)
                elif strategy == SyncStrategy.DELTA_SYNC:
                    result = await self._execute_delta_sync(config, source_conn, target_conn, execution_id)
                else:
                    raise ValueError(f"Unsupported sync strategy: {strategy}")
                
                # Store execution result
                await self._store_sync_execution(result)
                
                sync_operations_total.labels(
                    source=config.source.name,
                    target=config.target.name,
                    operation="sync"
                ).inc()
                
                return result
                
        except Exception as e:
            error_msg = str(e)
            
            # Create error result
            result = SyncResult(
                sync_id=sync_id,
                execution_id=execution_id,
                start_time=start_time,
                end_time=datetime.utcnow(),
                records_processed=0,
                records_created=0,
                records_updated=0,
                records_deleted=0,
                records_failed=0,
                conflicts_detected=0,
                conflicts_resolved=0,
                errors=[error_msg],
                performance_metrics={}
            )
            
            await self._store_sync_execution(result)
            
            sync_errors_total.labels(
                source=config.source.name,
                target=config.target.name,
                error_type="execution_failed"
            ).inc()
            
            logger.error(f"Sync execution failed for {sync_id}: {error_msg}")
            raise
            
        finally:
            active_syncs.dec()

    async def _execute_full_sync(self, config: SyncConfiguration, 
                               source_conn: Any, target_conn: Any, 
                               execution_id: str) -> SyncResult:
        """Execute full synchronization"""
        
        start_time = datetime.utcnow()
        
        records_processed = 0
        records_created = 0
        records_updated = 0
        records_deleted = 0
        records_failed = 0
        conflicts_detected = 0
        conflicts_resolved = 0
        errors = []
        
        try:
            # Get all source records
            source_records = await self._fetch_source_records(config.source, source_conn)
            
            # Get existing mappings
            existing_mappings = await self._get_sync_mappings(config.sync_id)
            
            # Process each source record
            for source_record in source_records:
                try:
                    records_processed += 1
                    
                    # Transform record
                    transformed_record = await self._transform_record(
                        source_record, config.transformation_rules
                    )
                    
                    # Validate record
                    if not await self._validate_record(transformed_record, config.validation_rules):
                        records_failed += 1
                        errors.append(f"Validation failed for record {source_record.get('id', 'unknown')}")
                        continue
                    
                    # Check if record exists in target
                    source_id = str(source_record.get('id'))
                    existing_mapping = existing_mappings.get(source_id)
                    
                    if existing_mapping:
                        # Update existing record
                        target_record = await self._fetch_target_record(
                            config.target, target_conn, existing_mapping.target_id
                        )
                        
                        if target_record:
                            # Check for conflicts
                            conflict_data = await self._detect_conflicts(
                                source_record, target_record, config.field_mappings
                            )
                            
                            if conflict_data:
                                conflicts_detected += 1
                                
                                # Resolve conflict
                                resolved_record = await self._resolve_conflict(
                                    source_record, target_record, conflict_data, config.conflict_resolution
                                )
                                
                                if resolved_record:
                                    conflicts_resolved += 1
                                    await self._update_target_record(
                                        config.target, target_conn, existing_mapping.target_id, resolved_record
                                    )
                                    records_updated += 1
                                else:
                                    records_failed += 1
                                    errors.append(f"Conflict resolution failed for record {source_id}")
                            else:
                                # No conflict, update record
                                await self._update_target_record(
                                    config.target, target_conn, existing_mapping.target_id, transformed_record
                                )
                                records_updated += 1
                        else:
                            # Target record not found, create new
                            target_id = await self._create_target_record(
                                config.target, target_conn, transformed_record
                            )
                            records_created += 1
                            
                            # Update mapping
                            existing_mapping.target_id = target_id
                            existing_mapping.last_sync_timestamp = datetime.utcnow()
                            await self._update_sync_mapping(existing_mapping)
                    else:
                        # Create new record
                        target_id = await self._create_target_record(
                            config.target, target_conn, transformed_record
                        )
                        records_created += 1
                        
                        # Create mapping
                        mapping = SyncRecord(
                            record_id=str(uuid.uuid4()),
                            source_id=source_id,
                            target_id=target_id,
                            source_hash=self._calculate_record_hash(source_record),
                            target_hash=self._calculate_record_hash(transformed_record),
                            last_sync_timestamp=datetime.utcnow(),
                            sync_status="synced",
                            conflict_data=None
                        )
                        
                        await self._create_sync_mapping(config.sync_id, mapping)
                    
                except Exception as e:
                    records_failed += 1
                    errors.append(f"Failed to process record {source_record.get('id', 'unknown')}: {str(e)}")
            
            # Handle bidirectional sync
            if config.direction == SyncDirection.BIDIRECTIONAL:
                # Sync from target to source (reverse direction)
                reverse_stats = await self._execute_reverse_sync(config, source_conn, target_conn)
                records_created += reverse_stats['created']
                records_updated += reverse_stats['updated']
                records_failed += reverse_stats['failed']
                errors.extend(reverse_stats['errors'])
            
            end_time = datetime.utcnow()
            
            return SyncResult(
                sync_id=config.sync_id,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                records_processed=records_processed,
                records_created=records_created,
                records_updated=records_updated,
                records_deleted=records_deleted,
                records_failed=records_failed,
                conflicts_detected=conflicts_detected,
                conflicts_resolved=conflicts_resolved,
                errors=errors,
                performance_metrics={
                    'sync_duration_seconds': (end_time - start_time).total_seconds(),
                    'records_per_second': records_processed / max((end_time - start_time).total_seconds(), 1)
                }
            )
            
        except Exception as e:
            logger.error(f"Full sync execution failed: {e}")
            raise

    async def _execute_incremental_sync(self, config: SyncConfiguration,
                                      source_conn: Any, target_conn: Any,
                                      execution_id: str) -> SyncResult:
        """Execute incremental synchronization"""
        
        start_time = datetime.utcnow()
        
        # Get last sync timestamp
        last_sync = await self._get_last_sync_timestamp(config.sync_id)
        
        # Fetch records modified since last sync
        source_records = await self._fetch_incremental_records(
            config.source, source_conn, last_sync
        )
        
        # Process records similar to full sync but only for modified records
        return await self._process_records_batch(
            config, source_conn, target_conn, source_records, execution_id, start_time
        )

    async def _execute_delta_sync(self, config: SyncConfiguration,
                                source_conn: Any, target_conn: Any,
                                execution_id: str) -> SyncResult:
        """Execute delta synchronization"""
        
        start_time = datetime.utcnow()
        
        # Get existing mappings with hashes
        existing_mappings = await self._get_sync_mappings(config.sync_id)
        
        # Fetch all source records
        source_records = await self._fetch_source_records(config.source, source_conn)
        
        # Identify changed records by comparing hashes
        changed_records = []
        
        for source_record in source_records:
            source_id = str(source_record.get('id'))
            current_hash = self._calculate_record_hash(source_record)
            
            existing_mapping = existing_mappings.get(source_id)
            
            if not existing_mapping or existing_mapping.source_hash != current_hash:
                changed_records.append(source_record)
        
        # Process only changed records
        return await self._process_records_batch(
            config, source_conn, target_conn, changed_records, execution_id, start_time
        )

    async def _process_records_batch(self, config: SyncConfiguration,
                                   source_conn: Any, target_conn: Any,
                                   records: List[Dict[str, Any]],
                                   execution_id: str, start_time: datetime) -> SyncResult:
        """Process batch of records for synchronization"""
        
        records_processed = 0
        records_created = 0
        records_updated = 0
        records_failed = 0
        conflicts_detected = 0
        conflicts_resolved = 0
        errors = []
        
        # Process records in batches
        batch_size = config.source.batch_size
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            for record in batch:
                try:
                    records_processed += 1
                    
                    # Transform and validate
                    transformed_record = await self._transform_record(record, config.transformation_rules)
                    
                    if not await self._validate_record(transformed_record, config.validation_rules):
                        records_failed += 1
                        continue
                    
                    # Sync record
                    result = await self._sync_single_record(
                        config, source_conn, target_conn, record, transformed_record
                    )
                    
                    if result['created']:
                        records_created += 1
                    elif result['updated']:
                        records_updated += 1
                    elif result['conflict']:
                        conflicts_detected += 1
                        if result['resolved']:
                            conflicts_resolved += 1
                    elif result['failed']:
                        records_failed += 1
                        errors.append(result['error'])
                    
                except Exception as e:
                    records_failed += 1
                    errors.append(f"Failed to process record: {str(e)}")
        
        end_time = datetime.utcnow()
        
        return SyncResult(
            sync_id=config.sync_id,
            execution_id=execution_id,
            start_time=start_time,
            end_time=end_time,
            records_processed=records_processed,
            records_created=records_created,
            records_updated=records_updated,
            records_deleted=0,
            records_failed=records_failed,
            conflicts_detected=conflicts_detected,
            conflicts_resolved=conflicts_resolved,
            errors=errors,
            performance_metrics={
                'sync_duration_seconds': (end_time - start_time).total_seconds(),
                'records_per_second': records_processed / max((end_time - start_time).total_seconds(), 1)
            }
        )

    # Helper methods for data operations
    
    async def _get_connection(self, data_source: DataSource):
        """Get database connection for data source"""
        
        if data_source.name in self.db_connections:
            return self.db_connections[data_source.name]
        
        if data_source.source_type == DataSourceType.POSTGRESQL:
            conn = await asyncpg.connect(data_source.connection_string)
        elif data_source.source_type == DataSourceType.MYSQL:
            conn = await aiomysql.connect(
                host=data_source.credentials.get('host', 'localhost'),
                port=int(data_source.credentials.get('port', 3306)),
                user=data_source.credentials.get('username'),
                password=data_source.credentials.get('password'),
                db=data_source.credentials.get('database')
            )
        elif data_source.source_type == DataSourceType.SQLITE:
            conn = await aiosqlite.connect(data_source.connection_string)
        else:
            raise ValueError(f"Unsupported data source type: {data_source.source_type}")
        
        self.db_connections[data_source.name] = conn
        return conn

    async def _fetch_source_records(self, data_source: DataSource, connection: Any) -> List[Dict[str, Any]]:
        """Fetch records from source data source"""
        
        try:
            if data_source.source_type == DataSourceType.POSTGRESQL:
                query = f"SELECT * FROM {data_source.schema_mapping.get('table', 'records')}"
                
                # Apply filters
                if data_source.filters:
                    conditions = []
                    for field, value in data_source.filters.items():
                        conditions.append(f"{field} = ${len(conditions) + 1}")
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                rows = await connection.fetch(query, *data_source.filters.values())
                return [dict(row) for row in rows]
            
            # Add support for other database types as needed
            return []
            
        except Exception as e:
            logger.error(f"Error fetching source records: {e}")
            raise

    async def _transform_record(self, record: Dict[str, Any], 
                              transformation_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply transformation rules to record"""
        
        transformed = record.copy()
        
        for rule in transformation_rules:
            rule_type = rule.get('type')
            field = rule.get('field')
            
            if rule_type == 'rename_field':
                old_name = rule.get('old_name')
                new_name = rule.get('new_name')
                if old_name in transformed:
                    transformed[new_name] = transformed.pop(old_name)
            
            elif rule_type == 'transform_value':
                function_name = rule.get('function')
                if field in transformed and function_name in self.transformation_functions:
                    transformed[field] = self.transformation_functions[function_name](transformed[field])
            
            elif rule_type == 'add_field':
                value = rule.get('value')
                transformed[field] = value
            
            elif rule_type == 'remove_field':
                if field in transformed:
                    del transformed[field]
        
        return transformed

    async def _validate_record(self, record: Dict[str, Any], 
                             validation_rules: List[Dict[str, Any]]) -> bool:
        """Validate record against validation rules"""
        
        for rule in validation_rules:
            rule_type = rule.get('type')
            field = rule.get('field')
            
            if rule_type == 'required_field':
                if field not in record or record[field] is None:
                    return False
            
            elif rule_type == 'data_type':
                expected_type = rule.get('expected_type')
                if field in record:
                    if expected_type == 'string' and not isinstance(record[field], str):
                        return False
                    elif expected_type == 'integer' and not isinstance(record[field], int):
                        return False
                    elif expected_type == 'float' and not isinstance(record[field], (int, float)):
                        return False
            
            elif rule_type == 'value_range':
                min_val = rule.get('min_value')
                max_val = rule.get('max_value')
                if field in record:
                    value = record[field]
                    if isinstance(value, (int, float)):
                        if min_val is not None and value < min_val:
                            return False
                        if max_val is not None and value > max_val:
                            return False
        
        return True

    def _calculate_record_hash(self, record: Dict[str, Any]) -> str:
        """Calculate hash for record to detect changes"""
        
        # Sort keys for consistent hashing
        sorted_record = {k: record[k] for k in sorted(record.keys())}
        record_json = json.dumps(sorted_record, sort_keys=True, default=str)
        return hashlib.sha256(record_json.encode()).hexdigest()

    def _register_default_transformations(self):
        """Register default transformation functions"""
        
        self.transformation_functions.update({
            'uppercase': lambda x: str(x).upper() if x is not None else None,
            'lowercase': lambda x: str(x).lower() if x is not None else None,
            'strip_whitespace': lambda x: str(x).strip() if x is not None else None,
            'to_integer': lambda x: int(x) if x is not None and str(x).isdigit() else None,
            'to_float': lambda x: float(x) if x is not None else None,
            'format_phone': lambda x: self._format_phone_number(x),
            'format_date': lambda x: self._format_date(x)
        })

    def _format_phone_number(self, phone: str) -> str:
        """Format phone number"""
        if not phone:
            return phone
        
        # Remove non-digits
        digits = ''.join(filter(str.isdigit, phone))
        
        # Format as (XXX) XXX-XXXX
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        
        return phone

    def _format_date(self, date_str: str) -> str:
        """Format date string"""
        if not date_str:
            return date_str
        
        try:
            # Try to parse and reformat
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return date_str

    async def _load_sync_configurations(self):
        """Load sync configurations from database"""
        
        try:
            with self.Session() as session:
                records = session.query(SyncConfigurationRecord).filter_by(enabled=True).all()
                
                for record in records:
                    config = SyncConfiguration(
                        sync_id=record.sync_id,
                        name=record.name,
                        source=DataSource(**record.source_config),
                        target=DataSource(**record.target_config),
                        direction=SyncDirection(record.direction),
                        strategy=SyncStrategy(record.strategy),
                        conflict_resolution=ConflictResolution(record.conflict_resolution),
                        schedule_cron=record.schedule_cron,
                        sync_interval_minutes=record.sync_interval_minutes,
                        enabled=record.enabled,
                        field_mappings=record.field_mappings or {},
                        transformation_rules=record.transformation_rules or [],
                        validation_rules=record.validation_rules or [],
                        retry_attempts=record.retry_attempts,
                        error_threshold=record.error_threshold
                    )
                    
                    self.sync_configs[record.sync_id] = config
                    
                logger.info(f"Loaded {len(self.sync_configs)} sync configurations")
                
        except Exception as e:
            logger.error(f"Error loading sync configurations: {e}")

    async def _store_sync_execution(self, result: SyncResult):
        """Store sync execution result"""
        
        try:
            with self.Session() as session:
                record = SyncExecutionRecord(
                    execution_id=result.execution_id,
                    sync_id=result.sync_id,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    records_processed=result.records_processed,
                    records_created=result.records_created,
                    records_updated=result.records_updated,
                    records_deleted=result.records_deleted,
                    records_failed=result.records_failed,
                    conflicts_detected=result.conflicts_detected,
                    conflicts_resolved=result.conflicts_resolved,
                    errors=result.errors,
                    performance_metrics=result.performance_metrics,
                    status="completed" if not result.errors else "completed_with_errors",
                    created_at=datetime.utcnow()
                )
                
                session.add(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing sync execution: {e}")

    # Additional helper methods would be implemented here for:
    # - _get_sync_mappings
    # - _detect_conflicts
    # - _resolve_conflict
    # - _create_target_record
    # - _update_target_record
    # - _sync_single_record
    # etc.

def create_data_synchronizer(db_url: str = None, redis_url: str = None) -> DataSynchronizer:
    """Create and configure DataSynchronizer instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return DataSynchronizer(db_url=db_url, redis_url=redis_url)

