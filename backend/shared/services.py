"""
Insurance AI Agent System - Base Service Classes
Production-ready base services with error handling, logging, and monitoring
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Type, TypeVar, Generic
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import structlog
from prometheus_client import Counter, Histogram, Gauge
import redis.asyncio as redis
from pydantic import BaseModel, ValidationError

from .database import get_db_session, get_redis_client, CacheManager
from .models import Base
from .schemas import BaseSchema, PaginationParams, PaginatedResponse

# Setup structured logging
logger = structlog.get_logger(__name__)

# Prometheus metrics
service_operations_total = Counter('service_operations_total', 'Total service operations', ['service', 'operation', 'status'])
service_operation_duration = Histogram('service_operation_duration_seconds', 'Service operation duration', ['service', 'operation'])
service_cache_operations = Counter('service_cache_operations_total', 'Cache operations', ['operation', 'status'])

# Type variables
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class ServiceException(Exception):
    """Base service exception"""
    def __init__(self, message: str, code: str = "SERVICE_ERROR", details: Dict[str, Any] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

class ValidationException(ServiceException):
    """Validation exception"""
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field

class NotFoundException(ServiceException):
    """Not found exception"""
    def __init__(self, resource: str, identifier: str):
        super().__init__(f"{resource} not found: {identifier}", "NOT_FOUND")
        self.resource = resource
        self.identifier = identifier

class ConflictException(ServiceException):
    """Conflict exception"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "CONFLICT", details)

class PermissionException(ServiceException):
    """Permission exception"""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, "PERMISSION_DENIED")

class BaseService(ABC, Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base service class with CRUD operations, caching, and monitoring
    """
    
    def __init__(
        self,
        model: Type[ModelType],
        db_session: AsyncSession,
        cache_manager: Optional[CacheManager] = None,
        cache_ttl: int = 3600
    ):
        self.model = model
        self.db_session = db_session
        self.cache_manager = cache_manager
        self.cache_ttl = cache_ttl
        self.service_name = self.__class__.__name__.lower().replace('service', '')
        
    async def create(
        self,
        obj_in: CreateSchemaType,
        created_by: Optional[uuid.UUID] = None,
        organization_id: Optional[uuid.UUID] = None
    ) -> ModelType:
        """Create a new record"""
        operation_timer = service_operation_duration.labels(
            service=self.service_name, operation='create'
        ).time()
        
        try:
            with operation_timer:
                # Determine input type
                if isinstance(obj_in, BaseModel):
                    obj_data = obj_in.dict()
                    if hasattr(self.model, 'created_by') and created_by:
                        obj_data['created_by'] = created_by
                    if hasattr(self.model, 'organization_id') and organization_id:
                        obj_data['organization_id'] = organization_id
                    db_obj = self.model(**obj_data)
                elif isinstance(obj_in, self.model):
                    db_obj = obj_in
                    if hasattr(db_obj, 'created_by') and created_by:
                        setattr(db_obj, 'created_by', created_by)
                    if hasattr(db_obj, 'organization_id') and organization_id:
                        setattr(db_obj, 'organization_id', organization_id)
                elif isinstance(obj_in, dict):
                    obj_data = obj_in.copy()
                    if hasattr(self.model, 'created_by') and created_by:
                        obj_data.setdefault('created_by', created_by)
                    if hasattr(self.model, 'organization_id') and organization_id:
                        obj_data.setdefault('organization_id', organization_id)
                    db_obj = self.model(**obj_data)
                else:
                    raise ValidationException(
                        f"Unsupported input type: {type(obj_in)}"
                    )
                
                # Add to session and commit
                self.db_session.add(db_obj)
                await self.db_session.commit()
                await self.db_session.refresh(db_obj)
                
                # Invalidate cache
                if self.cache_manager:
                    await self._invalidate_cache(str(db_obj.id))
                
                # Log operation
                logger.info(
                    "Created record",
                    service=self.service_name,
                    model=self.model.__name__,
                    id=str(db_obj.id)
                )
                
                service_operations_total.labels(
                    service=self.service_name, operation='create', status='success'
                ).inc()
                
                return db_obj
                
        except IntegrityError as e:
            await self.db_session.rollback()
            service_operations_total.labels(
                service=self.service_name, operation='create', status='error'
            ).inc()
            raise ConflictException(f"Record already exists: {str(e)}")
            
        except ValidationError as e:
            await self.db_session.rollback()
            service_operations_total.labels(
                service=self.service_name, operation='create', status='error'
            ).inc()
            raise ValidationException(f"Validation error: {str(e)}")
            
        except Exception as e:
            await self.db_session.rollback()
            service_operations_total.labels(
                service=self.service_name, operation='create', status='error'
            ).inc()
            logger.error(
                "Failed to create record",
                service=self.service_name,
                error=str(e)
            )
            raise ServiceException(f"Failed to create record: {str(e)}")
    
    async def get(
        self,
        id: uuid.UUID,
        use_cache: bool = True,
        load_relationships: List[str] = None
    ) -> Optional[ModelType]:
        """Get a record by ID"""
        operation_timer = service_operation_duration.labels(
            service=self.service_name, operation='get'
        ).time()
        
        try:
            with operation_timer:
                # Try cache first
                if use_cache and self.cache_manager:
                    cached_obj = await self._get_from_cache(str(id))
                    if cached_obj:
                        service_cache_operations.labels(operation='hit', status='success').inc()
                        return cached_obj
                    service_cache_operations.labels(operation='miss', status='success').inc()
                
                # Build query
                query = select(self.model).where(self.model.id == id)
                
                # Add relationship loading
                if load_relationships:
                    for rel in load_relationships:
                        if hasattr(self.model, rel):
                            query = query.options(selectinload(getattr(self.model, rel)))
                
                # Execute query
                result = await self.db_session.execute(query)
                db_obj = result.scalar_one_or_none()
                
                if db_obj and use_cache and self.cache_manager:
                    await self._set_cache(str(id), db_obj)
                
                service_operations_total.labels(
                    service=self.service_name, operation='get', status='success'
                ).inc()
                
                return db_obj
                
        except Exception as e:
            service_operations_total.labels(
                service=self.service_name, operation='get', status='error'
            ).inc()
            logger.error(
                "Failed to get record",
                service=self.service_name,
                id=str(id),
                error=str(e)
            )
            raise ServiceException(f"Failed to get record: {str(e)}")

    async def get_by_field(
        self,
        field_name: str,
        field_value: Any,
        use_cache: bool = True
    ) -> Optional[ModelType]:
        """Get a record by any field"""
        operation_timer = service_operation_duration.labels(
            service=self.service_name, operation='get_by_field'
        ).time()
        
        try:
            with operation_timer:
                # Build query
                query = select(self.model).where(getattr(self.model, field_name) == field_value)
                result = await self.db_session.execute(query)
                obj = result.scalar_one_or_none()
                
                service_operations_total.labels(
                    service=self.service_name, operation='get_by_field', status='success'
                ).inc()
                
                return obj
                
        except Exception as e:
            service_operations_total.labels(
                service=self.service_name, operation='get_by_field', status='error'
            ).inc()
            logger.error(f"Error getting {self.model.__name__} by {field_name}", error=str(e))
            raise ServiceException(f"Failed to get {self.model.__name__} by {field_name}")
    
    async def get_multi(
        self,
        pagination: PaginationParams,
        filters: Dict[str, Any] = None,
        organization_id: Optional[uuid.UUID] = None
    ) -> PaginatedResponse:
        """Get multiple records with pagination and filtering"""
        operation_timer = service_operation_duration.labels(
            service=self.service_name, operation='get_multi'
        ).time()
        
        try:
            with operation_timer:
                # Build base query
                query = select(self.model)
                count_query = select(func.count(self.model.id))
                
                # Apply organization filter if model supports it
                if hasattr(self.model, 'organization_id') and organization_id:
                    query = query.where(self.model.organization_id == organization_id)
                    count_query = count_query.where(self.model.organization_id == organization_id)
                
                # Apply filters
                if filters:
                    filter_conditions = self._build_filter_conditions(filters)
                    if filter_conditions:
                        query = query.where(and_(*filter_conditions))
                        count_query = count_query.where(and_(*filter_conditions))
                
                # Apply sorting
                if pagination.sort_by and hasattr(self.model, pagination.sort_by):
                    sort_column = getattr(self.model, pagination.sort_by)
                    if pagination.sort_order == 'desc':
                        query = query.order_by(sort_column.desc())
                    else:
                        query = query.order_by(sort_column.asc())
                else:
                    # Default sort by created_at if available
                    if hasattr(self.model, 'created_at'):
                        query = query.order_by(self.model.created_at.desc())
                
                # Apply pagination
                offset = (pagination.page - 1) * pagination.size
                query = query.offset(offset).limit(pagination.size)
                
                # Execute queries
                result = await self.db_session.execute(query)
                items = result.scalars().all()
                
                count_result = await self.db_session.execute(count_query)
                total = count_result.scalar()
                
                # Calculate pagination info
                pages = (total + pagination.size - 1) // pagination.size
                
                service_operations_total.labels(
                    service=self.service_name, operation='get_multi', status='success'
                ).inc()
                
                return PaginatedResponse(
                    items=items,
                    total=total,
                    page=pagination.page,
                    size=pagination.size,
                    pages=pages
                )
                
        except Exception as e:
            service_operations_total.labels(
                service=self.service_name, operation='get_multi', status='error'
            ).inc()
            logger.error(
                "Failed to get multiple records",
                service=self.service_name,
                error=str(e)
            )
            raise ServiceException(f"Failed to get records: {str(e)}")
    
    async def update(
        self,
        id: uuid.UUID,
        obj_in: UpdateSchemaType,
        updated_by: Optional[uuid.UUID] = None
    ) -> Optional[ModelType]:
        """Update a record"""
        operation_timer = service_operation_duration.labels(
            service=self.service_name, operation='update'
        ).time()
        
        try:
            with operation_timer:
                # Get existing record
                db_obj = await self.get(id, use_cache=False)
                if not db_obj:
                    raise NotFoundException(self.model.__name__, str(id))
                
                # Convert update data to dict, excluding None values
                update_data = obj_in.dict(exclude_unset=True)
                
                # Add audit fields if model supports them
                if hasattr(self.model, 'updated_by') and updated_by:
                    update_data['updated_by'] = updated_by
                
                # Update fields
                for field, value in update_data.items():
                    if hasattr(db_obj, field):
                        setattr(db_obj, field, value)
                
                # Commit changes
                await self.db_session.commit()
                await self.db_session.refresh(db_obj)
                
                # Invalidate cache
                if self.cache_manager:
                    await self._invalidate_cache(str(id))
                
                logger.info(
                    "Updated record",
                    service=self.service_name,
                    model=self.model.__name__,
                    id=str(id)
                )
                
                service_operations_total.labels(
                    service=self.service_name, operation='update', status='success'
                ).inc()
                
                return db_obj
                
        except NotFoundException:
            raise
        except Exception as e:
            await self.db_session.rollback()
            service_operations_total.labels(
                service=self.service_name, operation='update', status='error'
            ).inc()
            logger.error(
                "Failed to update record",
                service=self.service_name,
                id=str(id),
                error=str(e)
            )
            raise ServiceException(f"Failed to update record: {str(e)}")
    
    async def delete(self, id: uuid.UUID) -> bool:
        """Delete a record"""
        operation_timer = service_operation_duration.labels(
            service=self.service_name, operation='delete'
        ).time()
        
        try:
            with operation_timer:
                # Check if record exists
                db_obj = await self.get(id, use_cache=False)
                if not db_obj:
                    raise NotFoundException(self.model.__name__, str(id))
                
                # Delete record
                await self.db_session.delete(db_obj)
                await self.db_session.commit()
                
                # Invalidate cache
                if self.cache_manager:
                    await self._invalidate_cache(str(id))
                
                logger.info(
                    "Deleted record",
                    service=self.service_name,
                    model=self.model.__name__,
                    id=str(id)
                )
                
                service_operations_total.labels(
                    service=self.service_name, operation='delete', status='success'
                ).inc()
                
                return True
                
        except NotFoundException:
            raise
        except Exception as e:
            await self.db_session.rollback()
            service_operations_total.labels(
                service=self.service_name, operation='delete', status='error'
            ).inc()
            logger.error(
                "Failed to delete record",
                service=self.service_name,
                id=str(id),
                error=str(e)
            )
            raise ServiceException(f"Failed to delete record: {str(e)}")
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> List:
        """Build SQLAlchemy filter conditions from filter dict"""
        conditions = []
        
        for field, value in filters.items():
            if not hasattr(self.model, field):
                continue
                
            column = getattr(self.model, field)
            
            if isinstance(value, dict):
                # Handle complex filters
                for operator, filter_value in value.items():
                    if operator == 'eq':
                        conditions.append(column == filter_value)
                    elif operator == 'ne':
                        conditions.append(column != filter_value)
                    elif operator == 'gt':
                        conditions.append(column > filter_value)
                    elif operator == 'gte':
                        conditions.append(column >= filter_value)
                    elif operator == 'lt':
                        conditions.append(column < filter_value)
                    elif operator == 'lte':
                        conditions.append(column <= filter_value)
                    elif operator == 'in':
                        conditions.append(column.in_(filter_value))
                    elif operator == 'not_in':
                        conditions.append(~column.in_(filter_value))
                    elif operator == 'like':
                        conditions.append(column.like(f"%{filter_value}%"))
                    elif operator == 'ilike':
                        conditions.append(column.ilike(f"%{filter_value}%"))
                    elif operator == 'is_null':
                        if filter_value:
                            conditions.append(column.is_(None))
                        else:
                            conditions.append(column.isnot(None))
            else:
                # Simple equality filter
                conditions.append(column == value)
        
        return conditions
    
    async def _get_from_cache(self, key: str) -> Optional[ModelType]:
        """Get object from cache"""
        try:
            if not self.cache_manager:
                return None
            
            cached_data = await self.cache_manager.get(key, namespace=self.service_name)
            if cached_data:
                # In a real implementation, you'd deserialize the cached object
                # For now, we'll skip caching complex objects
                pass
            return None
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None
    
    async def _set_cache(self, key: str, obj: ModelType) -> bool:
        """Set object in cache"""
        try:
            if not self.cache_manager:
                return False
            
            # In a real implementation, you'd serialize the object
            # For now, we'll skip caching complex objects
            return True
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False
    
    async def _invalidate_cache(self, key: str) -> bool:
        """Invalidate cache entry"""
        try:
            if not self.cache_manager:
                return False
            
            return await self.cache_manager.delete(key, namespace=self.service_name)
        except Exception as e:
            logger.warning("Cache invalidation failed", key=key, error=str(e))
            return False

class WorkflowService(BaseService):
    """Base workflow service with state management"""
    
    async def start_workflow(
        self,
        workflow_id: uuid.UUID,
        timeout_seconds: int = 300
    ) -> bool:
        """Start a workflow"""
        try:
            workflow = await self.get(workflow_id)
            if not workflow:
                raise NotFoundException("Workflow", str(workflow_id))
            
            if workflow.status != 'pending':
                raise ValidationException(f"Workflow is not in pending state: {workflow.status}")
            
            # Update workflow status
            workflow.status = 'running'
            workflow.started_at = datetime.utcnow()
            workflow.timeout_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)
            
            await self.db_session.commit()
            
            logger.info(
                "Started workflow",
                workflow_id=str(workflow_id),
                type=workflow.type
            )
            
            return True
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(
                "Failed to start workflow",
                workflow_id=str(workflow_id),
                error=str(e)
            )
            raise
    
    async def complete_workflow(
        self,
        workflow_id: uuid.UUID,
        output_data: Dict[str, Any],
        success: bool = True
    ) -> bool:
        """Complete a workflow"""
        try:
            workflow = await self.get(workflow_id)
            if not workflow:
                raise NotFoundException("Workflow", str(workflow_id))
            
            # Update workflow status
            workflow.status = 'completed' if success else 'failed'
            workflow.completed_at = datetime.utcnow()
            workflow.output_data = output_data
            
            await self.db_session.commit()
            
            logger.info(
                "Completed workflow",
                workflow_id=str(workflow_id),
                success=success
            )
            
            return True
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(
                "Failed to complete workflow",
                workflow_id=str(workflow_id),
                error=str(e)
            )
            raise
    
    async def fail_workflow(
        self,
        workflow_id: uuid.UUID,
        error_message: str,
        retry: bool = True
    ) -> bool:
        """Fail a workflow with optional retry"""
        try:
            workflow = await self.get(workflow_id)
            if not workflow:
                raise NotFoundException("Workflow", str(workflow_id))
            
            if retry and workflow.retry_count < workflow.max_retries:
                # Retry workflow
                workflow.retry_count += 1
                workflow.status = 'pending'
                workflow.error_message = error_message
                workflow.started_at = None
                workflow.timeout_at = None
            else:
                # Fail permanently
                workflow.status = 'failed'
                workflow.completed_at = datetime.utcnow()
                workflow.error_message = error_message
            
            await self.db_session.commit()
            
            logger.info(
                "Failed workflow",
                workflow_id=str(workflow_id),
                retry=retry,
                retry_count=workflow.retry_count
            )
            
            return True
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(
                "Failed to fail workflow",
                workflow_id=str(workflow_id),
                error=str(e)
            )
            raise

class AgentService(ABC):
    """Base agent service class"""
    
    def __init__(self, agent_name: str, version: str = "1.0.0"):
        self.agent_name = agent_name
        self.version = version
        self.logger = structlog.get_logger(agent_name)
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent input and return output"""
        pass
    
    async def execute(
        self,
        workflow_id: uuid.UUID,
        input_data: Dict[str, Any],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Execute agent with monitoring and error handling"""
        from .models import AgentExecution
        
        # Create execution record
        execution = AgentExecution(
            workflow_id=workflow_id,
            agent_name=self.agent_name,
            agent_version=self.version,
            input_data=input_data,
            status='running'
        )
        
        db_session.add(execution)
        await db_session.commit()
        await db_session.refresh(execution)
        
        start_time = datetime.utcnow()
        
        try:
            # Process input
            output_data = await self.process(input_data)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update execution record
            execution.status = 'completed'
            execution.output_data = output_data
            execution.execution_time_ms = int(execution_time)
            execution.completed_at = datetime.utcnow()
            
            await db_session.commit()
            
            self.logger.info(
                "Agent execution completed",
                execution_id=str(execution.id),
                execution_time_ms=execution.execution_time_ms
            )
            
            return output_data
            
        except Exception as e:
            # Update execution record with error
            execution.status = 'failed'
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            
            await db_session.commit()
            
            self.logger.error(
                "Agent execution failed",
                execution_id=str(execution.id),
                error=str(e)
            )
            
            raise ServiceException(f"Agent execution failed: {str(e)}")

class EventService:
    """Event service for publishing and subscribing to events"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = structlog.get_logger("event_service")
    
    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        channel: str = "default"
    ) -> bool:
        """Publish an event"""
        try:
            event = {
                "type": event_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "id": str(uuid.uuid4())
            }
            
            await self.redis_client.publish(channel, str(event))
            
            self.logger.info(
                "Published event",
                event_type=event_type,
                channel=channel
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to publish event",
                event_type=event_type,
                error=str(e)
            )
            return False
    
    async def subscribe(
        self,
        channels: List[str],
        callback: callable
    ):
        """Subscribe to events"""
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(*channels)
            
            self.logger.info(
                "Subscribed to channels",
                channels=channels
            )
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        await callback(message['channel'], message['data'])
                    except Exception as e:
                        self.logger.error(
                            "Event callback failed",
                            channel=message['channel'],
                            error=str(e)
                        )
                        
        except Exception as e:
            self.logger.error(
                "Subscription failed",
                channels=channels,
                error=str(e)
            )

# Utility functions
async def with_db_transaction(func):
    """Decorator for database transactions"""
    async def wrapper(*args, **kwargs):
        async with get_db_session() as session:
            try:
                result = await func(session, *args, **kwargs)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise
    return wrapper

def handle_service_errors(func):
    """Decorator for handling service errors"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ServiceException:
            raise
        except ValidationError as e:
            raise ValidationException(str(e))
        except IntegrityError as e:
            raise ConflictException(str(e))
        except Exception as e:
            logger.error("Unexpected service error", error=str(e))
            raise ServiceException(f"Unexpected error: {str(e)}")
    return wrapper

