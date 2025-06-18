"""
Insurance AI Agent System - Message Queue System
Production-ready message queue system for asynchronous communication
"""

import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor

# Redis and async support
import redis.asyncio as redis
import aioredis

# Database
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text

# Internal imports
from backend.shared.models import MessageQueue, QueuedMessage, ProcessingResult
from backend.shared.schemas import QueueMessage, TaskResult, AgentRequest
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

import structlog
logger = structlog.get_logger(__name__)

class QueueType(Enum):
    """Message queue types"""
    HIGH_PRIORITY = "high_priority"
    NORMAL_PRIORITY = "normal_priority"
    LOW_PRIORITY = "low_priority"
    DEAD_LETTER = "dead_letter"
    DELAYED = "delayed"
    BROADCAST = "broadcast"
    AGENT_TASKS = "agent_tasks"
    ORCHESTRATOR_COMMANDS = "orchestrator_commands"
    NOTIFICATIONS = "notifications"
    SYSTEM_EVENTS = "system_events"
    DOCUMENT_PROCESSING = "document_processing"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECKS = "compliance_checks"
    COMMUNICATION = "communication"
    WORKFLOW_UPDATES = "workflow_updates"

class MessageStatus(Enum):
    """Message processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"

@dataclass
class QueueMessage:
    """Message structure for queue system"""
    id: str
    queue_type: QueueType
    payload: Dict[str, Any]
    priority: int
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    status: MessageStatus = MessageStatus.PENDING
    processor_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "queue_type": self.queue_type.value,
            "payload": self.payload,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "processor_id": self.processor_id,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueMessage':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            queue_type=QueueType(data["queue_type"]),
            payload=data["payload"],
            priority=data["priority"],
            created_at=datetime.fromisoformat(data["created_at"]),
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]) if data.get("scheduled_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            status=MessageStatus(data.get("status", "pending")),
            processor_id=data.get("processor_id"),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )

class MessageProcessor:
    """Base class for message processors"""
    
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.logger = structlog.get_logger(f"processor_{processor_id}")
    
    async def process(self, message: QueueMessage) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Process a message
        
        Args:
            message: Message to process
            
        Returns:
            Tuple of (success, result, error)
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    async def can_process(self, message: QueueMessage) -> bool:
        """Check if this processor can handle the message"""
        return True
    
    async def on_success(self, message: QueueMessage, result: Dict[str, Any]):
        """Called when message processing succeeds"""
        metrics.increment_counter(
            "queue_messages_processed_total",
            labels={"queue_type": message.queue_type.value, "status": "success"},
        )
        audit_logger.log_system_event(
            "queue_message_processed",
            f"Message {message.id} processed",
            severity="info",
        )

    async def on_failure(self, message: QueueMessage, error: str):
        """Called when message processing fails"""
        metrics.increment_counter(
            "queue_messages_processed_total",
            labels={"queue_type": message.queue_type.value, "status": "failure"},
        )
        audit_logger.log_system_event(
            "queue_message_failed",
            f"Message {message.id} failed",
            severity="error",
            details={"error": error},
        )

class MessageQueueManager:
    """
    Comprehensive message queue manager using Redis
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = structlog.get_logger("message_queue_manager")
        
        # Queue configurations
        self.queue_configs = {
            QueueType.HIGH_PRIORITY: {"max_size": 10000, "timeout": 1},
            QueueType.NORMAL_PRIORITY: {"max_size": 50000, "timeout": 5},
            QueueType.LOW_PRIORITY: {"max_size": 100000, "timeout": 30},
            QueueType.DEAD_LETTER: {"max_size": 10000, "timeout": None},
            QueueType.DELAYED: {"max_size": 50000, "timeout": None},
            QueueType.BROADCAST: {"max_size": 10000, "timeout": 1},
            QueueType.AGENT_TASKS: {"max_size": 20000, "timeout": 10},
            QueueType.ORCHESTRATOR_COMMANDS: {"max_size": 5000, "timeout": 2},
            QueueType.NOTIFICATIONS: {"max_size": 50000, "timeout": 5},
            QueueType.SYSTEM_EVENTS: {"max_size": 20000, "timeout": 5},
            QueueType.DOCUMENT_PROCESSING: {"max_size": 10000, "timeout": 60},
            QueueType.RISK_ASSESSMENT: {"max_size": 5000, "timeout": 30},
            QueueType.COMPLIANCE_CHECKS: {"max_size": 5000, "timeout": 20},
            QueueType.COMMUNICATION: {"max_size": 20000, "timeout": 10},
            QueueType.WORKFLOW_UPDATES: {"max_size": 10000, "timeout": 5}
        }
        
        # Message processors
        self.processors: Dict[QueueType, List[MessageProcessor]] = {}
        
        # Worker tasks
        self.worker_tasks: Dict[QueueType, List[asyncio.Task]] = {}
        self.worker_count = {
            QueueType.HIGH_PRIORITY: 5,
            QueueType.NORMAL_PRIORITY: 10,
            QueueType.LOW_PRIORITY: 3,
            QueueType.AGENT_TASKS: 8,
            QueueType.ORCHESTRATOR_COMMANDS: 3,
            QueueType.NOTIFICATIONS: 5,
            QueueType.SYSTEM_EVENTS: 3,
            QueueType.DOCUMENT_PROCESSING: 4,
            QueueType.RISK_ASSESSMENT: 2,
            QueueType.COMPLIANCE_CHECKS: 2,
            QueueType.COMMUNICATION: 4,
            QueueType.WORKFLOW_UPDATES: 3
        }
        
        # Background tasks
        self.cleanup_task = None
        self.delayed_message_task = None
        self.metrics_task = None
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_retried": 0,
            "active_workers": 0,
            "queue_sizes": {}
        }
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def start(self):
        """Start the message queue manager"""
        try:
            self.logger.info("Starting message queue manager")
            
            # Start worker tasks for each queue type
            for queue_type, worker_count in self.worker_count.items():
                if queue_type not in self.worker_tasks:
                    self.worker_tasks[queue_type] = []
                
                for i in range(worker_count):
                    task = asyncio.create_task(self._worker(queue_type, f"worker_{i}"))
                    self.worker_tasks[queue_type].append(task)
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
            self.delayed_message_task = asyncio.create_task(self._process_delayed_messages())
            self.metrics_task = asyncio.create_task(self._update_metrics())
            
            # Register default processors
            self._register_default_processors()
            
            self.logger.info("Message queue manager started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start message queue manager", error=str(e))
            raise
    
    async def stop(self):
        """Stop the message queue manager"""
        try:
            self.logger.info("Stopping message queue manager")
            
            # Cancel all worker tasks
            for queue_type, tasks in self.worker_tasks.items():
                for task in tasks:
                    task.cancel()
            
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.delayed_message_task:
                self.delayed_message_task.cancel()
            if self.metrics_task:
                self.metrics_task.cancel()
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            self.logger.info("Message queue manager stopped")
            
        except Exception as e:
            self.logger.error("Error stopping message queue manager", error=str(e))
    
    async def send_message(
        self,
        queue_type: QueueType,
        payload: Dict[str, Any],
        priority: int = 0,
        delay_seconds: Optional[int] = None,
        expires_in_seconds: Optional[int] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message to a queue
        
        Args:
            queue_type: Target queue type
            payload: Message payload
            priority: Message priority (higher = more important)
            delay_seconds: Delay before processing (for scheduled messages)
            expires_in_seconds: Message expiration time
            max_retries: Maximum retry attempts
            metadata: Additional metadata
            
        Returns:
            Message ID
        """
        
        try:
            # Generate message ID
            message_id = str(uuid.uuid4())
            
            # Calculate timestamps
            now = datetime.utcnow()
            scheduled_at = now + timedelta(seconds=delay_seconds) if delay_seconds else now
            expires_at = now + timedelta(seconds=expires_in_seconds) if expires_in_seconds else None
            
            # Create message
            message = QueueMessage(
                id=message_id,
                queue_type=queue_type,
                payload=payload,
                priority=priority,
                created_at=now,
                scheduled_at=scheduled_at,
                expires_at=expires_at,
                max_retries=max_retries,
                metadata=metadata or {}
            )
            
            # Determine target queue
            if delay_seconds and delay_seconds > 0:
                # Send to delayed queue
                await self._send_to_delayed_queue(message)
            else:
                # Send to immediate queue
                await self._send_to_queue(message)
            
            # Update statistics
            self.stats["messages_sent"] += 1
            
            self.logger.info(
                "Message sent to queue",
                message_id=message_id,
                queue_type=queue_type.value,
                priority=priority,
                delay_seconds=delay_seconds
            )
            
            return message_id
            
        except Exception as e:
            self.logger.error("Failed to send message to queue", error=str(e))
            raise ServiceException(f"Failed to send message: {str(e)}")
    
    async def get_message(self, queue_type: QueueType, timeout: Optional[int] = None) -> Optional[QueueMessage]:
        """
        Get a message from a queue (blocking)
        
        Args:
            queue_type: Queue to get message from
            timeout: Timeout in seconds (None for blocking)
            
        Returns:
            Message or None if timeout
        """
        
        try:
            queue_name = self._get_queue_name(queue_type)
            
            # Use timeout from configuration if not specified
            if timeout is None:
                timeout = self.queue_configs[queue_type]["timeout"]
            
            # Get message from Redis queue
            if timeout:
                result = await self.redis_client.blpop(queue_name, timeout=timeout)
            else:
                result = await self.redis_client.lpop(queue_name)
            
            if not result:
                return None
            
            # Parse message
            if isinstance(result, tuple):
                _, message_data = result
            else:
                message_data = result
            
            message_dict = json.loads(message_data)
            message = QueueMessage.from_dict(message_dict)
            
            # Update message status
            message.status = MessageStatus.PROCESSING
            message.processor_id = f"worker_{asyncio.current_task().get_name()}"
            
            # Store processing state
            await self._store_message_state(message)
            
            return message
            
        except Exception as e:
            self.logger.error("Failed to get message from queue", queue_type=queue_type.value, error=str(e))
            return None
    
    async def complete_message(self, message: QueueMessage, result: Optional[Dict[str, Any]] = None):
        """
        Mark a message as completed
        
        Args:
            message: Message to complete
            result: Processing result
        """
        
        try:
            message.status = MessageStatus.COMPLETED
            message.result = result
            
            # Store final state
            await self._store_message_state(message)
            
            # Update statistics
            self.stats["messages_processed"] += 1
            
            self.logger.info(
                "Message completed",
                message_id=message.id,
                queue_type=message.queue_type.value
            )
            
        except Exception as e:
            self.logger.error("Failed to complete message", message_id=message.id, error=str(e))
    
    async def fail_message(self, message: QueueMessage, error: str, retry: bool = True):
        """
        Mark a message as failed
        
        Args:
            message: Message that failed
            error: Error description
            retry: Whether to retry the message
        """
        
        try:
            message.error = error
            
            if retry and message.retry_count < message.max_retries:
                # Retry the message
                message.retry_count += 1
                message.status = MessageStatus.RETRYING
                
                # Calculate retry delay (exponential backoff)
                delay_seconds = min(300, 2 ** message.retry_count)  # Max 5 minutes
                
                # Reschedule message
                message.scheduled_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
                await self._send_to_delayed_queue(message)
                
                # Update statistics
                self.stats["messages_retried"] += 1
                
                self.logger.warning(
                    "Message failed, retrying",
                    message_id=message.id,
                    retry_count=message.retry_count,
                    delay_seconds=delay_seconds,
                    error=error
                )
            else:
                # Send to dead letter queue
                message.status = MessageStatus.DEAD_LETTER
                await self._send_to_dead_letter_queue(message)
                
                # Update statistics
                self.stats["messages_failed"] += 1
                
                self.logger.error(
                    "Message failed permanently",
                    message_id=message.id,
                    retry_count=message.retry_count,
                    error=error
                )
            
            # Store state
            await self._store_message_state(message)
            
        except Exception as e:
            self.logger.error("Failed to handle message failure", message_id=message.id, error=str(e))
    
    def register_processor(self, queue_type: QueueType, processor: MessageProcessor):
        """Register a message processor for a queue type"""
        if queue_type not in self.processors:
            self.processors[queue_type] = []
        self.processors[queue_type].append(processor)
        
        self.logger.info(
            "Processor registered",
            queue_type=queue_type.value,
            processor_id=processor.processor_id
        )
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            # Update queue sizes
            for queue_type in QueueType:
                queue_name = self._get_queue_name(queue_type)
                size = await self.redis_client.llen(queue_name)
                self.stats["queue_sizes"][queue_type.value] = size
            
            # Count active workers
            active_workers = sum(len(tasks) for tasks in self.worker_tasks.values())
            self.stats["active_workers"] = active_workers
            
            return self.stats.copy()
            
        except Exception as e:
            self.logger.error("Failed to get queue stats", error=str(e))
            return {}
    
    async def _worker(self, queue_type: QueueType, worker_id: str):
        """Worker task to process messages from a queue"""
        self.logger.info("Worker started", queue_type=queue_type.value, worker_id=worker_id)
        
        while True:
            try:
                # Get message from queue
                message = await self.get_message(queue_type)
                if not message:
                    continue
                
                # Check if message has expired
                if message.expires_at and datetime.utcnow() > message.expires_at:
                    await self.fail_message(message, "Message expired", retry=False)
                    continue
                
                # Find appropriate processor
                processor = await self._find_processor(queue_type, message)
                if not processor:
                    await self.fail_message(message, "No processor available")
                    continue
                
                # Process message
                try:
                    success, result, error = await processor.process(message)
                    
                    if success:
                        await self.complete_message(message, result)
                        await processor.on_success(message, result)
                    else:
                        await self.fail_message(message, error or "Processing failed")
                        await processor.on_failure(message, error or "Processing failed")
                        
                except Exception as e:
                    error_msg = f"Processor exception: {str(e)}"
                    await self.fail_message(message, error_msg)
                    await processor.on_failure(message, error_msg)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Worker error", queue_type=queue_type.value, worker_id=worker_id, error=str(e))
                await asyncio.sleep(1)  # Brief pause before retrying
        
        self.logger.info("Worker stopped", queue_type=queue_type.value, worker_id=worker_id)
    
    async def _find_processor(self, queue_type: QueueType, message: QueueMessage) -> Optional[MessageProcessor]:
        """Find an appropriate processor for a message"""
        if queue_type not in self.processors:
            return None
        
        for processor in self.processors[queue_type]:
            if await processor.can_process(message):
                return processor
        
        return None
    
    async def _send_to_queue(self, message: QueueMessage):
        """Send message to immediate processing queue"""
        queue_name = self._get_queue_name(message.queue_type)
        message_data = json.dumps(message.to_dict())
        
        # Use priority queue (higher priority = lower score)
        score = -message.priority
        await self.redis_client.zadd(queue_name, {message_data: score})
    
    async def _send_to_delayed_queue(self, message: QueueMessage):
        """Send message to delayed processing queue"""
        delayed_queue_name = f"delayed:{message.queue_type.value}"
        message_data = json.dumps(message.to_dict())
        
        # Use scheduled time as score
        score = message.scheduled_at.timestamp()
        await self.redis_client.zadd(delayed_queue_name, {message_data: score})
    
    async def _send_to_dead_letter_queue(self, message: QueueMessage):
        """Send message to dead letter queue"""
        dead_letter_queue = self._get_queue_name(QueueType.DEAD_LETTER)
        message_data = json.dumps(message.to_dict())
        await self.redis_client.lpush(dead_letter_queue, message_data)
    
    def _get_queue_name(self, queue_type: QueueType) -> str:
        """Get Redis queue name for queue type"""
        return f"queue:{queue_type.value}"
    
    async def _store_message_state(self, message: QueueMessage):
        """Store message state in Redis"""
        try:
            state_key = f"message_state:{message.id}"
            state_data = json.dumps(message.to_dict())
            
            # Store with expiration (24 hours)
            await self.redis_client.setex(state_key, 86400, state_data)
            
        except Exception as e:
            self.logger.warning("Failed to store message state", message_id=message.id, error=str(e))
    
    async def _cleanup_expired_messages(self):
        """Background task to clean up expired messages"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = time.time()
                
                # Clean up expired messages from all queues
                for queue_type in QueueType:
                    queue_name = self._get_queue_name(queue_type)
                    
                    # Get all messages and check expiration
                    messages = await self.redis_client.zrange(queue_name, 0, -1, withscores=True)
                    
                    expired_messages = []
                    for message_data, score in messages:
                        try:
                            message_dict = json.loads(message_data)
                            if message_dict.get("expires_at"):
                                expires_at = datetime.fromisoformat(message_dict["expires_at"])
                                if datetime.utcnow() > expires_at:
                                    expired_messages.append(message_data)
                        except Exception:
                            continue
                    
                    # Remove expired messages
                    if expired_messages:
                        await self.redis_client.zrem(queue_name, *expired_messages)
                        self.logger.info(
                            "Cleaned up expired messages",
                            queue_type=queue_type.value,
                            count=len(expired_messages)
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in message cleanup", error=str(e))
    
    async def _process_delayed_messages(self):
        """Background task to move delayed messages to processing queues"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                current_time = time.time()
                
                # Process delayed messages for all queue types
                for queue_type in QueueType:
                    delayed_queue_name = f"delayed:{queue_type.value}"
                    
                    # Get messages that are ready to process
                    ready_messages = await self.redis_client.zrangebyscore(
                        delayed_queue_name, 0, current_time, withscores=True
                    )
                    
                    for message_data, score in ready_messages:
                        try:
                            # Move to processing queue
                            message_dict = json.loads(message_data)
                            message = QueueMessage.from_dict(message_dict)
                            
                            await self._send_to_queue(message)
                            await self.redis_client.zrem(delayed_queue_name, message_data)
                            
                        except Exception as e:
                            self.logger.error("Error processing delayed message", error=str(e))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in delayed message processing", error=str(e))
    
    async def _update_metrics(self):
        """Background task to update metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update queue size metrics
                for queue_type in QueueType:
                    queue_name = self._get_queue_name(queue_type)
                    size = await self.redis_client.llen(queue_name)
                    
                    # Record metric
                    metrics.gauge(
                        "queue_size",
                        size,
                        tags={"queue_type": queue_type.value}
                    )
                
                # Record processing metrics
                metrics.counter("messages_sent", self.stats["messages_sent"])
                metrics.counter("messages_processed", self.stats["messages_processed"])
                metrics.counter("messages_failed", self.stats["messages_failed"])
                metrics.counter("messages_retried", self.stats["messages_retried"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error updating metrics", error=str(e))
    
    def _register_default_processors(self):
        """Register default message processors"""
        
        class DefaultProcessor(MessageProcessor):
            """Default processor for unhandled messages"""
            
            async def process(self, message: QueueMessage) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
                # Default processing - just log the message
                logger.info("Processing message with default processor", message_id=message.id)
                return True, {"processed_by": "default_processor"}, None
        
        # Register default processor for system events
        default_processor = DefaultProcessor("default_system_processor")
        self.register_processor(QueueType.SYSTEM_EVENTS, default_processor)

# Factory function
async def create_message_queue_manager(redis_client: redis.Redis) -> MessageQueueManager:
    """Create and start message queue manager"""
    manager = MessageQueueManager(redis_client)
    await manager.start()
    return manager

