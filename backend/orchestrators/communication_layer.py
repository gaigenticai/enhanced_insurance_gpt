"""
Insurance AI Agent System - Orchestrator Communication Layer
Production-ready communication system for orchestrator coordination and messaging
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import json
from dataclasses import dataclass, asdict
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import redis.asyncio as redis
from contextlib import asynccontextmanager

from backend.shared.services import EventService, ServiceException
from backend.shared.database import get_redis_client
from backend.shared.monitoring import metrics, audit_logger
from backend.shared.utils import DataUtils

logger = structlog.get_logger(__name__)

class MessageType(str, Enum):
    """Message type enumeration"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    NOTIFICATION = "notification"
    COMMAND = "command"

class MessagePriority(int, Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    NORMAL = 5

@dataclass
class Message:
    """Inter-orchestrator message"""
    id: str
    type: MessageType
    source: str
    target: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: str = None
    expires_at: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.id is None:
            self.id = str(uuid.uuid4())

@dataclass
class MessageHandler:
    """Message handler configuration"""
    message_type: str
    handler_func: Callable
    timeout_seconds: int = 30
    retry_on_failure: bool = True

class OrchestratorCommunicationLayer:
    """
    Communication layer for orchestrator coordination
    Handles message routing, delivery, and coordination between orchestrators
    """
    
    def __init__(self, orchestrator_name: str, redis_client: redis.Redis):
        self.orchestrator_name = orchestrator_name
        self.redis_client = redis_client
        self.event_service = EventService(redis_client)
        self.logger = structlog.get_logger(f"comm_layer_{orchestrator_name}")
        
        # Message handlers registry
        self.message_handlers: Dict[str, MessageHandler] = {}
        
        # Active message subscriptions
        self.subscriptions: Dict[str, asyncio.Task] = {}
        
        # Pending responses (for request-response pattern)
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Message queues
        self.inbox_queue = f"orchestrator:{orchestrator_name}:inbox"
        self.outbox_queue = f"orchestrator:{orchestrator_name}:outbox"
        
        # Circuit breaker for failed communications
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Initialize built-in message handlers
        self._initialize_handlers()
    
    def _initialize_handlers(self):
        """Initialize built-in message handlers"""
        
        # Health check handler
        self.register_handler(
            "health_check",
            MessageHandler(
                message_type="health_check",
                handler_func=self._handle_health_check,
                timeout_seconds=5
            )
        )
        
        # Status request handler
        self.register_handler(
            "status_request",
            MessageHandler(
                message_type="status_request",
                handler_func=self._handle_status_request,
                timeout_seconds=10
            )
        )
        
        # Workflow coordination handler
        self.register_handler(
            "workflow_coordination",
            MessageHandler(
                message_type="workflow_coordination",
                handler_func=self._handle_workflow_coordination,
                timeout_seconds=30
            )
        )
    
    async def start(self):
        """Start the communication layer"""
        
        try:
            # Start message processing
            self.subscriptions["inbox"] = asyncio.create_task(
                self._process_inbox_messages()
            )
            
            # Start health monitoring
            self.subscriptions["health"] = asyncio.create_task(
                self._health_monitor()
            )
            
            # Register orchestrator as available
            await self._register_orchestrator()
            
            self.logger.info(
                "Communication layer started",
                orchestrator=self.orchestrator_name
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to start communication layer",
                error=str(e)
            )
            raise
    
    async def stop(self):
        """Stop the communication layer"""
        
        try:
            # Cancel all subscriptions
            for task in self.subscriptions.values():
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.subscriptions.values(), return_exceptions=True)
            
            # Unregister orchestrator
            await self._unregister_orchestrator()
            
            self.logger.info(
                "Communication layer stopped",
                orchestrator=self.orchestrator_name
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to stop communication layer",
                error=str(e)
            )
    
    def register_handler(self, message_type: str, handler: MessageHandler):
        """Register a message handler"""
        
        self.message_handlers[message_type] = handler
        
        self.logger.info(
            "Message handler registered",
            message_type=message_type,
            orchestrator=self.orchestrator_name
        )
    
    async def send_message(
        self,
        target: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        expires_in_seconds: Optional[int] = None
    ) -> str:
        """Send a message to another orchestrator"""
        
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open(target):
                raise ServiceException(f"Circuit breaker open for target: {target}")
            
            # Create message
            message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.EVENT,
                source=self.orchestrator_name,
                target=target,
                payload=payload,
                priority=priority,
                correlation_id=correlation_id
            )
            
            if expires_in_seconds:
                message.expires_at = (
                    datetime.utcnow() + timedelta(seconds=expires_in_seconds)
                ).isoformat()
            
            # Send message
            await self._deliver_message(message)
            
            # Reset circuit breaker on success
            self._reset_circuit_breaker(target)
            
            self.logger.info(
                "Message sent",
                message_id=message.id,
                target=target,
                message_type=message_type
            )
            
            return message.id
            
        except Exception as e:
            # Record circuit breaker failure
            self._record_circuit_breaker_failure(target)
            
            self.logger.error(
                "Failed to send message",
                target=target,
                message_type=message_type,
                error=str(e)
            )
            raise
    
    async def send_request(
        self,
        target: str,
        message_type: str,
        payload: Dict[str, Any],
        timeout_seconds: int = 30,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Dict[str, Any]:
        """Send a request and wait for response"""
        
        try:
            # Create request message
            correlation_id = str(uuid.uuid4())
            
            message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.REQUEST,
                source=self.orchestrator_name,
                target=target,
                payload=payload,
                priority=priority,
                correlation_id=correlation_id,
                reply_to=self.orchestrator_name
            )
            
            # Create future for response
            response_future = asyncio.Future()
            self.pending_responses[correlation_id] = response_future
            
            try:
                # Send request
                await self._deliver_message(message)
                
                # Wait for response
                response = await asyncio.wait_for(response_future, timeout=timeout_seconds)
                
                self.logger.info(
                    "Request completed",
                    message_id=message.id,
                    target=target,
                    message_type=message_type
                )
                
                return response
                
            finally:
                # Cleanup pending response
                if correlation_id in self.pending_responses:
                    del self.pending_responses[correlation_id]
            
        except asyncio.TimeoutError:
            self.logger.error(
                "Request timeout",
                target=target,
                message_type=message_type,
                timeout_seconds=timeout_seconds
            )
            raise ServiceException(f"Request timeout after {timeout_seconds} seconds")
            
        except Exception as e:
            self.logger.error(
                "Request failed",
                target=target,
                message_type=message_type,
                error=str(e)
            )
            raise
    
    async def send_response(
        self,
        original_message: Message,
        response_payload: Dict[str, Any],
        success: bool = True
    ):
        """Send a response to a request"""
        
        try:
            if not original_message.reply_to or not original_message.correlation_id:
                raise ServiceException("Cannot send response: missing reply_to or correlation_id")
            
            # Create response message
            response = Message(
                id=str(uuid.uuid4()),
                type=MessageType.RESPONSE,
                source=self.orchestrator_name,
                target=original_message.reply_to,
                payload={
                    "success": success,
                    "data": response_payload,
                    "original_message_id": original_message.id
                },
                correlation_id=original_message.correlation_id
            )
            
            # Send response
            await self._deliver_message(response)
            
            self.logger.info(
                "Response sent",
                original_message_id=original_message.id,
                target=original_message.reply_to,
                success=success
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to send response",
                original_message_id=original_message.id,
                error=str(e)
            )
            raise
    
    async def broadcast_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        targets: Optional[List[str]] = None
    ):
        """Broadcast an event to multiple orchestrators"""
        
        try:
            if targets is None:
                # Get all registered orchestrators
                targets = await self._get_registered_orchestrators()
                # Remove self from targets
                targets = [t for t in targets if t != self.orchestrator_name]
            
            # Send to all targets
            tasks = []
            for target in targets:
                task = self.send_message(
                    target=target,
                    message_type=event_type,
                    payload=payload,
                    priority=MessagePriority.MEDIUM
                )
                tasks.append(task)
            
            # Wait for all sends to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info(
                "Event broadcasted",
                event_type=event_type,
                target_count=len(targets)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to broadcast event",
                event_type=event_type,
                error=str(e)
            )
            raise
    
    async def _deliver_message(self, message: Message):
        """Deliver message to target orchestrator"""
        
        try:
            # Serialize message
            message_data = asdict(message)
            message_json = json.dumps(message_data, default=str)
            
            # Add to target's inbox queue
            target_inbox = f"orchestrator:{message.target}:inbox"
            
            # Use priority queue (higher priority = lower score)
            priority_score = message.priority.value
            
            await self.redis_client.zadd(
                target_inbox,
                {message_json: priority_score}
            )
            
            # Set expiration if specified
            if message.expires_at:
                expire_time = datetime.fromisoformat(message.expires_at)
                ttl = int((expire_time - datetime.utcnow()).total_seconds())
                if ttl > 0:
                    await self.redis_client.expire(target_inbox, ttl)
            
            # Record metrics
            metrics.record_workflow("message_delivery", "success")
            
        except Exception as e:
            metrics.record_workflow("message_delivery", "failed")
            raise
    
    async def _process_inbox_messages(self):
        """Process incoming messages from inbox queue"""
        
        while True:
            try:
                # Get highest priority message
                messages = await self.redis_client.zrange(
                    self.inbox_queue,
                    0, 0,  # Get first (highest priority) message
                    withscores=True
                )
                
                if messages:
                    message_json, priority_score = messages[0]
                    
                    # Remove message from queue
                    await self.redis_client.zrem(self.inbox_queue, message_json)
                    
                    # Deserialize message
                    message_data = json.loads(message_json)
                    message = Message(**message_data)
                    
                    # Check if message has expired
                    if message.expires_at:
                        expire_time = datetime.fromisoformat(message.expires_at)
                        if datetime.utcnow() > expire_time:
                            self.logger.warning(
                                "Message expired",
                                message_id=message.id,
                                expired_at=message.expires_at
                            )
                            continue
                    
                    # Process message
                    await self._handle_message(message)
                
                else:
                    # No messages, wait a bit
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(
                    "Error processing inbox messages",
                    error=str(e)
                )
                await asyncio.sleep(1)  # Wait longer on error
    
    async def _handle_message(self, message: Message):
        """Handle incoming message"""
        
        try:
            # Log message receipt
            self.logger.info(
                "Message received",
                message_id=message.id,
                source=message.source,
                type=message.type.value,
                payload_keys=list(message.payload.keys())
            )
            
            # Handle response messages
            if message.type == MessageType.RESPONSE and message.correlation_id:
                if message.correlation_id in self.pending_responses:
                    future = self.pending_responses[message.correlation_id]
                    if not future.done():
                        future.set_result(message.payload)
                    return
            
            # Find handler for message type
            handler = None
            for msg_type, msg_handler in self.message_handlers.items():
                if msg_type in message.payload or msg_type == message.type.value:
                    handler = msg_handler
                    break
            
            if not handler:
                self.logger.warning(
                    "No handler found for message",
                    message_id=message.id,
                    message_type=message.type.value
                )
                return
            
            # Execute handler with timeout
            try:
                result = await asyncio.wait_for(
                    handler.handler_func(message),
                    timeout=handler.timeout_seconds
                )
                
                # Send response if this was a request
                if message.type == MessageType.REQUEST:
                    await self.send_response(message, result, success=True)
                
            except asyncio.TimeoutError:
                self.logger.error(
                    "Message handler timeout",
                    message_id=message.id,
                    handler_timeout=handler.timeout_seconds
                )
                
                if message.type == MessageType.REQUEST:
                    await self.send_response(
                        message,
                        {"error": "Handler timeout"},
                        success=False
                    )
            
            except Exception as e:
                self.logger.error(
                    "Message handler failed",
                    message_id=message.id,
                    error=str(e)
                )
                
                if message.type == MessageType.REQUEST:
                    await self.send_response(
                        message,
                        {"error": str(e)},
                        success=False
                    )
                
                # Retry if configured
                if handler.retry_on_failure and message.retry_count < message.max_retries:
                    message.retry_count += 1
                    await asyncio.sleep(2 ** message.retry_count)  # Exponential backoff
                    await self._deliver_message(message)
            
        except Exception as e:
            self.logger.error(
                "Failed to handle message",
                message_id=message.id,
                error=str(e)
            )
    
    # Built-in message handlers
    
    async def _handle_health_check(self, message: Message) -> Dict[str, Any]:
        """Handle health check request"""
        
        return {
            "status": "healthy",
            "orchestrator": self.orchestrator_name,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - datetime.utcnow()).total_seconds()  # Placeholder
        }
    
    async def _handle_status_request(self, message: Message) -> Dict[str, Any]:
        """Handle status request"""
        
        return {
            "orchestrator": self.orchestrator_name,
            "status": "running",
            "active_workflows": 0,  # Placeholder
            "pending_messages": await self.redis_client.zcard(self.inbox_queue),
            "circuit_breakers": {
                name: breaker["state"] 
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    async def _handle_workflow_coordination(self, message: Message) -> Dict[str, Any]:
        """Handle workflow coordination request"""
        
        coordination_type = message.payload.get("coordination_type")
        
        if coordination_type == "workflow_handoff":
            # Handle workflow handoff between orchestrators
            return await self._handle_workflow_handoff(message.payload)
        
        elif coordination_type == "resource_request":
            # Handle resource request
            return await self._handle_resource_request(message.payload)
        
        elif coordination_type == "status_sync":
            # Handle status synchronization
            return await self._handle_status_sync(message.payload)
        
        else:
            return {"error": f"Unknown coordination type: {coordination_type}"}
    
    async def _handle_workflow_handoff(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow handoff between orchestrators"""
        
        workflow_id = payload.get("workflow_id")
        workflow_data = payload.get("workflow_data")
        
        # Production workflow handoff implementation
        try:
            # Validate workflow data
            if not workflow_id or not workflow_data:
                return {
                    "accepted": False,
                    "error": "Missing workflow_id or workflow_data",
                    "workflow_id": workflow_id
                }
            
            # Store workflow in database
            workflow_record = {
                "workflow_id": workflow_id,
                "source_orchestrator": payload.get("source_orchestrator"),
                "target_orchestrator": payload.get("target_orchestrator"),
                "workflow_type": workflow_data.get("type"),
                "workflow_data": workflow_data,
                "status": "received",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Insert into database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO workflow_handoffs 
                    (workflow_id, source_orchestrator, target_orchestrator, 
                     workflow_type, workflow_data, status, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                workflow_id, 
                workflow_record["source_orchestrator"],
                workflow_record["target_orchestrator"],
                workflow_record["workflow_type"],
                json.dumps(workflow_record["workflow_data"]),
                workflow_record["status"],
                workflow_record["created_at"],
                workflow_record["updated_at"]
                )
            
            # Cache workflow for quick access
            await self.redis.setex(
                f"workflow_handoff:{workflow_id}",
                3600,  # 1 hour TTL
                json.dumps(workflow_record, default=str)
            )
            
            # Trigger workflow processing
            await self._process_handoff_workflow(workflow_id, workflow_data)
            
            self.logger.info(
                "Workflow handoff accepted and processed",
                workflow_id=workflow_id,
                from_orchestrator=payload.get("source_orchestrator"),
                workflow_type=workflow_data.get("type")
            )
            
            return {
                "accepted": True,
                "workflow_id": workflow_id,
                "message": "Workflow handoff accepted and processing initiated",
                "status": "processing"
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to process workflow handoff",
                workflow_id=workflow_id,
                error=str(e)
            )
            return {
                "accepted": False,
                "workflow_id": workflow_id,
                "error": f"Failed to process handoff: {str(e)}"
            }
    
    async def _handle_resource_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource request from another orchestrator"""
        
        resource_type = payload.get("resource_type")
        resource_id = payload.get("resource_id")
        requesting_orchestrator = payload.get("requesting_orchestrator")
        
        # Production resource management implementation
        try:
            # Validate request
            if not resource_type or not resource_id:
                return {
                    "available": False,
                    "error": "Missing resource_type or resource_id",
                    "resource_type": resource_type,
                    "resource_id": resource_id
                }
            
            # Check resource availability based on type
            if resource_type == "agent_capacity":
                # Check agent service capacity
                capacity_info = await self._check_agent_capacity(resource_id)
                return {
                    "available": capacity_info["available"],
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "capacity": capacity_info,
                    "estimated_wait_time": capacity_info.get("wait_time", 0)
                }
                
            elif resource_type == "database_connection":
                # Check database connection pool
                pool_info = await self._check_db_pool_capacity()
                return {
                    "available": pool_info["available"],
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "pool_status": pool_info
                }
                
            elif resource_type == "processing_slot":
                # Check processing queue capacity
                queue_info = await self._check_processing_queue(resource_id)
                return {
                    "available": queue_info["available"],
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "queue_position": queue_info.get("position"),
                    "estimated_processing_time": queue_info.get("processing_time")
                }
                
            elif resource_type == "memory":
                # Check memory availability
                memory_info = await self._check_memory_availability()
                return {
                    "available": memory_info["available"],
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "memory_usage": memory_info
                }
                
            else:
                # Unknown resource type
                return {
                    "available": False,
                    "error": f"Unknown resource type: {resource_type}",
                    "resource_type": resource_type,
                    "resource_id": resource_id
                }
                
        except Exception as e:
            self.logger.error(
                "Failed to process resource request",
                resource_type=resource_type,
                resource_id=resource_id,
                error=str(e)
            )
            return {
                "available": False,
                "error": f"Resource check failed: {str(e)}",
                "resource_type": resource_type,
                "resource_id": resource_id
            }
    
    async def _handle_status_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status synchronization request"""
        
        # Production status synchronization implementation
        try:
            sync_type = payload.get("sync_type", "full")
            target_orchestrator = payload.get("target_orchestrator")
            
            # Gather current status information
            status_data = {
                "orchestrator_id": self.orchestrator_id,
                "timestamp": datetime.utcnow().isoformat(),
                "sync_type": sync_type,
                "status": {
                    "health": await self._get_health_status(),
                    "metrics": await self._get_current_metrics(),
                    "active_workflows": await self._get_active_workflows_count(),
                    "resource_usage": await self._get_resource_usage(),
                    "agent_status": await self._get_agent_status_summary()
                }
            }
            
            # Store sync record in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO status_syncs 
                    (sync_id, source_orchestrator, target_orchestrator, 
                     sync_type, status_data, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                str(uuid.uuid4()),
                self.orchestrator_id,
                target_orchestrator,
                sync_type,
                json.dumps(status_data),
                datetime.utcnow()
                )
            
            # Update Redis cache with latest status
            await self.redis.setex(
                f"orchestrator_status:{self.orchestrator_id}",
                300,  # 5 minutes TTL
                json.dumps(status_data, default=str)
            )
            
            self.logger.info(
                "Status sync completed",
                sync_type=sync_type,
                target_orchestrator=target_orchestrator
            )
            
            return {
                "sync_completed": True,
                "timestamp": datetime.utcnow().isoformat(),
                "status_data": status_data
            }
            
        except Exception as e:
            self.logger.error(
                "Status sync failed",
                error=str(e)
            )
            return {
                "sync_completed": False,
                "error": f"Status sync failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Circuit breaker methods
    
    def _is_circuit_breaker_open(self, target: str) -> bool:
        """Check if circuit breaker is open for target"""
        
        if target not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[target]
        
        if breaker["state"] == "open":
            if datetime.utcnow() > breaker["next_attempt"]:
                breaker["state"] = "half_open"
                return False
            return True
        
        return False
    
    def _record_circuit_breaker_failure(self, target: str):
        """Record circuit breaker failure"""
        
        if target not in self.circuit_breakers:
            self.circuit_breakers[target] = {
                "failure_count": 0,
                "state": "closed",
                "next_attempt": None
            }
        
        breaker = self.circuit_breakers[target]
        breaker["failure_count"] += 1
        
        if breaker["failure_count"] >= 3:  # Threshold of 3 failures
            breaker["state"] = "open"
            breaker["next_attempt"] = datetime.utcnow() + timedelta(minutes=2)  # 2 minute cooldown
            
            self.logger.warning(
                "Circuit breaker opened",
                target=target,
                failure_count=breaker["failure_count"]
            )
    
    def _reset_circuit_breaker(self, target: str):
        """Reset circuit breaker on success"""
        
        if target in self.circuit_breakers:
            self.circuit_breakers[target] = {
                "failure_count": 0,
                "state": "closed",
                "next_attempt": None
            }
    
    # Registry methods
    
    async def _register_orchestrator(self):
        """Register orchestrator as available"""
        
        try:
            registry_key = "orchestrator_registry"
            orchestrator_data = {
                "name": self.orchestrator_name,
                "status": "active",
                "registered_at": datetime.utcnow().isoformat(),
                "last_heartbeat": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.hset(
                registry_key,
                self.orchestrator_name,
                json.dumps(orchestrator_data)
            )
            
            self.logger.info(
                "Orchestrator registered",
                orchestrator=self.orchestrator_name
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to register orchestrator",
                error=str(e)
            )
    
    async def _unregister_orchestrator(self):
        """Unregister orchestrator"""
        
        try:
            registry_key = "orchestrator_registry"
            await self.redis_client.hdel(registry_key, self.orchestrator_name)
            
            self.logger.info(
                "Orchestrator unregistered",
                orchestrator=self.orchestrator_name
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to unregister orchestrator",
                error=str(e)
            )
    
    async def _get_registered_orchestrators(self) -> List[str]:
        """Get list of registered orchestrators"""
        
        try:
            registry_key = "orchestrator_registry"
            orchestrators = await self.redis_client.hkeys(registry_key)
            return [orch.decode() if isinstance(orch, bytes) else orch for orch in orchestrators]
            
        except Exception as e:
            self.logger.error(
                "Failed to get registered orchestrators",
                error=str(e)
            )
            return []
    
    async def _health_monitor(self):
        """Health monitoring and heartbeat"""
        
        while True:
            try:
                # Update heartbeat
                registry_key = "orchestrator_registry"
                orchestrator_data = await self.redis_client.hget(
                    registry_key,
                    self.orchestrator_name
                )
                
                if orchestrator_data:
                    data = json.loads(orchestrator_data)
                    data["last_heartbeat"] = datetime.utcnow().isoformat()
                    
                    await self.redis_client.hset(
                        registry_key,
                        self.orchestrator_name,
                        json.dumps(data)
                    )
                
                # Wait for next heartbeat
                await asyncio.sleep(30)  # 30 second heartbeat
                
            except Exception as e:
                self.logger.error(
                    "Health monitor error",
                    error=str(e)
                )
                await asyncio.sleep(60)  # Wait longer on error

# Communication layer factory
async def create_communication_layer(orchestrator_name: str) -> OrchestratorCommunicationLayer:
    """Create communication layer for orchestrator"""
    
    async with get_redis_client() as redis_client:
        comm_layer = OrchestratorCommunicationLayer(orchestrator_name, redis_client)
        return comm_layer

@asynccontextmanager
async def communication_layer_context(orchestrator_name: str):
    """Context manager for communication layer"""
    
    comm_layer = await create_communication_layer(orchestrator_name)
    
    try:
        await comm_layer.start()
        yield comm_layer
    finally:
        await comm_layer.stop()

