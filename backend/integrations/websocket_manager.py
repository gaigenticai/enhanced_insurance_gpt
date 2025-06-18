"""
Insurance AI Agent System - WebSocket Manager
Production-ready WebSocket communication system for real-time updates
"""

import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable, Union
from enum import Enum
import weakref
from dataclasses import dataclass, asdict
import logging

# FastAPI and WebSocket
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocketState
import websockets

# Database and Redis
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text

# Internal imports
from backend.shared.models import User, Organization, WebSocketConnection, SystemEvent
from backend.shared.schemas import WebSocketMessage, EventNotification, SystemStatus
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client
from backend.shared.monitoring import metrics, performance_monitor, audit_logger
from backend.shared.utils import DataUtils, ValidationUtils

import structlog
logger = structlog.get_logger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    SYSTEM_STATUS = "system_status"
    CLAIM_UPDATE = "claim_update"
    POLICY_UPDATE = "policy_update"
    AGENT_STATUS = "agent_status"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    AUTHENTICATION = "authentication"
    SUBSCRIPTION = "subscription"
    UNSUBSCRIPTION = "unsubscription"
    BROADCAST = "broadcast"
    PRIVATE_MESSAGE = "private_message"
    WORKFLOW_UPDATE = "workflow_update"
    DOCUMENT_PROCESSED = "document_processed"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_ALERT = "compliance_alert"

class SubscriptionType(Enum):
    """Subscription types for WebSocket clients"""
    ALL_EVENTS = "all_events"
    CLAIMS = "claims"
    POLICIES = "policies"
    AGENTS = "agents"
    NOTIFICATIONS = "notifications"
    WORKFLOWS = "workflows"
    SYSTEM_STATUS = "system_status"
    USER_SPECIFIC = "user_specific"
    ORGANIZATION_SPECIFIC = "organization_specific"

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: float
    message_id: str
    sender_id: Optional[str] = None
    recipient_id: Optional[str] = None
    subscription_type: Optional[SubscriptionType] = None
    priority: int = 0  # 0 = normal, 1 = high, 2 = critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "subscription_type": self.subscription_type.value if self.subscription_type else None,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create from dictionary"""
        return cls(
            type=MessageType(data["type"]),
            data=data["data"],
            timestamp=data["timestamp"],
            message_id=data["message_id"],
            sender_id=data.get("sender_id"),
            recipient_id=data.get("recipient_id"),
            subscription_type=SubscriptionType(data["subscription_type"]) if data.get("subscription_type") else None,
            priority=data.get("priority", 0)
        )

@dataclass
class ConnectionInfo:
    """WebSocket connection information"""
    connection_id: str
    user_id: Optional[str]
    organization_id: Optional[str]
    websocket: WebSocket
    subscriptions: Set[SubscriptionType]
    connected_at: datetime
    last_heartbeat: datetime
    is_authenticated: bool = False
    metadata: Dict[str, Any] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class WebSocketManager:
    """
    Comprehensive WebSocket manager for real-time communication
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = structlog.get_logger("websocket_manager")
        
        # Active connections
        self.connections: Dict[str, ConnectionInfo] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.organization_connections: Dict[str, Set[str]] = {}  # org_id -> connection_ids
        
        # Message queues and handlers
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.subscription_handlers: Dict[SubscriptionType, List[Callable]] = {}
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 300  # seconds
        self.max_connections_per_user = 5
        self.max_message_size = 1024 * 1024  # 1MB
        
        # Background tasks
        self.cleanup_task = None
        self.heartbeat_task = None
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0
        }
    
    async def start(self):
        """Start the WebSocket manager"""
        try:
            self.logger.info("Starting WebSocket manager")
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_connections())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            # Register default message handlers
            self._register_default_handlers()
            
            self.logger.info("WebSocket manager started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start WebSocket manager", error=str(e))
            raise
    
    async def stop(self):
        """Stop the WebSocket manager"""
        try:
            self.logger.info("Stopping WebSocket manager")
            
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            
            # Close all connections
            for connection_id in list(self.connections.keys()):
                await self.disconnect(connection_id)
            
            self.logger.info("WebSocket manager stopped")
            
        except Exception as e:
            self.logger.error("Error stopping WebSocket manager", error=str(e))
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None, organization_id: Optional[str] = None) -> str:
        """
        Accept a new WebSocket connection
        
        Args:
            websocket: WebSocket connection
            user_id: Optional user ID for authenticated connections
            organization_id: Optional organization ID
            
        Returns:
            Connection ID
        """
        
        try:
            # Generate connection ID
            connection_id = str(uuid.uuid4())
            
            # Check connection limits
            if user_id and user_id in self.user_connections:
                if len(self.user_connections[user_id]) >= self.max_connections_per_user:
                    await websocket.close(code=1008, reason="Too many connections")
                    raise ServiceException("Too many connections for user")
            
            # Accept WebSocket connection
            await websocket.accept()
            
            # Create connection info
            connection_info = ConnectionInfo(
                connection_id=connection_id,
                user_id=user_id,
                organization_id=organization_id,
                websocket=websocket,
                subscriptions=set(),
                connected_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                is_authenticated=user_id is not None,
                client_ip=websocket.client.host if websocket.client else None,
                user_agent=websocket.headers.get("user-agent")
            )
            
            # Store connection
            self.connections[connection_id] = connection_info
            
            # Update user and organization mappings
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
            
            if organization_id:
                if organization_id not in self.organization_connections:
                    self.organization_connections[organization_id] = set()
                self.organization_connections[organization_id].add(connection_id)
            
            # Update statistics
            self.stats["total_connections"] += 1
            self.stats["active_connections"] = len(self.connections)
            
            # Send welcome message
            await self.send_message(
                connection_id,
                MessageType.SYSTEM_STATUS,
                {
                    "status": "connected",
                    "connection_id": connection_id,
                    "server_time": datetime.utcnow().isoformat(),
                    "heartbeat_interval": self.heartbeat_interval
                }
            )
            
            # Log connection
            self.logger.info(
                "WebSocket connection established",
                connection_id=connection_id,
                user_id=user_id,
                organization_id=organization_id
            )
            
            # Record in database
            await self._record_connection(connection_info)
            
            return connection_id
            
        except Exception as e:
            self.logger.error("Failed to establish WebSocket connection", error=str(e))
            self.stats["errors"] += 1
            raise
    
    async def disconnect(self, connection_id: str, code: int = 1000, reason: str = "Normal closure"):
        """
        Disconnect a WebSocket connection
        
        Args:
            connection_id: Connection ID to disconnect
            code: WebSocket close code
            reason: Reason for disconnection
        """
        
        try:
            if connection_id not in self.connections:
                return
            
            connection_info = self.connections[connection_id]
            
            # Close WebSocket if still open
            if connection_info.websocket.client_state == WebSocketState.CONNECTED:
                await connection_info.websocket.close(code=code, reason=reason)
            
            # Remove from mappings
            if connection_info.user_id and connection_info.user_id in self.user_connections:
                self.user_connections[connection_info.user_id].discard(connection_id)
                if not self.user_connections[connection_info.user_id]:
                    del self.user_connections[connection_info.user_id]
            
            if connection_info.organization_id and connection_info.organization_id in self.organization_connections:
                self.organization_connections[connection_info.organization_id].discard(connection_id)
                if not self.organization_connections[connection_info.organization_id]:
                    del self.organization_connections[connection_info.organization_id]
            
            # Remove connection
            del self.connections[connection_id]
            
            # Update statistics
            self.stats["active_connections"] = len(self.connections)
            
            # Log disconnection
            self.logger.info(
                "WebSocket connection closed",
                connection_id=connection_id,
                user_id=connection_info.user_id,
                reason=reason
            )
            
            # Record in database
            await self._record_disconnection(connection_info)
            
        except Exception as e:
            self.logger.error("Error disconnecting WebSocket", connection_id=connection_id, error=str(e))
    
    async def send_message(
        self,
        connection_id: str,
        message_type: MessageType,
        data: Dict[str, Any],
        priority: int = 0
    ) -> bool:
        """
        Send message to a specific connection
        
        Args:
            connection_id: Target connection ID
            message_type: Type of message
            data: Message data
            priority: Message priority (0=normal, 1=high, 2=critical)
            
        Returns:
            True if message was sent successfully
        """
        
        try:
            if connection_id not in self.connections:
                self.logger.warning("Attempted to send message to non-existent connection", connection_id=connection_id)
                return False
            
            connection_info = self.connections[connection_id]
            
            # Check if connection is still active
            if connection_info.websocket.client_state != WebSocketState.CONNECTED:
                await self.disconnect(connection_id, reason="Connection lost")
                return False
            
            # Create message
            message = WebSocketMessage(
                type=message_type,
                data=data,
                timestamp=time.time(),
                message_id=str(uuid.uuid4()),
                priority=priority
            )
            
            # Send message
            message_json = json.dumps(message.to_dict())
            
            # Check message size
            if len(message_json.encode('utf-8')) > self.max_message_size:
                self.logger.warning("Message too large", connection_id=connection_id, size=len(message_json))
                return False
            
            await connection_info.websocket.send_text(message_json)
            
            # Update statistics
            self.stats["messages_sent"] += 1
            
            # Log high priority messages
            if priority > 0:
                self.logger.info(
                    "High priority message sent",
                    connection_id=connection_id,
                    message_type=message_type.value,
                    priority=priority
                )
            
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id, reason="Client disconnected")
            return False
        except Exception as e:
            self.logger.error("Failed to send WebSocket message", connection_id=connection_id, error=str(e))
            self.stats["errors"] += 1
            return False
    
    async def broadcast_message(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
        subscription_type: Optional[SubscriptionType] = None,
        user_ids: Optional[List[str]] = None,
        organization_ids: Optional[List[str]] = None,
        exclude_connections: Optional[List[str]] = None,
        priority: int = 0
    ) -> int:
        """
        Broadcast message to multiple connections
        
        Args:
            message_type: Type of message
            data: Message data
            subscription_type: Only send to connections with this subscription
            user_ids: Only send to specific users
            organization_ids: Only send to specific organizations
            exclude_connections: Connection IDs to exclude
            priority: Message priority
            
        Returns:
            Number of connections that received the message
        """
        
        try:
            target_connections = set()
            exclude_connections = set(exclude_connections or [])
            
            # Determine target connections
            if user_ids:
                for user_id in user_ids:
                    if user_id in self.user_connections:
                        target_connections.update(self.user_connections[user_id])
            elif organization_ids:
                for org_id in organization_ids:
                    if org_id in self.organization_connections:
                        target_connections.update(self.organization_connections[org_id])
            else:
                # Broadcast to all connections
                target_connections = set(self.connections.keys())
            
            # Filter by subscription type
            if subscription_type:
                filtered_connections = set()
                for connection_id in target_connections:
                    if connection_id in self.connections:
                        connection_info = self.connections[connection_id]
                        if subscription_type in connection_info.subscriptions or SubscriptionType.ALL_EVENTS in connection_info.subscriptions:
                            filtered_connections.add(connection_id)
                target_connections = filtered_connections
            
            # Remove excluded connections
            target_connections -= exclude_connections
            
            # Send messages
            sent_count = 0
            for connection_id in target_connections:
                if await self.send_message(connection_id, message_type, data, priority):
                    sent_count += 1
            
            self.logger.info(
                "Broadcast message sent",
                message_type=message_type.value,
                target_count=len(target_connections),
                sent_count=sent_count
            )
            
            return sent_count
            
        except Exception as e:
            self.logger.error("Failed to broadcast message", error=str(e))
            self.stats["errors"] += 1
            return 0
    
    async def subscribe(self, connection_id: str, subscription_type: SubscriptionType) -> bool:
        """
        Subscribe connection to a specific type of messages
        
        Args:
            connection_id: Connection ID
            subscription_type: Type of subscription
            
        Returns:
            True if subscription was successful
        """
        
        try:
            if connection_id not in self.connections:
                return False
            
            connection_info = self.connections[connection_id]
            connection_info.subscriptions.add(subscription_type)
            
            # Send confirmation
            await self.send_message(
                connection_id,
                MessageType.SUBSCRIPTION,
                {
                    "subscription_type": subscription_type.value,
                    "status": "subscribed"
                }
            )
            
            self.logger.info(
                "Connection subscribed",
                connection_id=connection_id,
                subscription_type=subscription_type.value
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to subscribe connection", connection_id=connection_id, error=str(e))
            return False
    
    async def unsubscribe(self, connection_id: str, subscription_type: SubscriptionType) -> bool:
        """
        Unsubscribe connection from a specific type of messages
        
        Args:
            connection_id: Connection ID
            subscription_type: Type of subscription to remove
            
        Returns:
            True if unsubscription was successful
        """
        
        try:
            if connection_id not in self.connections:
                return False
            
            connection_info = self.connections[connection_id]
            connection_info.subscriptions.discard(subscription_type)
            
            # Send confirmation
            await self.send_message(
                connection_id,
                MessageType.UNSUBSCRIPTION,
                {
                    "subscription_type": subscription_type.value,
                    "status": "unsubscribed"
                }
            )
            
            self.logger.info(
                "Connection unsubscribed",
                connection_id=connection_id,
                subscription_type=subscription_type.value
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to unsubscribe connection", connection_id=connection_id, error=str(e))
            return False
    
    async def handle_message(self, connection_id: str, message_data: str):
        """
        Handle incoming message from WebSocket connection
        
        Args:
            connection_id: Source connection ID
            message_data: Raw message data
        """
        
        try:
            if connection_id not in self.connections:
                return
            
            connection_info = self.connections[connection_id]
            
            # Parse message
            try:
                message_dict = json.loads(message_data)
                message = WebSocketMessage.from_dict(message_dict)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                await self.send_message(
                    connection_id,
                    MessageType.ERROR,
                    {"error": "Invalid message format", "details": str(e)}
                )
                return
            
            # Update statistics
            self.stats["messages_received"] += 1
            
            # Update last heartbeat for heartbeat messages
            if message.type == MessageType.HEARTBEAT:
                connection_info.last_heartbeat = datetime.utcnow()
                await self.send_message(
                    connection_id,
                    MessageType.HEARTBEAT,
                    {"status": "pong", "server_time": datetime.utcnow().isoformat()}
                )
                return
            
            # Handle subscription/unsubscription messages
            if message.type == MessageType.SUBSCRIPTION:
                subscription_type = SubscriptionType(message.data.get("subscription_type"))
                await self.subscribe(connection_id, subscription_type)
                return
            elif message.type == MessageType.UNSUBSCRIPTION:
                subscription_type = SubscriptionType(message.data.get("subscription_type"))
                await self.unsubscribe(connection_id, subscription_type)
                return
            
            # Call registered message handlers
            if message.type in self.message_handlers:
                for handler in self.message_handlers[message.type]:
                    try:
                        await handler(connection_id, message, connection_info)
                    except Exception as e:
                        self.logger.error("Message handler error", handler=str(handler), error=str(e))
            
            # Log important messages
            if message.priority > 0:
                self.logger.info(
                    "High priority message received",
                    connection_id=connection_id,
                    message_type=message.type.value,
                    priority=message.priority
                )
            
        except Exception as e:
            self.logger.error("Error handling WebSocket message", connection_id=connection_id, error=str(e))
            self.stats["errors"] += 1
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for a specific message type"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def register_subscription_handler(self, subscription_type: SubscriptionType, handler: Callable):
        """Register a handler for subscription events"""
        if subscription_type not in self.subscription_handlers:
            self.subscription_handlers[subscription_type] = []
        self.subscription_handlers[subscription_type].append(handler)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            **self.stats,
            "connections_by_user": {
                user_id: len(connections) 
                for user_id, connections in self.user_connections.items()
            },
            "connections_by_organization": {
                org_id: len(connections) 
                for org_id, connections in self.organization_connections.items()
            },
            "subscriptions": {
                connection_id: [sub.value for sub in info.subscriptions]
                for connection_id, info in self.connections.items()
            }
        }
    
    async def _cleanup_connections(self):
        """Background task to clean up stale connections"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection_info in self.connections.items():
                    # Check for timeout
                    if (current_time - connection_info.last_heartbeat).total_seconds() > self.connection_timeout:
                        stale_connections.append(connection_id)
                    # Check if WebSocket is still connected
                    elif connection_info.websocket.client_state != WebSocketState.CONNECTED:
                        stale_connections.append(connection_id)
                
                # Clean up stale connections
                for connection_id in stale_connections:
                    await self.disconnect(connection_id, reason="Connection timeout or lost")
                
                if stale_connections:
                    self.logger.info("Cleaned up stale connections", count=len(stale_connections))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in connection cleanup", error=str(e))
    
    async def _heartbeat_monitor(self):
        """Background task to monitor connection health"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send heartbeat to all connections
                heartbeat_data = {
                    "server_time": datetime.utcnow().isoformat(),
                    "active_connections": len(self.connections)
                }
                
                for connection_id in list(self.connections.keys()):
                    await self.send_message(connection_id, MessageType.HEARTBEAT, heartbeat_data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in heartbeat monitor", error=str(e))
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        
        async def handle_authentication(connection_id: str, message: WebSocketMessage, connection_info: ConnectionInfo):
            """Handle authentication messages"""
            # This would integrate with the authentication system
            # For now, just acknowledge
            await self.send_message(
                connection_id,
                MessageType.AUTHENTICATION,
                {"status": "acknowledged"}
            )
        
        async def handle_error(connection_id: str, message: WebSocketMessage, connection_info: ConnectionInfo):
            """Handle error messages"""
            self.logger.warning(
                "Client reported error",
                connection_id=connection_id,
                error=message.data
            )
        
        # Register handlers
        self.register_message_handler(MessageType.AUTHENTICATION, handle_authentication)
        self.register_message_handler(MessageType.ERROR, handle_error)
    
    async def _record_connection(self, connection_info: ConnectionInfo):
        """Record connection in database"""
        try:
            record = WebSocketConnection(
                user_id=connection_info.user_id,
                organization_id=connection_info.organization_id,
                connection_id=connection_info.connection_id,
                client_ip=connection_info.client_ip,
                user_agent=connection_info.user_agent,
                metadata=connection_info.metadata or {},
            )
            self.db_session.add(record)
            await self.db_session.commit()
        except Exception as e:
            self.logger.warning("Failed to record connection in database", error=str(e))

    async def _record_disconnection(self, connection_info: ConnectionInfo):
        """Record disconnection in database"""
        try:
            result = await self.db_session.execute(
                select(WebSocketConnection).where(WebSocketConnection.connection_id == connection_info.connection_id)
            )
            ws_conn = result.scalars().first()
            if ws_conn:
                ws_conn.is_active = False
                ws_conn.disconnected_at = datetime.utcnow()
                await self.db_session.commit()
        except Exception as e:
            self.logger.warning("Failed to record disconnection in database", error=str(e))

# Factory function
async def create_websocket_manager(redis_client: redis.Redis) -> WebSocketManager:
    """Create and start WebSocket manager"""
    manager = WebSocketManager(redis_client)
    await manager.start()
    return manager

