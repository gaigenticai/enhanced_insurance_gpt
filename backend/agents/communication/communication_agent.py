"""
Communication Agent - Production Ready Implementation
Handles customer communication, notifications, and multi-channel messaging for insurance operations
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

# Third-party imports
import aiohttp
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioException

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
messages_sent_total = Counter('messages_sent_total', 'Total messages sent', ['channel', 'status'])
message_processing_duration = Histogram('message_processing_duration_seconds', 'Time spent processing messages')
active_communications = Gauge('active_communications', 'Number of active communication sessions')

Base = declarative_base()

class MessageChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    PHONE = "phone"
    POSTAL = "postal"

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 3
    HIGH = 7
    URGENT = 10

class MessageStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CommunicationType(Enum):
    NOTIFICATION = "notification"
    REMINDER = "reminder"
    ALERT = "alert"
    CONFIRMATION = "confirmation"
    MARKETING = "marketing"
    SUPPORT = "support"

@dataclass
class CommunicationMessage:
    """Represents a communication message with all metadata"""
    message_id: str
    recipient_id: str
    channel: MessageChannel
    priority: MessagePriority
    communication_type: CommunicationType
    subject: str
    content: str
    template_id: Optional[str] = None
    template_data: Optional[Dict[str, Any]] = None
    scheduled_at: Optional[datetime] = None
    created_at: datetime = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    status: MessageStatus = MessageStatus.PENDING
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CommunicationPreferences:
    """User communication preferences"""
    user_id: str
    email_enabled: bool = True
    sms_enabled: bool = True
    push_enabled: bool = True
    phone_enabled: bool = False
    marketing_enabled: bool = True
    notification_hours_start: int = 9  # 9 AM
    notification_hours_end: int = 21   # 9 PM
    timezone: str = "UTC"
    preferred_language: str = "en"
    do_not_disturb: bool = False

@dataclass
class ContactInfo:
    """Contact information for a user"""
    user_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    postal_address: Optional[Dict[str, str]] = None
    preferred_channel: MessageChannel = MessageChannel.EMAIL

class CommunicationMessageModel(Base):
    """SQLAlchemy model for persisting communication messages"""
    __tablename__ = 'communication_messages'
    
    message_id = Column(String, primary_key=True)
    recipient_id = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    priority = Column(Integer, nullable=False)
    communication_type = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    template_id = Column(String)
    template_data = Column(JSON)
    scheduled_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    sent_at = Column(DateTime)
    delivered_at = Column(DateTime)
    status = Column(String, nullable=False)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    model_metadata = Column(JSON)

class CommunicationAgent:
    """
    Production-ready Communication Agent for Insurance AI System
    Handles customer communication, notifications, and multi-channel messaging
    """
    
    def __init__(self, db_url: str, redis_url: str, config: Dict[str, Any]):
        self.db_url = db_url
        self.redis_url = redis_url
        self.config = config
        
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis setup
        self.redis_client = redis.from_url(redis_url)
        
        # Message management
        self.pending_messages: Dict[str, CommunicationMessage] = {}
        self.user_preferences: Dict[str, CommunicationPreferences] = {}
        self.contact_info: Dict[str, ContactInfo] = {}
        
        # Channel handlers
        self.email_config = config.get('email', {})
        self.sms_config = config.get('sms', {})
        self.push_config = config.get('push', {})
        
        # Initialize Twilio client if configured
        self.twilio_client = None
        if self.sms_config.get('twilio_account_sid') and self.sms_config.get('twilio_auth_token'):
            self.twilio_client = TwilioClient(
                self.sms_config['twilio_account_sid'],
                self.sms_config['twilio_auth_token']
            )
        
        # Template management
        self.message_templates: Dict[str, Dict[str, Any]] = {}
        self._load_default_templates()
        
        # Processing
        self.processing_queue = asyncio.Queue()
        self.processing_active = False
        
        logger.info("CommunicationAgent initialized successfully")

    def _load_default_templates(self):
        """Load default message templates"""
        
        self.message_templates = {
            "policy_approved": {
                "subject": "Policy Approved - {{policy_number}}",
                "email_content": """
Dear {{customer_name}},

Great news! Your insurance policy application has been approved.

Policy Details:
- Policy Number: {{policy_number}}
- Coverage Amount: {{coverage_amount}}
- Premium: {{premium_amount}}
- Effective Date: {{effective_date}}

Your policy documents will be sent to you shortly. If you have any questions, please don't hesitate to contact us.

Best regards,
{{company_name}} Team
                """,
                "sms_content": "Good news! Your policy {{policy_number}} has been approved. Coverage starts {{effective_date}}. Documents coming soon."
            },
            "policy_rejected": {
                "subject": "Policy Application Update - {{policy_number}}",
                "email_content": """
Dear {{customer_name}},

Thank you for your interest in our insurance coverage. After careful review, we are unable to approve your application at this time.

Reason: {{rejection_reason}}

If you would like to discuss this decision or explore alternative options, please contact our underwriting team at {{contact_number}}.

Best regards,
{{company_name}} Team
                """,
                "sms_content": "Your policy application {{policy_number}} requires additional review. Please call {{contact_number}} for details."
            },
            "claim_received": {
                "subject": "Claim Received - {{claim_number}}",
                "email_content": """
Dear {{customer_name}},

We have received your insurance claim and wanted to confirm the details:

Claim Details:
- Claim Number: {{claim_number}}
- Date of Loss: {{loss_date}}
- Estimated Amount: {{claim_amount}}
- Status: Under Review

Our claims team will review your submission and contact you within 2-3 business days with next steps.

You can track your claim status online at {{portal_url}} using claim number {{claim_number}}.

Best regards,
{{company_name}} Claims Team
                """,
                "sms_content": "Claim {{claim_number}} received. Under review. Track at {{portal_url}}. We'll contact you in 2-3 days."
            },
            "claim_approved": {
                "subject": "Claim Approved - {{claim_number}}",
                "email_content": """
Dear {{customer_name}},

Great news! Your insurance claim has been approved.

Claim Details:
- Claim Number: {{claim_number}}
- Approved Amount: {{approved_amount}}
- Payment Method: {{payment_method}}
- Expected Payment Date: {{payment_date}}

{{#has_deductible}}
Please note that your deductible of {{deductible_amount}} has been applied.
{{/has_deductible}}

If you have any questions about your claim or payment, please contact us at {{contact_number}}.

Best regards,
{{company_name}} Claims Team
                """,
                "sms_content": "Claim {{claim_number}} approved! ${{approved_amount}} payment coming {{payment_date}}."
            },
            "payment_reminder": {
                "subject": "Payment Reminder - {{policy_number}}",
                "email_content": """
Dear {{customer_name}},

This is a friendly reminder that your insurance premium payment is due soon.

Payment Details:
- Policy Number: {{policy_number}}
- Amount Due: {{amount_due}}
- Due Date: {{due_date}}
- Payment Method: {{payment_method}}

To avoid any interruption in coverage, please make your payment by {{due_date}}.

You can pay online at {{payment_url}} or call us at {{contact_number}}.

Best regards,
{{company_name}} Billing Team
                """,
                "sms_content": "Payment reminder: ${{amount_due}} due {{due_date}} for policy {{policy_number}}. Pay at {{payment_url}}"
            },
            "appointment_reminder": {
                "subject": "Appointment Reminder - {{appointment_date}}",
                "email_content": """
Dear {{customer_name}},

This is a reminder of your upcoming appointment:

Appointment Details:
- Date: {{appointment_date}}
- Time: {{appointment_time}}
- Type: {{appointment_type}}
- Location: {{appointment_location}}
- Agent: {{agent_name}}

If you need to reschedule or have any questions, please call us at {{contact_number}}.

Best regards,
{{company_name}} Team
                """,
                "sms_content": "Reminder: {{appointment_type}} appointment {{appointment_date}} at {{appointment_time}} with {{agent_name}}."
            }
        }
        
        logger.info(f"Loaded {len(self.message_templates)} default message templates")

    async def send_message(self, 
                          recipient_id: str,
                          channel: MessageChannel,
                          subject: str,
                          content: str,
                          priority: MessagePriority = MessagePriority.NORMAL,
                          communication_type: CommunicationType = CommunicationType.NOTIFICATION,
                          scheduled_at: Optional[datetime] = None,
                          template_id: Optional[str] = None,
                          template_data: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Send a message through specified channel"""
        
        message_id = str(uuid.uuid4())
        
        message = CommunicationMessage(
            message_id=message_id,
            recipient_id=recipient_id,
            channel=channel,
            priority=priority,
            communication_type=communication_type,
            subject=subject,
            content=content,
            template_id=template_id,
            template_data=template_data,
            scheduled_at=scheduled_at,
            metadata=metadata or {}
        )
        
        # Check user preferences
        if not await self._check_user_preferences(recipient_id, channel, communication_type):
            message.status = MessageStatus.CANCELLED
            message.error_message = "User preferences do not allow this communication"
            await self._store_message(message)
            return message_id
        
        # Apply template if specified
        if template_id:
            await self._apply_template(message)
        
        # Store message
        await self._store_message(message)
        
        # Queue for processing
        if scheduled_at is None or scheduled_at <= datetime.utcnow():
            await self.processing_queue.put(message_id)
        else:
            # Schedule for later
            await self._schedule_message(message)
        
        logger.info(f"Queued message {message_id} for {channel.value} to {recipient_id}")
        return message_id

    async def send_template_message(self,
                                   recipient_id: str,
                                   template_id: str,
                                   template_data: Dict[str, Any],
                                   channel: Optional[MessageChannel] = None,
                                   priority: MessagePriority = MessagePriority.NORMAL,
                                   communication_type: CommunicationType = CommunicationType.NOTIFICATION) -> str:
        """Send a message using a predefined template"""
        
        if template_id not in self.message_templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.message_templates[template_id]
        
        # Use user's preferred channel if not specified
        if channel is None:
            contact = await self._get_contact_info(recipient_id)
            channel = contact.preferred_channel if contact else MessageChannel.EMAIL
        
        # Get template content for channel
        content_key = f"{channel.value}_content"
        if content_key not in template:
            content_key = "email_content"  # Fallback to email content
        
        return await self.send_message(
            recipient_id=recipient_id,
            channel=channel,
            subject=template["subject"],
            content=template[content_key],
            priority=priority,
            communication_type=communication_type,
            template_id=template_id,
            template_data=template_data
        )

    async def _apply_template(self, message: CommunicationMessage):
        """Apply template data to message content using Jinja2"""
        
        if not message.template_id or not message.template_data:
            return
        
        template = self.message_templates.get(message.template_id)
        if not template:
            return
        
        from jinja2 import Environment, DictLoader, select_autoescape
        
        # Create Jinja2 environment
        env = Environment(
            loader=DictLoader({message.template_id: template}),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        def currency_filter(value):
            try:
                return f"${float(value):,.2f}"
            except:
                return str(value)
        
        def date_filter(value, format='%B %d, %Y'):
            try:
                if isinstance(value, str):
                    from dateutil import parser
                    value = parser.parse(value)
                return value.strftime(format)
            except:
                return str(value)
        
        env.filters['currency'] = currency_filter
        env.filters['date'] = date_filter
        
        # Get content for the specific channel
        channel_key = f"{message.channel.value}_content"
        if channel_key in template:
            content_template = template[channel_key]
        else:
            content_template = template.get("email_content", template.get("content", message.content))
        
        # Render templates
        subject_template = env.from_string(template["subject"])
        content_template = env.from_string(content_template)
        
        message.subject = subject_template.render(**message.template_data)
        message.content = content_template.render(**message.template_data)

    async def _check_user_preferences(self, 
                                    user_id: str, 
                                    channel: MessageChannel, 
                                    communication_type: CommunicationType) -> bool:
        """Check if user preferences allow this communication"""
        
        preferences = await self._get_user_preferences(user_id)
        if not preferences:
            return True  # Allow if no preferences set
        
        # Check if user has opted out of this channel
        if channel == MessageChannel.EMAIL and not preferences.email_enabled:
            return False
        elif channel == MessageChannel.SMS and not preferences.sms_enabled:
            return False
        elif channel == MessageChannel.PUSH and not preferences.push_enabled:
            return False
        elif channel == MessageChannel.PHONE and not preferences.phone_enabled:
            return False
        
        # Check marketing preferences
        if communication_type == CommunicationType.MARKETING and not preferences.marketing_enabled:
            return False
        
        # Check do not disturb
        if preferences.do_not_disturb:
            return False
        
        # Check notification hours (for non-urgent messages)
        current_hour = datetime.utcnow().hour
        if (communication_type != CommunicationType.ALERT and 
            not (preferences.notification_hours_start <= current_hour <= preferences.notification_hours_end)):
            return False
        
        return True

    async def start_processing(self):
        """Start processing messages from the queue"""
        
        if self.processing_active:
            return
        
        self.processing_active = True
        
        while self.processing_active:
            try:
                # Get message from queue with timeout
                message_id = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                message = await self._get_message(message_id)
                if message:
                    await self._process_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, message: CommunicationMessage):
        """Process a single message"""
        
        with message_processing_duration.time():
            try:
                active_communications.inc()
                
                logger.info(f"Processing message {message.message_id} via {message.channel.value}")
                
                # Get contact information
                contact = await self._get_contact_info(message.recipient_id)
                if not contact:
                    raise ValueError(f"No contact information found for user {message.recipient_id}")
                
                # Send via appropriate channel
                if message.channel == MessageChannel.EMAIL:
                    await self._send_email(message, contact)
                elif message.channel == MessageChannel.SMS:
                    await self._send_sms(message, contact)
                elif message.channel == MessageChannel.PUSH:
                    await self._send_push_notification(message, contact)
                elif message.channel == MessageChannel.IN_APP:
                    await self._send_in_app_notification(message, contact)
                else:
                    raise ValueError(f"Unsupported channel: {message.channel}")
                
                message.status = MessageStatus.SENT
                message.sent_at = datetime.utcnow()
                
                messages_sent_total.labels(channel=message.channel.value, status='sent').inc()
                logger.info(f"Message {message.message_id} sent successfully")
                
            except Exception as e:
                message.status = MessageStatus.FAILED
                message.error_message = str(e)
                
                # Retry logic
                if message.retry_count < message.max_retries:
                    message.retry_count += 1
                    message.status = MessageStatus.PENDING
                    
                    # Exponential backoff
                    delay = 2 ** message.retry_count
                    await asyncio.sleep(delay)
                    await self.processing_queue.put(message.message_id)
                    
                    logger.warning(f"Message {message.message_id} failed, retrying ({message.retry_count}/{message.max_retries})")
                else:
                    messages_sent_total.labels(channel=message.channel.value, status='failed').inc()
                    logger.error(f"Message {message.message_id} failed permanently: {e}")
            
            finally:
                active_communications.dec()
                await self._store_message(message)

    async def _send_email(self, message: CommunicationMessage, contact: ContactInfo):
        """Send email message"""
        
        if not contact.email:
            raise ValueError("No email address available")
        
        smtp_config = self.email_config
        if not smtp_config.get('smtp_server'):
            raise ValueError("Email not configured")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_config['from_address']
        msg['To'] = contact.email
        msg['Subject'] = message.subject
        
        # Add body
        msg.attach(MIMEText(message.content, 'plain'))
        
        # Send email
        context = ssl.create_default_context()
        
        with smtplib.SMTP(smtp_config['smtp_server'], smtp_config.get('smtp_port', 587)) as server:
            server.starttls(context=context)
            if smtp_config.get('username'):
                server.login(smtp_config['username'], smtp_config['password'])
            
            server.send_message(msg)
        
        logger.info(f"Email sent to {contact.email}")

    async def _send_sms(self, message: CommunicationMessage, contact: ContactInfo):
        """Send SMS message"""
        
        if not contact.phone:
            raise ValueError("No phone number available")
        
        if not self.twilio_client:
            raise ValueError("SMS not configured")
        
        try:
            # Send SMS via Twilio
            twilio_message = self.twilio_client.messages.create(
                body=message.content,
                from_=self.sms_config['from_number'],
                to=contact.phone
            )
            
            # Store Twilio message SID in metadata
            message.metadata['twilio_sid'] = twilio_message.sid
            
            logger.info(f"SMS sent to {contact.phone}")
            
        except TwilioException as e:
            raise Exception(f"Twilio error: {e}")

    async def _send_push_notification(self, message: CommunicationMessage, contact: ContactInfo):
        """Send push notification"""
        
        if not contact.push_token:
            raise ValueError("No push token available")
        
        push_config = self.push_config
        if not push_config.get('service_url'):
            raise ValueError("Push notifications not configured")
        
        # Prepare push notification payload
        payload = {
            'to': contact.push_token,
            'title': message.subject,
            'body': message.content,
            'data': {
                'message_id': message.message_id,
                'type': message.communication_type.value
            }
        }
        
        # Send via push service (Firebase, APNs, etc.)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                push_config['service_url'],
                json=payload,
                headers={
                    'Authorization': f"key={push_config['api_key']}",
                    'Content-Type': 'application/json'
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Push notification failed: {error_text}")
        
        logger.info(f"Push notification sent to {contact.push_token}")

    async def _send_in_app_notification(self, message: CommunicationMessage, contact: ContactInfo):
        """Send in-app notification"""
        
        # Store in Redis for real-time delivery
        notification_data = {
            'message_id': message.message_id,
            'title': message.subject,
            'content': message.content,
            'type': message.communication_type.value,
            'timestamp': datetime.utcnow().isoformat(),
            'read': False
        }
        
        # Store in user's notification queue
        self.redis_client.lpush(
            f"notifications:{message.recipient_id}",
            json.dumps(notification_data)
        )
        
        # Set expiration (30 days)
        self.redis_client.expire(f"notifications:{message.recipient_id}", 30 * 24 * 3600)
        
        # Publish to real-time channel
        self.redis_client.publish(
            f"user_notifications:{message.recipient_id}",
            json.dumps(notification_data)
        )
        
        logger.info(f"In-app notification sent to user {message.recipient_id}")

    async def _get_user_preferences(self, user_id: str) -> Optional[CommunicationPreferences]:
        """Get user communication preferences"""
        
        if user_id in self.user_preferences:
            return self.user_preferences[user_id]
        
        # Try to load from Redis
        try:
            prefs_data = self.redis_client.get(f"user_preferences:{user_id}")
            if prefs_data:
                data = json.loads(prefs_data)
                preferences = CommunicationPreferences(**data)
                self.user_preferences[user_id] = preferences
                return preferences
        except Exception as e:
            logger.error(f"Failed to load user preferences: {e}")
        
        return None

    async def _get_contact_info(self, user_id: str) -> Optional[ContactInfo]:
        """Get user contact information"""
        
        if user_id in self.contact_info:
            return self.contact_info[user_id]
        
        # Try to load from Redis
        try:
            contact_data = self.redis_client.get(f"contact_info:{user_id}")
            if contact_data:
                data = json.loads(contact_data)
                contact = ContactInfo(**data)
                self.contact_info[user_id] = contact
                return contact
        except Exception as e:
            logger.error(f"Failed to load contact info: {e}")
        
        return None

    async def _store_message(self, message: CommunicationMessage):
        """Store message in database"""
        
        try:
            with self.Session() as session:
                model = CommunicationMessageModel(
                    message_id=message.message_id,
                    recipient_id=message.recipient_id,
                    channel=message.channel.value,
                    priority=message.priority.value,
                    communication_type=message.communication_type.value,
                    subject=message.subject,
                    content=message.content,
                    template_id=message.template_id,
                    template_data=message.template_data,
                    scheduled_at=message.scheduled_at,
                    created_at=message.created_at,
                    sent_at=message.sent_at,
                    delivered_at=message.delivered_at,
                    status=message.status.value,
                    error_message=message.error_message,
                    retry_count=message.retry_count,
                    max_retries=message.max_retries,
                    metadata=message.metadata
                )
                
                session.merge(model)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store message: {e}")

    async def _get_message(self, message_id: str) -> Optional[CommunicationMessage]:
        """Get message by ID"""
        
        try:
            with self.Session() as session:
                model = session.query(CommunicationMessageModel).filter(
                    CommunicationMessageModel.message_id == message_id
                ).first()
                
                if model:
                    return CommunicationMessage(
                        message_id=model.message_id,
                        recipient_id=model.recipient_id,
                        channel=MessageChannel(model.channel),
                        priority=MessagePriority(model.priority),
                        communication_type=CommunicationType(model.communication_type),
                        subject=model.subject,
                        content=model.content,
                        template_id=model.template_id,
                        template_data=model.template_data,
                        scheduled_at=model.scheduled_at,
                        created_at=model.created_at,
                        sent_at=model.sent_at,
                        delivered_at=model.delivered_at,
                        status=MessageStatus(model.status),
                        error_message=model.error_message,
                        retry_count=model.retry_count,
                        max_retries=model.max_retries,
                        metadata=model.metadata
                    )
        except Exception as e:
            logger.error(f"Failed to get message: {e}")
        
        return None

    async def _schedule_message(self, message: CommunicationMessage):
        """Schedule message for future delivery"""
        
        # Store scheduled message in Redis with expiration
        schedule_data = {
            'message_id': message.message_id,
            'scheduled_at': message.scheduled_at.isoformat()
        }
        
        # Calculate delay in seconds
        delay = (message.scheduled_at - datetime.utcnow()).total_seconds()
        
        self.redis_client.setex(
            f"scheduled_message:{message.message_id}",
            int(delay) + 60,  # Add buffer
            json.dumps(schedule_data)
        )
        
        logger.info(f"Scheduled message {message.message_id} for {message.scheduled_at}")

    async def update_user_preferences(self, user_id: str, preferences: CommunicationPreferences):
        """Update user communication preferences"""
        
        self.user_preferences[user_id] = preferences
        
        # Store in Redis
        try:
            prefs_data = asdict(preferences)
            self.redis_client.setex(
                f"user_preferences:{user_id}",
                3600 * 24 * 30,  # 30 days
                json.dumps(prefs_data)
            )
            
            logger.info(f"Updated preferences for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to store user preferences: {e}")

    async def update_contact_info(self, user_id: str, contact: ContactInfo):
        """Update user contact information"""
        
        self.contact_info[user_id] = contact
        
        # Store in Redis
        try:
            contact_data = asdict(contact)
            self.redis_client.setex(
                f"contact_info:{user_id}",
                3600 * 24 * 30,  # 30 days
                json.dumps(contact_data, default=str)
            )
            
            logger.info(f"Updated contact info for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to store contact info: {e}")

    async def get_message_history(self, recipient_id: str, limit: int = 50) -> List[CommunicationMessage]:
        """Get message history for a user"""
        
        try:
            with self.Session() as session:
                models = session.query(CommunicationMessageModel).filter(
                    CommunicationMessageModel.recipient_id == recipient_id
                ).order_by(CommunicationMessageModel.created_at.desc()).limit(limit).all()
                
                messages = []
                for model in models:
                    message = CommunicationMessage(
                        message_id=model.message_id,
                        recipient_id=model.recipient_id,
                        channel=MessageChannel(model.channel),
                        priority=MessagePriority(model.priority),
                        communication_type=CommunicationType(model.communication_type),
                        subject=model.subject,
                        content=model.content,
                        template_id=model.template_id,
                        template_data=model.template_data,
                        scheduled_at=model.scheduled_at,
                        created_at=model.created_at,
                        sent_at=model.sent_at,
                        delivered_at=model.delivered_at,
                        status=MessageStatus(model.status),
                        error_message=model.error_message,
                        retry_count=model.retry_count,
                        max_retries=model.max_retries,
                        metadata=model.metadata
                    )
                    messages.append(message)
                
                return messages
                
        except Exception as e:
            logger.error(f"Failed to get message history: {e}")
            return []

    async def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        
        try:
            with self.Session() as session:
                # Total messages by status
                status_counts = {}
                for status in MessageStatus:
                    count = session.query(CommunicationMessageModel).filter(
                        CommunicationMessageModel.status == status.value
                    ).count()
                    status_counts[status.value] = count
                
                # Messages by channel
                channel_counts = {}
                for channel in MessageChannel:
                    count = session.query(CommunicationMessageModel).filter(
                        CommunicationMessageModel.channel == channel.value
                    ).count()
                    channel_counts[channel.value] = count
                
                # Recent activity (last 24 hours)
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                recent_count = session.query(CommunicationMessageModel).filter(
                    CommunicationMessageModel.created_at >= recent_cutoff
                ).count()
                
                return {
                    "total_messages": sum(status_counts.values()),
                    "status_breakdown": status_counts,
                    "channel_breakdown": channel_counts,
                    "recent_24h": recent_count,
                    "active_communications": len(self.pending_messages),
                    "templates_available": len(self.message_templates)
                }
                
        except Exception as e:
            logger.error(f"Failed to get communication statistics: {e}")
            return {}

    def stop_processing(self):
        """Stop message processing"""
        
        self.processing_active = False
        logger.info("Message processing stopped")

    async def shutdown(self):
        """Graceful shutdown of the communication agent"""
        
        logger.info("Shutting down CommunicationAgent...")
        
        # Stop processing
        self.stop_processing()
        
        # Wait for pending messages to complete
        timeout = 30
        start_time = asyncio.get_event_loop().time()
        
        while self.pending_messages and (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(1)
        
        logger.info("CommunicationAgent shutdown complete")

# Factory function
def create_communication_agent(db_url: str = None, redis_url: str = None, config: Dict[str, Any] = None) -> CommunicationAgent:
    """Create and configure a CommunicationAgent instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    if not config:
        config = {
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'from_address': 'noreply@insurance-ai.com',
                'username': os.getenv('EMAIL_USERNAME'),
                'password': os.getenv('EMAIL_PASSWORD')
            },
            'sms': {
                'twilio_account_sid': os.getenv('TWILIO_ACCOUNT_SID'),
                'twilio_auth_token': os.getenv('TWILIO_AUTH_TOKEN'),
                'from_number': os.getenv('TWILIO_FROM_NUMBER')
            },
            'push': {
                'service_url': 'https://fcm.googleapis.com/fcm/send',
                'api_key': os.getenv('FCM_API_KEY')
            }
        }
    
    return CommunicationAgent(db_url=db_url, redis_url=redis_url, config=config)

# Example usage
if __name__ == "__main__":
    async def test_communication_agent():
        """Test the communication agent functionality"""
        
        agent = create_communication_agent()
        
        # Update contact info
        await agent.update_contact_info("user123", ContactInfo(
            user_id="user123",
            email="customer@example.com",
            phone="+1234567890",
            preferred_channel=MessageChannel.EMAIL
        ))
        
        # Send template message
        message_id = await agent.send_template_message(
            recipient_id="user123",
            template_id="policy_approved",
            template_data={
                "customer_name": "John Doe",
                "policy_number": "POL123456",
                "coverage_amount": "$100,000",
                "premium_amount": "$1,200/year",
                "effective_date": "2024-02-01",
                "company_name": "Insurance AI"
            }
        )
        
        print(f"Sent message: {message_id}")
        
        # Start processing
        await agent.start_processing()
    
    # Run test
    # asyncio.run(test_communication_agent())

