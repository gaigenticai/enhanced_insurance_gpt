"""
Notification Manager - Production Ready Implementation
Manages real-time notifications, delivery tracking, and notification preferences
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import websockets
import redis
from concurrent.futures import ThreadPoolExecutor

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
notifications_delivered_total = Counter('notifications_delivered_total', 'Total notifications delivered', ['type', 'channel'])
notification_delivery_duration = Histogram('notification_delivery_duration_seconds', 'Time to deliver notifications')
active_notification_sessions = Gauge('active_notification_sessions', 'Number of active notification sessions')

class NotificationType(Enum):
    REAL_TIME = "real_time"
    BATCH = "batch"
    SCHEDULED = "scheduled"
    TRIGGERED = "triggered"

class NotificationPriority(Enum):
    LOW = 1
    NORMAL = 3
    HIGH = 7
    CRITICAL = 10

@dataclass
class NotificationRule:
    """Defines rules for when and how to send notifications"""
    rule_id: str
    name: str
    description: str
    trigger_event: str
    conditions: Dict[str, Any]
    notification_template: str
    channels: List[str]
    priority: NotificationPriority
    delay_seconds: int = 0
    batch_window_seconds: int = 0
    max_frequency_per_hour: int = 0
    is_active: bool = True

@dataclass
class NotificationSession:
    """Represents an active notification session for a user"""
    session_id: str
    user_id: str
    connection_type: str  # websocket, sse, polling
    connected_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]

@dataclass
class NotificationDelivery:
    """Tracks delivery of a notification"""
    delivery_id: str
    notification_id: str
    user_id: str
    channel: str
    status: str  # pending, delivered, failed, read
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    error_message: Optional[str] = None

class NotificationManager:
    """
    Production-ready Notification Manager
    Handles real-time notifications, delivery tracking, and user sessions
    """
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = redis.from_url(redis_url)
        
        # Session management
        self.active_sessions: Dict[str, NotificationSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        
        # Notification rules
        self.notification_rules: Dict[str, NotificationRule] = {}
        
        # WebSocket connections
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.processing_active = False
        
        # Load notification rules
        self._load_notification_rules()
        
        logger.info("NotificationManager initialized successfully")

    def _load_notification_rules(self):
        """Load predefined notification rules"""
        
        rules = [
            NotificationRule(
                rule_id="policy_status_change",
                name="Policy Status Change",
                description="Notify when policy status changes",
                trigger_event="policy.status.changed",
                conditions={"status": {"in": ["approved", "rejected", "pending_review"]}},
                notification_template="policy_status_notification",
                channels=["email", "in_app", "push"],
                priority=NotificationPriority.HIGH
            ),
            NotificationRule(
                rule_id="claim_update",
                name="Claim Update",
                description="Notify when claim status updates",
                trigger_event="claim.status.changed",
                conditions={},
                notification_template="claim_update_notification",
                channels=["email", "sms", "in_app"],
                priority=NotificationPriority.HIGH
            ),
            NotificationRule(
                rule_id="payment_due",
                name="Payment Due Reminder",
                description="Remind about upcoming payment",
                trigger_event="payment.due.reminder",
                conditions={"days_until_due": {"lte": 7}},
                notification_template="payment_reminder",
                channels=["email", "sms"],
                priority=NotificationPriority.NORMAL,
                delay_seconds=0
            ),
            NotificationRule(
                rule_id="document_required",
                name="Document Required",
                description="Notify when additional documents are needed",
                trigger_event="document.required",
                conditions={},
                notification_template="document_request",
                channels=["email", "in_app"],
                priority=NotificationPriority.NORMAL
            ),
            NotificationRule(
                rule_id="fraud_alert",
                name="Fraud Alert",
                description="Critical fraud detection alert",
                trigger_event="fraud.detected",
                conditions={"confidence": {"gte": 0.8}},
                notification_template="fraud_alert",
                channels=["email", "sms", "in_app", "push"],
                priority=NotificationPriority.CRITICAL,
                max_frequency_per_hour=1
            ),
            NotificationRule(
                rule_id="system_maintenance",
                name="System Maintenance",
                description="Notify about scheduled maintenance",
                trigger_event="system.maintenance.scheduled",
                conditions={},
                notification_template="maintenance_notification",
                channels=["email", "in_app"],
                priority=NotificationPriority.NORMAL,
                batch_window_seconds=3600  # Batch for 1 hour
            )
        ]
        
        for rule in rules:
            self.notification_rules[rule.rule_id] = rule
        
        logger.info(f"Loaded {len(rules)} notification rules")

    async def register_session(self, 
                             user_id: str, 
                             connection_type: str,
                             metadata: Dict[str, Any] = None) -> str:
        """Register a new notification session"""
        
        session_id = str(uuid.uuid4())
        session = NotificationSession(
            session_id=session_id,
            user_id=user_id,
            connection_type=connection_type,
            connected_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.active_sessions[session_id] = session
        
        # Track user sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)
        
        # Store in Redis for persistence
        await self._store_session(session)
        
        active_notification_sessions.inc()
        logger.info(f"Registered notification session {session_id} for user {user_id}")
        
        return session_id

    async def unregister_session(self, session_id: str):
        """Unregister a notification session"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        user_id = session.user_id
        
        # Remove from tracking
        del self.active_sessions[session_id]
        
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
        
        # Remove WebSocket connection if exists
        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]
        
        # Remove from Redis
        self.redis_client.delete(f"notification_session:{session_id}")
        
        active_notification_sessions.dec()
        logger.info(f"Unregistered notification session {session_id}")

    async def register_websocket(self, session_id: str, websocket: websockets.WebSocketServerProtocol):
        """Register a WebSocket connection for real-time notifications"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        self.websocket_connections[session_id] = websocket
        
        # Update session activity
        session = self.active_sessions[session_id]
        session.last_activity = datetime.utcnow()
        await self._store_session(session)
        
        logger.info(f"Registered WebSocket for session {session_id}")

    async def send_notification(self,
                               user_id: str,
                               title: str,
                               content: str,
                               notification_type: NotificationType = NotificationType.REAL_TIME,
                               priority: NotificationPriority = NotificationPriority.NORMAL,
                               channels: List[str] = None,
                               metadata: Dict[str, Any] = None) -> str:
        """Send a notification to a user"""
        
        notification_id = str(uuid.uuid4())
        
        notification_data = {
            'notification_id': notification_id,
            'user_id': user_id,
            'title': title,
            'content': content,
            'type': notification_type.value,
            'priority': priority.value,
            'channels': channels or ['in_app'],
            'metadata': metadata or {},
            'created_at': datetime.utcnow().isoformat(),
            'status': 'pending'
        }
        
        # Store notification
        await self._store_notification(notification_data)
        
        # Send via specified channels
        for channel in notification_data['channels']:
            await self._send_via_channel(notification_data, channel)
        
        logger.info(f"Sent notification {notification_id} to user {user_id}")
        return notification_id

    async def trigger_notification(self, 
                                  event: str, 
                                  data: Dict[str, Any],
                                  user_id: Optional[str] = None):
        """Trigger notifications based on event and rules"""
        
        triggered_notifications = []
        
        for rule in self.notification_rules.values():
            if not rule.is_active or rule.trigger_event != event:
                continue
            
            # Check conditions
            if not self._check_rule_conditions(rule, data):
                continue
            
            # Check frequency limits
            if rule.max_frequency_per_hour > 0:
                if not await self._check_frequency_limit(rule, user_id or data.get('user_id')):
                    continue
            
            # Determine target users
            target_users = [user_id] if user_id else self._get_target_users(rule, data)
            
            for target_user in target_users:
                notification_id = await self._create_rule_notification(rule, data, target_user)
                triggered_notifications.append(notification_id)
        
        logger.info(f"Triggered {len(triggered_notifications)} notifications for event {event}")
        return triggered_notifications

    def _check_rule_conditions(self, rule: NotificationRule, data: Dict[str, Any]) -> bool:
        """Check if rule conditions are met"""
        
        if not rule.conditions:
            return True
        
        for condition_key, condition_value in rule.conditions.items():
            if condition_key not in data:
                return False
            
            data_value = data[condition_key]
            
            if isinstance(condition_value, dict):
                if "in" in condition_value and data_value not in condition_value["in"]:
                    return False
                if "not_in" in condition_value and data_value in condition_value["not_in"]:
                    return False
                if "gte" in condition_value and data_value < condition_value["gte"]:
                    return False
                if "lte" in condition_value and data_value > condition_value["lte"]:
                    return False
                if "gt" in condition_value and data_value <= condition_value["gt"]:
                    return False
                if "lt" in condition_value and data_value >= condition_value["lt"]:
                    return False
            else:
                if data_value != condition_value:
                    return False
        
        return True

    async def _check_frequency_limit(self, rule: NotificationRule, user_id: str) -> bool:
        """Check if frequency limit allows sending notification"""
        
        if not user_id or rule.max_frequency_per_hour <= 0:
            return True
        
        # Check Redis for recent notifications
        key = f"notification_frequency:{rule.rule_id}:{user_id}"
        count = self.redis_client.get(key)
        
        if count and int(count) >= rule.max_frequency_per_hour:
            return False
        
        return True

    def _get_target_users(self, rule: NotificationRule, data: Dict[str, Any]) -> List[str]:
        """Get target users for a notification rule"""
        
        # Production implementation with database query and rule-based targeting
        target_users = []
        
        try:
            # Direct user targeting
            if 'user_id' in data:
                target_users.append(data['user_id'])
            elif 'user_ids' in data:
                target_users.extend(data['user_ids'])
            
            # Role-based targeting
            if rule.target_roles:
                from backend.shared.database import get_db_session
                from backend.shared.models import User
                
                session = get_db_session()
                try:
                    role_users = session.query(User).filter(
                        User.roles.any(role_name=rule.target_roles),
                        User.is_active == True,
                        User.notification_preferences.contains({'enabled': True})
                    ).all()
                    
                    target_users.extend([user.id for user in role_users])
                finally:
                    session.close()
            
            # Department-based targeting
            if rule.target_departments:
                from backend.shared.database import get_db_session
                from backend.shared.models import User
                
                session = get_db_session()
                try:
                    dept_users = session.query(User).filter(
                        User.department.in_(rule.target_departments),
                        User.is_active == True,
                        User.notification_preferences.contains({'enabled': True})
                    ).all()
                    
                    target_users.extend([user.id for user in dept_users])
                finally:
                    session.close()
            
            # Remove duplicates while preserving order
            seen = set()
            unique_users = []
            for user_id in target_users:
                if user_id not in seen:
                    seen.add(user_id)
                    unique_users.append(user_id)
            
            return unique_users
            
        except Exception as e:
            logger.error(f"Error getting target users: {e}")
            # Fallback to basic targeting
            if 'user_id' in data:
                return [data['user_id']]
            elif 'user_ids' in data:
                return data['user_ids']
            else:
                return []

    async def _create_rule_notification(self, 
                                       rule: NotificationRule, 
                                       data: Dict[str, Any], 
                                       user_id: str) -> str:
        """Create notification based on rule"""
        
        # Generate notification content from template
        title, content = await self._generate_notification_content(rule.notification_template, data)
        
        # Apply delay if specified
        if rule.delay_seconds > 0:
            await asyncio.sleep(rule.delay_seconds)
        
        # Send notification
        notification_id = await self.send_notification(
            user_id=user_id,
            title=title,
            content=content,
            notification_type=NotificationType.TRIGGERED,
            priority=rule.priority,
            channels=rule.channels,
            metadata={'rule_id': rule.rule_id, 'trigger_data': data}
        )
        
        # Update frequency counter
        if rule.max_frequency_per_hour > 0:
            key = f"notification_frequency:{rule.rule_id}:{user_id}"
            self.redis_client.incr(key)
            self.redis_client.expire(key, 3600)  # 1 hour
        
        return notification_id

    async def _generate_notification_content(self, 
                                           template_id: str, 
                                           data: Dict[str, Any]) -> tuple[str, str]:
        """Generate notification content from template using Jinja2"""
        
        from jinja2 import Environment, DictLoader, select_autoescape
        
        templates = {
            "policy_status_notification": {
                "title": "Policy Status Update - {{ policy_number }}",
                "content": """Dear {{ customer_name }},

Your insurance policy {{ policy_number }} status has been updated.

New Status: {{ status|title }}
{% if status == 'approved' %}
Congratulations! Your policy is now active.
Coverage Amount: ${{ coverage_amount|default('N/A') }}
Effective Date: {{ effective_date|default('N/A') }}
{% elif status == 'rejected' %}
Unfortunately, we cannot approve your policy at this time.
Reason: {{ rejection_reason|default('Please contact us for details') }}
{% elif status == 'pending_review' %}
Your policy is currently under review by our underwriting team.
Expected completion: {{ review_completion_date|default('2-3 business days') }}
{% endif %}

For questions, contact us at {{ support_phone|default('1-800-INSURANCE') }}.

Best regards,
{{ company_name|default('Insurance AI') }} Team"""
            },
            "claim_update_notification": {
                "title": "Claim Update - {{ claim_number }}",
                "content": """Dear {{ customer_name }},

Your insurance claim {{ claim_number }} has been updated.

Claim Details:
- Claim Number: {{ claim_number }}
- Status: {{ status|title }}
- Date of Loss: {{ loss_date }}
{% if status == 'approved' %}
- Approved Amount: ${{ approved_amount }}
- Payment Method: {{ payment_method|default('Direct Deposit') }}
- Expected Payment: {{ payment_date }}
{% elif status == 'under_review' %}
- Estimated Review Time: {{ review_time|default('5-7 business days') }}
- Additional Documents Needed: {{ required_documents|default('None') }}
{% elif status == 'rejected' %}
- Rejection Reason: {{ rejection_reason }}
- Appeal Deadline: {{ appeal_deadline }}
{% endif %}

Track your claim at {{ portal_url|default('https://portal.insurance-ai.com') }}

Best regards,
{{ company_name|default('Insurance AI') }} Claims Team"""
            },
            "payment_reminder": {
                "title": "Payment Reminder - {{ policy_number }}",
                "content": """Dear {{ customer_name }},

Your insurance premium payment is due soon.

Payment Details:
- Policy Number: {{ policy_number }}
- Amount Due: ${{ amount }}
- Due Date: {{ due_date }}
- Payment Method: {{ payment_method|default('Auto-Pay') }}

{% if days_until_due <= 3 %}
⚠️ URGENT: Payment is due in {{ days_until_due }} day(s).
{% elif days_until_due <= 7 %}
Reminder: Payment is due in {{ days_until_due }} days.
{% endif %}

Pay online: {{ payment_url|default('https://pay.insurance-ai.com') }}
Phone: {{ payment_phone|default('1-800-PAY-NOW') }}

Avoid coverage interruption - pay by {{ due_date }}.

Best regards,
{{ company_name|default('Insurance AI') }} Billing Team"""
            },
            "document_request": {
                "title": "Documents Required - {{ request_type|default('Policy Processing') }}",
                "content": """Dear {{ customer_name }},

We need additional documents to process your {{ request_type|default('request') }}.

Required Documents:
{% for doc in document_list %}
- {{ doc.name }}{% if doc.description %}: {{ doc.description }}{% endif %}
{% endfor %}

Submission Options:
1. Upload online: {{ upload_url|default('https://upload.insurance-ai.com') }}
2. Email: {{ document_email|default('documents@insurance-ai.com') }}
3. Fax: {{ fax_number|default('1-800-FAX-DOCS') }}

Deadline: {{ submission_deadline }}
Reference: {{ reference_number|default(policy_number) }}

Questions? Call {{ support_phone|default('1-800-SUPPORT') }}

Best regards,
{{ company_name|default('Insurance AI') }} Team"""
            },
            "fraud_alert": {
                "title": "🚨 Security Alert - Immediate Action Required",
                "content": """SECURITY ALERT

Dear {{ customer_name }},

We detected suspicious activity on your account:

Alert Details:
- Detection Time: {{ detection_time }}
- Activity Type: {{ activity_type }}
- Risk Level: {{ risk_level|upper }}
- Location: {{ suspicious_location|default('Unknown') }}

IMMEDIATE ACTIONS REQUIRED:
1. Change your password immediately
2. Review recent account activity
3. Contact us to verify legitimate transactions

Secure Login: {{ secure_login_url|default('https://secure.insurance-ai.com') }}
24/7 Fraud Hotline: {{ fraud_hotline|default('1-800-FRAUD-HELP') }}

If you recognize this activity, please confirm at {{ confirmation_url|default('https://secure.insurance-ai.com/confirm') }}

Your account security is our priority.

{{ company_name|default('Insurance AI') }} Security Team"""
            },
            "maintenance_notification": {
                "title": "Scheduled Maintenance - {{ maintenance_date }}",
                "content": """Dear Valued Customer,

We will be performing scheduled system maintenance to improve your experience.

Maintenance Schedule:
- Date: {{ maintenance_date }}
- Start Time: {{ start_time }}
- End Time: {{ end_time }}
- Duration: {{ duration|default('Approximately 2 hours') }}

Affected Services:
{% for service in affected_services %}
- {{ service.name }}{% if service.impact %} ({{ service.impact }}){% endif %}
{% endfor %}

During maintenance:
- Online portal may be unavailable
- Mobile app access limited
- Phone wait times may be longer

Emergency Contact: {{ emergency_phone|default('1-800-EMERGENCY') }}

We apologize for any inconvenience.

Best regards,
{{ company_name|default('Insurance AI') }} IT Team"""
            }
        }
        
        # Create Jinja2 environment with security settings
        env = Environment(
            loader=DictLoader(templates),
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
        
        # Get template
        if template_id not in templates:
            template_id = "policy_status_notification"  # Default fallback
        
        template_data = templates[template_id]
        title_template = env.from_string(template_data["title"])
        content_template = env.from_string(template_data["content"])
        
        # Render with data
        title = title_template.render(**data)
        content = content_template.render(**data)
        
        return title, content

    async def _send_via_channel(self, notification_data: Dict[str, Any], channel: str):
        """Send notification via specific channel"""
        
        with notification_delivery_duration.time():
            try:
                if channel == "in_app":
                    await self._send_in_app_notification(notification_data)
                elif channel == "websocket":
                    await self._send_websocket_notification(notification_data)
                elif channel == "email":
                    await self._send_email_notification(notification_data)
                elif channel == "sms":
                    await self._send_sms_notification(notification_data)
                elif channel == "push":
                    await self._send_push_notification(notification_data)
                
                # Track delivery
                await self._track_delivery(notification_data, channel, "delivered")
                
                notifications_delivered_total.labels(
                    type=notification_data['type'], 
                    channel=channel
                ).inc()
                
            except Exception as e:
                logger.error(f"Failed to send notification via {channel}: {e}")
                await self._track_delivery(notification_data, channel, "failed", str(e))

    async def _send_in_app_notification(self, notification_data: Dict[str, Any]):
        """Send in-app notification"""
        
        user_id = notification_data['user_id']
        
        # Store in user's notification queue
        self.redis_client.lpush(
            f"notifications:{user_id}",
            json.dumps(notification_data)
        )
        
        # Set expiration (30 days)
        self.redis_client.expire(f"notifications:{user_id}", 30 * 24 * 3600)
        
        # Publish to real-time channel
        self.redis_client.publish(
            f"user_notifications:{user_id}",
            json.dumps(notification_data)
        )

    async def _send_websocket_notification(self, notification_data: Dict[str, Any]):
        """Send notification via WebSocket"""
        
        user_id = notification_data['user_id']
        
        if user_id not in self.user_sessions:
            return
        
        # Send to all active WebSocket sessions for the user
        for session_id in self.user_sessions[user_id]:
            if session_id in self.websocket_connections:
                websocket = self.websocket_connections[session_id]
                
                try:
                    await websocket.send(json.dumps(notification_data))
                except websockets.exceptions.ConnectionClosed:
                    # Remove closed connection
                    await self.unregister_session(session_id)
                except Exception as e:
                    logger.error(f"Failed to send WebSocket notification: {e}")

    async def _send_email_notification(self, notification_data: Dict[str, Any]):
        """Send email notification via SMTP"""
        
        import smtplib
        import ssl
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        import os
        
        # Get user contact info
        user_id = notification_data['user_id']
        contact_data = self.redis_client.get(f"contact_info:{user_id}")
        
        if not contact_data:
            raise ValueError(f"No contact information found for user {user_id}")
        
        contact_info = json.loads(contact_data)
        email = contact_info.get('email')
        
        if not email:
            raise ValueError(f"No email address found for user {user_id}")
        
        # Email configuration
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_username = os.getenv('SMTP_USERNAME')
        smtp_password = os.getenv('SMTP_PASSWORD')
        from_address = os.getenv('FROM_EMAIL', 'noreply@insurance-ai.com')
        
        if not smtp_username or not smtp_password:
            raise ValueError("SMTP credentials not configured")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = from_address
        msg['To'] = email
        msg['Subject'] = notification_data['title']
        
        # Create HTML and plain text versions
        text_content = notification_data['content']
        content_html = notification_data['content'].replace('\n', '<br>')
        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
              <h2 style="color: #2c5aa0;">{notification_data['title']}</h2>
              <div style="background: #f9f9f9; padding: 20px; border-radius: 5px;">
                {content_html}
              </div>
              <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
              <p style="font-size: 12px; color: #666;">
                This is an automated message from Insurance AI System.
                <br>Please do not reply to this email.
              </p>
            </div>
          </body>
        </html>
        """
        
        # Attach parts
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        context = ssl.create_default_context()
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Email notification sent to {email}")

    async def _send_sms_notification(self, notification_data: Dict[str, Any]):
        """Send SMS notification via Twilio"""
        
        from twilio.rest import Client as TwilioClient
        from twilio.base.exceptions import TwilioException
        import os
        
        # Get user contact info
        user_id = notification_data['user_id']
        contact_data = self.redis_client.get(f"contact_info:{user_id}")
        
        if not contact_data:
            raise ValueError(f"No contact information found for user {user_id}")
        
        contact_info = json.loads(contact_data)
        phone = contact_info.get('phone')
        
        if not phone:
            raise ValueError(f"No phone number found for user {user_id}")
        
        # Twilio configuration
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        from_number = os.getenv('TWILIO_FROM_NUMBER')
        
        if not account_sid or not auth_token or not from_number:
            raise ValueError("Twilio credentials not configured")
        
        client = TwilioClient(account_sid, auth_token)
        
        # Create SMS content (limit to 160 characters for standard SMS)
        sms_content = notification_data['content']
        if len(sms_content) > 160:
            sms_content = sms_content[:157] + "..."
        
        try:
            message = client.messages.create(
                body=sms_content,
                from_=from_number,
                to=phone
            )
            
            # Store message SID for tracking
            notification_data['metadata']['twilio_sid'] = message.sid
            
            logger.info(f"SMS notification sent to {phone}, SID: {message.sid}")
            
        except TwilioException as e:
            raise Exception(f"Twilio SMS error: {str(e)}")

    async def _send_push_notification(self, notification_data: Dict[str, Any]):
        """Send push notification via Firebase Cloud Messaging"""
        
        import aiohttp
        import os
        
        # Get user contact info
        user_id = notification_data['user_id']
        contact_data = self.redis_client.get(f"contact_info:{user_id}")
        
        if not contact_data:
            raise ValueError(f"No contact information found for user {user_id}")
        
        contact_info = json.loads(contact_data)
        push_token = contact_info.get('push_token')
        
        if not push_token:
            raise ValueError(f"No push token found for user {user_id}")
        
        # FCM configuration
        fcm_server_key = os.getenv('FCM_SERVER_KEY')
        fcm_url = "https://fcm.googleapis.com/fcm/send"
        
        if not fcm_server_key:
            raise ValueError("FCM server key not configured")
        
        # Create push notification payload
        payload = {
            "to": push_token,
            "notification": {
                "title": notification_data['title'],
                "body": notification_data['content'][:100] + "..." if len(notification_data['content']) > 100 else notification_data['content'],
                "icon": "ic_notification",
                "sound": "default",
                "click_action": "FLUTTER_NOTIFICATION_CLICK"
            },
            "data": {
                "notification_id": notification_data['notification_id'],
                "type": notification_data['type'],
                "user_id": user_id,
                "timestamp": notification_data['created_at']
            },
            "priority": "high" if notification_data['priority'] >= 7 else "normal"
        }
        
        headers = {
            "Authorization": f"key={fcm_server_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(fcm_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('success') == 1:
                        logger.info(f"Push notification sent successfully to {push_token}")
                    else:
                        error = result.get('results', [{}])[0].get('error', 'Unknown error')
                        raise Exception(f"FCM error: {error}")
                else:
                    error_text = await response.text()
                    raise Exception(f"FCM HTTP error {response.status}: {error_text}")

    async def _track_delivery(self, 
                            notification_data: Dict[str, Any], 
                            channel: str, 
                            status: str,
                            error_message: str = None):
        """Track notification delivery"""
        
        delivery = NotificationDelivery(
            delivery_id=str(uuid.uuid4()),
            notification_id=notification_data['notification_id'],
            user_id=notification_data['user_id'],
            channel=channel,
            status=status,
            delivered_at=datetime.utcnow() if status == "delivered" else None,
            error_message=error_message
        )
        
        # Store delivery record
        delivery_data = asdict(delivery)
        # Convert datetime to ISO string
        if delivery_data['delivered_at']:
            delivery_data['delivered_at'] = delivery_data['delivered_at'].isoformat()
        
        self.redis_client.setex(
            f"notification_delivery:{delivery.delivery_id}",
            3600 * 24 * 7,  # 7 days
            json.dumps(delivery_data)
        )

    async def _store_notification(self, notification_data: Dict[str, Any]):
        """Store notification data"""
        
        self.redis_client.setex(
            f"notification:{notification_data['notification_id']}",
            3600 * 24 * 30,  # 30 days
            json.dumps(notification_data)
        )

    async def _store_session(self, session: NotificationSession):
        """Store session data"""
        
        session_data = asdict(session)
        # Convert datetime to ISO string
        session_data['connected_at'] = session_data['connected_at'].isoformat()
        session_data['last_activity'] = session_data['last_activity'].isoformat()
        
        self.redis_client.setex(
            f"notification_session:{session.session_id}",
            3600 * 24,  # 24 hours
            json.dumps(session_data)
        )

    async def get_user_notifications(self, 
                                   user_id: str, 
                                   limit: int = 50, 
                                   unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get notifications for a user"""
        
        notifications = []
        
        # Get from Redis
        notification_data = self.redis_client.lrange(f"notifications:{user_id}", 0, limit - 1)
        
        for data in notification_data:
            notification = json.loads(data)
            
            if unread_only and notification.get('read', False):
                continue
            
            notifications.append(notification)
        
        return notifications

    async def mark_notification_read(self, user_id: str, notification_id: str):
        """Mark a notification as read"""
        
        # Update in Redis
        notifications = self.redis_client.lrange(f"notifications:{user_id}", 0, -1)
        
        for i, data in enumerate(notifications):
            notification = json.loads(data)
            if notification['notification_id'] == notification_id:
                notification['read'] = True
                notification['read_at'] = datetime.utcnow().isoformat()
                
                # Update in Redis
                self.redis_client.lset(f"notifications:{user_id}", i, json.dumps(notification))
                break
        
        logger.info(f"Marked notification {notification_id} as read for user {user_id}")

    async def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        
        return {
            "active_sessions": len(self.active_sessions),
            "websocket_connections": len(self.websocket_connections),
            "notification_rules": len(self.notification_rules),
            "users_with_sessions": len(self.user_sessions)
        }

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            # Consider session expired if no activity for 1 hour
            if (current_time - session.last_activity).total_seconds() > 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.unregister_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def start_processing(self):
        """Start background processing"""
        
        if self.processing_active:
            return
        
        self.processing_active = True
        
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())
        
        logger.info("NotificationManager processing started")

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired sessions and notifications"""
        
        while self.processing_active:
            try:
                await self.cleanup_expired_sessions()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)

    def stop_processing(self):
        """Stop background processing"""
        
        self.processing_active = False
        logger.info("NotificationManager processing stopped")

    async def shutdown(self):
        """Graceful shutdown"""
        
        logger.info("Shutting down NotificationManager...")
        
        self.stop_processing()
        
        # Close all WebSocket connections
        for websocket in self.websocket_connections.values():
            try:
                await websocket.close()
            except:
                pass
        
        # Clear sessions
        self.active_sessions.clear()
        self.user_sessions.clear()
        self.websocket_connections.clear()
        
        logger.info("NotificationManager shutdown complete")

# Factory function
def create_notification_manager(redis_url: str = None) -> NotificationManager:
    """Create and configure a NotificationManager instance"""
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return NotificationManager(redis_url=redis_url)

# Example usage
if __name__ == "__main__":
    async def test_notification_manager():
        """Test the notification manager functionality"""
        
        manager = create_notification_manager()
        
        # Register session
        session_id = await manager.register_session("user123", "websocket")
        
        # Send notification
        notification_id = await manager.send_notification(
            user_id="user123",
            title="Test Notification",
            content="This is a test notification",
            channels=["in_app", "websocket"]
        )
        
        # Trigger event-based notification
        await manager.trigger_notification(
            event="policy.status.changed",
            data={
                "user_id": "user123",
                "policy_number": "POL123456",
                "status": "approved"
            }
        )
        
        print(f"Sent notifications for session {session_id}")
        
        # Start processing
        manager.start_processing()
    
    # Run test
    # asyncio.run(test_notification_manager())

