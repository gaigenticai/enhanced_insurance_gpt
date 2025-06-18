"""
Insurance AI Agent System - Monitoring and Metrics
Production-ready monitoring system with Prometheus metrics
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import gc
import json

# Prometheus metrics
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server, push_to_gateway
)

# Database and Redis
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text

# Internal imports
from backend.shared.models import SystemMetric, PerformanceLog, AlertRule
from backend.shared.schemas import MetricData, AlertConfig, SystemHealth
from backend.shared.services import BaseService, ServiceException
from backend.shared.database import get_db_session, get_redis_client, get_database_manager
from backend.shared.utils import DataUtils, ValidationUtils

import structlog
logger = structlog.get_logger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = None
    buckets: List[float] = None  # For histograms
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = []

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 100"
    threshold: float
    severity: AlertSeverity
    duration: int = 300  # seconds
    description: str = ""
    enabled: bool = True

class MetricsCollector:
    """
    Comprehensive metrics collection system
    """
    
    def __init__(self, redis_client: redis.Redis, registry: Optional[CollectorRegistry] = None):
        self.redis_client = redis_client
        self.registry = registry or CollectorRegistry()
        self.logger = structlog.get_logger("metrics_collector")
        
        # Metric storage
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_states: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_data = defaultdict(lambda: deque(maxlen=1000))
        
        # Background tasks
        self.collection_task = None
        self.alert_task = None
        
        # Initialize core metrics
        self._initialize_core_metrics()
    
    async def start(self):
        """Start the metrics collector"""
        try:
            self.logger.info("Starting metrics collector")
            
            # Start background tasks
            self.collection_task = asyncio.create_task(self._collect_system_metrics())
            self.alert_task = asyncio.create_task(self._check_alerts())
            
            # Start Prometheus HTTP server
            start_http_server(8000, registry=self.registry)
            
            self.logger.info("Metrics collector started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start metrics collector", error=str(e))
            raise
    
    async def stop(self):
        """Stop the metrics collector"""
        try:
            self.logger.info("Stopping metrics collector")
            
            # Cancel background tasks
            if self.collection_task:
                self.collection_task.cancel()
            if self.alert_task:
                self.alert_task.cancel()
            
            self.logger.info("Metrics collector stopped")
            
        except Exception as e:
            self.logger.error("Error stopping metrics collector", error=str(e))
    
    def register_metric(self, definition: MetricDefinition):
        """Register a new metric"""
        try:
            if definition.name in self.metrics:
                self.logger.warning("Metric already registered", metric=definition.name)
                return
            
            # Create Prometheus metric
            if definition.metric_type == MetricType.COUNTER:
                metric = Counter(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.HISTOGRAM:
                buckets = definition.buckets or [0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                metric = Histogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=buckets,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.SUMMARY:
                metric = Summary(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unknown metric type: {definition.metric_type}")
            
            self.metrics[definition.name] = metric
            self.metric_definitions[definition.name] = definition
            
            self.logger.info("Metric registered", metric=definition.name, type=definition.metric_type.value)
            
        except Exception as e:
            self.logger.error("Failed to register metric", metric=definition.name, error=str(e))
            raise
    
    def increment_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        try:
            if name not in self.metrics:
                self.logger.warning("Unknown metric", metric=name)
                return
            
            metric = self.metrics[name]
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
                
        except Exception as e:
            self.logger.error("Failed to increment counter", metric=name, error=str(e))
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        try:
            if name not in self.metrics:
                self.logger.warning("Unknown metric", metric=name)
                return
            
            metric = self.metrics[name]
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
                
        except Exception as e:
            self.logger.error("Failed to set gauge", metric=name, error=str(e))
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a histogram metric"""
        try:
            if name not in self.metrics:
                self.logger.warning("Unknown metric", metric=name)
                return
            
            metric = self.metrics[name]
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
                
        except Exception as e:
            self.logger.error("Failed to observe histogram", metric=name, error=str(e))
    
    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a summary metric"""
        try:
            if name not in self.metrics:
                self.logger.warning("Unknown metric", metric=name)
                return
            
            metric = self.metrics[name]
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
                
        except Exception as e:
            self.logger.error("Failed to observe summary", metric=name, error=str(e))
    
    def time_function(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator to time function execution"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        duration = time.time() - start_time
                        self.observe_histogram(metric_name, duration, labels)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        duration = time.time() - start_time
                        self.observe_histogram(metric_name, duration, labels)
                return sync_wrapper
        return decorator
    
    def register_alert_rule(self, rule: AlertRule):
        """Register an alert rule"""
        self.alert_rules[rule.name] = rule
        self.alert_states[rule.name] = {
            "triggered": False,
            "triggered_at": None,
            "last_check": None,
            "consecutive_violations": 0
        }
        
        self.logger.info("Alert rule registered", rule=rule.name, metric=rule.metric_name)
    
    async def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current metric value"""
        try:
            if name not in self.metrics:
                return None
            
            metric = self.metrics[name]
            sample = None
            if labels:
                metric_obj = metric.labels(**labels)
            else:
                metric_obj = metric

            if hasattr(metric_obj, "_value"):
                sample = metric_obj._value.get()
            elif hasattr(metric_obj, "_sum"):
                sample = metric_obj._sum.get()

            return float(sample) if sample is not None else None
            
        except Exception as e:
            self.logger.error("Failed to get metric value", metric=name, error=str(e))
            return None
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Database connection status
            db_healthy = True
            try:
                # Test database connection
                async with get_db_session() as session:
                    await session.execute(text("SELECT 1"))
            except Exception:
                db_healthy = False
            
            # Redis connection status
            redis_healthy = True
            try:
                await self.redis_client.ping()
            except Exception:
                redis_healthy = False
            
            # Active alerts
            active_alerts = [
                rule_name for rule_name, state in self.alert_states.items()
                if state["triggered"]
            ]
            
            # Overall health status
            health_status = "healthy"
            if not db_healthy or not redis_healthy:
                health_status = "unhealthy"
            elif cpu_percent > 80 or memory.percent > 80 or disk.percent > 80:
                health_status = "degraded"
            elif active_alerts:
                health_status = "warning"
            
            return {
                "status": health_status,
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "uptime": time.time() - psutil.boot_time()
                },
                "services": {
                    "database": "healthy" if db_healthy else "unhealthy",
                    "redis": "healthy" if redis_healthy else "unhealthy"
                },
                "alerts": {
                    "active_count": len(active_alerts),
                    "active_alerts": active_alerts
                }
            }
            
        except Exception as e:
            self.logger.error("Failed to get system health", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            self.logger.error("Failed to export metrics", error=str(e))
            return ""
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics"""
        
        # HTTP request metrics
        self.register_metric(MetricDefinition(
            name="http_requests_total",
            metric_type=MetricType.COUNTER,
            description="Total number of HTTP requests",
            labels=["method", "endpoint", "status_code"]
        ))
        
        self.register_metric(MetricDefinition(
            name="http_request_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="HTTP request duration in seconds",
            labels=["method", "endpoint"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ))
        
        # Database metrics
        self.register_metric(MetricDefinition(
            name="database_connections_active",
            metric_type=MetricType.GAUGE,
            description="Number of active database connections"
        ))
        
        self.register_metric(MetricDefinition(
            name="database_query_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Database query duration in seconds",
            labels=["operation"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        ))
        
        # Agent metrics
        self.register_metric(MetricDefinition(
            name="agent_tasks_total",
            metric_type=MetricType.COUNTER,
            description="Total number of agent tasks",
            labels=["agent_type", "status"]
        ))
        
        self.register_metric(MetricDefinition(
            name="agent_task_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Agent task duration in seconds",
            labels=["agent_type"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
        ))
        
        # Queue metrics
        self.register_metric(MetricDefinition(
            name="queue_size",
            metric_type=MetricType.GAUGE,
            description="Number of messages in queue",
            labels=["queue_type"]
        ))
        
        self.register_metric(MetricDefinition(
            name="queue_processing_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Queue message processing duration",
            labels=["queue_type"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
        ))

        self.register_metric(MetricDefinition(
            name="queue_messages_processed_total",
            metric_type=MetricType.COUNTER,
            description="Total queue messages processed",
            labels=["queue_type", "status"]
        ))
        
        # System metrics
        self.register_metric(MetricDefinition(
            name="system_cpu_percent",
            metric_type=MetricType.GAUGE,
            description="System CPU usage percentage"
        ))
        
        self.register_metric(MetricDefinition(
            name="system_memory_percent",
            metric_type=MetricType.GAUGE,
            description="System memory usage percentage"
        ))
        
        self.register_metric(MetricDefinition(
            name="system_disk_percent",
            metric_type=MetricType.GAUGE,
            description="System disk usage percentage"
        ))
        
        # WebSocket metrics
        self.register_metric(MetricDefinition(
            name="websocket_connections_active",
            metric_type=MetricType.GAUGE,
            description="Number of active WebSocket connections"
        ))
        
        self.register_metric(MetricDefinition(
            name="websocket_messages_total",
            metric_type=MetricType.COUNTER,
            description="Total WebSocket messages",
            labels=["direction", "message_type"]
        ))
        
        # API metrics
        self.register_metric(MetricDefinition(
            name="external_api_calls_total",
            metric_type=MetricType.COUNTER,
            description="Total external API calls",
            labels=["provider", "endpoint", "status"]
        ))
        
        self.register_metric(MetricDefinition(
            name="external_api_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="External API call duration",
            labels=["provider", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        ))
        
        # Initialize default alert rules
        self._initialize_alert_rules()
    
    def _initialize_alert_rules(self):
        """Initialize default alert rules"""
        
        # High CPU usage
        self.register_alert_rule(AlertRule(
            name="high_cpu_usage",
            metric_name="system_cpu_percent",
            condition="> 80",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration=300,
            description="CPU usage is above 80%"
        ))
        
        # High memory usage
        self.register_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="system_memory_percent",
            condition="> 85",
            threshold=85.0,
            severity=AlertSeverity.WARNING,
            duration=300,
            description="Memory usage is above 85%"
        ))
        
        # High disk usage
        self.register_alert_rule(AlertRule(
            name="high_disk_usage",
            metric_name="system_disk_percent",
            condition="> 90",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
            duration=60,
            description="Disk usage is above 90%"
        ))
        
        # High error rate
        self.register_alert_rule(AlertRule(
            name="high_error_rate",
            metric_name="http_requests_total",
            condition="> 0.05",  # 5% error rate
            threshold=0.05,
            severity=AlertSeverity.ERROR,
            duration=180,
            description="HTTP error rate is above 5%"
        ))
    
    async def _collect_system_metrics(self):
        """Background task to collect system metrics"""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.set_gauge("system_cpu_percent", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.set_gauge("system_memory_percent", memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.set_gauge("system_disk_percent", disk.percent)
                
                # Database connections
                try:
                    manager = await get_database_manager()
                    pool = manager._engine.pool if manager._engine else None
                    connections = pool.checkedout() if pool else 0
                    self.set_gauge("database_connections_active", connections)
                except Exception:
                    self.set_gauge("database_connections_active", 0)
                
                # Store performance data
                self.performance_data["cpu"].append({
                    "timestamp": time.time(),
                    "value": cpu_percent
                })
                self.performance_data["memory"].append({
                    "timestamp": time.time(),
                    "value": memory.percent
                })
                self.performance_data["disk"].append({
                    "timestamp": time.time(),
                    "value": disk.percent
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error collecting system metrics", error=str(e))
    
    async def _check_alerts(self):
        """Background task to check alert rules"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                
                for rule_name, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue
                    
                    state = self.alert_states[rule_name]
                    
                    # Get current metric value
                    metric_value = await self.get_metric_value(rule.metric_name)
                    if metric_value is None:
                        continue
                    
                    # Check condition
                    condition_met = self._evaluate_condition(metric_value, rule.condition, rule.threshold)
                    
                    if condition_met:
                        state["consecutive_violations"] += 1
                        
                        # Check if alert should be triggered
                        if (not state["triggered"] and 
                            state["consecutive_violations"] >= rule.duration // 60):
                            
                            state["triggered"] = True
                            state["triggered_at"] = current_time
                            
                            await self._trigger_alert(rule, metric_value)
                    else:
                        # Reset violation count
                        state["consecutive_violations"] = 0
                        
                        # Clear alert if it was triggered
                        if state["triggered"]:
                            state["triggered"] = False
                            state["triggered_at"] = None
                            
                            await self._clear_alert(rule)
                    
                    state["last_check"] = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error checking alerts", error=str(e))
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        try:
            if condition.startswith(">"):
                return value > threshold
            elif condition.startswith("<"):
                return value < threshold
            elif condition.startswith(">="):
                return value >= threshold
            elif condition.startswith("<="):
                return value <= threshold
            elif condition.startswith("=="):
                return value == threshold
            elif condition.startswith("!="):
                return value != threshold
            else:
                return False
        except Exception:
            return False
    
    async def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger an alert"""
        try:
            alert_data = {
                "rule_name": rule.name,
                "metric_name": rule.metric_name,
                "current_value": value,
                "threshold": rule.threshold,
                "severity": rule.severity.value,
                "description": rule.description,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store alert in Redis
            alert_key = f"alert:{rule.name}"
            await self.redis_client.setex(
                alert_key,
                3600,  # 1 hour expiration
                json.dumps(alert_data)
            )
            
            # Log alert
            self.logger.warning(
                "Alert triggered",
                rule=rule.name,
                metric=rule.metric_name,
                value=value,
                threshold=rule.threshold,
                severity=rule.severity.value
            )
            
            # Here you would integrate with alerting systems like:
            # - Email notifications
            # - Slack/Teams webhooks
            # - PagerDuty
            # - SMS alerts
            
        except Exception as e:
            self.logger.error("Failed to trigger alert", rule=rule.name, error=str(e))
    
    async def _clear_alert(self, rule: AlertRule):
        """Clear an alert"""
        try:
            # Remove alert from Redis
            alert_key = f"alert:{rule.name}"
            await self.redis_client.delete(alert_key)
            
            # Log alert clearance
            self.logger.info("Alert cleared", rule=rule.name, metric=rule.metric_name)
            
        except Exception as e:
            self.logger.error("Failed to clear alert", rule=rule.name, error=str(e))

# Factory function
async def create_metrics_collector(redis_client: redis.Redis) -> MetricsCollector:
    """Create and start metrics collector"""
    collector = MetricsCollector(redis_client)
    await collector.start()
    return collector

# Global metrics instance (to be initialized by the application)
metrics: Optional[MetricsCollector] = None

def get_metrics() -> MetricsCollector:
    """Get the global metrics instance"""
    if metrics is None:
        raise RuntimeError("Metrics collector not initialized")
    return metrics

