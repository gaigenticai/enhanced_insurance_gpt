"""
Insurance AI Agent System - Logging and Monitoring Utilities
Production-ready logging, monitoring, and error handling framework
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager, asynccontextmanager
import uuid
import traceback
import json
from pathlib import Path
import structlog
from structlog.stdlib import LoggerFactory
from structlog.processors import JSONRenderer, TimeStamper, add_log_level, StackInfoRenderer
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server, CollectorRegistry
import psutil
import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# =============================================================================
# STRUCTURED LOGGING SETUP
# =============================================================================

class CustomJSONRenderer:
    """Custom JSON renderer for structured logging"""
    
    def __call__(self, logger, method_name, event_dict):
        """Render log entry as JSON"""
        # Add timestamp if not present
        if 'timestamp' not in event_dict:
            event_dict['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Add log level
        event_dict['level'] = method_name.upper()
        
        # Add service info
        event_dict['service'] = 'insurance-ai-system'
        
        # Add request ID if available
        if hasattr(structlog.contextvars, 'get_context'):
            context = structlog.contextvars.get_context()
            if 'request_id' in context:
                event_dict['request_id'] = context['request_id']
        
        return json.dumps(event_dict, default=str, ensure_ascii=False)

def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None
):
    """Setup structured logging configuration"""
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
    ]
    
    if log_format == "json":
        processors.append(CustomJSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# =============================================================================
# REQUEST CONTEXT MANAGEMENT
# =============================================================================

class RequestContext:
    """Request context manager for tracking requests across services"""
    
    def __init__(self):
        self.request_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.organization_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
    
    def set_request_id(self, request_id: str):
        """Set request ID"""
        self.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
    
    def set_user_context(self, user_id: str, organization_id: str = None):
        """Set user context"""
        self.user_id = user_id
        self.organization_id = organization_id
        structlog.contextvars.bind_contextvars(
            user_id=user_id,
            organization_id=organization_id
        )
    
    def add_metadata(self, **kwargs):
        """Add metadata to context"""
        self.metadata.update(kwargs)
        structlog.contextvars.bind_contextvars(**kwargs)
    
    def start_request(self):
        """Start request timing"""
        self.start_time = datetime.now(timezone.utc)
    
    def get_duration(self) -> Optional[float]:
        """Get request duration in seconds"""
        if self.start_time:
            return (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return None

# Global request context
request_context = RequestContext()

@contextmanager
def request_context_manager(request_id: str = None, user_id: str = None, organization_id: str = None):
    """Context manager for request tracking"""
    if not request_id:
        request_id = str(uuid.uuid4())
    
    old_context = structlog.contextvars.get_context().copy()
    
    try:
        request_context.set_request_id(request_id)
        if user_id:
            request_context.set_user_context(user_id, organization_id)
        request_context.start_request()
        
        yield request_context
        
    finally:
        # Restore old context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**old_context)

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

class MetricsCollector:
    """Centralized metrics collector"""
    
    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections_active = Gauge(
            'db_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['operation'],
            registry=self.registry
        )
        
        self.db_queries_total = Counter(
            'db_queries_total',
            'Total database queries',
            ['operation', 'status'],
            registry=self.registry
        )
        
        # Agent metrics
        self.agent_executions_total = Counter(
            'agent_executions_total',
            'Total agent executions',
            ['agent_name', 'status'],
            registry=self.registry
        )
        
        self.agent_execution_duration = Histogram(
            'agent_execution_duration_seconds',
            'Agent execution duration',
            ['agent_name'],
            registry=self.registry
        )
        
        # Workflow metrics
        self.workflows_total = Counter(
            'workflows_total',
            'Total workflows',
            ['type', 'status'],
            registry=self.registry
        )
        
        self.workflow_duration = Histogram(
            'workflow_duration_seconds',
            'Workflow duration',
            ['type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio',
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['path'],
            registry=self.registry
        )
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_db_query(self, operation: str, duration: float, success: bool = True):
        """Record database query metrics"""
        status = 'success' if success else 'error'
        
        self.db_queries_total.labels(
            operation=operation,
            status=status
        ).inc()
        
        self.db_query_duration.labels(operation=operation).observe(duration)
    
    def record_agent_execution(self, agent_name: str, duration: float, success: bool = True):
        """Record agent execution metrics"""
        status = 'success' if success else 'error'
        
        self.agent_executions_total.labels(
            agent_name=agent_name,
            status=status
        ).inc()
        
        self.agent_execution_duration.labels(agent_name=agent_name).observe(duration)
    
    def record_workflow(self, workflow_type: str, status: str, duration: float = None):
        """Record workflow metrics"""
        self.workflows_total.labels(
            type=workflow_type,
            status=status
        ).inc()
        
        if duration is not None:
            self.workflow_duration.labels(type=workflow_type).observe(duration)
    
    def record_cache_operation(self, operation: str, success: bool = True):
        """Record cache operation metrics"""
        status = 'success' if success else 'error'
        
        self.cache_operations_total.labels(
            operation=operation,
            status=status
        ).inc()
    
    def update_system_metrics(self):
        """Update system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            
            # Disk usage
            for disk in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(disk.mountpoint)
                    self.system_disk_usage.labels(path=disk.mountpoint).set(usage.used)
                except (PermissionError, OSError):
                    pass
                    
        except Exception as e:
            logger = structlog.get_logger(__name__)
            logger.warning("Failed to update system metrics", error=str(e))

# Global metrics collector
metrics = MetricsCollector()

# =============================================================================
# ERROR HANDLING AND MONITORING
# =============================================================================

class ErrorTracker:
    """Error tracking and monitoring"""
    
    def __init__(self):
        self.error_counter = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'service', 'severity']
        )
        
        self.logger = structlog.get_logger(__name__)
    
    def track_error(
        self,
        error: Exception,
        service: str = "unknown",
        severity: str = "error",
        context: Dict[str, Any] = None
    ):
        """Track an error with context"""
        error_type = type(error).__name__
        
        # Increment error counter
        self.error_counter.labels(
            error_type=error_type,
            service=service,
            severity=severity
        ).inc()
        
        # Log error with context
        log_data = {
            "error_type": error_type,
            "error_message": str(error),
            "service": service,
            "severity": severity,
            "traceback": traceback.format_exc()
        }
        
        if context:
            log_data.update(context)
        
        if severity == "critical":
            self.logger.critical("Critical error occurred", **log_data)
        elif severity == "error":
            self.logger.error("Error occurred", **log_data)
        elif severity == "warning":
            self.logger.warning("Warning occurred", **log_data)
        else:
            self.logger.info("Info event", **log_data)
    
    def track_exception(self, exc_info=None, **kwargs):
        """Track exception from current context"""
        if exc_info is None:
            exc_info = sys.exc_info()
        
        if exc_info[1] is not None:
            self.track_error(exc_info[1], **kwargs)

# Global error tracker
error_tracker = ErrorTracker()

# =============================================================================
# HEALTH MONITORING
# =============================================================================

class HealthMonitor:
    """System health monitoring"""
    
    def __init__(self):
        self.health_status = Gauge(
            'service_health_status',
            'Service health status (1=healthy, 0=unhealthy)',
            ['service']
        )
        
        self.logger = structlog.get_logger(__name__)
        self._checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check"""
        self._checks[name] = check_func
    
    async def check_database_health(self, db_session: AsyncSession) -> bool:
        """Check database health"""
        try:
            await db_session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error("Database health check failed", error=str(e))
            return False
    
    async def check_redis_health(self, redis_client: redis.Redis) -> bool:
        """Check Redis health"""
        try:
            await redis_client.ping()
            return True
        except Exception as e:
            self.logger.error("Redis health check failed", error=str(e))
            return False
    
    async def check_external_service_health(self, service_url: str) -> bool:
        """Check external service health"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{service_url}/health")
                return response.status_code == 200
        except Exception as e:
            self.logger.error("External service health check failed", url=service_url, error=str(e))
            return False
    
    async def run_all_checks(self) -> Dict[str, bool]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self._checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = result
                self.health_status.labels(service=name).set(1 if result else 0)
                
            except Exception as e:
                self.logger.error("Health check failed", check=name, error=str(e))
                results[name] = False
                self.health_status.labels(service=name).set(0)
        
        return results
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                },
                "boot_time": psutil.boot_time(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except Exception as e:
            self.logger.error("Failed to get system info", error=str(e))
            return {}

# Global health monitor
health_monitor = HealthMonitor()

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self):
        self.response_time_summary = Summary(
            'response_time_seconds',
            'Response time summary',
            ['operation']
        )
        
        self.logger = structlog.get_logger(__name__)
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str):
        """Monitor operation performance"""
        start_time = datetime.now(timezone.utc)
        
        try:
            yield
            
        finally:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.response_time_summary.labels(operation=operation_name).observe(duration)
            
            self.logger.info(
                "Operation completed",
                operation=operation_name,
                duration_seconds=duration
            )
    
    def profile_function(self, func_name: str = None):
        """Decorator for profiling functions"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    async with self.monitor_operation(name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = datetime.now(timezone.utc)
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                        self.response_time_summary.labels(operation=name).observe(duration)
                        
                        self.logger.info(
                            "Function completed",
                            function=name,
                            duration_seconds=duration
                        )
                return sync_wrapper
            
        return decorator

# Global performance monitor
performance_monitor = PerformanceMonitor()

# =============================================================================
# AUDIT LOGGING
# =============================================================================

class AuditLogger:
    """Audit logging for compliance and security"""
    
    def __init__(self):
        self.logger = structlog.get_logger("audit")
    
    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource_type: str = None,
        resource_id: str = None,
        details: Dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log user action for audit trail"""
        audit_data = {
            "audit_type": "user_action",
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if details:
            audit_data["details"] = details
        
        self.logger.info("User action", **audit_data)
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "info",
        details: Dict[str, Any] = None
    ):
        """Log system event"""
        audit_data = {
            "audit_type": "system_event",
            "event_type": event_type,
            "description": description,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if details:
            audit_data["details"] = details
        
        if severity == "critical":
            self.logger.critical("System event", **audit_data)
        elif severity == "error":
            self.logger.error("System event", **audit_data)
        elif severity == "warning":
            self.logger.warning("System event", **audit_data)
        else:
            self.logger.info("System event", **audit_data)
    
    def log_security_event(
        self,
        event_type: str,
        user_id: str = None,
        ip_address: str = None,
        details: Dict[str, Any] = None,
        severity: str = "warning"
    ):
        """Log security event"""
        audit_data = {
            "audit_type": "security_event",
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if details:
            audit_data["details"] = details
        
        if severity == "critical":
            self.logger.critical("Security event", **audit_data)
        elif severity == "error":
            self.logger.error("Security event", **audit_data)
        else:
            self.logger.warning("Security event", **audit_data)

# Global audit logger
audit_logger = AuditLogger()

# =============================================================================
# MONITORING SETUP FUNCTIONS
# =============================================================================

def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server"""
    try:
        start_http_server(port, registry=metrics.registry)
        logger = structlog.get_logger(__name__)
        logger.info("Metrics server started", port=port)
    except Exception as e:
        logger = structlog.get_logger(__name__)
        logger.error("Failed to start metrics server", port=port, error=str(e))

async def setup_monitoring():
    """Setup monitoring and health checks"""
    # Register default health checks
    health_monitor.register_check("system", lambda: True)
    
    # Start background task for system metrics
    async def update_metrics_loop():
        while True:
            try:
                metrics.update_system_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger = structlog.get_logger(__name__)
                logger.error("Failed to update system metrics", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    asyncio.create_task(update_metrics_loop())

# =============================================================================
# DECORATOR UTILITIES
# =============================================================================

def log_function_call(include_args: bool = False, include_result: bool = False):
    """Decorator to log function calls"""
    def decorator(func):
        logger = structlog.get_logger(func.__module__)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                log_data = {"function": func.__name__}
                
                if include_args:
                    log_data["args"] = str(args)
                    log_data["kwargs"] = str(kwargs)
                
                logger.info("Function called", **log_data)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    if include_result:
                        log_data["result"] = str(result)
                    
                    logger.info("Function completed", **log_data)
                    return result
                    
                except Exception as e:
                    logger.error("Function failed", function=func.__name__, error=str(e))
                    raise
                    
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                log_data = {"function": func.__name__}
                
                if include_args:
                    log_data["args"] = str(args)
                    log_data["kwargs"] = str(kwargs)
                
                logger.info("Function called", **log_data)
                
                try:
                    result = func(*args, **kwargs)
                    
                    if include_result:
                        log_data["result"] = str(result)
                    
                    logger.info("Function completed", **log_data)
                    return result
                    
                except Exception as e:
                    logger.error("Function failed", function=func.__name__, error=str(e))
                    raise
                    
            return sync_wrapper
            
    return decorator

def monitor_performance(operation_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with performance_monitor.monitor_operation(name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = datetime.now(timezone.utc)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                    performance_monitor.response_time_summary.labels(operation=name).observe(duration)
            return sync_wrapper
            
    return decorator

