"""
Task Scheduler - Production Ready Implementation
Handles scheduling, queuing, and execution of automated tasks with cron-like functionality
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import croniter
import pytz
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import heapq

# Database and caching
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
scheduled_tasks_total = Counter('scheduled_tasks_total', 'Total scheduled tasks', ['task_type', 'status'])
task_execution_duration = Histogram('task_execution_duration_seconds', 'Time spent executing scheduled tasks')
active_scheduled_tasks = Gauge('active_scheduled_tasks', 'Number of active scheduled tasks')
scheduler_queue_size = Gauge('scheduler_queue_size', 'Number of tasks in scheduler queue')

Base = declarative_base()

class ScheduleType(Enum):
    ONE_TIME = "one_time"
    RECURRING = "recurring"
    CRON = "cron"
    INTERVAL = "interval"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10

@dataclass
class ScheduledTask:
    """Represents a scheduled task with all scheduling information"""
    task_id: str
    name: str
    description: str
    schedule_type: ScheduleType
    task_function: str  # Function name or endpoint
    parameters: Dict[str, Any]
    priority: TaskPriority
    created_at: datetime
    next_run_at: datetime
    last_run_at: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    is_active: bool = True
    timezone: str = "UTC"
    
    # Scheduling configuration
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    
    # Execution configuration
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 60
    
    # Metadata
    tags: List[str] = None
    owner: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class TaskExecution:
    """Tracks execution of a scheduled task"""
    execution_id: str
    task_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    retry_count: int = 0

class ScheduledTaskModel(Base):
    """SQLAlchemy model for persisting scheduled tasks"""
    __tablename__ = 'scheduled_tasks'
    
    task_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    schedule_type = Column(String, nullable=False)
    task_function = Column(String, nullable=False)
    parameters = Column(JSON)
    priority = Column(Integer, default=5)
    created_at = Column(DateTime, nullable=False)
    next_run_at = Column(DateTime, nullable=False)
    last_run_at = Column(DateTime)
    run_count = Column(Integer, default=0)
    max_runs = Column(Integer)
    is_active = Column(Boolean, default=True)
    timezone = Column(String, default="UTC")
    cron_expression = Column(String)
    interval_seconds = Column(Integer)
    timeout_seconds = Column(Integer, default=300)
    max_retries = Column(Integer, default=3)
    retry_delay_seconds = Column(Integer, default=60)
    tags = Column(JSON)
    owner = Column(String)

class TaskScheduler:
    """
    Production-ready Task Scheduler
    Handles scheduling, queuing, and execution of automated tasks with cron-like functionality
    """
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis setup
        self.redis_client = redis.from_url(redis_url)
        
        # Task management
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.task_queue = []  # Priority queue (heapq)
        self.active_executions: Dict[str, TaskExecution] = {}
        
        # Execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Task functions registry
        self.task_functions: Dict[str, Callable] = {}
        
        # Load existing tasks from database
        self._load_tasks_from_database()
        
        logger.info("TaskScheduler initialized successfully")

    def _load_tasks_from_database(self):
        """Load existing scheduled tasks from database"""
        
        try:
            with self.Session() as session:
                task_models = session.query(ScheduledTaskModel).filter(
                    ScheduledTaskModel.is_active == True
                ).all()
                
                for task_model in task_models:
                    task = self._model_to_task(task_model)
                    self.scheduled_tasks[task.task_id] = task
                    
                    # Add to queue if due
                    if task.next_run_at <= datetime.utcnow():
                        self._add_to_queue(task)
                
                logger.info(f"Loaded {len(task_models)} scheduled tasks from database")
                
        except Exception as e:
            logger.error(f"Failed to load tasks from database: {e}")

    def _model_to_task(self, model: ScheduledTaskModel) -> ScheduledTask:
        """Convert database model to ScheduledTask"""
        
        return ScheduledTask(
            task_id=model.task_id,
            name=model.name,
            description=model.description,
            schedule_type=ScheduleType(model.schedule_type),
            task_function=model.task_function,
            parameters=model.parameters or {},
            priority=TaskPriority(model.priority),
            created_at=model.created_at,
            next_run_at=model.next_run_at,
            last_run_at=model.last_run_at,
            run_count=model.run_count,
            max_runs=model.max_runs,
            is_active=model.is_active,
            timezone=model.timezone,
            cron_expression=model.cron_expression,
            interval_seconds=model.interval_seconds,
            timeout_seconds=model.timeout_seconds,
            max_retries=model.max_retries,
            retry_delay_seconds=model.retry_delay_seconds,
            tags=model.tags or [],
            owner=model.owner
        )

    def _task_to_model(self, task: ScheduledTask) -> ScheduledTaskModel:
        """Convert ScheduledTask to database model"""
        
        return ScheduledTaskModel(
            task_id=task.task_id,
            name=task.name,
            description=task.description,
            schedule_type=task.schedule_type.value,
            task_function=task.task_function,
            parameters=task.parameters,
            priority=task.priority.value,
            created_at=task.created_at,
            next_run_at=task.next_run_at,
            last_run_at=task.last_run_at,
            run_count=task.run_count,
            max_runs=task.max_runs,
            is_active=task.is_active,
            timezone=task.timezone,
            cron_expression=task.cron_expression,
            interval_seconds=task.interval_seconds,
            timeout_seconds=task.timeout_seconds,
            max_retries=task.max_retries,
            retry_delay_seconds=task.retry_delay_seconds,
            tags=task.tags,
            owner=task.owner
        )

    def register_task_function(self, name: str, function: Callable):
        """Register a task function that can be scheduled"""
        
        self.task_functions[name] = function
        logger.info(f"Registered task function: {name}")

    def schedule_one_time_task(self, 
                              name: str,
                              task_function: str,
                              parameters: Dict[str, Any],
                              run_at: datetime,
                              priority: TaskPriority = TaskPriority.NORMAL,
                              **kwargs) -> str:
        """Schedule a one-time task"""
        
        task_id = str(uuid.uuid4())
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            description=kwargs.get('description', ''),
            schedule_type=ScheduleType.ONE_TIME,
            task_function=task_function,
            parameters=parameters,
            priority=priority,
            created_at=datetime.utcnow(),
            next_run_at=run_at,
            max_runs=1,
            timezone=kwargs.get('timezone', 'UTC'),
            timeout_seconds=kwargs.get('timeout_seconds', 300),
            max_retries=kwargs.get('max_retries', 3),
            tags=kwargs.get('tags', []),
            owner=kwargs.get('owner')
        )
        
        self._save_task(task)
        
        # Add to queue if due
        if task.next_run_at <= datetime.utcnow():
            self._add_to_queue(task)
        
        scheduled_tasks_total.labels(task_type='one_time', status='created').inc()
        logger.info(f"Scheduled one-time task {task_id}: {name}")
        
        return task_id

    def schedule_recurring_task(self,
                               name: str,
                               task_function: str,
                               parameters: Dict[str, Any],
                               cron_expression: str,
                               priority: TaskPriority = TaskPriority.NORMAL,
                               **kwargs) -> str:
        """Schedule a recurring task using cron expression"""
        
        task_id = str(uuid.uuid4())
        
        # Calculate next run time
        tz = pytz.timezone(kwargs.get('timezone', 'UTC'))
        cron = croniter.croniter(cron_expression, datetime.now(tz))
        next_run = cron.get_next(datetime)
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            description=kwargs.get('description', ''),
            schedule_type=ScheduleType.CRON,
            task_function=task_function,
            parameters=parameters,
            priority=priority,
            created_at=datetime.utcnow(),
            next_run_at=next_run,
            cron_expression=cron_expression,
            timezone=kwargs.get('timezone', 'UTC'),
            max_runs=kwargs.get('max_runs'),
            timeout_seconds=kwargs.get('timeout_seconds', 300),
            max_retries=kwargs.get('max_retries', 3),
            tags=kwargs.get('tags', []),
            owner=kwargs.get('owner')
        )
        
        self._save_task(task)
        
        # Add to queue if due
        if task.next_run_at <= datetime.utcnow():
            self._add_to_queue(task)
        
        scheduled_tasks_total.labels(task_type='cron', status='created').inc()
        logger.info(f"Scheduled recurring task {task_id}: {name} with cron '{cron_expression}'")
        
        return task_id

    def schedule_interval_task(self,
                              name: str,
                              task_function: str,
                              parameters: Dict[str, Any],
                              interval_seconds: int,
                              priority: TaskPriority = TaskPriority.NORMAL,
                              **kwargs) -> str:
        """Schedule a task to run at regular intervals"""
        
        task_id = str(uuid.uuid4())
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            description=kwargs.get('description', ''),
            schedule_type=ScheduleType.INTERVAL,
            task_function=task_function,
            parameters=parameters,
            priority=priority,
            created_at=datetime.utcnow(),
            next_run_at=datetime.utcnow() + timedelta(seconds=interval_seconds),
            interval_seconds=interval_seconds,
            timezone=kwargs.get('timezone', 'UTC'),
            max_runs=kwargs.get('max_runs'),
            timeout_seconds=kwargs.get('timeout_seconds', 300),
            max_retries=kwargs.get('max_retries', 3),
            tags=kwargs.get('tags', []),
            owner=kwargs.get('owner')
        )
        
        self._save_task(task)
        
        scheduled_tasks_total.labels(task_type='interval', status='created').inc()
        logger.info(f"Scheduled interval task {task_id}: {name} every {interval_seconds} seconds")
        
        return task_id

    def _save_task(self, task: ScheduledTask):
        """Save task to database and memory"""
        
        try:
            with self.Session() as session:
                model = self._task_to_model(task)
                session.merge(model)
                session.commit()
                
            self.scheduled_tasks[task.task_id] = task
            active_scheduled_tasks.inc()
            
        except Exception as e:
            logger.error(f"Failed to save task {task.task_id}: {e}")
            raise

    def _add_to_queue(self, task: ScheduledTask):
        """Add task to execution queue"""
        
        # Use negative priority for max-heap behavior (higher priority first)
        priority = -task.priority.value
        heapq.heappush(self.task_queue, (priority, task.next_run_at, task.task_id))
        scheduler_queue_size.inc()
        
        logger.debug(f"Added task {task.task_id} to queue")

    def start_scheduler(self):
        """Start the task scheduler"""
        
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        
        def scheduler_loop():
            """Main scheduler loop"""
            
            while self.scheduler_running:
                try:
                    self._process_scheduled_tasks()
                    time.sleep(1)  # Check every second
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    time.sleep(5)  # Wait longer on error
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Task scheduler started")

    def _process_scheduled_tasks(self):
        """Process tasks that are due for execution"""
        
        current_time = datetime.utcnow()
        
        # Process due tasks from queue
        while self.task_queue:
            priority, scheduled_time, task_id = self.task_queue[0]
            
            if scheduled_time > current_time:
                break  # No more due tasks
            
            # Remove from queue
            heapq.heappop(self.task_queue)
            scheduler_queue_size.dec()
            
            # Execute task
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                if task.is_active:
                    self._execute_task_async(task)
        
        # Check for new tasks that became due
        self._check_for_due_tasks()

    def _check_for_due_tasks(self):
        """Check for tasks that became due and add them to queue"""
        
        current_time = datetime.utcnow()
        
        for task in self.scheduled_tasks.values():
            if (task.is_active and 
                task.next_run_at <= current_time and 
                task.task_id not in [exec.task_id for exec in self.active_executions.values()]):
                
                # Check if task is already in queue
                task_in_queue = any(tid == task.task_id for _, _, tid in self.task_queue)
                
                if not task_in_queue:
                    self._add_to_queue(task)

    def _execute_task_async(self, task: ScheduledTask):
        """Execute task asynchronously"""
        
        execution_id = str(uuid.uuid4())
        execution = TaskExecution(
            execution_id=execution_id,
            task_id=task.task_id,
            started_at=datetime.utcnow()
        )
        
        self.active_executions[execution_id] = execution
        
        # Submit to thread pool
        future = self.executor.submit(self._execute_task_sync, task, execution)
        future.add_done_callback(lambda f: self._task_execution_completed(f, task, execution))
        
        logger.info(f"Started execution {execution_id} for task {task.task_id}")

    def _execute_task_sync(self, task: ScheduledTask, execution: TaskExecution):
        """Execute task synchronously in thread pool"""
        
        with task_execution_duration.time():
            try:
                logger.info(f"Executing task {task.task_id}: {task.name}")
                
                # Get task function
                if task.task_function not in self.task_functions:
                    raise ValueError(f"Task function '{task.task_function}' not registered")
                
                task_func = self.task_functions[task.task_function]
                
                # Execute with timeout
                result = task_func(task.parameters)
                
                execution.status = "completed"
                execution.result = result
                
                scheduled_tasks_total.labels(task_type=task.schedule_type.value, status='completed').inc()
                logger.info(f"Task {task.task_id} completed successfully")
                
            except Exception as e:
                execution.status = "failed"
                execution.error_message = str(e)
                
                scheduled_tasks_total.labels(task_type=task.schedule_type.value, status='failed').inc()
                logger.error(f"Task {task.task_id} failed: {e}")
                
            finally:
                execution.completed_at = datetime.utcnow()
                execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()

    def _task_execution_completed(self, future, task: ScheduledTask, execution: TaskExecution):
        """Handle task execution completion"""
        
        try:
            # Update task statistics
            task.last_run_at = execution.started_at
            task.run_count += 1
            
            # Schedule next run if recurring
            if task.schedule_type in [ScheduleType.CRON, ScheduleType.INTERVAL] and task.is_active:
                if not task.max_runs or task.run_count < task.max_runs:
                    self._schedule_next_run(task)
                else:
                    # Max runs reached, deactivate
                    task.is_active = False
                    logger.info(f"Task {task.task_id} reached max runs ({task.max_runs}), deactivated")
            elif task.schedule_type == ScheduleType.ONE_TIME:
                # One-time task completed, deactivate
                task.is_active = False
            
            # Save updated task
            self._save_task(task)
            
            # Store execution result
            self._store_execution_result(execution)
            
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
        
        finally:
            # Remove from active executions
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]

    def _schedule_next_run(self, task: ScheduledTask):
        """Schedule the next run for a recurring task"""
        
        if task.schedule_type == ScheduleType.CRON:
            tz = pytz.timezone(task.timezone)
            cron = croniter.croniter(task.cron_expression, datetime.now(tz))
            task.next_run_at = cron.get_next(datetime)
            
        elif task.schedule_type == ScheduleType.INTERVAL:
            task.next_run_at = datetime.utcnow() + timedelta(seconds=task.interval_seconds)
        
        # Add to queue
        self._add_to_queue(task)
        
        logger.debug(f"Scheduled next run for task {task.task_id} at {task.next_run_at}")

    def _store_execution_result(self, execution: TaskExecution):
        """Store execution result in Redis"""
        
        try:
            execution_data = asdict(execution)
            
            # Convert datetime objects to ISO strings
            for key, value in execution_data.items():
                if isinstance(value, datetime):
                    execution_data[key] = value.isoformat() if value else None
            
            self.redis_client.setex(
                f"task_execution:{execution.execution_id}",
                3600 * 24,  # 24 hours TTL
                json.dumps(execution_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to store execution result: {e}")

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get scheduled task by ID"""
        return self.scheduled_tasks.get(task_id)

    def get_all_tasks(self, active_only: bool = True) -> List[ScheduledTask]:
        """Get all scheduled tasks"""
        
        if active_only:
            return [task for task in self.scheduled_tasks.values() if task.is_active]
        else:
            return list(self.scheduled_tasks.values())

    def get_tasks_by_tag(self, tag: str) -> List[ScheduledTask]:
        """Get tasks by tag"""
        
        return [task for task in self.scheduled_tasks.values() 
                if tag in task.tags and task.is_active]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        
        if task_id not in self.scheduled_tasks:
            return False
        
        task = self.scheduled_tasks[task_id]
        task.is_active = False
        
        try:
            self._save_task(task)
            active_scheduled_tasks.dec()
            
            scheduled_tasks_total.labels(task_type=task.schedule_type.value, status='cancelled').inc()
            logger.info(f"Cancelled task {task_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task"""
        
        if task_id not in self.scheduled_tasks:
            return False
        
        task = self.scheduled_tasks[task_id]
        task.is_active = False
        
        try:
            self._save_task(task)
            logger.info(f"Paused task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause task {task_id}: {e}")
            return False

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task"""
        
        if task_id not in self.scheduled_tasks:
            return False
        
        task = self.scheduled_tasks[task_id]
        task.is_active = True
        
        # Recalculate next run time if needed
        if task.schedule_type == ScheduleType.CRON:
            tz = pytz.timezone(task.timezone)
            cron = croniter.croniter(task.cron_expression, datetime.now(tz))
            task.next_run_at = cron.get_next(datetime)
        elif task.schedule_type == ScheduleType.INTERVAL:
            task.next_run_at = datetime.utcnow() + timedelta(seconds=task.interval_seconds)
        
        try:
            self._save_task(task)
            
            # Add to queue if due
            if task.next_run_at <= datetime.utcnow():
                self._add_to_queue(task)
            
            logger.info(f"Resumed task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume task {task_id}: {e}")
            return False

    def get_execution_history(self, task_id: str, limit: int = 10) -> List[TaskExecution]:
        """Get execution history for a task"""
        
        try:
            keys = self.redis_client.keys(f"task_execution:*")
            executions = []
            
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    execution_data = json.loads(data)
                    if execution_data.get('task_id') == task_id:
                        # Convert ISO strings back to datetime
                        for key_name in ['started_at', 'completed_at']:
                            if execution_data.get(key_name):
                                execution_data[key_name] = datetime.fromisoformat(execution_data[key_name])
                        
                        executions.append(TaskExecution(**execution_data))
            
            # Sort by started_at descending and limit
            executions.sort(key=lambda x: x.started_at, reverse=True)
            return executions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get execution history: {e}")
            return []

    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        
        active_tasks = len([t for t in self.scheduled_tasks.values() if t.is_active])
        
        task_counts_by_type = {}
        for task in self.scheduled_tasks.values():
            if task.is_active:
                task_type = task.schedule_type.value
                task_counts_by_type[task_type] = task_counts_by_type.get(task_type, 0) + 1
        
        return {
            "total_tasks": len(self.scheduled_tasks),
            "active_tasks": active_tasks,
            "tasks_by_type": task_counts_by_type,
            "queue_size": len(self.task_queue),
            "active_executions": len(self.active_executions),
            "registered_functions": len(self.task_functions)
        }

    def stop_scheduler(self):
        """Stop the task scheduler"""
        
        if not self.scheduler_running:
            return
        
        self.scheduler_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        logger.info("Task scheduler stopped")

    def shutdown(self):
        """Graceful shutdown of the task scheduler"""
        
        logger.info("Shutting down TaskScheduler...")
        
        # Stop scheduler
        self.stop_scheduler()
        
        # Wait for active executions to complete
        timeout = 30  # seconds
        start_time = time.time()
        
        while self.active_executions and time.time() - start_time < timeout:
            time.sleep(1)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("TaskScheduler shutdown complete")

# Factory function
def create_task_scheduler(db_url: str = None, redis_url: str = None) -> TaskScheduler:
    """Create and configure a TaskScheduler instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    return TaskScheduler(db_url=db_url, redis_url=redis_url)

# Example task functions
def example_backup_task(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Example backup task"""
    logger.info("Running backup task")
    return {"status": "completed", "backup_size": "1.2GB"}

def example_cleanup_task(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Example cleanup task"""
    logger.info("Running cleanup task")
    return {"status": "completed", "files_cleaned": 150}

def example_report_task(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Example report generation task"""
    logger.info("Running report generation task")
    return {"status": "completed", "report_path": "/reports/daily_report.pdf"}

# Example usage
if __name__ == "__main__":
    def test_task_scheduler():
        """Test the task scheduler functionality"""
        
        scheduler = create_task_scheduler()
        
        # Register task functions
        scheduler.register_task_function("backup", example_backup_task)
        scheduler.register_task_function("cleanup", example_cleanup_task)
        scheduler.register_task_function("report", example_report_task)
        
        # Schedule tasks
        scheduler.schedule_recurring_task(
            name="Daily Backup",
            task_function="backup",
            parameters={"target": "/backup"},
            cron_expression="0 2 * * *",  # Daily at 2 AM
            description="Daily system backup"
        )
        
        scheduler.schedule_interval_task(
            name="Cleanup Temp Files",
            task_function="cleanup",
            parameters={"directory": "/tmp"},
            interval_seconds=3600,  # Every hour
            description="Clean up temporary files"
        )
        
        scheduler.schedule_one_time_task(
            name="Generate Monthly Report",
            task_function="report",
            parameters={"month": "2024-01"},
            run_at=datetime.utcnow() + timedelta(minutes=5),
            description="Generate monthly insurance report"
        )
        
        # Start scheduler
        scheduler.start_scheduler()
        
        print("Task scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(10)
                stats = scheduler.get_scheduler_statistics()
                print(f"Scheduler stats: {stats}")
        except KeyboardInterrupt:
            scheduler.shutdown()
    
    # Run test
    # test_task_scheduler()

