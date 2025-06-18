"""
Workflow Automation Agent
Handles automated workflow execution, task scheduling, and process orchestration
"""

from backend.agents.automation.automation_agent import AutomationAgent
from backend.agents.automation.workflow_executor import WorkflowExecutor
from backend.agents.automation.task_scheduler import TaskScheduler

__all__ = ['AutomationAgent', 'WorkflowExecutor', 'TaskScheduler']

