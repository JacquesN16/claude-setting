"""
Shared data models for the three-agent pipeline system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from datetime import datetime
import uuid


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskStep:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: Priority = Priority.MEDIUM
    estimated_time_minutes: int = 10
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    notes: str = ""


@dataclass
class ExecutionPlan:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str = ""
    task_analysis: str = ""
    steps: List[TaskStep] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    total_estimated_time: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    created_by: AgentType = AgentType.PLANNER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "task_analysis": self.task_analysis,
            "steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "priority": step.priority.value,
                    "estimated_time_minutes": step.estimated_time_minutes,
                    "dependencies": step.dependencies,
                    "status": step.status.value,
                    "created_at": step.created_at.isoformat(),
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "notes": step.notes
                } for step in self.steps
            ],
            "deliverables": self.deliverables,
            "risks": self.risks,
            "total_estimated_time": self.total_estimated_time,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by.value
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class CodeOutput:
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    functions_created: int = 0
    classes_created: int = 0
    tests_created: int = 0
    documentation_files: List[str] = field(default_factory=list)
    execution_log: List[str] = field(default_factory=list)
    errors_encountered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReviewIssue:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    line_number: Optional[int] = None
    category: str = ""  # security, performance, readability, etc.
    severity: str = ""  # low, medium, high, critical
    description: str = ""
    suggestion: str = ""
    auto_fixable: bool = False
    rule_violated: str = ""


@dataclass
class ReviewReport:
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    overall_score: float = 0.0
    grade: str = ""  # excellent, good, acceptable, needs_improvement, unacceptable
    issues: List[ReviewIssue] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    approval_status: str = ""  # approved, approved_with_changes, rejected
    code_output_analyzed: Optional[CodeOutput] = None
    review_time_minutes: float = 0.0
    reviewed_at: datetime = field(default_factory=datetime.now)
    reviewed_by: AgentType = AgentType.REVIEWER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "issues": [
                {
                    "id": issue.id,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "category": issue.category,
                    "severity": issue.severity,
                    "description": issue.description,
                    "suggestion": issue.suggestion,
                    "auto_fixable": issue.auto_fixable,
                    "rule_violated": issue.rule_violated
                } for issue in self.issues
            ],
            "strengths": self.strengths,
            "recommendations": self.recommendations,
            "approval_status": self.approval_status,
            "review_time_minutes": self.review_time_minutes,
            "reviewed_at": self.reviewed_at.isoformat(),
            "reviewed_by": self.reviewed_by.value
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PipelineRun:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    execution_plan: Optional[ExecutionPlan] = None
    code_output: Optional[CodeOutput] = None
    review_report: Optional[ReviewReport] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_time_minutes: float = 0.0
    current_agent: Optional[AgentType] = None
    error_message: str = ""
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_description": self.task_description,
            "status": self.status.value,
            "execution_plan": self.execution_plan.to_dict() if self.execution_plan else None,
            "code_output": self.code_output.__dict__ if self.code_output else None,
            "review_report": self.review_report.to_dict() if self.review_report else None,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_time_minutes": self.total_time_minutes,
            "current_agent": self.current_agent.value if self.current_agent else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AgentMessage:
    from_agent: AgentType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    to_agent: Optional[AgentType] = None
    message_type: str = ""  # plan, code_output, review_report, error, etc.
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    pipeline_run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_agent": self.from_agent.value,
            "to_agent": self.to_agent.value if self.to_agent else None,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "pipeline_run_id": self.pipeline_run_id
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)