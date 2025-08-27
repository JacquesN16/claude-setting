"""
Shared utilities for the three-agent pipeline system.
"""

import json
import yaml
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import hashlib
import tempfile
import shutil

from .models import (
    ExecutionPlan, CodeOutput, ReviewReport, PipelineRun, 
    TaskStatus, AgentType, AgentMessage
)


class PipelineLogger:
    """Centralized logging for the pipeline system."""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("pipeline")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, agent: Optional[AgentType] = None):
        prefix = f"[{agent.value.upper()}] " if agent else ""
        self.logger.info(f"{prefix}{message}")
    
    def error(self, message: str, agent: Optional[AgentType] = None):
        prefix = f"[{agent.value.upper()}] " if agent else ""
        self.logger.error(f"{prefix}{message}")
    
    def debug(self, message: str, agent: Optional[AgentType] = None):
        prefix = f"[{agent.value.upper()}] " if agent else ""
        self.logger.debug(f"{prefix}{message}")


class ConfigLoader:
    """Loads and manages configuration files."""
    
    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Failed to load YAML file {file_path}: {str(e)}")
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            raise Exception(f"Failed to load JSON file {file_path}: {str(e)}")
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2):
        """Save data to JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=indent, default=str)
        except Exception as e:
            raise Exception(f"Failed to save JSON file {file_path}: {str(e)}")


class FileManager:
    """Manages file operations for the pipeline."""
    
    def __init__(self, base_output_dir: str = "./pipeline_output"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_run_directory(self, run_id: str) -> Path:
        """Create a directory for a specific pipeline run."""
        run_dir = self.base_output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (run_dir / "code_output").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)
        
        return run_dir
    
    def save_execution_plan(self, plan: ExecutionPlan, run_dir: Path):
        """Save execution plan to file."""
        plan_file = run_dir / "execution_plan.json"
        ConfigLoader.save_json(plan.to_dict(), plan_file)
    
    def save_review_report(self, report: ReviewReport, run_dir: Path):
        """Save review report to file."""
        report_file = run_dir / "reports" / "review_report.json"
        ConfigLoader.save_json(report.to_dict(), report_file)
    
    def save_pipeline_run(self, run: PipelineRun, run_dir: Path):
        """Save pipeline run metadata to file."""
        run_file = run_dir / "pipeline_run.json"
        ConfigLoader.save_json(run.to_dict(), run_file)
    
    def copy_code_files(self, source_files: List[str], run_dir: Path):
        """Copy generated code files to run directory."""
        code_dir = run_dir / "code_output"
        
        for file_path in source_files:
            if os.path.exists(file_path):
                dest_path = code_dir / os.path.basename(file_path)
                # Only copy if source and destination are different
                if os.path.abspath(file_path) != os.path.abspath(str(dest_path)):
                    shutil.copy2(file_path, dest_path)


class TaskTimer:
    """Utility for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in minutes."""
        if self.start_time is None:
            return 0.0
        
        self.end_time = time.time()
        elapsed_seconds = self.end_time - self.start_time
        return elapsed_seconds / 60.0  # Convert to minutes


class MessageQueue:
    """Simple message queue for inter-agent communication."""
    
    def __init__(self):
        self.messages: List[AgentMessage] = []
    
    def send(self, message: AgentMessage):
        """Add message to queue."""
        self.messages.append(message)
    
    def receive(self, agent_type: AgentType) -> List[AgentMessage]:
        """Get messages for specific agent."""
        messages = [
            msg for msg in self.messages 
            if msg.to_agent == agent_type or msg.to_agent is None
        ]
        # Remove received messages
        self.messages = [
            msg for msg in self.messages 
            if msg.to_agent != agent_type and msg.to_agent is not None
        ]
        return messages
    
    def clear(self):
        """Clear all messages."""
        self.messages.clear()


class CodeAnalyzer:
    """Utility for analyzing code files."""
    
    @staticmethod
    def count_lines_of_code(file_path: str) -> int:
        """Count non-empty lines in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file.readlines()]
                return len([line for line in lines if line and not line.startswith('#')])
        except:
            return 0
    
    @staticmethod
    def count_functions(file_path: str) -> int:
        """Count function definitions in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return content.count('def ')
        except:
            return 0
    
    @staticmethod
    def count_classes(file_path: str) -> int:
        """Count class definitions in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return content.count('class ')
        except:
            return 0
    
    @staticmethod
    def analyze_directory(directory_path: str) -> CodeOutput:
        """Analyze all Python files in a directory."""
        code_output = CodeOutput()
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    code_output.files_created.append(file_path)
                    code_output.lines_of_code += CodeAnalyzer.count_lines_of_code(file_path)
                    code_output.functions_created += CodeAnalyzer.count_functions(file_path)
                    code_output.classes_created += CodeAnalyzer.count_classes(file_path)
                elif file.endswith(('.md', '.rst', '.txt')):
                    code_output.documentation_files.append(os.path.join(root, file))
        
        return code_output


class ValidationUtils:
    """Validation utilities for pipeline data."""
    
    @staticmethod
    def validate_execution_plan(plan_dict: Dict[str, Any]) -> bool:
        """Validate execution plan structure."""
        required_fields = ['task_id', 'task_description', 'steps']
        return all(field in plan_dict for field in required_fields)
    
    @staticmethod
    def validate_review_report(report_dict: Dict[str, Any]) -> bool:
        """Validate review report structure."""
        required_fields = ['review_id', 'overall_score', 'grade', 'approval_status']
        return all(field in report_dict for field in required_fields)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations."""
        import re
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        return filename


def create_temp_directory() -> str:
    """Create a temporary directory for pipeline operations."""
    return tempfile.mkdtemp(prefix="pipeline_")


def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def generate_file_hash(file_path: str) -> str:
    """Generate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return ""


def ensure_directory_exists(directory_path: str):
    """Ensure directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)