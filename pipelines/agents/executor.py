"""
Execution Agent for the Three-Agent Pipeline System.

This agent implements the execution plan created by the Planning Agent.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid

from shared.models import (
    ExecutionPlan, TaskStep, CodeOutput, TaskStatus, AgentType
)
from shared.utils import (
    PipelineLogger, TaskTimer, FileManager, CodeAnalyzer, 
    ensure_directory_exists, create_temp_directory, cleanup_temp_directory
)
from shared.memory import PipelineMemoryManager


class ExecutionAgent:
    """
    Agent responsible for executing the plan and implementing code.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 memory_manager: Optional[PipelineMemoryManager] = None):
        self.config = config or {}
        self.logger = PipelineLogger()
        self.agent_type = AgentType.EXECUTOR
        self.file_manager = FileManager()
        self.execution_log = []
        self.temp_directory = None
        self.memory = memory_manager
        
    def execute_plan(self, execution_plan: ExecutionPlan, output_directory: str = "./code_output") -> CodeOutput:
        """
        Execute the provided execution plan step by step.
        
        Args:
            execution_plan: The plan to execute
            output_directory: Directory to store generated code
            
        Returns:
            CodeOutput object with execution results
        """
        self.logger.info(f"Starting execution of plan: {execution_plan.task_id}", self.agent_type)
        
        timer = TaskTimer()
        timer.start()
        
        try:
            # Set up working directory
            self.temp_directory = create_temp_directory()
            ensure_directory_exists(output_directory)
            
            # Initialize code output tracking
            code_output = CodeOutput()
            code_output.execution_log.append(f"Started execution at {datetime.now().isoformat()}")
            
            # Execute steps in order
            for step in execution_plan.steps:
                try:
                    self.logger.info(f"Executing step: {step.description}", self.agent_type)
                    step_result = self._execute_step(step, execution_plan, output_directory)
                    
                    # Update step status
                    step.status = TaskStatus.COMPLETED
                    step.completed_at = datetime.now()
                    
                    # Update code output
                    self._update_code_output(code_output, step_result, step)
                    
                    self.logger.info(f"Completed step: {step.description}", self.agent_type)
                    
                except Exception as e:
                    step.status = TaskStatus.FAILED
                    error_msg = f"Step failed: {step.description} - {str(e)}"
                    self.logger.error(error_msg, self.agent_type)
                    code_output.errors_encountered.append(error_msg)
                    
                    # Decide whether to continue or fail
                    if step.priority.value == "critical":
                        raise Exception(f"Critical step failed: {error_msg}")
                    else:
                        code_output.warnings.append(f"Non-critical step failed: {step.description}")
            
            # Finalize code output
            elapsed_time = timer.stop()
            code_output.execution_log.append(f"Completed execution in {elapsed_time:.2f} minutes")
            
            # Analyze generated code
            if os.path.exists(output_directory):
                analyzed_output = CodeAnalyzer.analyze_directory(output_directory)
                code_output.lines_of_code = analyzed_output.lines_of_code
                code_output.functions_created = analyzed_output.functions_created
                code_output.classes_created = analyzed_output.classes_created
                code_output.files_created.extend(analyzed_output.files_created)
                code_output.documentation_files.extend(analyzed_output.documentation_files)
            
            self.logger.info(f"Execution completed successfully in {elapsed_time:.2f} minutes", self.agent_type)
            
            # Store results in memory if available
            if self.memory:
                self.memory.remember_execution_result(execution_plan, code_output)
            
            return code_output
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            self.logger.error(error_msg, self.agent_type)
            raise
        finally:
            # Cleanup temporary directory
            if self.temp_directory:
                cleanup_temp_directory(self.temp_directory)
    
    def _execute_step(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> Dict[str, Any]:
        """Execute a single step and return results."""
        step_type = self._determine_step_type(step.description)
        
        if step_type == "analysis":
            return self._execute_analysis_step(step, plan, output_dir)
        elif step_type == "setup":
            return self._execute_setup_step(step, plan, output_dir)
        elif step_type == "implementation":
            return self._execute_implementation_step(step, plan, output_dir)
        elif step_type == "testing":
            return self._execute_testing_step(step, plan, output_dir)
        elif step_type == "documentation":
            return self._execute_documentation_step(step, plan, output_dir)
        else:
            return self._execute_generic_step(step, plan, output_dir)
    
    def _determine_step_type(self, description: str) -> str:
        """Determine the type of step based on its description."""
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in ['analyze', 'analysis', 'examine', 'study']):
            return "analysis"
        elif any(keyword in description_lower for keyword in ['setup', 'install', 'configure', 'initialize']):
            return "setup"
        elif any(keyword in description_lower for keyword in ['implement', 'create', 'build', 'develop', 'design']):
            return "implementation"
        elif any(keyword in description_lower for keyword in ['test', 'testing', 'validate', 'verify']):
            return "testing"
        elif any(keyword in description_lower for keyword in ['document', 'documentation', 'comment']):
            return "documentation"
        else:
            return "generic"
    
    def _execute_analysis_step(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> Dict[str, Any]:
        """Execute analysis step (understanding existing code)."""
        result = {
            "step_type": "analysis",
            "files_analyzed": [],
            "findings": []
        }
        
        # Look for existing code patterns
        current_dir = os.getcwd()
        
        # Find relevant files to analyze
        relevant_files = self._find_relevant_files(current_dir, plan.task_description)
        result["files_analyzed"] = relevant_files
        
        # Analyze patterns and conventions
        conventions = self._analyze_code_conventions(relevant_files)
        result["findings"] = conventions
        
        # Log findings
        analysis_log = f"Analysis completed: Found {len(relevant_files)} relevant files"
        self.execution_log.append(analysis_log)
        
        return result
    
    def _execute_setup_step(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> Dict[str, Any]:
        """Execute setup step (environment preparation)."""
        result = {
            "step_type": "setup",
            "actions_taken": [],
            "directories_created": []
        }
        
        # Create necessary directories
        directories_to_create = self._identify_required_directories(plan.task_description)
        
        for directory in directories_to_create:
            full_path = os.path.join(output_dir, directory)
            ensure_directory_exists(full_path)
            result["directories_created"].append(full_path)
            result["actions_taken"].append(f"Created directory: {directory}")
        
        # Create __init__.py files for Python packages
        if any(".py" in deliverable for deliverable in plan.deliverables):
            for directory in result["directories_created"]:
                init_file = os.path.join(directory, "__init__.py")
                if not os.path.exists(init_file):
                    with open(init_file, 'w') as f:
                        f.write('"""Package initialization."""\n')
                    result["actions_taken"].append(f"Created __init__.py in {directory}")
        
        return result
    
    def _execute_implementation_step(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> Dict[str, Any]:
        """Execute implementation step (actual code writing)."""
        result = {
            "step_type": "implementation",
            "files_created": [],
            "functions_implemented": [],
            "classes_implemented": []
        }
        
        # Determine what to implement based on step description and task
        implementation_type = self._determine_implementation_type(step.description, plan.task_description)
        
        if implementation_type == "api":
            files_created = self._implement_api_code(step, plan, output_dir)
        elif implementation_type == "database":
            files_created = self._implement_database_code(step, plan, output_dir)
        elif implementation_type == "frontend":
            files_created = self._implement_frontend_code(step, plan, output_dir)
        elif implementation_type == "utility":
            files_created = self._implement_utility_code(step, plan, output_dir)
        else:
            files_created = self._implement_generic_code(step, plan, output_dir)
        
        result["files_created"] = files_created
        
        # Analyze what was implemented
        for file_path in files_created:
            if os.path.exists(file_path):
                result["functions_implemented"].extend(self._extract_function_names(file_path))
                result["classes_implemented"].extend(self._extract_class_names(file_path))
        
        return result
    
    def _execute_testing_step(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> Dict[str, Any]:
        """Execute testing step (create and run tests)."""
        result = {
            "step_type": "testing",
            "test_files_created": [],
            "tests_passed": 0,
            "tests_failed": 0
        }
        
        # Create test files for implemented code
        test_files = self._create_test_files(plan, output_dir)
        result["test_files_created"] = test_files
        
        # Run tests if possible
        test_results = self._run_tests(test_files)
        result.update(test_results)
        
        return result
    
    def _execute_documentation_step(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> Dict[str, Any]:
        """Execute documentation step (create/update documentation)."""
        result = {
            "step_type": "documentation",
            "documentation_files": [],
            "comments_added": 0
        }
        
        # Add docstrings to Python files
        python_files = self._find_python_files(output_dir)
        for file_path in python_files:
            comments_added = self._add_documentation_to_file(file_path)
            result["comments_added"] += comments_added
        
        # Create README if needed
        if not os.path.exists(os.path.join(output_dir, "README.md")):
            readme_path = self._create_readme(plan, output_dir)
            if readme_path:
                result["documentation_files"].append(readme_path)
        
        return result
    
    def _execute_generic_step(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> Dict[str, Any]:
        """Execute a generic step when type cannot be determined."""
        return {
            "step_type": "generic",
            "description": step.description,
            "status": "completed",
            "notes": "Generic step executed based on description"
        }
    
    def _find_relevant_files(self, directory: str, task_description: str) -> List[str]:
        """Find files relevant to the current task."""
        relevant_files = []
        
        # Common file patterns to analyze
        patterns = ['*.py', '*.js', '*.ts', '*.java', '*.go', '*.rs']
        
        for pattern in patterns:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(pattern[1:]):  # Remove the *
                        full_path = os.path.join(root, file)
                        if self._is_file_relevant(full_path, task_description):
                            relevant_files.append(full_path)
                
                # Limit depth to avoid scanning too deep
                if len(relevant_files) > 20:
                    break
        
        return relevant_files[:10]  # Limit to 10 most relevant files
    
    def _is_file_relevant(self, file_path: str, task_description: str) -> bool:
        """Check if a file is relevant to the current task."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                task_lower = task_description.lower()
                
                # Look for task-related keywords in the file
                keywords = [word for word in task_lower.split() if len(word) > 3]
                relevance_score = sum(1 for keyword in keywords if keyword in content)
                
                return relevance_score > 0
        except:
            return False
    
    def _analyze_code_conventions(self, files: List[str]) -> List[str]:
        """Analyze code conventions from existing files."""
        conventions = []
        
        if not files:
            return ["No existing files found for convention analysis"]
        
        # Analyze Python files for conventions
        python_files = [f for f in files if f.endswith('.py')]
        if python_files:
            conventions.extend(self._analyze_python_conventions(python_files))
        
        # Analyze JavaScript files for conventions
        js_files = [f for f in files if f.endswith(('.js', '.ts'))]
        if js_files:
            conventions.extend(self._analyze_js_conventions(js_files))
        
        return conventions or ["Standard coding conventions will be applied"]
    
    def _analyze_python_conventions(self, files: List[str]) -> List[str]:
        """Analyze Python-specific conventions."""
        conventions = []
        
        try:
            sample_file = files[0]
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for docstring style
            if '"""' in content:
                conventions.append("Using triple-quote docstrings")
            
            # Check for type hints
            if '->' in content or ': ' in content:
                conventions.append("Using type hints")
            
            # Check for naming conventions
            if any(word.startswith('_') for word in content.split()):
                conventions.append("Using underscore for private members")
                
        except:
            pass
        
        return conventions or ["Standard Python conventions"]
    
    def _analyze_js_conventions(self, files: List[str]) -> List[str]:
        """Analyze JavaScript/TypeScript conventions."""
        conventions = []
        
        try:
            sample_file = files[0]
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'const ' in content:
                conventions.append("Preferring const declarations")
            
            if '=>' in content:
                conventions.append("Using arrow functions")
                
        except:
            pass
        
        return conventions or ["Standard JavaScript conventions"]
    
    def _identify_required_directories(self, task_description: str) -> List[str]:
        """Identify directories that need to be created."""
        directories = []
        description_lower = task_description.lower()
        
        # API-related directories
        if any(keyword in description_lower for keyword in ['api', 'endpoint', 'route']):
            directories.extend(['routes', 'controllers', 'middleware'])
        
        # Database-related directories
        if any(keyword in description_lower for keyword in ['database', 'model', 'schema']):
            directories.extend(['models', 'migrations'])
        
        # Testing directories
        if 'test' in description_lower:
            directories.append('tests')
        
        # Frontend directories
        if any(keyword in description_lower for keyword in ['frontend', 'ui', 'component']):
            directories.extend(['components', 'styles'])
        
        # Default directories
        if not directories:
            directories = ['src', 'tests']
        
        return list(set(directories))  # Remove duplicates
    
    def _determine_implementation_type(self, step_description: str, task_description: str) -> str:
        """Determine what type of code to implement."""
        combined = (step_description + " " + task_description).lower()
        
        if any(keyword in combined for keyword in ['api', 'endpoint', 'route', 'handler']):
            return "api"
        elif any(keyword in combined for keyword in ['database', 'model', 'schema', 'migration']):
            return "database"
        elif any(keyword in combined for keyword in ['frontend', 'ui', 'component', 'interface']):
            return "frontend"
        elif any(keyword in combined for keyword in ['utility', 'helper', 'util', 'common']):
            return "utility"
        else:
            return "generic"
    
    def _implement_api_code(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> List[str]:
        """Implement API-related code."""
        files_created = []
        
        # Create a simple API handler
        api_file = os.path.join(output_dir, "api_handler.py")
        api_content = self._generate_api_code_template(plan.task_description)
        
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(api_content)
        
        files_created.append(api_file)
        return files_created
    
    def _implement_database_code(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> List[str]:
        """Implement database-related code."""
        files_created = []
        
        # Create a database model
        model_file = os.path.join(output_dir, "models.py")
        model_content = self._generate_database_code_template(plan.task_description)
        
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(model_content)
        
        files_created.append(model_file)
        return files_created
    
    def _implement_frontend_code(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> List[str]:
        """Implement frontend-related code."""
        files_created = []
        
        # Create a component file
        component_file = os.path.join(output_dir, "component.js")
        component_content = self._generate_frontend_code_template(plan.task_description)
        
        with open(component_file, 'w', encoding='utf-8') as f:
            f.write(component_content)
        
        files_created.append(component_file)
        return files_created
    
    def _implement_utility_code(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> List[str]:
        """Implement utility code."""
        files_created = []
        
        # Create a utility file
        util_file = os.path.join(output_dir, "utils.py")
        util_content = self._generate_utility_code_template(plan.task_description)
        
        with open(util_file, 'w', encoding='utf-8') as f:
            f.write(util_content)
        
        files_created.append(util_file)
        return files_created
    
    def _implement_generic_code(self, step: TaskStep, plan: ExecutionPlan, output_dir: str) -> List[str]:
        """Implement generic code when type is unclear."""
        files_created = []
        
        # Create a main implementation file
        main_file = os.path.join(output_dir, "main.py")
        main_content = self._generate_generic_code_template(plan.task_description, step.description)
        
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(main_content)
        
        files_created.append(main_file)
        return files_created
    
    def _generate_api_code_template(self, task_description: str) -> str:
        """Generate API code template."""
        return '''"""
API Handler Implementation

Generated for task: {task_description}
"""

from typing import Dict, Any, Optional
import json


class APIHandler:
    """Main API handler class."""
    
    def __init__(self):
        self.routes = {{}}
    
    def handle_request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle incoming API request.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: Request path
            data: Request data
            
        Returns:
            Response dictionary
        """
        try:
            # Route handling logic
            if path in self.routes:
                handler = self.routes[path].get(method.lower())
                if handler:
                    return handler(data)
            
            return {{
                "status": "error",
                "message": f"Route {{method}} {{path}} not found",
                "code": 404
            }}
        
        except Exception as e:
            return {{
                "status": "error",
                "message": str(e),
                "code": 500
            }}
    
    def register_route(self, method: str, path: str, handler):
        """Register a new route handler."""
        if path not in self.routes:
            self.routes[path] = {{}}
        self.routes[path][method.lower()] = handler


def create_api_handler() -> APIHandler:
    """Factory function to create API handler."""
    handler = APIHandler()
    
    # Register default routes
    def health_check(data):
        return {{"status": "ok", "message": "API is running"}}
    
    handler.register_route("GET", "/health", health_check)
    
    return handler


if __name__ == "__main__":
    api = create_api_handler()
    print("API handler created successfully")
'''.format(task_description=task_description)
    
    def _generate_database_code_template(self, task_description: str) -> str:
        """Generate database code template."""
        return '''"""
Database Models Implementation

Generated for task: {task_description}
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class DatabaseModel:
    """Base database model class."""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {{
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }}
    
    def save(self):
        """Save model to database."""
        self.updated_at = datetime.now()
        # Database save logic would go here
        pass
    
    @classmethod
    def find_by_id(cls, model_id: int):
        """Find model by ID."""
        # Database query logic would go here
        pass


class DataManager:
    """Database operations manager."""
    
    def __init__(self):
        self.connection = None
    
    def connect(self, connection_string: str):
        """Connect to database."""
        # Database connection logic would go here
        self.connection = connection_string
        return True
    
    def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Execute database query."""
        # Query execution logic would go here
        return []
    
    def create_tables(self):
        """Create database tables."""
        # Table creation logic would go here
        pass


if __name__ == "__main__":
    manager = DataManager()
    print("Database models created successfully")
'''.format(task_description=task_description)
    
    def _generate_frontend_code_template(self, task_description: str) -> str:
        """Generate frontend code template."""
        return '''/**
 * Frontend Component Implementation
 * 
 * Generated for task: {task_description}
 */

class Component {{
    constructor(element, options = {{}}) {{
        this.element = element;
        this.options = {{
            ...this.defaultOptions(),
            ...options
        }};
        this.init();
    }}
    
    defaultOptions() {{
        return {{
            className: 'component',
            autoInit: true
        }};
    }}
    
    init() {{
        this.setupElements();
        this.bindEvents();
        if (this.options.autoInit) {{
            this.render();
        }}
    }}
    
    setupElements() {{
        this.element.classList.add(this.options.className);
    }}
    
    bindEvents() {{
        this.element.addEventListener('click', (e) => {{
            this.handleClick(e);
        }});
    }}
    
    handleClick(event) {{
        console.log('Component clicked', event);
    }}
    
    render() {{
        // Rendering logic goes here
        this.element.innerHTML = '<div>Component rendered successfully</div>';
    }}
    
    destroy() {{
        // Cleanup logic
        this.element.removeEventListener('click', this.handleClick);
    }}
}}

// Factory function
function createComponent(selector, options) {{
    const element = document.querySelector(selector);
    if (element) {{
        return new Component(element, options);
    }}
    return null;
}}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = {{ Component, createComponent }};
}}
'''.format(task_description=task_description)
    
    def _generate_utility_code_template(self, task_description: str) -> str:
        """Generate utility code template."""
        return '''"""
Utility Functions Implementation

Generated for task: {task_description}
"""

from typing import Any, Dict, List, Optional, Union
import json
import datetime


def format_datetime(dt: datetime.datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime object to string."""
    return dt.strftime(format_string)


def parse_json_safe(json_string: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON string."""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None


def sanitize_string(input_string: str) -> str:
    """Sanitize string input for safe processing."""
    if not isinstance(input_string, str):
        return str(input_string)
    
    # Remove potentially harmful characters
    sanitized = input_string.replace('<', '&lt;').replace('>', '&gt;')
    sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
    
    return sanitized.strip()


def validate_email(email: str) -> bool:
    """Basic email validation."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def deep_merge_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def chunk_list(input_list: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    chunks = []
    for i in range(0, len(input_list), chunk_size):
        chunks.append(input_list[i:i + chunk_size])
    return chunks


class ConfigManager:
    """Simple configuration manager."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = {{}}
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Config file {{config_file}} not found")
        except json.JSONDecodeError:
            print(f"Invalid JSON in config file {{config_file}}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {{}}
            config = config[k]
        
        config[keys[-1]] = value


if __name__ == "__main__":
    print("Utility functions loaded successfully")
'''.format(task_description=task_description)
    
    def _generate_generic_code_template(self, task_description: str, step_description: str) -> str:
        """Generate generic code template."""
        return '''"""
Generic Implementation

Task: {task_description}
Step: {step_description}
"""

from typing import Any, Dict, List, Optional


class Implementation:
    """Main implementation class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {{}}
        self.setup()
    
    def setup(self):
        """Set up the implementation."""
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the main functionality."""
        # Implementation logic goes here
        return "Implementation completed successfully"
    
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        return True
    
    def cleanup(self):
        """Clean up resources."""
        pass


def main():
    """Main entry point."""
    impl = Implementation()
    result = impl.execute()
    print(f"Result: {{result}}")
    impl.cleanup()


if __name__ == "__main__":
    main()
'''.format(task_description=task_description, step_description=step_description)
    
    def _extract_function_names(self, file_path: str) -> List[str]:
        """Extract function names from a Python file."""
        functions = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            import re
            # Find function definitions
            pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            matches = re.findall(pattern, content)
            functions.extend(matches)
        except:
            pass
        
        return functions
    
    def _extract_class_names(self, file_path: str) -> List[str]:
        """Extract class names from a Python file."""
        classes = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            import re
            # Find class definitions
            pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[(\:]'
            matches = re.findall(pattern, content)
            classes.extend(matches)
        except:
            pass
        
        return classes
    
    def _create_test_files(self, plan: ExecutionPlan, output_dir: str) -> List[str]:
        """Create test files for the implemented code."""
        test_files = []
        
        # Find Python files to test
        python_files = self._find_python_files(output_dir)
        
        for py_file in python_files[:3]:  # Limit to first 3 files
            test_file_path = self._create_test_file(py_file, output_dir)
            if test_file_path:
                test_files.append(test_file_path)
        
        return test_files
    
    def _find_python_files(self, directory: str) -> List[str]:
        """Find Python files in directory."""
        python_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def _create_test_file(self, source_file: str, output_dir: str) -> Optional[str]:
        """Create a test file for a source file."""
        try:
            # Generate test file name
            basename = os.path.basename(source_file)
            test_name = f"test_{basename}"
            test_path = os.path.join(output_dir, "tests", test_name)
            
            # Ensure test directory exists
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            
            # Generate test content
            test_content = self._generate_test_content(source_file)
            
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            return test_path
        except:
            return None
    
    def _generate_test_content(self, source_file: str) -> str:
        """Generate test content for a source file."""
        module_name = os.path.splitext(os.path.basename(source_file))[0]
        
        return f'''"""
Tests for {module_name} module.
"""

import unittest
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import {module_name}
except ImportError as e:
    print(f"Warning: Could not import {{module_name}}: {{e}}")
    {module_name} = None


class Test{module_name.title()}(unittest.TestCase):
    """Test cases for {module_name} module."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_module_import(self):
        """Test that the module can be imported."""
        self.assertIsNotNone({module_name}, "Module should be importable")
    
    def test_basic_functionality(self):
        """Test basic functionality exists."""
        # This is a placeholder test
        # Add specific tests based on the actual implementation
        self.assertTrue(True, "Basic functionality test")
    
    @unittest.skipIf({module_name} is None, "Module not available")
    def test_module_has_expected_attributes(self):
        """Test that module has expected attributes."""
        # Add tests for specific classes, functions, or variables
        # Example:
        # self.assertTrue(hasattr({module_name}, 'SomeClass'))
        # self.assertTrue(hasattr({module_name}, 'some_function'))
        pass


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
'''
    
    def _run_tests(self, test_files: List[str]) -> Dict[str, Any]:
        """Run test files and return results."""
        results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_output": []
        }
        
        for test_file in test_files:
            try:
                # Run the test file
                result = subprocess.run([
                    'python', '-m', 'pytest', test_file, '-v'
                ], capture_output=True, text=True, timeout=30)
                
                output = result.stdout + result.stderr
                results["test_output"].append(f"Test file: {test_file}\n{output}")
                
                if result.returncode == 0:
                    results["tests_passed"] += 1
                else:
                    results["tests_failed"] += 1
                    
                results["tests_run"] += 1
                
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fallback to basic unittest if pytest is not available
                try:
                    result = subprocess.run([
                        'python', test_file
                    ], capture_output=True, text=True, timeout=30)
                    
                    output = result.stdout + result.stderr
                    results["test_output"].append(f"Test file: {test_file}\n{output}")
                    
                    if "FAILED" not in output and "ERROR" not in output:
                        results["tests_passed"] += 1
                    else:
                        results["tests_failed"] += 1
                        
                    results["tests_run"] += 1
                    
                except:
                    results["test_output"].append(f"Could not run test file: {test_file}")
                    results["tests_failed"] += 1
        
        return results
    
    def _add_documentation_to_file(self, file_path: str) -> int:
        """Add documentation to a Python file."""
        comments_added = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            modified_lines = []
            i = 0
            
            while i < len(lines):
                line = lines[i]
                modified_lines.append(line)
                
                # Add docstring to functions without them
                if line.strip().startswith('def ') and ':' in line:
                    # Check if next few lines contain docstring
                    has_docstring = False
                    for j in range(i + 1, min(i + 4, len(lines))):
                        if '"""' in lines[j] or "'''" in lines[j]:
                            has_docstring = True
                            break
                    
                    if not has_docstring:
                        function_name = line.strip().split('(')[0].replace('def ', '')
                        docstring = f'        """{function_name.replace("_", " ").title()}."""\n'
                        modified_lines.append(docstring)
                        comments_added += 1
                
                # Add docstring to classes without them
                elif line.strip().startswith('class ') and ':' in line:
                    # Check if next few lines contain docstring
                    has_docstring = False
                    for j in range(i + 1, min(i + 4, len(lines))):
                        if '"""' in lines[j] or "'''" in lines[j]:
                            has_docstring = True
                            break
                    
                    if not has_docstring:
                        class_name = line.strip().split('(')[0].replace('class ', '').rstrip(':')
                        docstring = f'    """{class_name} class."""\n'
                        modified_lines.append(docstring)
                        comments_added += 1
                
                i += 1
            
            # Write back if comments were added
            if comments_added > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(modified_lines)
                    
        except:
            pass
        
        return comments_added
    
    def _create_readme(self, plan: ExecutionPlan, output_dir: str) -> Optional[str]:
        """Create a README.md file."""
        try:
            readme_path = os.path.join(output_dir, "README.md")
            
            # Format datetime outside of f-string to avoid backslash issues
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format steps, deliverables, and risks outside of f-string
            steps_text = "\n".join([f"{i+1}. {step.description}" for i, step in enumerate(plan.steps)])
            deliverables_text = "\n".join([f"- {deliverable}" for deliverable in plan.deliverables])
            risks_text = "\n".join([f"- {risk}" for risk in plan.risks])
            
            readme_content = """# Implementation

## Overview
{task_description}

## Task Analysis
{task_analysis}

## Implementation Steps
{steps}

## Deliverables
{deliverables}

## Potential Risks
{risks}

## Generated Files
This implementation was generated by the Claude Code Pipeline System.

## Usage
To use this implementation:
1. Review the generated code
2. Run the tests
3. Integrate with your existing codebase

## Testing
Run tests using:
```bash
python -m pytest tests/
```

Generated on: {current_time}
""".format(
                task_description=plan.task_description,
                task_analysis=plan.task_analysis,
                steps=steps_text,
                deliverables=deliverables_text,
                risks=risks_text,
                current_time=current_time
            )
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            return readme_path
            
        except:
            return None
    
    def _update_code_output(self, code_output: CodeOutput, step_result: Dict[str, Any], step: TaskStep):
        """Update code output with step results."""
        step_type = step_result.get("step_type", "unknown")
        
        # Update files created
        if "files_created" in step_result:
            code_output.files_created.extend(step_result["files_created"])
        
        if "test_files_created" in step_result:
            code_output.tests_created += len(step_result["test_files_created"])
        
        if "documentation_files" in step_result:
            code_output.documentation_files.extend(step_result["documentation_files"])
        
        # Update execution log
        log_entry = f"Completed {step_type} step: {step.description}"
        code_output.execution_log.append(log_entry)
        self.execution_log.append(log_entry)
        
        # Update function and class counts
        if "functions_implemented" in step_result:
            code_output.functions_created += len(step_result["functions_implemented"])
        
        if "classes_implemented" in step_result:
            code_output.classes_created += len(step_result["classes_implemented"])