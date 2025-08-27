"""
Planning Agent for the Three-Agent Pipeline System.

This agent analyzes tasks and creates detailed execution plans.
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from shared.models import (
    ExecutionPlan, TaskStep, Priority, TaskStatus, AgentType
)
from shared.utils import PipelineLogger, TaskTimer
from shared.memory import PipelineMemoryManager


class PlanningAgent:
    """
    Agent responsible for analyzing tasks and creating execution plans.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 memory_manager: Optional[PipelineMemoryManager] = None):
        self.config = config or {}
        self.logger = PipelineLogger()
        self.agent_type = AgentType.PLANNER
        self.memory = memory_manager
        
    def create_plan(self, task_description: str, requirements: Optional[List[str]] = None) -> ExecutionPlan:
        """
        Create a detailed execution plan from task description.
        
        Args:
            task_description: The main task to be executed
            requirements: Optional list of specific requirements
            
        Returns:
            ExecutionPlan object with detailed steps and metadata
        """
        self.logger.info(f"Creating execution plan for: {task_description[:100]}...", self.agent_type)
        
        timer = TaskTimer()
        timer.start()
        
        try:
            # Get insights from memory if available
            memory_insights = {}
            if self.memory:
                memory_insights = self.memory.get_planning_insights(task_description)
                self.logger.info(f"Retrieved planning insights: {memory_insights.get('similar_tasks_count', 0)} similar tasks found", self.agent_type)
            
            # Analyze the task (enhanced with memory)
            task_analysis = self._analyze_task(task_description, requirements, memory_insights)
            
            # Break down into steps (with memory-informed patterns)
            steps = self._create_execution_steps(task_description, task_analysis, memory_insights)
            
            # Identify deliverables
            deliverables = self._identify_deliverables(task_description, steps)
            
            # Assess risks (enhanced with historical data)
            risks = self._assess_risks(task_description, steps, memory_insights)
            
            # Calculate total time (with memory-based adjustments)
            total_time = sum(step.estimated_time_minutes for step in steps)
            if memory_insights.get('time_estimation_adjustment'):
                total_time = int(total_time * memory_insights['time_estimation_adjustment'])
            
            # Create execution plan
            plan = ExecutionPlan(
                task_description=task_description,
                task_analysis=task_analysis,
                steps=steps,
                deliverables=deliverables,
                risks=risks,
                total_estimated_time=total_time,
                created_by=self.agent_type
            )
            
            elapsed_time = timer.stop()
            self.logger.info(f"Plan created in {elapsed_time:.2f} minutes with {len(steps)} steps", self.agent_type)
            
            # Store results in memory if available
            if self.memory:
                self.memory.remember_planning_result(task_description, plan)
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create execution plan: {str(e)}", self.agent_type)
            raise
    
    def _analyze_task(self, task_description: str, requirements: Optional[List[str]] = None, 
                     memory_insights: Dict[str, Any] = None) -> str:
        """Analyze the task to understand its scope and complexity."""
        
        analysis_points = []
        
        # Detect task type
        task_type = self._detect_task_type(task_description)
        analysis_points.append(f"Task Type: {task_type}")
        
        # Analyze complexity
        complexity = self._assess_complexity(task_description)
        analysis_points.append(f"Complexity: {complexity}")
        
        # Identify key components
        components = self._identify_components(task_description)
        if components:
            analysis_points.append(f"Key Components: {', '.join(components)}")
        
        # Identify technologies/frameworks
        technologies = self._identify_technologies(task_description)
        if technologies:
            analysis_points.append(f"Technologies: {', '.join(technologies)}")
        
        # Add requirements analysis
        if requirements:
            analysis_points.append(f"Specific Requirements: {len(requirements)} items specified")
            
        # Add memory insights if available
        if memory_insights:
            if memory_insights.get('similar_tasks_count', 0) > 0:
                analysis_points.append(f"Historical Context: {memory_insights['similar_tasks_count']} similar tasks in memory")
            if memory_insights.get('average_estimation_accuracy'):
                acc = memory_insights['average_estimation_accuracy']
                analysis_points.append(f"Estimation Confidence: {acc:.1%} based on history")
        
        return ". ".join(analysis_points)
    
    def _detect_task_type(self, description: str) -> str:
        """Detect the type of development task."""
        description_lower = description.lower()
        
        # API/Backend patterns
        if any(keyword in description_lower for keyword in ['api', 'endpoint', 'rest', 'graphql', 'backend', 'server']):
            return "API/Backend Development"
        
        # Frontend patterns
        if any(keyword in description_lower for keyword in ['frontend', 'ui', 'interface', 'react', 'vue', 'angular', 'component']):
            return "Frontend Development"
        
        # Database patterns
        if any(keyword in description_lower for keyword in ['database', 'sql', 'mongodb', 'schema', 'migration']):
            return "Database Development"
        
        # Testing patterns
        if any(keyword in description_lower for keyword in ['test', 'testing', 'unittest', 'integration']):
            return "Testing"
        
        # Refactoring patterns
        if any(keyword in description_lower for keyword in ['refactor', 'optimize', 'improve', 'restructure']):
            return "Refactoring/Optimization"
        
        # Bug fix patterns
        if any(keyword in description_lower for keyword in ['fix', 'bug', 'error', 'issue', 'problem']):
            return "Bug Fix"
        
        # Feature development
        if any(keyword in description_lower for keyword in ['feature', 'functionality', 'implement', 'add', 'create']):
            return "Feature Development"
        
        return "General Development"
    
    def _assess_complexity(self, description: str) -> str:
        """Assess the complexity of the task."""
        description_lower = description.lower()
        
        complexity_indicators = {
            'high': ['complex', 'advanced', 'multiple', 'integration', 'system', 'architecture', 'scalable'],
            'medium': ['implement', 'create', 'build', 'develop', 'design', 'several'],
            'low': ['simple', 'basic', 'single', 'small', 'quick', 'minor', 'fix']
        }
        
        scores = {'high': 0, 'medium': 0, 'low': 0}
        
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in description_lower:
                    scores[level] += 1
        
        # Determine complexity based on scores
        if scores['high'] > scores['medium'] and scores['high'] > scores['low']:
            return "High"
        elif scores['low'] > scores['medium'] and scores['low'] > scores['high']:
            return "Low"
        else:
            return "Medium"
    
    def _identify_components(self, description: str) -> List[str]:
        """Identify key components mentioned in the task."""
        components = []
        description_lower = description.lower()
        
        component_keywords = {
            'Authentication': ['auth', 'login', 'authentication', 'user management'],
            'Database': ['database', 'sql', 'mongodb', 'data storage'],
            'API': ['api', 'endpoint', 'rest', 'graphql'],
            'Frontend': ['frontend', 'ui', 'interface', 'web page'],
            'Testing': ['test', 'testing', 'unit test', 'integration test'],
            'Configuration': ['config', 'configuration', 'settings'],
            'Security': ['security', 'authorization', 'permissions'],
            'File Handling': ['file', 'upload', 'download', 'storage'],
            'Validation': ['validation', 'validate', 'check'],
            'Error Handling': ['error', 'exception', 'error handling']
        }
        
        for component, keywords in component_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                components.append(component)
        
        return components
    
    def _identify_technologies(self, description: str) -> List[str]:
        """Identify technologies/frameworks mentioned in the task."""
        technologies = []
        description_lower = description.lower()
        
        tech_keywords = {
            'Python': ['python', 'flask', 'django', 'fastapi', 'sqlalchemy'],
            'JavaScript': ['javascript', 'js', 'node', 'express', 'react', 'vue', 'angular'],
            'Database': ['postgresql', 'mysql', 'mongodb', 'redis', 'sqlite'],
            'Cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'Testing': ['pytest', 'unittest', 'jest', 'mocha'],
            'Web': ['html', 'css', 'bootstrap', 'tailwind']
        }
        
        for tech, keywords in tech_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                technologies.append(tech)
        
        return technologies
    
    def _create_execution_steps(self, task_description: str, task_analysis: str, 
                               memory_insights: Dict[str, Any] = None) -> List[TaskStep]:
        """Create detailed execution steps based on task analysis."""
        steps = []
        
        # Always start with setup/analysis steps
        steps.extend(self._create_setup_steps(task_description))
        
        # Add task-specific implementation steps
        steps.extend(self._create_implementation_steps(task_description, task_analysis))
        
        # Add testing and validation steps
        steps.extend(self._create_testing_steps(task_description))
        
        # Add documentation steps
        steps.extend(self._create_documentation_steps(task_description))
        
        # Set dependencies
        self._set_step_dependencies(steps)
        
        return steps
    
    def _create_setup_steps(self, task_description: str) -> List[TaskStep]:
        """Create initial setup and analysis steps."""
        steps = []
        
        # Always include codebase analysis
        steps.append(TaskStep(
            description="Analyze existing codebase structure and patterns",
            priority=Priority.HIGH,
            estimated_time_minutes=10,
            notes="Understanding existing code is crucial for consistent implementation"
        ))
        
        # Environment setup if needed
        if any(keyword in task_description.lower() for keyword in ['new', 'create', 'setup', 'initialize']):
            steps.append(TaskStep(
                description="Set up development environment and dependencies",
                priority=Priority.MEDIUM,
                estimated_time_minutes=15,
                notes="May include package installation and configuration"
            ))
        
        return steps
    
    def _create_implementation_steps(self, task_description: str, task_analysis: str) -> List[TaskStep]:
        """Create implementation-specific steps based on task type."""
        steps = []
        description_lower = task_description.lower()
        
        # API/Backend implementation steps
        if 'api' in description_lower or 'backend' in description_lower:
            steps.extend([
                TaskStep(
                    description="Design API endpoints and data models",
                    priority=Priority.HIGH,
                    estimated_time_minutes=20
                ),
                TaskStep(
                    description="Implement core business logic and handlers",
                    priority=Priority.HIGH,
                    estimated_time_minutes=30
                ),
                TaskStep(
                    description="Add input validation and error handling",
                    priority=Priority.MEDIUM,
                    estimated_time_minutes=15
                )
            ])
        
        # Database implementation steps
        if any(keyword in description_lower for keyword in ['database', 'model', 'schema']):
            steps.extend([
                TaskStep(
                    description="Design database schema and relationships",
                    priority=Priority.HIGH,
                    estimated_time_minutes=25
                ),
                TaskStep(
                    description="Implement database models and migrations",
                    priority=Priority.HIGH,
                    estimated_time_minutes=20
                )
            ])
        
        # Frontend implementation steps
        if 'frontend' in description_lower or 'ui' in description_lower:
            steps.extend([
                TaskStep(
                    description="Design user interface and component structure",
                    priority=Priority.HIGH,
                    estimated_time_minutes=25
                ),
                TaskStep(
                    description="Implement UI components and styling",
                    priority=Priority.MEDIUM,
                    estimated_time_minutes=35
                ),
                TaskStep(
                    description="Add client-side validation and interactions",
                    priority=Priority.MEDIUM,
                    estimated_time_minutes=20
                )
            ])
        
        # Generic implementation step if no specific type detected
        if not steps:
            steps.append(TaskStep(
                description="Implement core functionality as specified",
                priority=Priority.HIGH,
                estimated_time_minutes=30,
                notes="Primary implementation step"
            ))
        
        return steps
    
    def _create_testing_steps(self, task_description: str) -> List[TaskStep]:
        """Create testing and validation steps."""
        steps = []
        
        # Always include basic testing
        steps.append(TaskStep(
            description="Write and execute unit tests for core functionality",
            priority=Priority.HIGH,
            estimated_time_minutes=20,
            notes="Essential for code quality and reliability"
        ))
        
        # Add integration testing for complex tasks
        if any(keyword in task_description.lower() for keyword in ['api', 'system', 'integration', 'multiple']):
            steps.append(TaskStep(
                description="Perform integration testing with existing systems",
                priority=Priority.MEDIUM,
                estimated_time_minutes=15,
                notes="Ensure new code works with existing codebase"
            ))
        
        return steps
    
    def _create_documentation_steps(self, task_description: str) -> List[TaskStep]:
        """Create documentation steps."""
        return [
            TaskStep(
                description="Update code documentation and comments",
                priority=Priority.MEDIUM,
                estimated_time_minutes=10,
                notes="Maintain code documentation for future developers"
            )
        ]
    
    def _set_step_dependencies(self, steps: List[TaskStep]):
        """Set dependencies between steps based on logical order."""
        for i, step in enumerate(steps):
            if i > 0:
                # Most steps depend on the previous step
                if 'test' not in step.description.lower() or i < len(steps) - 2:
                    step.dependencies = [steps[i-1].id]
                elif 'test' in step.description.lower():
                    # Testing steps depend on implementation steps
                    implementation_steps = [s for s in steps[:i] 
                                          if 'implement' in s.description.lower() or 'create' in s.description.lower()]
                    if implementation_steps:
                        step.dependencies = [implementation_steps[-1].id]
    
    def _identify_deliverables(self, task_description: str, steps: List[TaskStep]) -> List[str]:
        """Identify expected deliverables based on task and steps."""
        deliverables = []
        description_lower = task_description.lower()
        
        # Code deliverables
        if any(keyword in description_lower for keyword in ['implement', 'create', 'build', 'develop']):
            deliverables.append("Implemented code following project conventions")
        
        # API deliverables
        if 'api' in description_lower:
            deliverables.extend([
                "Functional API endpoints",
                "API documentation or OpenAPI specification"
            ])
        
        # Database deliverables
        if any(keyword in description_lower for keyword in ['database', 'model', 'schema']):
            deliverables.extend([
                "Database schema and models",
                "Data migration scripts if needed"
            ])
        
        # Testing deliverables
        if any('test' in step.description.lower() for step in steps):
            deliverables.append("Comprehensive unit and integration tests")
        
        # Documentation deliverables
        if any('document' in step.description.lower() for step in steps):
            deliverables.append("Updated code documentation and comments")
        
        # Default deliverables
        if not deliverables:
            deliverables = [
                "Working implementation of requested functionality",
                "Clean, well-documented code"
            ]
        
        return deliverables
    
    def _assess_risks(self, task_description: str, steps: List[TaskStep], 
                     memory_insights: Dict[str, Any] = None) -> List[str]:
        """Identify potential risks and challenges."""
        risks = []
        description_lower = task_description.lower()
        
        # Complexity risks
        if len(steps) > 8:
            risks.append("High complexity may require additional time and careful coordination")
        
        # Integration risks
        if any(keyword in description_lower for keyword in ['integration', 'existing', 'system']):
            risks.append("Integration with existing code may reveal unexpected dependencies")
        
        # Database risks
        if any(keyword in description_lower for keyword in ['database', 'migration', 'schema']):
            risks.append("Database changes may require careful migration planning")
        
        # API risks
        if 'api' in description_lower:
            risks.append("API changes may affect existing clients or systems")
        
        # Performance risks
        if any(keyword in description_lower for keyword in ['performance', 'scale', 'optimize']):
            risks.append("Performance requirements may need iterative optimization")
        
        # Security risks
        if any(keyword in description_lower for keyword in ['auth', 'security', 'user', 'access']):
            risks.append("Security implementation requires careful review and testing")
        
        # Add memory-informed risks
        if memory_insights and memory_insights.get('common_risks'):
            risks.extend(memory_insights['common_risks'])
        
        # Default risks
        if not risks:
            risks = [
                "Standard development risks: requirement changes, technical challenges",
                "Code quality issues if best practices are not followed"
            ]
        
        return risks
    
    def validate_plan(self, plan: ExecutionPlan) -> bool:
        """Validate that the execution plan is complete and logical."""
        try:
            # Check required fields
            if not plan.task_description or not plan.steps:
                return False
            
            # Check step dependencies
            step_ids = {step.id for step in plan.steps}
            for step in plan.steps:
                if step.dependencies:
                    if not all(dep_id in step_ids for dep_id in step.dependencies):
                        return False
            
            # Check time estimates
            if plan.total_estimated_time <= 0:
                return False
            
            return True
            
        except Exception:
            return False