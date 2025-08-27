"""
Review Agent for the Three-Agent Pipeline System.

This agent reviews code output from the Execution Agent and provides quality assessment.
"""

import os
import ast
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from shared.models import (
    CodeOutput, ReviewReport, ReviewIssue, AgentType
)
from shared.utils import PipelineLogger, TaskTimer, ConfigLoader
from shared.memory import PipelineMemoryManager


class ReviewAgent:
    """
    Agent responsible for reviewing and validating implemented code.
    """
    
    def __init__(self, review_rules_path: str, config: Optional[Dict[str, Any]] = None,
                 memory_manager: Optional[PipelineMemoryManager] = None):
        self.config = config or {}
        self.logger = PipelineLogger()
        self.agent_type = AgentType.REVIEWER
        self.review_rules = ConfigLoader.load_json(review_rules_path)
        self.memory = memory_manager
        
    def review_code(self, code_output: CodeOutput, code_directory: str) -> ReviewReport:
        """
        Review code output and generate a comprehensive review report.
        
        Args:
            code_output: CodeOutput object with execution results
            code_directory: Directory containing the generated code
            
        Returns:
            ReviewReport with assessment and recommendations
        """
        self.logger.info(f"Starting code review for {len(code_output.files_created)} files", self.agent_type)
        
        timer = TaskTimer()
        timer.start()
        
        try:
            # Initialize review report
            review_report = ReviewReport(
                code_output_analyzed=code_output,
                reviewed_by=self.agent_type
            )
            
            # Review each file
            all_issues = []
            for file_path in code_output.files_created:
                if os.path.exists(file_path):
                    file_issues = self._review_file(file_path)
                    all_issues.extend(file_issues)
            
            # Analyze overall code structure
            structure_issues = self._review_code_structure(code_directory)
            all_issues.extend(structure_issues)
            
            # Check security concerns
            security_issues = self._review_security(code_directory)
            all_issues.extend(security_issues)
            
            # Assess performance
            performance_issues = self._review_performance(code_directory)
            all_issues.extend(performance_issues)
            
            # Set issues in report
            review_report.issues = all_issues
            
            # Calculate overall score
            review_report.overall_score = self._calculate_overall_score(all_issues)
            review_report.grade = self._determine_grade(review_report.overall_score)
            review_report.approval_status = self._determine_approval_status(review_report.overall_score, all_issues)
            
            # Generate strengths and recommendations
            review_report.strengths = self._identify_strengths(code_output, code_directory)
            review_report.recommendations = self._generate_recommendations(all_issues, code_output)
            
            # Set review time
            review_report.review_time_minutes = timer.stop()
            
            self.logger.info(f"Review completed with score {review_report.overall_score:.1f}/100", self.agent_type)
            
            # Store results in memory if available
            if self.memory:
                self.memory.remember_review_result(code_output, review_report)
            
            return review_report
            
        except Exception as e:
            self.logger.error(f"Review failed: {str(e)}", self.agent_type)
            raise
    
    def _review_file(self, file_path: str) -> List[ReviewIssue]:
        """Review a single file and return issues found."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check file-specific issues
            if file_path.endswith('.py'):
                issues.extend(self._review_python_file(file_path, content))
            elif file_path.endswith(('.js', '.ts')):
                issues.extend(self._review_javascript_file(file_path, content))
            else:
                issues.extend(self._review_generic_file(file_path, content))
            
            # Check common issues for all files
            issues.extend(self._check_common_issues(file_path, content))
            
        except Exception as e:
            issues.append(ReviewIssue(
                file_path=file_path,
                category="file_access",
                severity="medium",
                description=f"Could not read file: {str(e)}",
                suggestion="Ensure file exists and has proper permissions"
            ))
        
        return issues
    
    def _review_python_file(self, file_path: str, content: str) -> List[ReviewIssue]:
        """Review Python-specific issues."""
        issues = []
        lines = content.split('\\n')
        
        try:
            # Parse AST for deeper analysis
            tree = ast.parse(content)
            
            # Check function complexity
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    issues.extend(self._check_function_complexity(node, file_path, lines))
                elif isinstance(node, ast.ClassDef):
                    issues.extend(self._check_class_design(node, file_path, lines))
            
        except SyntaxError as e:
            issues.append(ReviewIssue(
                file_path=file_path,
                line_number=e.lineno,
                category="syntax",
                severity="critical",
                description=f"Syntax error: {e.msg}",
                suggestion="Fix syntax error before proceeding"
            ))
        
        # Check Python-specific rules
        issues.extend(self._check_python_conventions(file_path, content, lines))
        issues.extend(self._check_python_security(file_path, content, lines))
        
        return issues
    
    def _review_javascript_file(self, file_path: str, content: str) -> List[ReviewIssue]:
        """Review JavaScript/TypeScript-specific issues."""
        issues = []
        lines = content.split('\\n')
        
        # Check JavaScript conventions
        issues.extend(self._check_js_conventions(file_path, content, lines))
        issues.extend(self._check_js_security(file_path, content, lines))
        
        return issues
    
    def _review_generic_file(self, file_path: str, content: str) -> List[ReviewIssue]:
        """Review generic file issues."""
        issues = []
        lines = content.split('\\n')
        
        # Check basic file quality
        if len(lines) > self.review_rules.get('code_quality', {}).get('max_file_length', 300):
            issues.append(ReviewIssue(
                file_path=file_path,
                category="file_length",
                severity="medium",
                description=f"File is too long ({len(lines)} lines)",
                suggestion="Consider breaking this file into smaller modules"
            ))
        
        return issues
    
    def _check_common_issues(self, file_path: str, content: str) -> List[ReviewIssue]:
        """Check issues common to all file types."""
        issues = []
        lines = content.split('\\n')
        
        # Check for secrets or sensitive data
        sensitive_patterns = [
            (r'password\\s*=\\s*["\'].*["\']', "password"),
            (r'secret\\s*=\\s*["\'].*["\']', "secret"),
            (r'api_key\\s*=\\s*["\'].*["\']', "api_key"),
            (r'token\\s*=\\s*["\'].*["\']', "token"),
        ]
        
        for i, line in enumerate(lines):
            for pattern, secret_type in sensitive_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(ReviewIssue(
                        file_path=file_path,
                        line_number=i + 1,
                        category="security",
                        severity="high",
                        description=f"Potential {secret_type} exposure in code",
                        suggestion=f"Move {secret_type} to environment variables or secure config",
                        rule_violated="check_secrets_exposure"
                    ))
        
        # Check for TODO/FIXME comments
        todo_pattern = r'(TODO|FIXME|XXX|HACK):'
        for i, line in enumerate(lines):
            if re.search(todo_pattern, line, re.IGNORECASE):
                issues.append(ReviewIssue(
                    file_path=file_path,
                    line_number=i + 1,
                    category="maintenance",
                    severity="low",
                    description="TODO/FIXME comment found",
                    suggestion="Address TODO items before production deployment"
                ))
        
        return issues
    
    def _check_function_complexity(self, node: ast.FunctionDef, file_path: str, lines: List[str]) -> List[ReviewIssue]:
        """Check function complexity issues."""
        issues = []
        
        # Check function length
        max_length = self.review_rules.get('simplicity', {}).get('max_function_length', 20)
        func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 10
        
        if func_lines > max_length:
            issues.append(ReviewIssue(
                file_path=file_path,
                line_number=node.lineno,
                category="complexity",
                severity="medium",
                description=f"Function '{node.name}' is too long ({func_lines} lines)",
                suggestion="Break this function into smaller, focused functions",
                rule_violated="max_function_length"
            ))
        
        # Check parameter count
        max_params = self.review_rules.get('simplicity', {}).get('max_function_parameters', 4)
        param_count = len(node.args.args)
        
        if param_count > max_params:
            issues.append(ReviewIssue(
                file_path=file_path,
                line_number=node.lineno,
                category="complexity",
                severity="medium",
                description=f"Function '{node.name}' has too many parameters ({param_count})",
                suggestion="Consider using a configuration object or breaking the function apart",
                rule_violated="max_function_parameters"
            ))
        
        # Check nesting depth
        max_nesting = self.review_rules.get('simplicity', {}).get('max_nesting_depth', 3)
        nesting_depth = self._calculate_nesting_depth(node)
        
        if nesting_depth > max_nesting:
            issues.append(ReviewIssue(
                file_path=file_path,
                line_number=node.lineno,
                category="complexity",
                severity="medium",
                description=f"Function '{node.name}' has excessive nesting ({nesting_depth} levels)",
                suggestion="Reduce nesting by using early returns or extracting nested logic",
                rule_violated="max_nesting_depth"
            ))
        
        return issues
    
    def _check_class_design(self, node: ast.ClassDef, file_path: str, lines: List[str]) -> List[ReviewIssue]:
        """Check class design issues."""
        issues = []
        
        # Check if class has docstring
        has_docstring = (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str))
        
        if not has_docstring:
            issues.append(ReviewIssue(
                file_path=file_path,
                line_number=node.lineno,
                category="documentation",
                severity="low",
                description=f"Class '{node.name}' missing docstring",
                suggestion="Add a docstring to explain the class purpose",
                rule_violated="documentation_required"
            ))
        
        # Check method count (too many methods might indicate SRP violation)
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if len(methods) > 15:
            issues.append(ReviewIssue(
                file_path=file_path,
                line_number=node.lineno,
                category="design",
                severity="medium",
                description=f"Class '{node.name}' has many methods ({len(methods)})",
                suggestion="Consider breaking this class into smaller, focused classes",
                rule_violated="single_responsibility_principle"
            ))
        
        return issues
    
    def _check_python_conventions(self, file_path: str, content: str, lines: List[str]) -> List[ReviewIssue]:
        """Check Python naming and style conventions."""
        issues = []
        
        # Check for snake_case in function and variable names
        if self.review_rules.get('readability', {}).get('consistent_naming_convention') == 'snake_case':
            # Check function names
            func_pattern = r'def\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\('
            for i, line in enumerate(lines):
                matches = re.findall(func_pattern, line)
                for func_name in matches:
                    if not re.match(r'^[a-z_][a-z0-9_]*$', func_name):
                        issues.append(ReviewIssue(
                            file_path=file_path,
                            line_number=i + 1,
                            category="naming",
                            severity="low",
                            description=f"Function '{func_name}' should use snake_case naming",
                            suggestion=f"Rename to follow snake_case convention",
                            rule_violated="consistent_naming_convention",
                            auto_fixable=True
                        ))
            
            # Check class names (should be PascalCase)
            class_pattern = r'class\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\s*[\\(:]'
            for i, line in enumerate(lines):
                matches = re.findall(class_pattern, line)
                for class_name in matches:
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                        issues.append(ReviewIssue(
                            file_path=file_path,
                            line_number=i + 1,
                            category="naming",
                            severity="low",
                            description=f"Class '{class_name}' should use PascalCase naming",
                            suggestion="Rename to follow PascalCase convention",
                            rule_violated="consistent_naming_convention",
                            auto_fixable=True
                        ))
        
        # Check for type hints if required
        if self.review_rules.get('code_quality', {}).get('type_hints_required'):
            func_pattern = r'def\\s+[a-zA-Z_][a-zA-Z0-9_]*\\s*\\([^)]*\\)\\s*:'
            for i, line in enumerate(lines):
                if re.search(func_pattern, line) and '->' not in line:
                    # Skip special methods and property decorators
                    if not any(special in line for special in ['__init__', '__str__', '__repr__', '@property']):
                        issues.append(ReviewIssue(
                            file_path=file_path,
                            line_number=i + 1,
                            category="type_hints",
                            severity="low",
                            description="Function missing return type hint",
                            suggestion="Add return type hint for better code clarity",
                            rule_violated="type_hints_required"
                        ))
        
        return issues
    
    def _check_python_security(self, file_path: str, content: str, lines: List[str]) -> List[ReviewIssue]:
        """Check Python security issues."""
        issues = []
        
        # Check for eval/exec usage
        dangerous_functions = ['eval(', 'exec(', 'execfile(', 'compile(']
        for i, line in enumerate(lines):
            for func in dangerous_functions:
                if func in line:
                    issues.append(ReviewIssue(
                        file_path=file_path,
                        line_number=i + 1,
                        category="security",
                        severity="high",
                        description=f"Dangerous function '{func.rstrip('(')}' found",
                        suggestion="Avoid using eval/exec as they can execute arbitrary code",
                        rule_violated="check_security_vulnerabilities"
                    ))
        
        # Check for SQL injection patterns
        sql_patterns = [
            r'\\+\\s*["\']\\s*\\+',  # String concatenation in queries
            r'%\\s*["\']\\s*%',      # String formatting in queries
            r'\\.format\\(',          # .format() in queries
        ]
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['select', 'insert', 'update', 'delete']):
                for pattern in sql_patterns:
                    if re.search(pattern, line):
                        issues.append(ReviewIssue(
                            file_path=file_path,
                            line_number=i + 1,
                            category="security",
                            severity="high",
                            description="Potential SQL injection vulnerability",
                            suggestion="Use parameterized queries or ORM methods",
                            rule_violated="check_sql_injection"
                        ))
        
        return issues
    
    def _check_js_conventions(self, file_path: str, content: str, lines: List[str]) -> List[ReviewIssue]:
        """Check JavaScript/TypeScript conventions."""
        issues = []
        
        # Check for var usage (should prefer const/let)
        for i, line in enumerate(lines):
            if re.search(r'\\bvar\\s+', line):
                issues.append(ReviewIssue(
                    file_path=file_path,
                    line_number=i + 1,
                    category="conventions",
                    severity="low",
                    description="Usage of 'var' found",
                    suggestion="Use 'const' or 'let' instead of 'var'",
                    auto_fixable=True
                ))
        
        return issues
    
    def _check_js_security(self, file_path: str, content: str, lines: List[str]) -> List[ReviewIssue]:
        """Check JavaScript security issues."""
        issues = []
        
        # Check for innerHTML usage
        for i, line in enumerate(lines):
            if 'innerHTML' in line and '=' in line:
                issues.append(ReviewIssue(
                    file_path=file_path,
                    line_number=i + 1,
                    category="security",
                    severity="medium",
                    description="Direct innerHTML assignment found",
                    suggestion="Use textContent or properly sanitize HTML to prevent XSS",
                    rule_violated="check_xss_vulnerabilities"
                ))
        
        return issues
    
    def _review_code_structure(self, code_directory: str) -> List[ReviewIssue]:
        """Review overall code structure and organization."""
        issues = []
        
        # Check for proper directory structure
        expected_dirs = ['tests']
        for expected_dir in expected_dirs:
            if not os.path.exists(os.path.join(code_directory, expected_dir)):
                issues.append(ReviewIssue(
                    file_path=code_directory,
                    category="structure",
                    severity="medium",
                    description=f"Missing '{expected_dir}' directory",
                    suggestion=f"Create {expected_dir} directory for better organization"
                ))
        
        # Check for __init__.py files in Python packages
        python_dirs = []
        for root, dirs, files in os.walk(code_directory):
            if any(f.endswith('.py') for f in files):
                python_dirs.append(root)
        
        for py_dir in python_dirs:
            if py_dir != code_directory:  # Skip root directory
                init_file = os.path.join(py_dir, '__init__.py')
                if not os.path.exists(init_file):
                    issues.append(ReviewIssue(
                        file_path=py_dir,
                        category="structure",
                        severity="low",
                        description="Python package missing __init__.py",
                        suggestion="Add __init__.py file to make directory a proper Python package",
                        auto_fixable=True
                    ))
        
        return issues
    
    def _review_security(self, code_directory: str) -> List[ReviewIssue]:
        """Review security aspects across all files."""
        issues = []
        
        # Check for hardcoded secrets across all files
        for root, dirs, files in os.walk(code_directory):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.json', '.yaml', '.yml')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check for hardcoded URLs, keys, etc.
                        security_patterns = [
                            (r'http://[^\\s]+', "HTTP URL (should use HTTPS)"),
                            (r'[a-zA-Z0-9]{32,}', "Potential hardcoded token/key"),
                        ]
                        
                        lines = content.split('\\n')
                        for i, line in enumerate(lines):
                            for pattern, description in security_patterns:
                                if re.search(pattern, line):
                                    issues.append(ReviewIssue(
                                        file_path=file_path,
                                        line_number=i + 1,
                                        category="security",
                                        severity="medium",
                                        description=description,
                                        suggestion="Review for security implications"
                                    ))
                    
                    except:
                        continue
        
        return issues
    
    def _review_performance(self, code_directory: str) -> List[ReviewIssue]:
        """Review performance implications."""
        issues = []
        
        # Check for obvious performance issues in Python files
        for root, dirs, files in os.walk(code_directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        lines = content.split('\\n')
                        
                        # Check for nested loops
                        for i, line in enumerate(lines):
                            if 'for ' in line:
                                # Look for nested for loops in next few lines
                                for j in range(i + 1, min(i + 10, len(lines))):
                                    if 'for ' in lines[j] and lines[j].strip().startswith(' '):
                                        issues.append(ReviewIssue(
                                            file_path=file_path,
                                            line_number=i + 1,
                                            category="performance",
                                            severity="medium",
                                            description="Nested loops detected",
                                            suggestion="Consider optimizing algorithm complexity"
                                        ))
                                        break
                        
                        # Check for inefficient string concatenation
                        if '+=' in content and 'str' in content:
                            issues.append(ReviewIssue(
                                file_path=file_path,
                                category="performance",
                                severity="low",
                                description="Potential inefficient string concatenation",
                                suggestion="Consider using join() for multiple string concatenations"
                            ))
                    
                    except:
                        continue
        
        return issues
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth in a function."""
        max_depth = 0
        
        def visit_node(node, current_depth):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            # Increment depth for control structures
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                for child in ast.iter_child_nodes(node):
                    visit_node(child, current_depth + 1)
            else:
                for child in ast.iter_child_nodes(node):
                    visit_node(child, current_depth)
        
        for child in ast.iter_child_nodes(node):
            visit_node(child, 1)
        
        return max_depth
    
    def _calculate_overall_score(self, issues: List[ReviewIssue]) -> float:
        """Calculate overall code quality score."""
        if not issues:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {
            'critical': 20,
            'high': 10,
            'medium': 5,
            'low': 2
        }
        
        violation_weights = self.review_rules.get('violation_weights', {})
        
        total_deduction = 0
        for issue in issues:
            base_deduction = severity_weights.get(issue.severity, 2)
            category_weight = violation_weights.get(issue.category, 1)
            total_deduction += base_deduction * category_weight
        
        # Cap maximum deduction to avoid negative scores
        max_deduction = min(total_deduction, 95)
        score = 100.0 - max_deduction
        
        return max(score, 5.0)  # Minimum score of 5
    
    def _determine_grade(self, score: float) -> str:
        """Determine letter grade based on score."""
        scoring = self.review_rules.get('scoring', {})
        
        if score >= scoring.get('excellent', {}).get('min_score', 90):
            return 'excellent'
        elif score >= scoring.get('good', {}).get('min_score', 75):
            return 'good'
        elif score >= scoring.get('acceptable', {}).get('min_score', 60):
            return 'acceptable'
        elif score >= scoring.get('needs_improvement', {}).get('min_score', 40):
            return 'needs_improvement'
        else:
            return 'unacceptable'
    
    def _determine_approval_status(self, score: float, issues: List[ReviewIssue]) -> str:
        """Determine approval status based on score and critical issues."""
        # Check for critical issues
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        if critical_issues:
            return 'rejected'
        
        # Check for high severity security issues
        security_issues = [issue for issue in issues if issue.category == 'security' and issue.severity == 'high']
        if security_issues:
            return 'rejected'
        
        # Approve based on score
        if score >= 85:
            return 'approved'
        elif score >= 60:
            return 'approved_with_changes'
        else:
            return 'rejected'
    
    def _identify_strengths(self, code_output: CodeOutput, code_directory: str) -> List[str]:
        """Identify positive aspects of the code."""
        strengths = []
        
        # Check for good practices
        if code_output.tests_created > 0:
            strengths.append("Includes unit tests for quality assurance")
        
        if code_output.documentation_files:
            strengths.append("Includes documentation for maintainability")
        
        if code_output.functions_created > 0:
            strengths.append("Code is properly modularized with functions")
        
        if code_output.classes_created > 0:
            strengths.append("Uses object-oriented design principles")
        
        # Check for proper error handling
        has_error_handling = False
        for root, dirs, files in os.walk(code_directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if 'try:' in content or 'except' in content:
                            has_error_handling = True
                            break
                    except:
                        continue
            if has_error_handling:
                break
        
        if has_error_handling:
            strengths.append("Implements proper error handling")
        
        # Default strengths if none found
        if not strengths:
            strengths = ["Code compiles and runs without critical errors"]
        
        return strengths
    
    def _generate_recommendations(self, issues: List[ReviewIssue], code_output: CodeOutput) -> List[str]:
        """Generate actionable recommendations based on issues found."""
        recommendations = []
        
        # Group issues by category
        issue_categories = {}
        for issue in issues:
            if issue.category not in issue_categories:
                issue_categories[issue.category] = []
            issue_categories[issue.category].append(issue)
        
        # Generate category-specific recommendations
        for category, category_issues in issue_categories.items():
            if category == 'security':
                recommendations.append("Review and address security vulnerabilities before deployment")
            elif category == 'complexity':
                recommendations.append("Refactor complex functions to improve maintainability")
            elif category == 'documentation':
                recommendations.append("Add comprehensive documentation and comments")
            elif category == 'performance':
                recommendations.append("Optimize performance bottlenecks identified in the review")
            elif category == 'naming':
                recommendations.append("Follow consistent naming conventions throughout the codebase")
        
        # Add general recommendations based on review rules priorities
        priority_recommendations = []
        for priority in self.review_rules.get('review_priorities', []):
            if any(keyword in priority.lower() for keyword in ['understand', 'read']):
                if any(issue.category in ['complexity', 'naming'] for issue in issues):
                    priority_recommendations.append("Focus on code readability and simplicity")
                    break
        
        recommendations.extend(priority_recommendations)
        
        # Add default recommendations if none generated
        if not recommendations:
            recommendations = [
                "Continue following coding best practices",
                "Consider adding more comprehensive tests",
                "Keep code documentation up to date"
            ]
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_review_summary(self, review_report: ReviewReport) -> str:
        """Generate a human-readable review summary."""
        summary_lines = []
        
        summary_lines.append(f"# Code Review Report")
        summary_lines.append(f"")
        summary_lines.append(f"**Overall Score:** {review_report.overall_score:.1f}/100")
        summary_lines.append(f"**Grade:** {review_report.grade.replace('_', ' ').title()}")
        summary_lines.append(f"**Status:** {review_report.approval_status.replace('_', ' ').title()}")
        summary_lines.append(f"")
        
        # Issues summary
        if review_report.issues:
            summary_lines.append(f"## Issues Found ({len(review_report.issues)})")
            
            # Group by severity
            severity_groups = {}
            for issue in review_report.issues:
                if issue.severity not in severity_groups:
                    severity_groups[issue.severity] = []
                severity_groups[issue.severity].append(issue)
            
            for severity in ['critical', 'high', 'medium', 'low']:
                if severity in severity_groups:
                    summary_lines.append(f"")
                    summary_lines.append(f"### {severity.title()} ({len(severity_groups[severity])})")
                    for issue in severity_groups[severity][:5]:  # Show first 5
                        summary_lines.append(f"- {issue.description} ({issue.file_path}:{issue.line_number or 'N/A'})")
        
        # Strengths
        if review_report.strengths:
            summary_lines.append(f"")
            summary_lines.append(f"## Strengths")
            for strength in review_report.strengths:
                summary_lines.append(f"- {strength}")
        
        # Recommendations
        if review_report.recommendations:
            summary_lines.append(f"")
            summary_lines.append(f"## Recommendations")
            for rec in review_report.recommendations:
                summary_lines.append(f"- {rec}")
        
        summary_lines.append(f"")
        summary_lines.append(f"*Review completed in {review_report.review_time_minutes:.1f} minutes*")
        
        return "\\n".join(summary_lines)