"""
Main Pipeline Orchestrator for the Three-Agent Pipeline System.

This module coordinates the execution of all three agents in sequence.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add the pipeline directory to the Python path
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pipeline_dir)

from shared.models import PipelineRun, TaskStatus, AgentType
from shared.utils import (
    PipelineLogger, ConfigLoader, FileManager, TaskTimer, 
    MessageQueue, ensure_directory_exists
)
from shared.memory import PipelineMemoryManager
from agents.planner import PlanningAgent
from agents.executor import ExecutionAgent  
from agents.reviewer import ReviewAgent


class PipelineOrchestrator:
    """
    Main orchestrator that coordinates all three agents.
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(pipeline_dir, "basic-pipeline", "pipeline.yaml")
        self.config = ConfigLoader.load_yaml(self.config_path)
        
        self.logger = PipelineLogger(self.config.get('settings', {}).get('log_level', 'INFO'))
        self.file_manager = FileManager(self.config.get('settings', {}).get('output_directory', './pipeline_output'))
        self.message_queue = MessageQueue()
        
        # Initialize memory system
        memory_config = self.config.get('memory', {})
        self.memory_manager = PipelineMemoryManager(
            project_id=memory_config.get('project_id', 'default'),
            memory_config=memory_config
        )
        
        # Initialize agents with memory
        self.planner = PlanningAgent(
            self.config.get('agents', {}).get('planner', {}),
            memory_manager=self.memory_manager
        )
        self.executor = ExecutionAgent(
            self.config.get('agents', {}).get('executor', {}),
            memory_manager=self.memory_manager
        )
        
        # Initialize reviewer with rules and memory
        review_rules_path = os.path.join(pipeline_dir, "rules", "review_rules.json")
        self.reviewer = ReviewAgent(
            review_rules_path, 
            self.config.get('agents', {}).get('reviewer', {}),
            memory_manager=self.memory_manager
        )
        
    def run_pipeline(self, task_description: str, requirements: Optional[List[str]] = None) -> PipelineRun:
        """
        Run the complete three-agent pipeline.
        
        Args:
            task_description: The main task to be executed
            requirements: Optional list of specific requirements
            
        Returns:
            PipelineRun object with complete execution results
        """
        self.logger.info("Starting three-agent pipeline execution")
        
        # Initialize pipeline run
        pipeline_run = PipelineRun(task_description=task_description)
        run_timer = TaskTimer()
        run_timer.start()
        
        try:
            # Create run directory
            run_dir = self.file_manager.create_run_directory(pipeline_run.run_id)
            self.logger.info(f"Created run directory: {run_dir}")
            
            # Stage 1: Planning
            self.logger.info("=" * 60)
            self.logger.info("STAGE 1: PLANNING")
            self.logger.info("=" * 60)
            
            pipeline_run.status = TaskStatus.IN_PROGRESS
            pipeline_run.current_agent = AgentType.PLANNER
            
            execution_plan = self._run_planning_stage(task_description, requirements, run_dir)
            pipeline_run.execution_plan = execution_plan
            
            # Stage 2: Execution
            self.logger.info("=" * 60)
            self.logger.info("STAGE 2: EXECUTION")
            self.logger.info("=" * 60)
            
            pipeline_run.current_agent = AgentType.EXECUTOR
            
            code_output = self._run_execution_stage(execution_plan, run_dir)
            pipeline_run.code_output = code_output
            
            # Stage 3: Review
            self.logger.info("=" * 60)
            self.logger.info("STAGE 3: REVIEW")
            self.logger.info("=" * 60)
            
            pipeline_run.current_agent = AgentType.REVIEWER
            
            review_report = self._run_review_stage(code_output, run_dir)
            pipeline_run.review_report = review_report
            
            # Finalize pipeline run
            pipeline_run.status = TaskStatus.COMPLETED
            pipeline_run.completed_at = datetime.now()
            pipeline_run.total_time_minutes = run_timer.stop()
            pipeline_run.current_agent = None
            
            # Save pipeline run metadata
            self.file_manager.save_pipeline_run(pipeline_run, run_dir)
            
            # Generate final summary
            self._generate_final_summary(pipeline_run, run_dir)
            
            self.logger.info("=" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info(f"Total execution time: {pipeline_run.total_time_minutes:.2f} minutes")
            self.logger.info(f"Final grade: {review_report.grade}")
            self.logger.info(f"Approval status: {review_report.approval_status}")
            self.logger.info(f"Results saved to: {run_dir}")
            
            return pipeline_run
            
        except Exception as e:
            pipeline_run.status = TaskStatus.FAILED
            pipeline_run.error_message = str(e)
            pipeline_run.completed_at = datetime.now()
            pipeline_run.total_time_minutes = run_timer.stop()
            
            self.logger.error(f"Pipeline failed: {str(e)}")
            
            # Try to save partial results
            try:
                self.file_manager.save_pipeline_run(pipeline_run, run_dir)
            except:
                pass
            
            raise
    
    def _run_planning_stage(self, task_description: str, requirements: Optional[List[str]], run_dir: Path):
        """Run the planning stage."""
        try:
            self.logger.info("Planning agent analyzing task...")
            
            # Create execution plan
            execution_plan = self.planner.create_plan(task_description, requirements)
            
            # Validate plan
            if not self.planner.validate_plan(execution_plan):
                raise Exception("Generated execution plan failed validation")
            
            # Save execution plan
            self.file_manager.save_execution_plan(execution_plan, run_dir)
            
            self.logger.info(f"Planning completed: {len(execution_plan.steps)} steps created")
            self.logger.info(f"Estimated time: {execution_plan.total_estimated_time} minutes")
            
            return execution_plan
            
        except Exception as e:
            self.logger.error(f"Planning stage failed: {str(e)}")
            raise Exception(f"Planning stage failed: {str(e)}")
    
    def _run_execution_stage(self, execution_plan, run_dir: Path):
        """Run the execution stage."""
        try:
            self.logger.info("Execution agent implementing plan...")
            
            # Set up code output directory
            code_output_dir = run_dir / "code_output"
            ensure_directory_exists(str(code_output_dir))
            
            # Execute the plan
            code_output = self.executor.execute_plan(execution_plan, str(code_output_dir))
            
            # Copy generated files to run directory
            if code_output.files_created:
                self.file_manager.copy_code_files(code_output.files_created, run_dir)
            
            self.logger.info(f"Execution completed: {len(code_output.files_created)} files created")
            self.logger.info(f"Lines of code: {code_output.lines_of_code}")
            self.logger.info(f"Functions: {code_output.functions_created}, Classes: {code_output.classes_created}")
            
            # Log any errors or warnings
            if code_output.errors_encountered:
                self.logger.error(f"Execution errors: {len(code_output.errors_encountered)}")
                for error in code_output.errors_encountered[:3]:  # Log first 3 errors
                    self.logger.error(f"  - {error}")
            
            if code_output.warnings:
                self.logger.info(f"Execution warnings: {len(code_output.warnings)}")
            
            return code_output
            
        except Exception as e:
            self.logger.error(f"Execution stage failed: {str(e)}")
            raise Exception(f"Execution stage failed: {str(e)}")
    
    def _run_review_stage(self, code_output, run_dir: Path):
        """Run the review stage."""
        try:
            self.logger.info("Review agent analyzing code quality...")
            
            code_directory = str(run_dir / "code_output")
            
            # Perform code review
            review_report = self.reviewer.review_code(code_output, code_directory)
            
            # Save review report
            self.file_manager.save_review_report(review_report, run_dir)
            
            # Generate and save review summary
            review_summary = self.reviewer.generate_review_summary(review_report)
            summary_file = run_dir / "reports" / "review_summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(review_summary)
            
            self.logger.info(f"Review completed with score: {review_report.overall_score:.1f}/100")
            self.logger.info(f"Grade: {review_report.grade}, Status: {review_report.approval_status}")
            self.logger.info(f"Issues found: {len(review_report.issues)}")
            
            # Log critical issues
            critical_issues = [issue for issue in review_report.issues if issue.severity == 'critical']
            if critical_issues:
                self.logger.error(f"Critical issues found: {len(critical_issues)}")
                for issue in critical_issues[:3]:
                    self.logger.error(f"  - {issue.description}")
            
            return review_report
            
        except Exception as e:
            self.logger.error(f"Review stage failed: {str(e)}")
            raise Exception(f"Review stage failed: {str(e)}")
    
    def _generate_final_summary(self, pipeline_run: PipelineRun, run_dir: Path):
        """Generate final summary of pipeline execution."""
        try:
            summary_lines = []
            
            summary_lines.append("# Three-Agent Pipeline Execution Summary")
            summary_lines.append("")
            summary_lines.append(f"**Run ID:** {pipeline_run.run_id}")
            summary_lines.append(f"**Task:** {pipeline_run.task_description}")
            summary_lines.append(f"**Started:** {pipeline_run.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append(f"**Completed:** {pipeline_run.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append(f"**Total Time:** {pipeline_run.total_time_minutes:.2f} minutes")
            summary_lines.append(f"**Status:** {pipeline_run.status.value}")
            summary_lines.append("")
            
            # Planning results
            if pipeline_run.execution_plan:
                plan = pipeline_run.execution_plan
                summary_lines.append("## Planning Stage")
                summary_lines.append(f"- **Steps Created:** {len(plan.steps)}")
                summary_lines.append(f"- **Estimated Time:** {plan.total_estimated_time} minutes")
                summary_lines.append(f"- **Deliverables:** {len(plan.deliverables)}")
                summary_lines.append(f"- **Risks Identified:** {len(plan.risks)}")
                summary_lines.append("")
            
            # Execution results
            if pipeline_run.code_output:
                code = pipeline_run.code_output
                summary_lines.append("## Execution Stage")
                summary_lines.append(f"- **Files Created:** {len(code.files_created)}")
                summary_lines.append(f"- **Lines of Code:** {code.lines_of_code}")
                summary_lines.append(f"- **Functions:** {code.functions_created}")
                summary_lines.append(f"- **Classes:** {code.classes_created}")
                summary_lines.append(f"- **Tests:** {code.tests_created}")
                summary_lines.append(f"- **Documentation Files:** {len(code.documentation_files)}")
                if code.errors_encountered:
                    summary_lines.append(f"- **Errors:** {len(code.errors_encountered)}")
                if code.warnings:
                    summary_lines.append(f"- **Warnings:** {len(code.warnings)}")
                summary_lines.append("")
            
            # Review results
            if pipeline_run.review_report:
                review = pipeline_run.review_report
                summary_lines.append("## Review Stage")
                summary_lines.append(f"- **Overall Score:** {review.overall_score:.1f}/100")
                summary_lines.append(f"- **Grade:** {review.grade.replace('_', ' ').title()}")
                summary_lines.append(f"- **Approval Status:** {review.approval_status.replace('_', ' ').title()}")
                summary_lines.append(f"- **Issues Found:** {len(review.issues)}")
                
                # Break down issues by severity
                severity_counts = {}
                for issue in review.issues:
                    severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
                
                if severity_counts:
                    summary_lines.append("- **Issue Breakdown:**")
                    for severity in ['critical', 'high', 'medium', 'low']:
                        if severity in severity_counts:
                            summary_lines.append(f"  - {severity.title()}: {severity_counts[severity]}")
                
                summary_lines.append(f"- **Review Time:** {review.review_time_minutes:.2f} minutes")
                summary_lines.append("")
            
            # Files generated
            summary_lines.append("## Generated Files")
            summary_lines.append("```")
            summary_lines.append("pipeline_output/")
            summary_lines.append(f"└── {pipeline_run.run_id}/")
            summary_lines.append("    ├── execution_plan.json")
            summary_lines.append("    ├── pipeline_run.json")
            summary_lines.append("    ├── code_output/")
            summary_lines.append("    │   └── [generated code files]")
            summary_lines.append("    ├── reports/")
            summary_lines.append("    │   ├── review_report.json")
            summary_lines.append("    │   └── review_summary.md")
            summary_lines.append("    └── logs/")
            summary_lines.append("```")
            summary_lines.append("")
            
            # Next steps
            summary_lines.append("## Next Steps")
            if pipeline_run.review_report:
                if pipeline_run.review_report.approval_status == 'approved':
                    summary_lines.append("✅ Code is approved and ready for deployment")
                elif pipeline_run.review_report.approval_status == 'approved_with_changes':
                    summary_lines.append("⚠️  Code is approved with minor changes needed")
                    summary_lines.append("- Review the recommendations in the review report")
                    summary_lines.append("- Address any medium or high severity issues")
                else:
                    summary_lines.append("❌ Code requires significant improvements before deployment")
                    summary_lines.append("- Address all critical and high severity issues")
                    summary_lines.append("- Consider re-running the pipeline after fixes")
            
            summary_lines.append("")
            summary_lines.append("---")
            summary_lines.append(f"*Generated by Three-Agent Pipeline System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            
            # Save summary
            summary_file = run_dir / "SUMMARY.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("\\n".join(summary_lines))
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
    
    def retry_failed_run(self, run_id: str, max_retries: int = 3) -> PipelineRun:
        """Retry a failed pipeline run."""
        try:
            # Load previous run
            run_dir = self.file_manager.base_output_dir / run_id
            run_file = run_dir / "pipeline_run.json"
            
            if not run_file.exists():
                raise Exception(f"Run {run_id} not found")
            
            with open(run_file, 'r') as f:
                run_data = json.load(f)
            
            task_description = run_data.get('task_description', '')
            retry_count = run_data.get('retry_count', 0)
            
            if retry_count >= max_retries:
                raise Exception(f"Maximum retries ({max_retries}) exceeded for run {run_id}")
            
            self.logger.info(f"Retrying pipeline run {run_id} (attempt {retry_count + 1})")
            
            # Run pipeline with retry count
            new_run = self.run_pipeline(task_description)
            new_run.retry_count = retry_count + 1
            
            return new_run
            
        except Exception as e:
            self.logger.error(f"Retry failed: {str(e)}")
            raise
    
    def list_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent pipeline runs."""
        runs = []
        
        try:
            output_dir = self.file_manager.base_output_dir
            if not output_dir.exists():
                return runs
            
            # Get all run directories
            run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            
            # Sort by creation time (most recent first)
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Load run metadata
            for run_dir in run_dirs[:limit]:
                run_file = run_dir / "pipeline_run.json"
                if run_file.exists():
                    try:
                        with open(run_file, 'r') as f:
                            run_data = json.load(f)
                        
                        runs.append({
                            'run_id': run_data.get('run_id', run_dir.name),
                            'task_description': run_data.get('task_description', 'Unknown')[:100] + '...',
                            'status': run_data.get('status', 'unknown'),
                            'started_at': run_data.get('started_at', ''),
                            'total_time_minutes': run_data.get('total_time_minutes', 0),
                            'grade': run_data.get('review_report', {}).get('grade', 'N/A') if run_data.get('review_report') else 'N/A'
                        })
                    except:
                        continue
            
        except Exception as e:
            self.logger.error(f"Failed to list runs: {str(e)}")
        
        return runs


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='Three-Agent Pipeline System')
    parser.add_argument('command', choices=['run', 'retry', 'list'], help='Command to execute')
    parser.add_argument('--task', type=str, help='Task description for run command')
    parser.add_argument('--run-id', type=str, help='Run ID for retry command')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--limit', type=int, default=10, help='Limit for list command')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(args.config)
        
        if args.command == 'run':
            if not args.task:
                print("Error: --task is required for run command")
                sys.exit(1)
            
            print(f"Running pipeline for task: {args.task}")
            result = orchestrator.run_pipeline(args.task)
            print(f"Pipeline completed with status: {result.status.value}")
            
        elif args.command == 'retry':
            if not args.run_id:
                print("Error: --run-id is required for retry command")
                sys.exit(1)
            
            print(f"Retrying pipeline run: {args.run_id}")
            result = orchestrator.retry_failed_run(args.run_id)
            print(f"Retry completed with status: {result.status.value}")
            
        elif args.command == 'list':
            print("Recent pipeline runs:")
            runs = orchestrator.list_runs(args.limit)
            
            if not runs:
                print("No pipeline runs found")
            else:
                print(f"{'Run ID':<12} {'Status':<12} {'Grade':<12} {'Task'}")
                print("-" * 80)
                for run in runs:
                    print(f"{run['run_id'][:11]:<12} {run['status']:<12} {run['grade']:<12} {run['task_description'][:40]}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()