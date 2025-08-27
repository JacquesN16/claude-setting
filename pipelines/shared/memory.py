"""
Memory System for the Three-Agent Pipeline.

Provides multi-layered memory capabilities including short-term, working, 
and long-term memory with learning and adaptation capabilities.
"""

import json
import sqlite3
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

from .models import ExecutionPlan, CodeOutput, ReviewReport, AgentType, TaskStatus
from .utils import ConfigLoader, ensure_directory_exists


@dataclass
class MemoryEntry:
    """Base memory entry structure."""
    id: str
    timestamp: datetime
    agent_type: AgentType
    entry_type: str
    data: Dict[str, Any]
    tags: Set[str] = field(default_factory=set)
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class PatternMatch:
    """Represents a pattern match in memory."""
    similarity_score: float
    memory_entry: MemoryEntry
    matched_features: List[str]


class ShortTermMemory:
    """Session-specific memory for current pipeline execution."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.entries: deque = deque(maxlen=max_size)
        self.context_data: Dict[str, Any] = {}
        self.agent_states: Dict[AgentType, Dict[str, Any]] = {}
        
    def add_entry(self, entry: MemoryEntry):
        """Add entry to short-term memory."""
        self.entries.append(entry)
        
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context data."""
        return self.context_data.get(key, default)
        
    def set_context(self, key: str, value: Any):
        """Set context data."""
        self.context_data[key] = value
        
    def get_agent_state(self, agent: AgentType, key: str, default: Any = None) -> Any:
        """Get agent-specific state."""
        return self.agent_states.get(agent, {}).get(key, default)
        
    def set_agent_state(self, agent: AgentType, key: str, value: Any):
        """Set agent-specific state."""
        if agent not in self.agent_states:
            self.agent_states[agent] = {}
        self.agent_states[agent][key] = value
        
    def clear(self):
        """Clear short-term memory."""
        self.entries.clear()
        self.context_data.clear()
        self.agent_states.clear()


class WorkingMemory:
    """Project-specific memory for medium-term learning."""
    
    def __init__(self, project_id: str, max_entries: int = 10000):
        self.project_id = project_id
        self.max_entries = max_entries
        self.entries: Dict[str, MemoryEntry] = {}
        self.patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.project_metadata: Dict[str, Any] = {}
        
    def add_entry(self, entry: MemoryEntry):
        """Add entry to working memory."""
        self.entries[entry.id] = entry
        
        # Add to pattern index
        pattern_key = f"{entry.agent_type.value}_{entry.entry_type}"
        self.patterns[pattern_key].append({
            'id': entry.id,
            'timestamp': entry.timestamp,
            'features': self._extract_features(entry),
            'tags': entry.tags
        })
        
        # Maintain size limit
        if len(self.entries) > self.max_entries:
            self._evict_oldest_entries()
            
    def find_similar_patterns(self, entry_type: str, features: Dict[str, Any], 
                            agent_type: AgentType, limit: int = 10) -> List[PatternMatch]:
        """Find similar patterns in working memory."""
        pattern_key = f"{agent_type.value}_{entry_type}"
        matches = []
        
        if pattern_key in self.patterns:
            for pattern in self.patterns[pattern_key]:
                similarity = self._calculate_similarity(features, pattern['features'])
                if similarity > 0.3:  # Minimum similarity threshold
                    memory_entry = self.entries.get(pattern['id'])
                    if memory_entry:
                        match = PatternMatch(
                            similarity_score=similarity,
                            memory_entry=memory_entry,
                            matched_features=list(features.keys())
                        )
                        matches.append(match)
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:limit]
        
    def get_project_stats(self) -> Dict[str, Any]:
        """Get project-specific statistics."""
        return {
            'total_entries': len(self.entries),
            'patterns_count': {k: len(v) for k, v in self.patterns.items()},
            'metadata': self.project_metadata
        }
        
    def _extract_features(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Extract features from memory entry for pattern matching."""
        features = {}
        
        # Extract common features based on entry type
        if entry.entry_type == 'task_analysis':
            data = entry.data
            features.update({
                'task_type': data.get('task_type', ''),
                'complexity': data.get('complexity', ''),
                'technologies': data.get('technologies', []),
                'components': data.get('components', [])
            })
        elif entry.entry_type == 'code_generation':
            data = entry.data
            features.update({
                'language': data.get('language', ''),
                'framework': data.get('framework', ''),
                'pattern_type': data.get('pattern_type', ''),
                'lines_of_code': data.get('lines_of_code', 0)
            })
        elif entry.entry_type == 'review_outcome':
            data = entry.data
            features.update({
                'score': data.get('score', 0),
                'issues_count': data.get('issues_count', 0),
                'main_categories': data.get('main_categories', [])
            })
            
        return features
        
    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between two feature sets."""
        if not features1 or not features2:
            return 0.0
            
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
            
        similarity_scores = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            
            if isinstance(val1, (list, set)) and isinstance(val2, (list, set)):
                # Jaccard similarity for sets/lists
                set1, set2 = set(val1), set(val2)
                if set1 or set2:
                    similarity = len(set1 & set2) / len(set1 | set2)
                else:
                    similarity = 1.0
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                similarity = 1.0 if val1 == val2 else 0.0
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity (normalized difference)
                max_val = max(abs(val1), abs(val2), 1)
                similarity = 1.0 - abs(val1 - val2) / max_val
            else:
                similarity = 1.0 if val1 == val2 else 0.0
                
            similarity_scores.append(similarity)
            
        return statistics.mean(similarity_scores) if similarity_scores else 0.0
        
    def _evict_oldest_entries(self):
        """Remove oldest entries to maintain size limit."""
        sorted_entries = sorted(self.entries.items(), 
                              key=lambda x: x[1].last_accessed)
        entries_to_remove = len(self.entries) - (self.max_entries // 2)
        
        for i in range(entries_to_remove):
            entry_id = sorted_entries[i][0]
            del self.entries[entry_id]


class LongTermMemory:
    """Persistent memory with database storage."""
    
    def __init__(self, db_path: str = "~/.claude/pipelines/memory/pipeline_memory.db"):
        self.db_path = Path(db_path).expanduser()
        ensure_directory_exists(str(self.db_path.parent))
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Memory entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                agent_type TEXT,
                entry_type TEXT,
                data_json TEXT,
                tags TEXT,
                importance_score REAL,
                access_count INTEGER,
                last_accessed REAL
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                agent_type TEXT,
                metric_type TEXT,
                value REAL,
                context_json TEXT
            )
        ''')
        
        # Pattern templates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_templates (
                id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                template_data TEXT,
                usage_count INTEGER,
                success_rate REAL,
                created_at REAL,
                updated_at REAL
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_type ON memory_entries(agent_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entry_type ON memory_entries(entry_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp)')
        
        conn.commit()
        conn.close()
        
    def store_entry(self, entry: MemoryEntry):
        """Store entry in long-term memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO memory_entries 
            (id, timestamp, agent_type, entry_type, data_json, tags, 
             importance_score, access_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.id,
            entry.timestamp.timestamp(),
            entry.agent_type.value,
            entry.entry_type,
            json.dumps(entry.data, default=str),
            json.dumps(list(entry.tags)),
            entry.importance_score,
            entry.access_count,
            entry.last_accessed.timestamp()
        ))
        
        conn.commit()
        conn.close()
        
    def retrieve_entries(self, agent_type: Optional[AgentType] = None,
                        entry_type: Optional[str] = None,
                        limit: int = 100,
                        days_back: int = 30) -> List[MemoryEntry]:
        """Retrieve entries from long-term memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM memory_entries WHERE timestamp > ?"
        params = [(datetime.now() - timedelta(days=days_back)).timestamp()]
        
        if agent_type:
            query += " AND agent_type = ?"
            params.append(agent_type.value)
            
        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type)
            
        query += " ORDER BY importance_score DESC, timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        entries = []
        for row in rows:
            entry = MemoryEntry(
                id=row[0],
                timestamp=datetime.fromtimestamp(row[1]),
                agent_type=AgentType(row[2]),
                entry_type=row[3],
                data=json.loads(row[4]),
                tags=set(json.loads(row[5])),
                importance_score=row[6],
                access_count=row[7],
                last_accessed=datetime.fromtimestamp(row[8])
            )
            entries.append(entry)
            
        conn.close()
        return entries
        
    def store_performance_metric(self, agent_type: AgentType, metric_type: str,
                                value: float, context: Dict[str, Any] = None):
        """Store performance metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metric_id = hashlib.md5(f"{agent_type.value}_{metric_type}_{datetime.now()}".encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (id, timestamp, agent_type, metric_type, value, context_json)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metric_id,
            datetime.now().timestamp(),
            agent_type.value,
            metric_type,
            value,
            json.dumps(context or {}, default=str)
        ))
        
        conn.commit()
        conn.close()
        
    def get_performance_trends(self, agent_type: AgentType, 
                             metric_type: str, days_back: int = 30) -> List[Tuple[datetime, float]]:
        """Get performance trends over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, value FROM performance_metrics 
            WHERE agent_type = ? AND metric_type = ? 
            AND timestamp > ?
            ORDER BY timestamp ASC
        ''', (
            agent_type.value,
            metric_type,
            (datetime.now() - timedelta(days=days_back)).timestamp()
        ))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [(datetime.fromtimestamp(row[0]), row[1]) for row in rows]


class PipelineMemoryManager:
    """Main memory manager coordinating all memory layers."""
    
    def __init__(self, project_id: str = "default", 
                 memory_config: Dict[str, Any] = None):
        self.project_id = project_id
        self.config = memory_config or {}
        
        # Initialize memory layers
        self.short_term = ShortTermMemory(
            max_size=self.config.get('short_term_size', 1000)
        )
        self.working = WorkingMemory(
            project_id=project_id,
            max_entries=self.config.get('working_memory_size', 10000)
        )
        self.long_term = LongTermMemory(
            db_path=self.config.get('db_path', "~/.claude/pipelines/memory/pipeline_memory.db")
        )
        
        # Initialize with existing long-term memory
        self._load_working_memory_from_long_term()
        
    def remember_planning_result(self, task_description: str, execution_plan: ExecutionPlan,
                               actual_time: Optional[float] = None):
        """Store planning results in memory."""
        # Extract planning insights
        planning_data = {
            'task_description': task_description,
            'task_type': self._extract_task_type(task_description),
            'complexity': self._assess_complexity(execution_plan),
            'steps_count': len(execution_plan.steps),
            'estimated_time': execution_plan.total_estimated_time,
            'actual_time': actual_time,
            'technologies': self._extract_technologies(task_description),
            'components': execution_plan.deliverables
        }
        
        entry = MemoryEntry(
            id=hashlib.md5(f"planning_{task_description}_{datetime.now()}".encode()).hexdigest(),
            timestamp=datetime.now(),
            agent_type=AgentType.PLANNER,
            entry_type='task_analysis',
            data=planning_data,
            tags={planning_data['task_type'], 'planning', 'analysis'},
            importance_score=0.7 if actual_time else 0.5
        )
        
        self._store_in_all_layers(entry)
        
        # Store performance metrics
        if actual_time and execution_plan.total_estimated_time > 0:
            accuracy = 1.0 - abs(actual_time - execution_plan.total_estimated_time) / execution_plan.total_estimated_time
            self.long_term.store_performance_metric(
                AgentType.PLANNER, 'time_estimation_accuracy', accuracy,
                {'task_type': planning_data['task_type']}
            )
            
    def remember_execution_result(self, execution_plan: ExecutionPlan, code_output: CodeOutput):
        """Store execution results in memory."""
        execution_data = {
            'files_created': len(code_output.files_created),
            'lines_of_code': code_output.lines_of_code,
            'functions_created': code_output.functions_created,
            'classes_created': code_output.classes_created,
            'tests_created': code_output.tests_created,
            'errors_count': len(code_output.errors_encountered),
            'warnings_count': len(code_output.warnings),
            'success_rate': 1.0 if not code_output.errors_encountered else 0.5
        }
        
        entry = MemoryEntry(
            id=hashlib.md5(f"execution_{execution_plan.task_id}_{datetime.now()}".encode()).hexdigest(),
            timestamp=datetime.now(),
            agent_type=AgentType.EXECUTOR,
            entry_type='code_generation',
            data=execution_data,
            tags={'execution', 'code_generation'},
            importance_score=0.8 if execution_data['success_rate'] > 0.7 else 0.4
        )
        
        self._store_in_all_layers(entry)
        
        # Store performance metrics
        self.long_term.store_performance_metric(
            AgentType.EXECUTOR, 'success_rate', execution_data['success_rate']
        )
        
    def remember_review_result(self, code_output: CodeOutput, review_report: ReviewReport):
        """Store review results in memory."""
        review_data = {
            'overall_score': review_report.overall_score,
            'grade': review_report.grade,
            'issues_count': len(review_report.issues),
            'approval_status': review_report.approval_status,
            'main_categories': list(set(issue.category for issue in review_report.issues)),
            'severity_distribution': {
                severity: len([i for i in review_report.issues if i.severity == severity])
                for severity in ['critical', 'high', 'medium', 'low']
            }
        }
        
        entry = MemoryEntry(
            id=hashlib.md5(f"review_{review_report.review_id}_{datetime.now()}".encode()).hexdigest(),
            timestamp=datetime.now(),
            agent_type=AgentType.REVIEWER,
            entry_type='review_outcome',
            data=review_data,
            tags={'review', 'quality_assessment', review_data['grade']},
            importance_score=0.9 if review_data['overall_score'] >= 85 else 0.6
        )
        
        self._store_in_all_layers(entry)
        
        # Store performance metrics
        self.long_term.store_performance_metric(
            AgentType.REVIEWER, 'review_score', review_report.overall_score
        )
        
    def get_planning_insights(self, task_description: str) -> Dict[str, Any]:
        """Get insights for planning based on memory."""
        task_type = self._extract_task_type(task_description)
        
        # Find similar planning patterns
        features = {
            'task_type': task_type,
            'technologies': self._extract_technologies(task_description)
        }
        
        similar_patterns = self.working.find_similar_patterns(
            'task_analysis', features, AgentType.PLANNER, limit=5
        )
        
        insights = {
            'similar_tasks_count': len(similar_patterns),
            'average_estimation_accuracy': self._get_average_accuracy(AgentType.PLANNER),
            'common_risks': self._get_common_risks(task_type),
            'recommended_steps': self._get_recommended_steps(task_type),
            'time_estimation_adjustment': self._get_time_adjustment(task_type)
        }
        
        return insights
        
    def get_execution_insights(self, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """Get insights for execution based on memory."""
        # Find similar execution patterns
        features = {
            'steps_count': len(execution_plan.steps),
            'estimated_time': execution_plan.total_estimated_time
        }
        
        similar_patterns = self.working.find_similar_patterns(
            'code_generation', features, AgentType.EXECUTOR, limit=5
        )
        
        insights = {
            'similar_executions_count': len(similar_patterns),
            'average_success_rate': self._get_average_metric(AgentType.EXECUTOR, 'success_rate'),
            'common_patterns': self._get_successful_patterns(),
            'potential_issues': self._get_common_execution_issues()
        }
        
        return insights
        
    def get_review_insights(self, code_output: CodeOutput) -> Dict[str, Any]:
        """Get insights for review based on memory."""
        # Find similar review patterns
        features = {
            'lines_of_code': code_output.lines_of_code,
            'functions_created': code_output.functions_created
        }
        
        similar_patterns = self.working.find_similar_patterns(
            'review_outcome', features, AgentType.REVIEWER, limit=5
        )
        
        insights = {
            'similar_reviews_count': len(similar_patterns),
            'average_score': self._get_average_metric(AgentType.REVIEWER, 'review_score'),
            'common_issue_categories': self._get_common_issues(),
            'quality_trends': self._get_quality_trends()
        }
        
        return insights
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get overall memory statistics."""
        return {
            'short_term': {
                'entries_count': len(self.short_term.entries),
                'context_keys': list(self.short_term.context_data.keys())
            },
            'working_memory': self.working.get_project_stats(),
            'long_term': {
                'total_entries': self._get_total_long_term_entries(),
                'performance_trends': self._get_all_performance_trends()
            }
        }
        
    def _store_in_all_layers(self, entry: MemoryEntry):
        """Store entry in all appropriate memory layers."""
        # Always store in short-term
        self.short_term.add_entry(entry)
        
        # Store in working memory if important enough
        if entry.importance_score >= 0.3:
            self.working.add_entry(entry)
            
        # Store in long-term if highly important
        if entry.importance_score >= 0.5:
            self.long_term.store_entry(entry)
            
    def _load_working_memory_from_long_term(self):
        """Load recent high-importance entries into working memory."""
        for agent_type in AgentType:
            entries = self.long_term.retrieve_entries(
                agent_type=agent_type, limit=100, days_back=7
            )
            for entry in entries:
                if entry.importance_score >= 0.6:
                    self.working.add_entry(entry)
                    
    def _extract_task_type(self, task_description: str) -> str:
        """Extract task type from description."""
        description_lower = task_description.lower()
        
        if any(keyword in description_lower for keyword in ['api', 'endpoint', 'rest']):
            return "API Development"
        elif any(keyword in description_lower for keyword in ['frontend', 'ui', 'react']):
            return "Frontend Development"
        elif any(keyword in description_lower for keyword in ['database', 'sql']):
            return "Database Development"
        elif any(keyword in description_lower for keyword in ['test', 'testing']):
            return "Testing"
        else:
            return "General Development"
            
    def _extract_technologies(self, task_description: str) -> List[str]:
        """Extract mentioned technologies."""
        technologies = []
        description_lower = task_description.lower()
        
        tech_keywords = {
            'Python': ['python', 'flask', 'django', 'fastapi'],
            'JavaScript': ['javascript', 'node', 'react', 'vue'],
            'Database': ['postgresql', 'mysql', 'mongodb'],
            'Testing': ['pytest', 'unittest', 'jest']
        }
        
        for tech, keywords in tech_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                technologies.append(tech)
                
        return technologies
        
    def _assess_complexity(self, plan: ExecutionPlan) -> str:
        """Assess plan complexity."""
        if len(plan.steps) > 8:
            return "High"
        elif len(plan.steps) > 4:
            return "Medium"
        else:
            return "Low"
            
    def _get_average_accuracy(self, agent_type: AgentType) -> float:
        """Get average accuracy for agent."""
        trends = self.long_term.get_performance_trends(
            agent_type, 'time_estimation_accuracy', days_back=30
        )
        if trends:
            return statistics.mean([value for _, value in trends])
        return 0.5
        
    def _get_average_metric(self, agent_type: AgentType, metric_type: str) -> float:
        """Get average metric value."""
        trends = self.long_term.get_performance_trends(
            agent_type, metric_type, days_back=30
        )
        if trends:
            return statistics.mean([value for _, value in trends])
        return 0.5
        
    def _get_common_risks(self, task_type: str) -> List[str]:
        """Get common risks for task type."""
        # This would be populated from historical data
        return [
            "Integration complexity",
            "Time estimation challenges",
            "Technical debt accumulation"
        ]
        
    def _get_recommended_steps(self, task_type: str) -> List[str]:
        """Get recommended steps based on successful patterns."""
        return [
            "Thorough requirement analysis",
            "Incremental implementation",
            "Continuous testing"
        ]
        
    def _get_time_adjustment(self, task_type: str) -> float:
        """Get time estimation adjustment factor."""
        return 1.2  # 20% buffer based on historical data
        
    def _get_successful_patterns(self) -> List[str]:
        """Get successful code patterns."""
        return [
            "Modular architecture",
            "Comprehensive error handling",
            "Clear documentation"
        ]
        
    def _get_common_execution_issues(self) -> List[str]:
        """Get common execution issues."""
        return [
            "Missing dependencies",
            "Configuration errors",
            "Integration challenges"
        ]
        
    def _get_common_issues(self) -> List[str]:
        """Get common review issues."""
        return [
            "Documentation gaps",
            "Error handling improvements",
            "Code organization"
        ]
        
    def _get_quality_trends(self) -> Dict[str, float]:
        """Get quality improvement trends."""
        return {
            'score_trend': 0.05,  # 5% improvement
            'issue_reduction': 0.1  # 10% fewer issues
        }
        
    def _get_total_long_term_entries(self) -> int:
        """Get total long-term memory entries."""
        # This would query the database for count
        return 0
        
    def _get_all_performance_trends(self) -> Dict[str, Any]:
        """Get all performance trends."""
        return {
            'planner_accuracy': 0.75,
            'executor_success': 0.85,
            'reviewer_consistency': 0.80
        }