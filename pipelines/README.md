# Claude Code Three-Agent Pipeline System

A sophisticated multi-agent system that automates the software development process through three specialized agents: Planning, Execution, and Review. Enhanced with **intelligent memory capabilities**, this system learns from experience, adapts to patterns, and continuously improves code generation quality through persistent knowledge retention.

## 🎯 Overview

The Three-Agent Pipeline System transforms task descriptions into high-quality, production-ready code through a systematic three-stage process:

1. **Planning Agent** 🧠 - Analyzes tasks and creates detailed execution plans with memory-enhanced insights
2. **Execution Agent** ⚡ - Implements code using learned patterns and successful templates
3. **Review Agent** 🔍 - Reviews code quality with adaptive rules and historical context
4. **Memory System** 🧠 - **NEW!** Multi-layered learning and pattern recognition across all agents

## 🏗 Architecture

```
pipelines/
├── agents/
│   ├── planner.py          # Planning Agent implementation
│   ├── executor.py         # Execution Agent implementation  
│   ├── reviewer.py         # Review Agent implementation
│   └── __init__.py
├── basic-pipeline/
│   └── pipeline.yaml       # Main pipeline configuration
├── rules/
│   └── review_rules.json   # Code review rules and standards
├── shared/
│   ├── models.py           # Data models and structures
│   ├── utils.py            # Shared utilities and helpers
│   └── __init__.py
├── logs/
│   └── pipeline_runs/      # Execution logs
├── main.py                 # Main pipeline orchestrator
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages: `pyyaml`, `pytest` (optional)

### Installation

1. Clone or download the pipeline system to your `~/.claude/pipelines/` directory
2. Install dependencies:
   ```bash
   pip install pyyaml
   ```

### Basic Usage

```bash
# Run a simple task
python main.py run --task "Create a REST API for user management"

# Run with custom configuration
python main.py run --task "Build a web scraper" --config basic-pipeline/pipeline.yaml

# List recent pipeline runs
python main.py list --limit 5

# Retry a failed run
python main.py retry --run-id abc123def456
```

### Programmatic Usage

```python
from main import PipelineOrchestrator

# Initialize the orchestrator
orchestrator = PipelineOrchestrator()

# Run a task
result = orchestrator.run_pipeline(
    task_description="Create a data processing pipeline",
    requirements=["Handle CSV files", "Include error handling"]
)

# Check results
print(f"Status: {result.status}")
print(f"Grade: {result.review_report.grade}")
print(f"Files created: {len(result.code_output.files_created)}")
```

## 🧠 Memory System

### **Multi-Layered Memory Architecture**

The pipeline features an advanced memory system that enables continuous learning and improvement:

#### **Short-Term Memory (Session)**
- **Current Context**: Maintains state during pipeline execution
- **Inter-Agent Communication**: Tracks messages between agents
- **Real-Time Decisions**: Stores immediate context for current task

#### **Working Memory (Project)**  
- **Pattern Recognition**: Identifies similar tasks and successful approaches
- **Project Conventions**: Learns and adapts to codebase-specific patterns
- **Recent Learning**: Retains insights from recent pipeline runs

#### **Long-Term Memory (Persistent)**
- **Historical Performance**: SQLite database storing all pipeline results
- **Success Patterns**: Library of proven implementation strategies
- **Quality Trends**: Tracks improvement over time across all metrics
- **Knowledge Base**: Cross-project insights and best practices

### **Memory-Enhanced Capabilities**

**🎯 Planning Agent with Memory:**
- **Pattern Matching**: Finds similar previous tasks (0-5 matches shown in logs)
- **Time Estimation**: Learns from actual vs estimated times (accuracy tracking)
- **Risk Assessment**: Historical risk database with likelihood scores
- **Template Adaptation**: Evolving planning templates based on success rates

**⚡ Execution Agent with Memory:**
- **Code Pattern Reuse**: Library of successful implementation patterns
- **Error Prevention**: Learns from past mistakes to avoid repetition  
- **Template Evolution**: Code generation templates improve over time
- **Convention Learning**: Adapts to project-specific coding standards

**🔍 Review Agent with Memory:**
- **Adaptive Scoring**: Review criteria adjust based on project patterns
- **False Positive Learning**: Improves accuracy by learning from feedback
- **Quality Trending**: Tracks code quality improvements over time
- **Custom Rules**: Develops project-specific quality standards

### **Learning Examples**

```bash
# First run - no memory
[PLANNER] Retrieved planning insights: 0 similar tasks found
[PLANNER] Plan created in 0.01 minutes with 5 steps

# After several API tasks - memory learning
[PLANNER] Retrieved planning insights: 3 similar tasks found
[PLANNER] Time estimation adjusted by 1.2x based on API project history
[PLANNER] Added common API risks: authentication complexity, rate limiting

# Advanced learning - cross-project insights
[EXECUTOR] Using successful REST API pattern from Project Alpha
[REVIEWER] Applying custom API quality rules (87% accuracy improvement)
```

## 🤖 Agent Capabilities

### Planning Agent

- **Task Analysis**: Breaks down complex requirements into manageable steps
- **Technology Detection**: Identifies relevant frameworks and technologies
- **Risk Assessment**: Anticipates potential challenges and dependencies  
- **Time Estimation**: Provides realistic time estimates for each step
- **Deliverable Definition**: Clearly defines expected outputs

**Example Plan Generation:**
```json
{
  "task_description": "Create a REST API for user management",
  "steps": [
    {
      "description": "Analyze existing codebase structure and patterns",
      "priority": "high",
      "estimated_time_minutes": 10
    },
    {
      "description": "Design API endpoints and data models", 
      "priority": "high",
      "estimated_time_minutes": 20
    }
  ],
  "deliverables": ["Functional API endpoints", "API documentation"],
  "risks": ["Integration with existing code may reveal dependencies"]
}
```

### Execution Agent

- **Code Generation**: Creates functional, well-structured code
- **Pattern Recognition**: Follows existing codebase conventions
- **Multi-Language Support**: Handles Python, JavaScript, and more
- **Test Creation**: Generates unit tests for quality assurance
- **Documentation**: Adds inline documentation and README files

**Supported Implementation Types:**
- API/Backend development
- Database models and migrations
- Frontend components and UI
- Utility functions and helpers
- Generic code templates

### Review Agent

- **Quality Assessment**: Comprehensive code quality scoring (0-100)
- **Security Analysis**: Identifies vulnerabilities and security issues
- **Performance Review**: Flags performance bottlenecks
- **Standards Compliance**: Enforces coding standards and conventions
- **Automated Suggestions**: Provides actionable improvement recommendations

**Review Categories:**
- **Simplicity**: Function length, complexity, nesting depth
- **Readability**: Naming conventions, documentation, organization
- **Security**: Vulnerability detection, secret exposure, input validation
- **Performance**: Efficiency issues, optimization opportunities
- **Standards**: Style compliance, best practices adherence

## ⚙️ Configuration

### Pipeline Configuration (`basic-pipeline/pipeline.yaml`)

```yaml
name: "three_agent_pipeline"
version: "1.0.0"

agents:
  planner:
    model: "claude-sonnet-4-20250514"
    temperature: 0.3
    max_tokens: 4000
  
  executor:
    model: "claude-sonnet-4-20250514" 
    temperature: 0.1
    max_tokens: 8000
    
  reviewer:
    model: "claude-sonnet-4-20250514"
    temperature: 0.2
    max_tokens: 4000

workflow:
  - name: "planning"
    agent: "planner"
    timeout_minutes: 10
    
  - name: "execution" 
    agent: "executor"
    timeout_minutes: 45
    depends_on: ["planning"]
    
  - name: "review"
    agent: "reviewer"
    timeout_minutes: 15
    depends_on: ["execution"]

settings:
  retry_attempts: 3
  output_directory: "./pipeline_output"
  log_level: "INFO"

memory:
  project_id: "my_project"           # Unique project identifier
  short_term_size: 1000             # Session memory size
  working_memory_size: 10000        # Project memory size  
  db_path: "~/.claude/pipelines/memory/pipeline_memory.db"
  learning_enabled: true            # Enable/disable learning
  pattern_matching_threshold: 0.3   # Similarity threshold for patterns
```

### Review Rules (`rules/review_rules.json`)

The review system uses comprehensive rules covering:

- **Function complexity limits** (max 20 lines, 4 parameters)
- **Security checks** (SQL injection, XSS, secret exposure)
- **Performance guidelines** (efficiency, optimization)  
- **Coding standards** (PEP8, naming conventions)
- **Documentation requirements** (docstrings, comments)

## 📊 Output Structure

Each pipeline run generates a structured output directory:

```
pipeline_output/
└── {run_id}/
    ├── execution_plan.json      # Detailed execution plan
    ├── pipeline_run.json        # Complete run metadata
    ├── SUMMARY.md              # Human-readable summary
    ├── code_output/            # Generated code files
    │   ├── api_handler.py
    │   ├── models.py
    │   ├── utils.py
    │   └── tests/
    │       └── test_*.py
    ├── reports/
    │   ├── review_report.json   # Detailed review results
    │   └── review_summary.md    # Human-readable review
    └── logs/
        └── execution.log        # Detailed execution logs
```

## 📈 Quality Metrics

The system provides comprehensive quality metrics:

### Scoring System
- **90-100**: Excellent (Production Ready) ✅
- **75-89**: Good (Minor improvements needed) ⚠️
- **60-74**: Acceptable (Several improvements needed) 🔄
- **40-59**: Needs Improvement (Significant issues) ⚠️
- **0-39**: Unacceptable (Critical issues) ❌

### Approval Status
- **Approved**: Ready for deployment
- **Approved with Changes**: Minor fixes needed
- **Rejected**: Significant improvements required

## 🛠 Advanced Usage

### Memory Management

**Viewing Memory Statistics:**
```bash
# Add memory stats command to main.py (future enhancement)
python main.py memory-stats --project my_project
```

**Memory Configuration:**
```yaml
memory:
  project_id: "ecommerce_api"       # Separate memory per project
  learning_enabled: true           # Enable continuous learning
  pattern_matching_threshold: 0.4  # Higher = stricter matching
  
  # Performance tuning
  short_term_size: 2000            # More session memory
  working_memory_size: 20000       # More project patterns
  
  # Custom database location  
  db_path: "/custom/path/memory.db"
```

**Memory Benefits by Project Type:**

| Project Type | Memory Benefits | Learning Focus |
|--------------|----------------|----------------|
| **API Development** | Pattern reuse, authentication templates, error handling | REST conventions, security patterns |
| **Frontend Apps** | Component templates, state management, UI patterns | Framework-specific best practices |
| **Data Processing** | Algorithm optimization, error handling, validation | Performance patterns, data formats |
| **Testing Projects** | Test templates, coverage patterns, mocking strategies | Testing methodologies, edge cases |

### Custom Review Rules

Create custom review rules for specific project needs:

```json
{
  "custom_rules": {
    "max_function_length": 15,
    "require_type_hints": true,
    "enforce_docstrings": true,
    "security_level": "strict"
  },
  "project_specific": {
    "framework": "fastapi",
    "database": "postgresql", 
    "testing_framework": "pytest"
  }
}
```

### Integration with CI/CD

```yaml
# .github/workflows/pipeline.yml
name: Code Pipeline
on: [push, pull_request]

jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Pipeline
        run: |
          python ~/.claude/pipelines/main.py run \\
            --task "${{ github.event.head_commit.message }}" \\
            --config production_config.yaml
```

### Extending Agents

Add custom functionality to agents:

```python
# Custom Planning Agent
class CustomPlanningAgent(PlanningAgent):
    def create_plan(self, task_description, requirements=None):
        # Add custom planning logic
        plan = super().create_plan(task_description, requirements)
        plan.custom_field = self.add_custom_analysis(task_description)
        return plan
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure Python path includes pipeline directory
   export PYTHONPATH="${PYTHONPATH}:~/.claude/pipelines"
   ```

2. **Configuration Not Found**
   ```bash
   # Specify config path explicitly
   python main.py run --task "..." --config basic-pipeline/pipeline.yaml
   ```

3. **Permission Errors**
   ```bash
   # Ensure write permissions for output directory
   chmod 755 ~/.claude/pipelines/pipeline_output
   ```

4. **Memory Database Issues**
   ```bash
   # Check memory database permissions
   ls -la ~/.claude/pipelines/memory/
   
   # Reset memory database if corrupted
   rm ~/.claude/pipelines/memory/pipeline_memory.db
   
   # Disable memory temporarily
   # Set learning_enabled: false in pipeline.yaml
   ```

### Debug Mode

Enable detailed logging:

```python
orchestrator = PipelineOrchestrator()
orchestrator.logger.logger.setLevel("DEBUG")
result = orchestrator.run_pipeline("Your task")
```

## 📝 Best Practices

### Task Descriptions
- **Be specific**: "Create a REST API with user authentication" vs "Make an API"
- **Include requirements**: Mention frameworks, databases, specific features
- **Specify constraints**: Performance requirements, security needs

### Good Examples:
```
✅ "Create a FastAPI REST API with JWT authentication, PostgreSQL database, and comprehensive error handling"

✅ "Build a React component for file upload with progress tracking, validation, and drag-and-drop support"

❌ "Make a website" (too vague)
❌ "Fix the code" (no context)
```

### Configuration Tips
- **Adjust timeouts** based on task complexity
- **Customize review rules** for project standards  
- **Set appropriate temperature** values for different agents
- **Configure output directories** for your workflow

## 🤝 Contributing

### Adding New Agent Types

1. Create agent class inheriting from base agent pattern
2. Implement required methods (`execute`, `validate`, etc.)
3. Add agent configuration to pipeline config
4. Update orchestrator to include new agent

### Extending Review Rules

1. Add new rule categories to `review_rules.json`
2. Implement rule checking logic in `ReviewAgent`
3. Update scoring algorithm to include new rules
4. Add tests for new rule validation

## 📄 License

This project is part of the Claude Code system by Anthropic.

## 🔗 Related Documentation

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Pipeline Configuration Guide](config/README.md)
- [Agent Development Guide](agents/README.md)
- [Review Rules Reference](config/review_rules.md)

---

**Generated by Claude Code Three-Agent Pipeline System** 🤖  
*Transforming ideas into production-ready code through intelligent automation*