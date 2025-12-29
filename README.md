# ORCA: ORchestrating Causal Agent 🐳

ORCA is a multi-agent system for automated causal analysis of data. It provides a comprehensive pipeline for data exploration, causal discovery, and causal inference using a team of specialized AI agents.

## 🚀 Quick Setup

### Option 1: New Server Setup (First Time)

```bash
# Clone repository
git clone <repository-url>
cd ORCA

# Set up environment variables
 .env
# Edit .env file with your configuration

# Initialize server (creates database, seed data, starts services)
./setup_server.sh
```

### Option 2: Connect to Existing Server

```bash
# Clone repository
git clone <repository-url>
cd ORCA
# Connect to existing server
./connect_server.sh
```

## 🏗️ Architecture

ORCA consists of several specialized agents working together:

- **Planner Agent**: Creates execution plans based on user queries
- **Executor Agent**: Executes the planned workflow
- **Data Explorer**: Analyzes and explores datasets
- **Causal Discovery**: Identifies causal relationships
- **Causal Inference**: Estimates causal effects
- **Report Generator**: Creates comprehensive reports

## 📊 Usage

```python
from main import ORCAMainAgent

# Initialize ORCA
agent = ORCAMainAgent(
    db_id="reef_db",
    db_type="postgresql",
    db_config={
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "postgres123",
        "database": "reef_db"
    }
)

# Initialize system
await agent.initialize_system()

# Execute query
result = await agent.execute_query("Analyze the causal relationships in this dataset")
```

## 🔧 Configuration

Edit `.env` file to configure:

- Database connection settings
- Redis configuration
- OpenAI API key
- Other system parameters

## 📁 Project Structure

```
ORCA/
├── agents/                 # Specialized agent implementations
├── core/                   # Core agent base classes and state
├── orchestration/          # Workflow orchestration
├── monitoring/             # Metrics and tracing
├── utils/                  # Utility functions and tools
├── REEF/                   # Sample database and seed data
├── setup_server.sh         # Server initialization script
├── connect_server.sh       # Server connection script
└── main.py                 # Main entry point
```

## 🛠️ Scripts

- **`setup_server.sh`**: Initialize new server (database, seed data, services)
- **`connect_server.sh`**: Connect to existing server and test ORCA system

## 📝 License

This project is licensed under the MIT License.
