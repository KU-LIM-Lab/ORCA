# ORCA: ORchestrating Causal Agent

This repository contains the **official implementation** of the paper  
**“ORCA: Orchestrating Causal Agent”**, which proposes an interactive multi-agent framework for end-to-end causal analysis on relational databases.

---

## Setup

### 1. Create conda environment

```bash
conda create -n ORCA python=3.11 -y
conda activate ORCA
```

### 2. Install dependencies and initialize

```bash
zsh setup_server.sh
```

### 3. Environment variables

Create a `.env` file in the project root with PostgreSQL, Redis, and API key settings.

```bash
cp env.example .env
```

---

## Running ORCA

### Run once (non-interactive)

```bash
python main.py \
  --query "Estimate the causal effect of coupon usage on total order amount" \
  --db-id reef_db \
```

Optional flags:
- `--interactive` : enable human-in-the-loop checkpoints

---

### Interactive session

```bash
python main.py --db-id reef_db --interactive
```

## License

MIT License