#!/usr/bin/env bash
# ORCA server bootstrap script
# Use this when setting up ORCA on a fresh machine for the first time.

set -e  # Exit immediately if a command fails (easier debugging)

echo "============================================================"
echo "🐳 ORCA setup"
echo "============================================================"

# ---- Conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate userstudy

echo "Using python: $(which python)"
echo "Using pip:    $(which pip)"

# ---- Python deps ----
echo "📦 Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements (assumes requirements.txt is valid for this OS)
python -m pip install -r requirements.txt

# (Optional) Fix commonly flaky deps explicitly
python -m pip install PyYAML psycopg2-binary

# ---- Node.js / npm (REEF seed) ----
echo "📋 Setting up Node.js environment..."

# Check nvm
if ! command -v nvm >/dev/null 2>&1; then
  echo "📦 nvm not found. Installing nvm..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

  # Load nvm into current shell
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
else
  echo "✅ nvm is already installed."
fi

# Install & use LTS Node
nvm install --lts
nvm use --lts

echo "Node: $(node -v)"
echo "npm:  $(npm -v)"

SEED_DIR="REEF/seed_R1"
cd "$SEED_DIR" || exit 1

if [ ! -f "package.json" ]; then
  echo "📦 package.json not found. Creating and installing seed dependencies..."
  npm init -y
  npm install @faker-js/faker pg dotenv uuid seedrandom
elif [ -f "package-lock.json" ]; then
  echo "📦 package-lock.json found → running npm ci"
  npm ci
else
  echo "📦 package.json found → running npm install"
  npm install
fi

cd - >/dev/null

echo "🚀 Starting ORCA server bootstrap..."

# ---- Step 1: env ----
echo "📋 Step 1: Environment variables"
if [ ! -f ".env" ]; then
    echo "⚠️  .env not found. Copying env.example → .env"
    cp env.example .env
    echo "⚠️  Please edit .env with your local configuration before continuing:"
    echo "   nano .env"
    echo "   # or"
    echo "   code .env"
    read -p "Press Enter to continue..."
else
    echo "✅ .env file exists."
fi

# ---- Step 2: Postgres init ----
echo "📋 Step 2: Initialize PostgreSQL database"
echo "Dropping and recreating database: reef_db"

dropdb --if-exists reef_db 2>/dev/null || true
createdb reef_db

echo "Running DDL..."
psql -d reef_db -f REEF/REEF_ddl_continuous.sql

echo "✅ Database schema created."

# ---- Step 3: Seed data ----
echo "📋 Step 3: Generate seed data"
node REEF/seed_R1/run_all_seeds.js
echo "✅ Seed data generated."

# ---- Step 4: Redis ----
echo "📋 Step 4: Redis check"
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is already running (port 6379)."
else
    echo "Redis is not running. Starting Redis..."

    if command -v redis-stack-server > /dev/null 2>&1; then
        echo "Using redis-stack-server..."
        redis-stack-server --daemonize yes
    elif command -v redis-server > /dev/null 2>&1; then
        echo "Using redis-server..."
        redis-server --daemonize yes
    else
        echo "❌ Redis server not found."
        echo "   Please install redis-stack-server or redis-server."
        exit 1
    fi

    sleep 1
    redis-cli ping > /dev/null 2>&1
    echo "✅ Redis started."
fi

# ---- Step 5: Connection tests ----
echo "📋 Step 5: Connection tests"

python - << 'PY'
import sys
sys.path.append(".")
from utils.settings import POSTGRES_CONFIG, REDIS_CONFIG
from utils.database import Database
import redis

print("🔍 Testing PostgreSQL...")
db = Database(db_type="postgresql", config=POSTGRES_CONFIG)
res = db.run_query("SELECT COUNT(*) as user_count FROM users;")
print(f"✅ PostgreSQL OK - users: {res[0][0]}")

print("🔍 Testing Redis...")
r = redis.Redis(**REDIS_CONFIG)
r.ping()
print("✅ Redis OK")

print("🎉 All connection tests passed!")
PY

# ---- Metadata ----
echo "📋 Generating ORCA metadata..."
python -m utils.data_prep.runner
echo "✅ Metadata generation completed."

echo ""
echo "🎉 ORCA server bootstrap completed!"
echo ""
echo "Server info:"
echo "- PostgreSQL: localhost:5432/reef_db"
echo "- Redis:      localhost:6379"
echo ""
