import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# relative to project root
ROOT_DIR = Path(__file__).resolve().parents[1]

# load .env
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# load config.yml
CONFIG_PATH = ROOT_DIR / "_config.yml"
with open(CONFIG_PATH, "r") as f:
    raw_config = yaml.safe_load(f)

# substitute environment variables
def resolve_env(value):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    return value

def resolve_nested(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = resolve_nested(v)
        else:
            d[k] = resolve_env(v)
    return d

# final config object
CONFIG = resolve_nested(raw_config)

# load DB settings
POSTGRES_CONFIG = CONFIG.get("database", {}).get("postgresql", {})
SQLITE_CONFIG = CONFIG.get("database", {}).get("sqlite", {})
REDIS_CONFIG = CONFIG.get("redis", {})