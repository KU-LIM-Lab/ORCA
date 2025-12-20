#!/bin/bash

# REEF 데이터를 이용한 ATE 데이터 생성 스크립트

set -e

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REEF_V2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REEF_V2_DIR/.." && pwd)"

# Python 경로 설정
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 기본값
QUERIES_FILE="${REEF_V2_DIR}/configs/ate_queries.yaml"
OUTPUT_FILE="${REEF_V2_DIR}/outputs/ate_results.json"
DB_NAME="reef_db"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --queries)
            QUERIES_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --db-name)
            DB_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --queries PATH    Path to queries YAML file (default: configs/ate_queries.yaml)"
            echo "  --output PATH     Output JSON file path (default: outputs/ate_results.json)"
            echo "  --db-name NAME    Database name (default: reef_db)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 출력 디렉토리 생성
OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUTPUT_DIR"

# Python 스크립트 실행
echo "🚀 Generating ATE data from REEF database..."
echo "   Queries file: $QUERIES_FILE"
echo "   Output file: $OUTPUT_FILE"
echo "   Database: $DB_NAME"
echo ""

cd "$PROJECT_ROOT"
python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from REEF_v2.src.generate_ate_data import generate_ate_data, load_queries_from_yaml
import json

queries = load_queries_from_yaml('$QUERIES_FILE')
results = generate_ate_data(
    queries=queries,
    db_name='$DB_NAME',
    output_path='$OUTPUT_FILE',
    verbose=True
)
"

echo ""
echo "✅ ATE data generation completed!"
echo "   Results saved to: $OUTPUT_FILE"

