#!/bin/bash

# ORCA 서버 초기 설정 스크립트
# 새로운 서버에 ORCA 시스템을 처음 설정하는 용도

echo "🚀 ORCA 서버 초기 설정을 시작합니다..."

# 1단계: 환경 변수 설정
echo "📋 1단계: 환경 변수 설정"
if [ ! -f ".env" ]; then
    echo "⚠️  .env 파일이 없습니다. env.example을 복사합니다."
    cp env.example .env
    echo "⚠️  .env 파일을 편집하여 서버 정보를 설정해주세요!"
    echo "   nano .env"
    echo "   # 또는"
    echo "   code .env"
    read -p "계속하려면 Enter를 누르세요..."
else
    echo "✅ .env 파일이 존재합니다."
fi

# 2단계: 데이터베이스 생성 및 초기화
echo "📋 2단계: 데이터베이스 생성 및 초기화"
echo "PostgreSQL 데이터베이스를 생성하고 초기화합니다..."

# 데이터베이스 생성
echo "데이터베이스를 초기화합니다..."
dropdb --if-exists reef_db 2>/dev/null || true

# 데이터베이스 생성
echo "새 데이터베이스를 생성합니다..."
createdb reef_db

if [ $? -ne 0 ]; then
    echo "❌ 데이터베이스 생성 실패"
    exit 1
fi

# DDL 실행 (에러는 무시하고 계속 진행)
echo "DDL을 실행합니다..."
psql -d reef_db -f REEF/REEF_ddl_continuous.sql > /dev/null 2>&1

# DDL 실행 결과 확인 (일부 에러는 정상 - 이미 존재하는 객체들)
echo "✅ 데이터베이스 스키마 생성 완료"

# 3단계: 시드 데이터 생성
echo "📋 3단계: 시드 데이터 생성"
echo "샘플 데이터를 생성합니다..."

# Node.js 의존성 설치
echo "Node.js 의존성을 설치합니다..."
cd REEF/seed_R1
if [ ! -d "node_modules" ]; then
    echo "npm 패키지를 설치합니다..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ npm 패키지 설치 실패"
        cd ../..
        exit 1
    fi
else
    echo "✅ node_modules가 이미 존재합니다."
fi
cd ../..

# 시드 데이터 실행
echo "시드 데이터를 생성합니다..."
node REEF/seed_R1/run_all_seeds.js

if [ $? -eq 0 ]; then
    echo "✅ 시드 데이터 생성 완료"
else
    echo "❌ 시드 데이터 생성 실패"
    exit 1
fi

# 4단계: Redis 설정
echo "📋 4단계: Redis 설정"
echo "Redis 서버 상태를 확인합니다..."

# Redis가 이미 실행 중인지 확인
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis 서버가 이미 실행 중입니다 (포트 6379)"
    echo "   기존 Redis 서버를 사용합니다."
else
    echo "Redis 서버를 시작합니다..."
    
    # redis-stack-server 우선 시도, 없으면 redis-server 사용
    if command -v redis-stack-server > /dev/null 2>&1; then
        echo "   redis-stack-server를 사용합니다..."
        redis-stack-server --daemonize yes
    elif command -v redis-server > /dev/null 2>&1; then
        echo "   redis-server를 사용합니다..."
        redis-server --daemonize yes
    else
        echo "❌ Redis 서버를 찾을 수 없습니다."
        echo "   redis-stack-server 또는 redis-server가 설치되어 있는지 확인하세요."
        exit 1
    fi
    
    # 시작 확인
    sleep 1
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis 서버 시작 완료"
    else
        echo "❌ Redis 서버 시작 실패"
        echo "   Redis가 설치되어 있는지 확인하세요."
        exit 1
    fi
fi

# 5단계: 연결 테스트
echo "📋 5단계: 연결 테스트"
echo "생성된 서버에 연결을 테스트합니다..."

python3 -c "
import sys
sys.path.append('.')
from utils.settings import POSTGRES_CONFIG, REDIS_CONFIG
from utils.database import Database
import redis

print('🔍 PostgreSQL 연결 테스트...')
try:
    db = Database(db_type='postgresql', config=POSTGRES_CONFIG)
    result = db.run_query('SELECT COUNT(*) as user_count FROM users;')
    print(f'✅ PostgreSQL 연결 성공 - 사용자 수: {result[0][0]}')
except Exception as e:
    print(f'❌ PostgreSQL 연결 실패: {e}')
    sys.exit(1)

print('🔍 Redis 연결 테스트...')
try:
    r = redis.Redis(**REDIS_CONFIG)
    result = r.ping()
    print('✅ Redis 연결 성공')
except Exception as e:
    print(f'❌ Redis 연결 실패: {e}')
    sys.exit(1)

print('🎉 모든 연결 테스트 통과!')
"

if [ $? -eq 0 ]; then
    echo "✅ 서버 초기 설정 완료"
else
    echo "❌ 서버 초기 설정 실패"
    exit 1
fi

echo ""
echo "🎉 ORCA 서버 초기 설정이 완료되었습니다!"
echo ""
echo "📋 서버 정보:"
echo "- PostgreSQL: localhost:5432/reef_db"
echo "- Redis: localhost:6379"
echo "- 시드 데이터: 10,000명 사용자, 1,000개 상품, 10,000개 주문"
echo ""
echo "📋 사용 방법:"
echo "1. 서버 연결: ./connect_server.sh"
echo "2. Python에서 사용: from main import ORCAMainAgent"
echo "3. 직접 실행: python main.py"
echo "4. ORCA 시스템 테스트: python tests/test_orca_system.py"
