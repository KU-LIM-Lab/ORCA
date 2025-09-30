#!/bin/bash

# ORCA 서버 연결 스크립트
# 기존에 설정된 서버에 연결하는 용도

echo "🔗 ORCA 서버에 연결합니다..."

# 1단계: 환경 변수 설정 확인
echo "📋 1단계: 환경 변수 설정 확인"
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

# 2단계: 서버 연결 테스트
echo "📋 2단계: 서버 연결 테스트"
echo "PostgreSQL과 Redis 서버에 연결을 테스트합니다..."

python3 -c "
import sys
sys.path.append('.')
from utils.settings import POSTGRES_CONFIG, REDIS_CONFIG
from utils.database import Database
import redis

print('🔍 PostgreSQL 서버 연결 테스트...')
try:
    db = Database(db_type='postgresql', config=POSTGRES_CONFIG)
    result = db.run_query('SELECT 1 as test;')
    print('✅ PostgreSQL 서버 연결 성공')
    print(f'   서버: {POSTGRES_CONFIG.get(\"host\", \"unknown\")}:{POSTGRES_CONFIG.get(\"port\", \"unknown\")}')
    print(f'   데이터베이스: {POSTGRES_CONFIG.get(\"dbname\", \"unknown\")}')
except Exception as e:
    print(f'❌ PostgreSQL 서버 연결 실패: {e}')
    print('   .env 파일의 PostgreSQL 설정을 확인하세요.')
    sys.exit(1)

print('🔍 Redis 서버 연결 테스트...')
try:
    r = redis.Redis(**REDIS_CONFIG)
    result = r.ping()
    print('✅ Redis 서버 연결 성공')
    print(f'   서버: {REDIS_CONFIG.get(\"host\", \"unknown\")}:{REDIS_CONFIG.get(\"port\", \"unknown\")}')
except Exception as e:
    print(f'❌ Redis 서버 연결 실패: {e}')
    print('   .env 파일의 Redis 설정을 확인하세요.')
    sys.exit(1)

print('🎉 모든 서버 연결 테스트 통과!')
"

if [ $? -eq 0 ]; then
    echo "✅ 서버 연결 성공"
else
    echo "❌ 서버 연결 실패"
    echo "   .env 파일의 설정을 확인하고 다시 시도하세요."
    exit 1
fi

echo ""
echo "🎉 ORCA 서버 연결이 완료되었습니다!"
echo ""
echo "📋 사용 방법:"
echo "1. Python에서 ORCA 사용:"
echo "   from main import ORCAMainAgent"
echo "   agent = ORCAMainAgent(db_id='reef_db', db_type='postgresql', db_config={...})"
echo "   await agent.initialize_system()"
echo "   result = await agent.execute_query('your query')"
echo ""
echo "2. 직접 실행:"
echo "   python main.py"
echo ""
echo "3. ORCA 시스템 테스트:"
echo "   python tests/test_orca_system.py"
