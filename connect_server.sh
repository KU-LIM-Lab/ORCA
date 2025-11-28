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

${PYTHON:-python} -c "
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
    
    # Redis 서버 정보 확인
    try:
        info = r.info('server')
        redis_version = info.get('redis_version', 'unknown')
        redis_mode = info.get('redis_mode', 'unknown')
        print(f'   버전: {redis_version} ({redis_mode})')
    except:
        pass
except Exception as e:
    print(f'❌ Redis 서버 연결 실패: {e}')
    print('   Redis 서버가 실행 중인지 확인하세요:')
    print('   - redis-stack-server (권장)')
    print('   - redis-server')
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

# 3단계: 메타데이터 확인 및 생성
echo ""
echo "📋 3단계: 메타데이터 확인 및 생성"
echo "데이터베이스 메타데이터를 확인합니다..."

${PYTHON:-python} -c "
import sys
sys.path.append('.')
from utils.redis_client import redis_client

db_id = 'reef_db'
metadata_key = f'{db_id}:metadata:table_names'

try:
    # 메타데이터 존재 여부 확인
    table_names = redis_client.smembers(metadata_key)
    # bytes를 문자열로 변환하여 확인
    if table_names and len(table_names) > 0:
        count = len(table_names)
        print(f'✅ 메타데이터가 존재합니다. ({count}개 테이블)')
        sys.exit(0)
    else:
        print('⚠️  메타데이터가 없습니다.')
        print('   메타데이터를 생성합니다...')
        sys.exit(1)
except Exception as e:
    print(f'⚠️  메타데이터 확인 중 오류: {e}')
    print('   메타데이터를 생성합니다...')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "📦 메타데이터 생성 중..."
    ${PYTHON:-python} -m utils.data_prep.runner
    if [ $? -eq 0 ]; then
        echo "✅ 메타데이터 생성 완료"
    else
        echo "⚠️  메타데이터 생성 중 오류가 발생했습니다."
        echo "   수동으로 실행할 수 있습니다: python -m utils.data_prep.runner"
    fi
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
