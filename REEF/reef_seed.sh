# echo "데이터베이스를 초기화합니다..."
dropdb --if-exists reef_db 2>/dev/null || true
createdb reef_db

# DDL 실행
echo "DDL을 실행합니다..."
psql -d reef_db -f REEF_ddl.sql

if [ $? -eq 0 ]; then
    echo "✅ 데이터베이스 스키마 생성 완료"
else
    echo "❌ 데이터베이스 스키마 생성 실패"
    exit 1
fi

# 3단계: 시드 데이터 생성
echo "📋 3단계: 시드 데이터 생성"
echo "샘플 데이터를 생성합니다..."

# 시드 데이터 실행
echo "시드 데이터를 생성합니다..."
node seed/run_all_seeds.js

if [ $? -eq 0 ]; then
    echo "✅ 시드 데이터 생성 완료"
else
    echo "❌ 시드 데이터 생성 실패"
    exit 1
fi