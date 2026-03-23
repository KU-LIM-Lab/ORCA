# echo "Initializing database..."
dropdb --if-exists reef_db 2>/dev/null || true
createdb reef_db

# Execute DDL
echo "Executing DDL..."
psql -d reef_db -f REEF/REEF_ddl.sql

if [ $? -eq 0 ]; then
    echo "✅ Database schema creation complete"
else
    echo "❌ Database schema creation failed"
    exit 1
fi

# Step 3: Generate seed data
echo "📋 Step 3: Generate seed data"
echo "Generating sample data..."

# Execute seed data
echo "Generating seed data..."
node REEF/seed/run_all_seeds.js

if [ $? -eq 0 ]; then
    echo "✅ Seed data generation complete"
else
    echo "❌ Seed data generation failed"
    exit 1
fi