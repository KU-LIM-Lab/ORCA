const { faker } = require('@faker-js/faker');
const getClient = require('./db');

const brandMap = {
  'Electronics': ['SAMSUNG', 'LG', 'APPLE', 'SONY', 'XIAOMI'],
  'Fashion': ['NIKE', 'ADIDAS', 'NEW BALANCE', 'PUMA', 'REEBOK'],
  'Home Appliances': ['CUCKOO', 'DYSON', 'BALMUDA', 'PHILIPS', 'ELECTROLUX'],
  'Books': ['MUNHAKDONGNE', 'MINUMSA', 'WISDOM HOUSE', 'HANGYOREH PUBLISHING', 'CHANGBI'],
  'Toys': ['LEGO', 'YOUNGTOYS', 'BANDAI', 'TAKARA TOMY', 'MIMI WORLD'],
  'Beauty': ['SULWHASOO', 'LANEIGE', 'INNISFREE', 'CLIO', 'MISSHA'],
  'Groceries': ['OTTOGI', 'NONGSHIM', 'CJ', 'PULMUONE', 'DONGWON'],
  'Furniture': ['HANSSEM', 'IKEA', 'ILOOM', 'DONGSEO FURNITURE', 'ACE BED'],
  'Sports': ['FILA', 'DESCENTE', 'UNDER ARMOUR', 'NIKE', 'CALLAWAY'],
  'Automotive': ['BULLSONE', 'CAREX', 'INAVI', 'CARTEM']
};

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log("Connected to DB. Inserting brands...");

  // 1) Get parent info from categories table
  const res = await client.query(`SELECT category_id, name, created_at, category_popularity_score FROM categories`);
  const categoryDict = {};
  res.rows.forEach(row => {
    categoryDict[row.name] = {
      id: row.category_id,
      created_at: row.created_at,
      popularity: row.category_popularity_score
    };
  });

  const today = new Date();

  function randomDateAfter(baseDate, minDays=0, maxDays=30) {
    const offset = faker.number.int({ min: minDays, max: maxDays });
    return new Date(baseDate.getTime() + offset * 24 * 60 * 60 * 1000);
  }

  for (const [categoryName, brandList] of Object.entries(brandMap)) {
    const category = categoryDict[categoryName];
    if (!category) {
      console.warn(`⚠ Category ${categoryName} not found in DB.`);
      continue;
    }

    for (const brandName of brandList) {
      const brand_id = faker.string.uuid();

      // SCM 1) created_at = category.created_at + U(0,30)
      const created_at = randomDateAfter(category.created_at, 0, 30);

      // SCM 2) updated_at = created_at
      const updated_at = created_at;

      // SCM 3) brand_strength_score = 0.8 * category_popularity_score + ε_B
      const epsilonB = faker.number.float({ mean: 0, stddev: 0.3 });
      const brand_strength_score = 0.8 * category.popularity + epsilonB;

      await client.query(`
        INSERT INTO brands (
          brand_id,
          category_id,
          brand_name,
          created_at,
          updated_at,
          brand_strength_score
        ) VALUES ($1, $2, $3, $4, $5, $6)
      `, [
        brand_id,
        category.id,
        brandName,
        created_at,
        updated_at,
        brand_strength_score
      ]);
    }
  }

  console.log("All brands inserted!");
  await client.end();
};