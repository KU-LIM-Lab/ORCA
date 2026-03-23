const { faker } = require('@faker-js/faker');
const getClient = require('./db'); 

const categoryNames = [
  'Electronics',
  'Fashion',
  'Home Appliances',
  'Books',
  'Toys',
  'Beauty',
  'Groceries',
  'Furniture',
  'Sports',
  'Automotive'
];

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log(`Connected. Inserting ${categoryNames.length} categories...`);

  const today = new Date();
  const DAY_MS = 24 * 60 * 60 * 1000;

  // Uniform(a, b) days ago → Return actual date
  const daysAgo = (minDays, maxDays) => {
    const d = faker.number.int({ min: minDays, max: maxDays });
    return new Date(today.getTime() - d * 24 * 60 * 60 * 1000);
  };

  const START = new Date("2024-01-01");

  function randomFromStart(startDate, minDays, maxDays) {
    const offsetDays = faker.number.int({ min: minDays, max: maxDays });
    return new Date(startDate.getTime() + offsetDays * DAY_MS);
  }

  for (const name of categoryNames) {

    const category_id = faker.string.uuid();
    const description = faker.commerce.productDescription();

    // 1) created_at = U(30, 365 days ago)
    const created_at = randomFromStart(START, 0, 30);

    // 2) updated_at = created_at
    const updated_at = created_at;

    // 3) category_popularity_score = N(0, 1)
    const category_popularity_score = faker.number.float({
      mean: 0,
      stddev: 1
    });

    // 4) parent_id — No parent category here (sample code provided below if you want to create hierarchy)
    const parent_id = null;

    try {
      await client.query(`
        INSERT INTO categories (
          category_id,
          parent_id,
          name,
          description,
          created_at,
          updated_at,
          category_popularity_score
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      `, [
        category_id,
        parent_id,
        name,
        description,
        created_at,
        updated_at,
        category_popularity_score
      ]);
    } catch (err) {
      console.error(`Error inserting category "${name}": ${err.message}`);
    }
  }

  console.log("All categories inserted!");
  await client.end();
};