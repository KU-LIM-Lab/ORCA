const { faker } = require("@faker-js/faker");
const getClient = require("./db");

const PRODUCT_COUNT = 1000;

// Product keyword list for each category
const productMap = {
  'Electronics': ['Smartphone', 'Laptop', 'Monitor', 'Earphones', 'Tablet'],
  'Fashion': ['Sneakers', 'Jacket', 'Shirt', 'Pants', 'Hat'],
  'Home Appliances': ['Vacuum Cleaner', 'Air Purifier', 'Refrigerator', 'Microwave', 'Washing Machine'],
  'Books': ['Novel', 'Self-Help Book', 'Essay Collection', 'Magazine', 'Comic Book'],
  'Toys': ['Model Kit', 'Doll', 'Puzzle', 'Building Blocks', 'Board Game'],
  'Beauty': ['Essence', 'Cream', 'Compact', 'Cushion Foundation', 'Lip Balm'],
  'Groceries': ['Instant Noodles', 'Instant Rice', 'Canned Tuna', 'Snacks', 'Tofu'],
  'Furniture': ['Bed', 'Desk', 'Drawer', 'Sofa', 'Table'],
  'Sports': ['Sportswear', 'Running Shoes', 'Golf Balls', 'Racket', 'Yoga Mat'],
  'Automotive': ['Engine Coating', 'Dash Cam', 'Air Freshener', 'Windshield Wipers', 'Car Wash Supplies'],
};

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log("Connected. Seeding products...");

  const DAY_MS = 24 * 60 * 60 * 1000;
  const today = new Date();

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  // ✅ Get brand, category + brand creation date info
  const res = await client.query(`
    SELECT 
      b.brand_id,
      b.brand_name,
      b.category_id,
      b.created_at AS brand_created_at,
      c.name AS category_name
    FROM brands b
    JOIN categories c ON b.category_id = c.category_id
  `);
  const brands = res.rows;

  for (let i = 0; i < PRODUCT_COUNT; i++) {
    const brand = faker.helpers.arrayElement(brands);
    const categoryName = brand.category_name;
    const keywords = productMap[categoryName];

    // Skip if category not in productMap
    if (!keywords) {
      console.warn(`⚠ No product keywords for category: ${categoryName}`);
      continue;
    }

    const productKeyword = faker.helpers.arrayElement(keywords);
    const productName = `${brand.brand_name} ${productKeyword}`;
    const productId = faker.string.uuid();

    // ───────── stock_quantity (exogenous) ─────────
    const stock_quantity = faker.number.int({ min: 10, max: 300 });

    const thumbnail_url = faker.image.urlPicsumPhotos();
    const description = faker.commerce.productDescription();

    // ───────── created_at = brand.created_at + U(0, 7 days) ─────────
    const brandCreated = new Date(brand.brand_created_at);
    const offsetDays = faker.number.int({ min: 0, max: 7 });
    let created_at = new Date(brandCreated.getTime() + offsetDays * DAY_MS);
    if (created_at > today) created_at = today;

    // ───────── is_active: a* = -1 + 1.5*I(stock>0) - 0.01*days_since_created + ε_a ─────────
    const days_since_created = (today.getTime() - created_at.getTime()) / DAY_MS;
    const I_stock = stock_quantity > 0 ? 1 : 0;
    const epsilonA = faker.number.float({ mean: 0, stddev: 1 });

    const aStar = -1 + 1.5 * I_stock - 0.01 * days_since_created + epsilonA;
    const pActive = sigmoid(aStar);
    const is_active = Math.random() < pActive;

    // ───────── updated_at = created_at (simple) ─────────
    const updated_at = created_at;

    try {
      await client.query(
        `
        INSERT INTO products (
          product_id,
          category_id,
          product_name,
          description,
          stock_quantity,
          thumbnail_url,
          is_active,
          created_at,
          updated_at
        ) VALUES (
          $1, $2, $3, $4, $5, $6, $7, $8, $9
        )
      `,
        [
          productId,
          brand.category_id,
          productName,
          description,
          stock_quantity,
          thumbnail_url,
          is_active,
          created_at,
          updated_at,
        ]
      );
    } catch (err) {
      console.error(`Error at row ${i}: ${err.message}`);
    }

    if (i % 200 === 0) console.log(`Inserted ${i} products...`);
  }

  console.log("All products inserted!");
  await client.end();
};