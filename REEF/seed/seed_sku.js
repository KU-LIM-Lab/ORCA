const { faker } = require('@faker-js/faker');
const getClient = require('./db');
const { v4: uuidv4 } = require('uuid');

const COLORS = ['Black', 'White', 'Red', 'Blue', 'Green', 'Gray'];
const OPTIONS = ['64GB', '128GB', '256GB', 'Small', 'Medium', 'Large'];
const VARIANT_NAMES = ['Standard', 'Pro', 'Lite', 'Plus'];

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log('Connected. Seeding skus...');

  const DAY_MS = 24 * 60 * 60 * 1000;
  const today = new Date();

  function clampPrice(x) {
    return x < 1 ? 1 : x;
  }

  // Get products + categories + brands info at once
  // Assuming products.product_name is "BrandName XXX", use prefix join
  const resProducts = await client.query(`
    SELECT
      p.product_id,
      p.product_name,
      p.category_id,
      p.stock_quantity,
      p.created_at       AS product_created_at,
      c.category_popularity_score,
      b.brand_strength_score
    FROM products p
    JOIN categories c ON p.category_id = c.category_id
    LEFT JOIN brands b 
      ON p.product_name LIKE b.brand_name || ' %'
  `);

  const products = resProducts.rows;

  if (products.length === 0) {
    console.error('🚨 No products found. Please seed products first.');
    await client.end();
    return;
  }

  const usedSkuCodes = new Set();
  let count = 0;
  const TARGET_SKU_COUNT = 1000;

  while (count < TARGET_SKU_COUNT) {
    const product = faker.helpers.arrayElement(products);

    const color = faker.helpers.arrayElement(COLORS);
    const option = faker.helpers.arrayElement(OPTIONS);
    const variantBase = faker.helpers.arrayElement(VARIANT_NAMES);
    const variantName = `${variantBase} ${option}`;

    // Simple SKU code: first 8 chars of product_id + color + option + variant
    const skuCode = `${product.product_id.slice(0, 8)}-${color}-${option}-${variantBase}`;
    if (usedSkuCodes.has(skuCode)) continue;
    usedSkuCodes.add(skuCode);

    // ───────── Price: price = 20 + 8*cat_pop + brand_strength + ε_p ─────────
    const catScore = product.category_popularity_score ?? 0;
    const brandScore = product.brand_strength_score ?? 0;
    const epsP = faker.number.float({ mean: 0, stddev: 5 });

    let price = 20 + 8 * catScore + brandScore + epsP;
    price = clampPrice(price);

    // ───────── Date: created_at = product.created_at, available_from = created_at ─────────
    const createdAt = new Date(product.product_created_at);
    let availableFrom = new Date(createdAt); // Set same

    // ───────── discontinued_at + is_active ─────────
    // If stock is 0, discontinue at some point, otherwise NULL
    let discontinuedAt = null;
    let isActive = true;

    if (product.stock_quantity <= 0) {
      // Discontinue at any time between creation and today
      const endTimestamp = today.getTime();
      const startTimestamp = createdAt.getTime();
      const randTime = faker.number.int({
        min: startTimestamp,
        max: endTimestamp,
      });
      discontinuedAt = new Date(randTime);
      isActive = false;
    } else {
      // If stock exists and today is before discontinued, then active
      isActive = true;
    }

    await client.query(
      `
      INSERT INTO sku (
        sku_id,
        product_id,
        sku_code,
        variant_name,
        color,
        variant_option,
        is_active,
        price,
        created_at,
        available_from,
        discontinued_at
      ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
    `,
      [
        uuidv4(),
        product.product_id,
        skuCode,
        variantName,
        color,
        option,
        isActive,
        price,
        createdAt,
        availableFrom,
        discontinuedAt,
      ]
    );

    count++;
  }

  console.log(`✅ ${count} skus inserted!`);
  await client.end();
};