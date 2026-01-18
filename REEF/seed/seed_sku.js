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

  // products + categories + brands 정보 한 번에 가져오기
  // products.product_name 이 "브랜드명 XXX" 라는 가정 하에 prefix join
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

    // 간단한 SKU 코드: product_id 앞 8자리 + color + option + variant
    const skuCode = `${product.product_id.slice(0, 8)}-${color}-${option}-${variantBase}`;
    if (usedSkuCodes.has(skuCode)) continue;
    usedSkuCodes.add(skuCode);

    // ───────── 가격: price = 20 + 8*cat_pop + brand_strength + ε_p ─────────
    const catScore = product.category_popularity_score ?? 0;
    const brandScore = product.brand_strength_score ?? 0;
    const epsP = faker.number.float({ mean: 0, stddev: 5 });

    let price = 20 + 8 * catScore + brandScore + epsP;
    price = clampPrice(price);

    // ───────── 날짜: created_at = product.created_at, available_from = created_at ─────────
    const createdAt = new Date(product.product_created_at);
    let availableFrom = new Date(createdAt); // 동일하게 설정

    // ───────── discontinued_at + is_active ─────────
    // 재고가 0이면 어떤 시점에 단종, 아니면 NULL
    let discontinuedAt = null;
    let isActive = true;

    if (product.stock_quantity <= 0) {
      // 생성 후 ~오늘 사이 아무 때나 단종시킴
      const endTimestamp = today.getTime();
      const startTimestamp = createdAt.getTime();
      const randTime = faker.number.int({
        min: startTimestamp,
        max: endTimestamp,
      });
      discontinuedAt = new Date(randTime);
      isActive = false;
    } else {
      // 재고가 있고, 오늘이 아직 discontinued 이전이면 활성
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