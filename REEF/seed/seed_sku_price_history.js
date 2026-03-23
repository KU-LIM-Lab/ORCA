const { faker } = require('@faker-js/faker');
const getClient = require('./db');

const ENTRY_COUNT = 500;

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log('Connected. Seeding sku_price_history...');

  const DAY_MS = 24 * 60 * 60 * 1000;

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  // 1. Get promotion list
  const promoRes = await client.query(`
    SELECT promo_id, discount_value, discount_type, start_at, end_at
    FROM promotion
  `);
  const promotions = promoRes.rows;

  // 2. Get SKU list
  const skuRes = await client.query(`
    SELECT sku_id, price
    FROM sku
  `);
  const skus = skuRes.rows;

  if (promotions.length === 0 || skus.length === 0) {
    console.error('🚨 No promotions or SKUs found. Please seed first.');
    await client.end();
    return;
  }

  const usedPairs = new Set();
  let count = 0;

  while (count < ENTRY_COUNT) {
    const promo = faker.helpers.arrayElement(promotions);
    const sku = faker.helpers.arrayElement(skus);

    const key = `${promo.promo_id}-${sku.sku_id}`;
    if (usedPairs.has(key)) continue;
    usedPairs.add(key);

    const originalPrice = Number(sku.price);

    // ───────── discount_price: Apply design formula ─────────
    const dv = Number(promo.discount_value); // amount or rate (0~1)
    const type = promo.discount_type;        // 'amount' or 'rate' (or 'percentage' compatible)
    const isAmount = type === 'amount';
    const isRate = type === 'rate' || type === 'percentage';

    // ε_d ~ N(0, 0.5^2)
    const epsD = faker.number.float({ mean: 0, stddev: 0.5 });

    let discountPriceStar;
    if (isAmount) {
      // Fixed amount discount: price - discount_value
      discountPriceStar = originalPrice - dv + epsD;
    } else if (isRate) {
      // Percentage discount: price * (1 - discount_value), assume dv is ratio like 0.05~0.3
      discountPriceStar = originalPrice * (1 - dv) + epsD;
    } else {
      // Exception: if type is invalid, no discount
      discountPriceStar = originalPrice + epsD;
    }

    let discountPrice = Math.max(0, discountPriceStar);

    // ───────── is_stackable: Stronger discounts cannot be stacked ─────────
    // D_norm = IF(type='amount', discount_value / price, discount_value)
    let D_norm;
    if (isAmount) {
      D_norm = dv / Math.max(originalPrice, 1e-6);
    } else {
      D_norm = dv;
    }

    // ε_s ~ N(0, 0.5^2) approximately
    const epsS = faker.number.float({ mean: 0, stddev: 0.5 });
    const S_star = 0.5 - 4.0 * D_norm + epsS;
    const pStack = sigmoid(S_star);
    const isStackable = Math.random() < pStack;

    // ───────── created_at = promotion.start_at + U(-3, 3 days) ─────────
    const promoStart = new Date(promo.start_at);
    const deltaDays = faker.number.int({ min: -3, max: 3 });
    const createdAt = new Date(promoStart.getTime() + deltaDays * DAY_MS);

    await client.query(
      `
      INSERT INTO sku_price_history (
        promo_id,
        sku_id,
        price,
        discount_price,
        start_at,
        end_at,
        is_stackable,
        created_at
      ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
    `,
      [
        promo.promo_id,
        sku.sku_id,
        originalPrice,
        discountPrice,
        promo.start_at,
        promo.end_at,
        isStackable,
        createdAt,
      ]
    );

    count++;
  }

  console.log(`✅ ${ENTRY_COUNT} sku_price_history records inserted!`);
  await client.end();
};