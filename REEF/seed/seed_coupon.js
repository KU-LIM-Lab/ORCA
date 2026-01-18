const { faker } = require('@faker-js/faker');
const getClient = require('./db');
const { v4: uuidv4 } = require('uuid');

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log('Connected. Seeding coupons with promotion linkage...');

  // 1. Load promotion list
  const promoRes = await client.query(`
    SELECT promo_id, discount_value, discount_type, start_at, end_at
    FROM promotion
  `);
  const promotions = promoRes.rows;

  if (promotions.length === 0) {
    console.error('🚨 No promotions found. Please seed them first.');
    await client.end();
    return;
  }

  const today = new Date();
  const couponList = [];

  for (let i = 0; i < promotions.length; i++) {
    const promo = promotions[i];

    // discount_type: 'amount' / 'rate' / (compatibility) 'fixed' / 'percentage'
    const type = promo.discount_type;
    const isRate =
      type === 'rate' ||
      type === 'percentage'; // Legacy schema compatibility
    const isAmount =
      type === 'amount' ||
      type === 'fixed';

    let discount_amount = 0;
    let discount_rate = 0;

    if (isAmount) {
      discount_amount = Number(promo.discount_value);     // e.g., 3~15
    } else if (isRate) {
      discount_rate = Number(promo.discount_value);       // e.g., 0.05~0.30
    }

    // ───────── discount_strength: Scalar representation of discount strength ─────────
    // amount: Slightly reduce amount, rate: Expand ratio to match similar scale
    let discount_strength;
    if (isAmount) {
      // 3~15  → approximately 0.3~1.5
      discount_strength = discount_amount / 10;
    } else if (isRate) {
      // 0.05~0.30 → approximately 0.5~3.0
      discount_strength = discount_rate * 10;
    } else {
      discount_strength = 0;
    }

    // ───────── min_order_amount = 20 + 1.5*discount_strength + ε_M ─────────
    const epsM = faker.number.float({ mean: 0, stddev: 3 });
    let min_order_amount =
      20 + 1.5 * discount_strength + epsM;
    if (min_order_amount < 0) min_order_amount = 0;
    // Can multiply by currency scale if needed (e.g., *1000)

    const start_date = promo.start_at;
    const expiration_date = promo.end_at;

    // ───────── is_active: 1 if within promotion period, else 0 ─────────
    const start = new Date(start_date);
    const end = new Date(expiration_date);
    const is_active = today >= start && today <= end;

    const isRateDesc = isRate;
    const desc = isRateDesc
      ? `${promo.discount_value * (promo.discount_value <= 1 ? 100 : 1)}% discount coupon`
      : `₩${discount_amount} discount coupon`;

    couponList.push({
      coupon_id: uuidv4(),
      code: `PROMO${i + 1}`,
      description: desc,
      discount_amount,
      discount_rate,
      discount_strength,
      min_order_amount,
      start_date,
      expiration_date,
      is_active,
      promo_id: promo.promo_id,
    });
  }

  // 2. DB insert
  for (const c of couponList) {
    await client.query(
      `
      INSERT INTO coupon (
        coupon_id,
        code,
        description,
        discount_amount,
        discount_rate,
        discount_strength,
        min_order_amount,
        start_date,
        expiration_date,
        is_active,
        promo_id
      ) VALUES (
        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11
      )
    `,
      [
        c.coupon_id,
        c.code,
        c.description,
        c.discount_amount,
        c.discount_rate,
        c.discount_strength,
        c.min_order_amount,
        c.start_date,
        c.expiration_date,
        c.is_active,
        c.promo_id,
      ]
    );
  }

  console.log(`✅ ${couponList.length} coupons inserted and linked with promotions!`);
  await client.end();
};