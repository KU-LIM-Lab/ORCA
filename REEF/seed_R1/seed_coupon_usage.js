const getClient = require('./db');
const { v4: uuidv4 } = require('uuid');

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log('Connected. Seeding coupon_usage...');

  // orders × coupon × user_coupons 조인
  const res = await client.query(`
    SELECT
      o.order_id,
      o.user_id,
      o.created_at      AS order_created_at,
      o.coupon_used     AS coupon_id,
      c.start_date,
      c.expiration_date
    FROM orders o
    JOIN coupon c
      ON o.coupon_used = c.coupon_id
    JOIN user_coupons uc
      ON uc.user_id   = o.user_id
     AND uc.coupon_id = o.coupon_used
     AND uc.is_used   = true
    WHERE o.coupon_used IS NOT NULL
  `);

  const rows = res.rows;

  if (rows.length === 0) {
    console.warn('⚠️ No orders with coupons found. Nothing to insert into coupon_usage.');
    await client.end();
    return;
  }

  let inserted = 0;

  for (const row of rows) {
    const {
      order_id,
      user_id,
      coupon_id,
      order_created_at,
      start_date,
      expiration_date,
    } = row;

    const usedAt = new Date(order_created_at);
    const start = new Date(start_date);
    const end = new Date(expiration_date);

    // 쿠폰 유효 기간 안에서만 사용 기록 생성
    if (usedAt < start || usedAt > end) {
      console.warn(
        `⏭️ Skip: order ${order_id} uses coupon ${coupon_id} outside validity window`
      );
      continue;
    }

    const usageId = uuidv4();

    await client.query(
      `
      INSERT INTO coupon_usage (
        usage_id,
        coupon_id,
        user_id,
        order_id,
        used_at
      ) VALUES ($1,$2,$3,$4,$5)
    `,
      [usageId, coupon_id, user_id, order_id, usedAt]
    );

    inserted += 1;
  }

  console.log(`✅ ${inserted} coupon_usage rows inserted!`);
  await client.end();
};