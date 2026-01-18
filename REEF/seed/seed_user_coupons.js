const { faker } = require('@faker-js/faker');
const getClient = require('./db');
const { v4: uuidv4 } = require('uuid');

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log("Connected. Seeding user_coupons...");

  const DAY_MS = 24 * 60 * 60 * 1000;

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  // 1) User info: is_active_score, is_active, created_at
  const resUsers = await client.query(
    'SELECT user_id, is_active_score, is_active, created_at FROM users'
  );

  const users = resUsers.rows;
  if (users.length === 0) {
    console.error('🚨 No users found.');
    await client.end();
    return;
  }

  const activeUsers = users.filter(u => u.is_active);
  const inactiveUsers = users.filter(u => !u.is_active);

  // 2) Coupon info: start_date, expiration_date, discount_strength
  const resCoupons = await client.query(`
    SELECT
      coupon_id,
      start_date,
      expiration_date,
      discount_strength
    FROM coupon
  `);

  const coupons = resCoupons.rows;
  if (coupons.length === 0) {
    console.error('🚨 No coupons found.');
    await client.end();
    return;
  }

  const assignments = [];

  for (const coupon of coupons) {
    const startDate = new Date(coupon.start_date);
    const expDate = new Date(coupon.expiration_date);

    if (startDate >= expDate) {
      console.warn(`⏭️ Skipping coupon ${coupon.coupon_id} (invalid date range)`);
      continue;
    }

    // About 100 users per coupon: higher proportion of active users (e.g., 70:30)
    const targetPerCoupon = 100;
    const numActive = Math.min(
      activeUsers.length,
      Math.round(targetPerCoupon * 0.7)
    );
    const numInactive = Math.min(
      inactiveUsers.length,
      targetPerCoupon - numActive
    );

    const sampledActive = faker.helpers.arrayElements(activeUsers, numActive);
    const sampledInactive = faker.helpers.arrayElements(inactiveUsers, numInactive);
    const sampledUsers = [...sampledActive, ...sampledInactive];

    for (const user of sampledUsers) {
      const { user_id, is_active_score, is_active, created_at } = user;
      const userCreatedAt = new Date(created_at);

      // ───────── 1) assigned_at: active users receive faster ─────────
      // Δ_assign = 20 - 10 * I(is_active) + ε_A  (days)
      const epsA = faker.number.float({ mean: 0, stddev: 5 });
      let deltaAssignDays = 20 - 10 * (is_active ? 1 : 0) + epsA;
      if (deltaAssignDays < 0) deltaAssignDays = 0;

      // Base is after coupon.start_date, also consider user signup date
      let candidate = new Date(startDate.getTime() + deltaAssignDays * DAY_MS);
      if (candidate < userCreatedAt) {
        candidate = userCreatedAt;
      }

      // If after expiration date, consider as never given coupon and skip
      if (candidate > expDate) continue;

      const assignedAt = candidate;

      // ───────── 2) is_used_score ─────────
      const daysToExp = (expDate.getTime() - assignedAt.getTime()) / DAY_MS;
      const discountStrength = Number(coupon.discount_strength ?? 0);

      const epsU = faker.number.float({ mean: 0, stddev: 1 });

      const is_used_score =
        -2.0 +
        0.9 * Number(is_active_score ?? 0) +
        0.4 * discountStrength +
        0.01 * Math.max(daysToExp, 0) +
        epsU;

      const p_used = sigmoid(is_used_score);
      const is_used = Math.random() < p_used ? 1 : 0;

      assignments.push({
        id: uuidv4(),
        user_id,
        coupon_id: coupon.coupon_id,
        assigned_at: assignedAt,
        is_used_score,
        is_used,
      });
    }
  }

  // 3) DB insert
  for (const u of assignments) {
    await client.query(
      `
      INSERT INTO user_coupons (
        id,
        user_id,
        coupon_id,
        assigned_at,
        is_used_score,
        is_used
      ) VALUES ($1, $2, $3, $4, $5, $6)
    `,
      [
        u.id,
        u.user_id,
        u.coupon_id,
        u.assigned_at,
        u.is_used_score,
        u.is_used,
      ]
    );
  }

  console.log(`✅ ${assignments.length} user_coupons inserted!`);
  await client.end();
};