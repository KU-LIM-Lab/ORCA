const { faker } = require('@faker-js/faker');
const getClient = require('./db');

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log("Connected. Seeding cart...");

  const DAY_MS = 24 * 60 * 60 * 1000;
  const today = new Date();

  // 1. 유저 및 SKU 목록 불러오기 (age, gender, is_active, created_at 포함)
  const userRes = await client.query(
    'SELECT user_id, age, gender, is_active, created_at FROM users'
  );
  const skuRes = await client.query('SELECT sku_id FROM sku');

  const users = userRes.rows;
  const skus = skuRes.rows;

  // 2. 전체 유저 중 90% 선택
  const sampledUsers = faker.helpers.arrayElements(
    users,
    Math.floor(users.length * 0.9)
  );

  for (const user of sampledUsers) {
    const isActive = user.is_active;             // boolean
    const age = user.age;                        // number
    const gender = user.gender;                  // 'M' / 'F' / null
    const userCreatedAt = new Date(user.created_at);

    // 한 유저당 1~5개의 SKU를 카트에 담게
    const productCount = faker.number.int({ min: 1, max: 5 });
    const sampledSkus = faker.helpers.arrayElements(skus, productCount);

    for (const s of sampledSkus) {
      // ───────── quantity: 1.5 + 0.05*(30-age) + 0.4*I(F) + 0.6*I(active) + ε_q ─────────
      const epsQ = faker.number.float({ mean: 0, stddev: 0.5 });
      let qStar =
        1.5 +
        0.05 * (30 - age) +
        0.4 * (gender === 'F' ? 1 : 0) +
        0.6 * (isActive ? 1 : 0) +
        epsQ;
      let quantity = Math.max(1, Math.round(qStar));

      // ───────── created_at: user.created_at + Δ_c (Δ_c = 60 - 30*I(active) + ε_c) ─────────
      const epsC = faker.number.float({ mean: 0, stddev: 10 });
      let deltaCreateDays = 60 - 30 * (isActive ? 1 : 0) + epsC;
      if (deltaCreateDays < 0) deltaCreateDays = 0;

      let createdAt = new Date(userCreatedAt.getTime() + deltaCreateDays * DAY_MS);
      if (createdAt > today) createdAt = today;

      // ───────── updated_at: created_at + S (S = 30 + 160*(1-I(active)) + ε_u) ─────────
      const epsU = faker.number.float({ mean: 0, stddev: 15 });
      let Sdays = 30 + 160 * (isActive ? 0 : 1) + epsU;
      if (Sdays < 0) Sdays = 0;

      let updatedAt = new Date(createdAt.getTime() + Sdays * DAY_MS);
      if (updatedAt > today) updatedAt = today;

      await client.query(
        `
        INSERT INTO cart (
          cart_id,
          user_id,
          sku_id,
          quantity,
          created_at,
          updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6)
      `,
        [
          faker.string.uuid(),
          user.user_id,
          s.sku_id,
          quantity,
          createdAt,
          updatedAt,
        ]
      );
    }
  }

  console.log("✅ Cart seeded successfully.");
  await client.end();
};