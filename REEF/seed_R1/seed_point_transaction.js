const { faker } = require('@faker-js/faker');
const getClient = require('./db');
const { v4: uuidv4 } = require('uuid');

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log('Connected. Seeding point transactions from orders...');

  // 주문 + 결제 정보 가져오기 (완료된 결제만)
  const res = await client.query(`
    SELECT
      o.order_id,
      o.user_id,
      o.total_amount,
      o.point_used,
      o.created_at       AS order_created_at,
      p.payment_date,
      p.payment_status
    FROM orders o
    JOIN payment p ON p.order_id = o.order_id
    WHERE p.payment_status = 'COMPLETED'
  `);

  const rows = res.rows;

  if (rows.length === 0) {
    console.warn('⚠️ No completed payments found. Nothing to seed for point_transaction.');
    await client.end();
    return;
  }

  let earnCount = 0;
  let usedCount = 0;

  for (const row of rows) {
    const {
      user_id,
      total_amount,
      point_used,
      payment_date,
      order_created_at,
    } = row;

    const baseTime = payment_date || order_created_at;

    // ───────── 1) earn 트랜잭션: 주문 금액 기반 적립 ─────────
    // earn_rate ~ U(0.005, 0.03) (0.5%~3%)
    const earnRate = faker.number.float({
      min: 0.005,
      max: 0.03,
      precision: 0.0001,
    });

    // ε_C ~ N(0, 5^2) 정도로 노이즈
    const epsC = faker.number.float({ mean: 0, stddev: 5 });

    let earnPointsStar = earnRate * Number(total_amount) + epsC;
    let earnPoints = Math.max(0, Math.round(earnPointsStar));

    if (earnPoints > 0) {
      await client.query(
        `
        INSERT INTO point_transaction (
          transaction_id,
          user_id,
          point_change,
          reason,
          transaction_at,
          type
        ) VALUES ($1,$2,$3,$4,$5,$6)
      `,
        [
          uuidv4(),
          user_id,
          earnPoints,
          '구매 적립',
          baseTime,
          'earn',
        ]
      );
      earnCount++;
    }

    // ───────── 2) used 트랜잭션: 주문에서 사용한 포인트 차감 ─────────
    if (Number(point_used) > 0) {
      const usedPoints = -Math.round(Number(point_used));

      await client.query(
        `
        INSERT INTO point_transaction (
          transaction_id,
          user_id,
          point_change,
          reason,
          transaction_at,
          type
        ) VALUES ($1,$2,$3,$4,$5,$6)
      `,
        [
          uuidv4(),
          user_id,
          usedPoints,
          '주문 사용',
          baseTime,
          'used',
        ]
      );
      usedCount++;
    }
  }

  console.log(`✅ point_transaction inserted. earn=${earnCount}, used=${usedCount}`);
  await client.end();
};