const { faker } = require('@faker-js/faker');
const getClient = require('./db');
const { v4: uuidv4 } = require('uuid');

const DAY_MS = 24 * 60 * 60 * 1000;

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// ordinal cutpoints: {1.5, 2.5, 3.5, 4.5}
function continuousToOrdinal(x) {
  if (x <= 1.5) return 1;
  if (x <= 2.5) return 2;
  if (x <= 3.5) return 3;
  if (x <= 4.5) return 4;
  return 5;
}

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log('Connected. Seeding reviews...');

  // 1) 유저 활동성 정보
  const userRes = await client.query(
    'SELECT user_id, is_active, is_active_score FROM users'
  );
  const userIsActive = {};
  const userScore = {};
  for (const row of userRes.rows) {
    userIsActive[row.user_id] = row.is_active;
    userScore[row.user_id] = row.is_active_score;
  }

  // 2) 주문 + 결제 + 배송 + 상품 정보
  //  - 주문 총액: orders.total_amount
  //  - 결제일: payment.payment_date
  //  - 배송완료일: shipping.delivered_at
  const orderItemRes = await client.query(`
    SELECT
      o.order_id,
      o.user_id,
      o.total_amount,
      oi.unit_price,
      oi.quantity,
      s.product_id,
      p.payment_date,
      sh.delivered_at
    FROM order_items oi
    JOIN orders o   ON o.order_id   = oi.order_id
    JOIN sku s      ON s.sku_id     = oi.sku_id
    JOIN payment p  ON p.order_id   = o.order_id
    LEFT JOIN shipping sh ON sh.order_id = o.order_id
    WHERE p.payment_status = 'COMPLETED'
  `);

  const usedPairs = new Set(); // user-product 한 번만 리뷰
  let count = 0;

  for (const row of orderItemRes.rows) {
    const {
      user_id,
      order_id,
      product_id,
      total_amount,
      payment_date,
      delivered_at
    } = row;

    // 배송 완료되지 않은 주문은 리뷰 생성 X
    if (!delivered_at || !payment_date) continue;

    const key = `${user_id}-${product_id}`;
    if (usedPairs.has(key)) continue;
    usedPairs.add(key);

    const is_active_score = userScore[user_id] ?? 0;
    const is_active = userIsActive[user_id] ?? false;

    const order_total = Number(total_amount || 0);
    const log_total = Math.log(order_total + 1);

    const payDate = new Date(payment_date);
    const delivDate = new Date(delivered_at);
    let delay_days = (delivDate.getTime() - payDate.getTime()) / DAY_MS;
    if (!Number.isFinite(delay_days) || delay_days < 0) delay_days = 0;

    // ───────── 2-1. 리뷰를 남길지 여부 (intent) ─────────
    // 활동성 + 주문규모 + 배송지연(지연되면 의욕↓) 반영
    const epsIntent = faker.number.float({ mean: 0, stddev: 1 });
    const review_intent_score =
      0.5 * (is_active ? 1 : 0) +
      0.2 * is_active_score +
      0.1 * log_total -
      0.02 * delay_days +
      epsIntent;

    const reviewProb = sigmoid(review_intent_score);
    if (Math.random() > reviewProb) continue;

    // ───────── 2-2. 연속 평점 score_cont ─────────
    const epsS = faker.number.float({ mean: 0, stddev: 0.7 });
    let score_cont =
      3.0 +
      0.2 * is_active_score +
      0.4 * log_total -
      0.05 * delay_days +
      epsS;

    // 1~5로 클램핑
    score_cont = Math.max(1, Math.min(5, score_cont));

    // ordinal 컷포인트 적용
    const score = continuousToOrdinal(score_cont);

    // ───────── 2-3. 리뷰 작성 시점: 배송 후 0~14일 ─────────
    const offsetDays = faker.number.int({ min: 0, max: 14 });
    const createdAt = new Date(delivDate.getTime() + offsetDays * DAY_MS);

    await client.query(
      `
      INSERT INTO review (
        review_id,
        order_id,
        product_id,
        user_id,
        title,
        content,
        score,
        score_cont,
        created_at
      ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
    `,
      [
        uuidv4(),
        order_id,
        product_id,
        user_id,
        faker.lorem.sentence(5),
        faker.lorem.paragraph(),
        score,
        score_cont,
        createdAt
      ]
    );

    count++;
    if (count >= 3000) break;
  }

  console.log(`✅ ${count} reviews inserted based on SCM (activity, amount, delay).`);
  await client.end();
};