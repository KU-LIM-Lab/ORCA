const { faker } = require('@faker-js/faker');
const getClient = require('./db');
const { v4: uuidv4 } = require('uuid');

const PROMO_COUNT = 100;

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log('Connected. Seeding promotions...');

  const baseDate = new Date('2024-01-01');
  const DAY_MS = 24 * 60 * 60 * 1000;

  for (let i = 0; i < PROMO_COUNT; i++) {
    const promoId = uuidv4();
    const name = faker.company.catchPhrase(); // 랜덤 프로모션 이름

    // discount_type: rate(정률) 쪽이 조금 더 자주 등장
    const discountType = Math.random() < 0.6 ? 'rate' : 'amount';

    // discount_value: type에 따라 구간 다르게
    let discountValue;
    if (discountType === 'amount') {
      // 정액 할인: 3 ~ 15 (예: 3,000 ~ 15,000원)
      discountValue = faker.number.float({ min: 3, max: 15, precision: 0.01 });
    } else {
      // 정률 할인: 5% ~ 30% (0.05 ~ 0.30)
      discountValue = faker.number.float({ min: 0.05, max: 0.30, precision: 0.0001 });
    }

    // start_at: 기준일 이후 0~365일 사이
    const offsetDays = faker.number.int({ min: 0, max: 365 });
    const startDate = new Date(baseDate.getTime() + offsetDays * DAY_MS);

    // duration_days ~ Uniform(14, 60), end_at = start_at + duration_days
    const durationDays = faker.number.int({ min: 14, max: 60 });
    const endDate = new Date(startDate.getTime() + durationDays * DAY_MS);

    await client.query(
      `
      INSERT INTO promotion (
        promo_id,
        name,
        discount_value,
        discount_type,
        start_at,
        end_at
      ) VALUES ($1, $2, $3, $4, $5, $6)
    `,
      [promoId, name, discountValue, discountType, startDate, endDate]
    );
  }

  console.log(`✅ ${PROMO_COUNT} promotions inserted!`);
  await client.end();
};