const { faker } = require('@faker-js/faker');
const getClient = require('./db');
const { v4: uuidv4 } = require('uuid');

const USER_COUNT = 10000;

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log(`Connected. Seeding ${USER_COUNT} users...`);

  const emailsUsed = new Set();
  const usernamesUsed = new Set();

  const today = new Date();

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  // 20·30·40대 비중을 크게 주는 age 샘플러
  function sampleAge() {
    const decades = [20, 30, 40, 50, 60];
    const weights = [0.35, 0.30, 0.20, 0.10, 0.05]; // 합 ≈ 1
    const r = Math.random();
    let acc = 0;
    let chosenDecade = decades[0];
    for (let i = 0; i < decades.length; i++) {
      acc += weights[i];
      if (r <= acc) {
        chosenDecade = decades[i];
        break;
      }
    }
    const minAge = chosenDecade;
    const maxAge = chosenDecade + 9;
    return faker.number.int({ min: minAge, max: maxAge });
  }

  for (let i = 0; i < USER_COUNT; i++) {
    const user_id = uuidv4();

    // 1%는 email/username 중복 허용
    const useDuplicate = Math.random() < 0.01;

    const email = useDuplicate
      ? faker.helpers.arrayElement([...emailsUsed])
      : faker.internet.email();
    const username = useDuplicate
      ? faker.helpers.arrayElement([...usernamesUsed])
      : faker.internet.username();

    emailsUsed.add(email);
    usernamesUsed.add(username);

    // 10%는 일부 NULL 값 포함
    const name = Math.random() < 0.1 ? null : faker.person.fullName();
    const address = Math.random() < 0.1 ? null : faker.location.streetAddress();
    const phone = Math.random() < 0.1 ? null : faker.phone.number();

    // ───────────────
    // SCM 기반 생성
    // ───────────────

    // signup_days_ago ~ Uniform(0, 3*365)
    const signup_days_ago = faker.number.int({ min: 0, max: 3 * 365 });

    // created_at = today - signup_days_ago
    const created_at = new Date(
      today.getTime() - signup_days_ago * 24 * 60 * 60 * 1000
    );

    // age: 20·30·40대 비중↑, 각 decade 내 U(d, d+9)
    const age = sampleAge();

    // birth_year = current_year - age (+ 작은 노이즈는 month/day로 표현)
    const birth = new Date(
      today.getFullYear() - age,
      faker.number.int({ min: 0, max: 11 }),
      faker.number.int({ min: 1, max: 28 })
    );

    // gender ~ Bernoulli(p_female)
    const pFemale = 0.65;
    const gender = Math.random() < pFemale ? 'F' : 'M';

    // avg_browsing_time = 35 - 0.6*(age-30) + 8*I(gender=F) + ε_B, ε_B~N(0,5^2)
    const epsilonB = faker.number.float({ mean: 0, stddev: 5 });
    let avg_browsing_time =
      35 - 0.6 * (age - 30) + 8 * (gender === 'F' ? 1 : 0) + epsilonB;
    if (avg_browsing_time < 1) avg_browsing_time = 1; // 하한

    // is_active_score = -1.0 + 0.04*avg_browsing_time - 0.004*signup_days_ago + ε_A
    const epsilonA = faker.number.float({ mean: 0, stddev: 1 });
    const is_active_score =
      -1.0 + 0.04 * avg_browsing_time - 0.004 * signup_days_ago + epsilonA;

    // is_active ~ Bernoulli(sigmoid(is_active_score))
    const pActive = sigmoid(is_active_score);
    const is_active = Math.random() < pActive;

    const updated_at = created_at; // 엑셀 정의에 맞게 동일하게 두기

    const point_balance = faker.number.int({ min: 100, max: 50000 });

    await client.query(
      `
      INSERT INTO users (
        user_id,
        username,
        password,
        name,
        email,
        phone,
        age,
        birth,
        gender,
        address,
        avg_browsing_time,
        is_active_score,
        is_active,
        created_at,
        updated_at,
        point_balance
      ) VALUES (
        $1, $2, $3, $4, $5, $6,
        $7, $8, $9, $10,
        $11, $12, $13,
        $14, $15, $16
      )
    `,
      [
        user_id,
        username,
        'hashed-password',
        name,
        email,
        phone,
        age,
        birth,
        gender,
        address,
        avg_browsing_time,
        is_active_score,
        is_active,
        created_at,
        updated_at,
        point_balance, // point_balance (2차 seeding에서 point_transaction로 업데이트 예정)
      ]
    );
  }

  console.log('All users inserted!');
  await client.end();
};