const { faker } = require('@faker-js/faker');
const getClient = require('./db');

const brandMap = {
  'Electronics': ['삼성전자', 'LG전자', '애플', '소니', '샤오미'],
  'Fashion': ['나이키', '아디다스', '뉴발란스', '푸마', '리복'],
  'Home Appliances': ['쿠쿠', '다이슨', '발뮤다', '필립스', '일렉트로룩스'],
  'Books': ['문학동네', '민음사', '위즈덤하우스', '한겨레출판', '창비'],
  'Toys': ['레고', '영실업', '반다이', '타카라토미', '미미월드'],
  'Beauty': ['설화수', '라네즈', '이니스프리', '클리오', '미샤'],
  'Groceries': ['오뚜기', '농심', 'CJ', '풀무원', '동원'],
  'Furniture': ['한샘', '이케아', '일룸', '동서가구', '에이스침대'],
  'Sports': ['휠라', '데상트', '언더아머', '나이키', '캘러웨이'],
  'Automotive': ['불스원', '카렉스', '아이나비', '카템']
};

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log("Connected to DB. Inserting brands...");

  // 1) categories 테이블에서 parent 정보 가져오기
  const res = await client.query(`SELECT category_id, name, created_at, category_popularity_score FROM categories`);
  const categoryDict = {};
  res.rows.forEach(row => {
    categoryDict[row.name] = {
      id: row.category_id,
      created_at: row.created_at,
      popularity: row.category_popularity_score
    };
  });

  const today = new Date();

  function randomDateAfter(baseDate, minDays=0, maxDays=30) {
    const offset = faker.number.int({ min: minDays, max: maxDays });
    return new Date(baseDate.getTime() + offset * 24 * 60 * 60 * 1000);
  }

  for (const [categoryName, brandList] of Object.entries(brandMap)) {
    const category = categoryDict[categoryName];
    if (!category) {
      console.warn(`⚠ Category ${categoryName} not found in DB.`);
      continue;
    }

    for (const brandName of brandList) {
      const brand_id = faker.string.uuid();

      // SCM 1) created_at = category.created_at + U(0,30)
      const created_at = randomDateAfter(category.created_at, 0, 30);

      // SCM 2) updated_at = created_at
      const updated_at = created_at;

      // SCM 3) brand_strength_score = 0.8 * category_popularity_score + ε_B
      const epsilonB = faker.number.float({ mean: 0, stddev: 0.3 });
      const brand_strength_score = 0.8 * category.popularity + epsilonB;

      await client.query(`
        INSERT INTO brands (
          brand_id,
          category_id,
          brand_name,
          created_at,
          updated_at,
          brand_strength_score
        ) VALUES ($1, $2, $3, $4, $5, $6)
      `, [
        brand_id,
        category.id,
        brandName,
        created_at,
        updated_at,
        brand_strength_score
      ]);
    }
  }

  console.log("All brands inserted!");
  await client.end();
};