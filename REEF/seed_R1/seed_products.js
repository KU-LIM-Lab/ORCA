const { faker } = require("@faker-js/faker");
const getClient = require("./db");

const PRODUCT_COUNT = 1000;

// 각 카테고리에 맞는 상품 키워드 리스트
const productMap = {
  Electronics: ["스마트폰", "노트북", "모니터", "이어폰", "태블릿"],
  Fashion: ["운동화", "재킷", "셔츠", "바지", "모자"],
  "Home Appliances": ["청소기", "공기청정기", "냉장고", "전자레인지", "세탁기"],
  Books: ["소설", "자기계발서", "에세이", "잡지", "만화"],
  Toys: ["프라모델", "인형", "퍼즐", "블록", "보드게임"],
  Beauty: ["에센스", "크림", "팩트", "쿠션", "립밤"],
  Groceries: ["라면", "햇반", "참치", "과자", "두부"],
  Furniture: ["침대", "책상", "서랍장", "소파", "테이블"],
  Sports: ["운동복", "러닝화", "골프공", "라켓", "요가매트"],
  Automotive: ["엔진코팅제", "블랙박스", "방향제", "와이퍼", "세차용품"],
};

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log("Connected. Seeding products...");

  const DAY_MS = 24 * 60 * 60 * 1000;
  const today = new Date();

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  // ✅ 브랜드, 카테고리 + 브랜드 생성일 정보까지 가져오기
  const res = await client.query(`
    SELECT 
      b.brand_id,
      b.brand_name,
      b.category_id,
      b.created_at AS brand_created_at,
      c.name AS category_name
    FROM brands b
    JOIN categories c ON b.category_id = c.category_id
  `);
  const brands = res.rows;

  for (let i = 0; i < PRODUCT_COUNT; i++) {
    const brand = faker.helpers.arrayElement(brands);
    const categoryName = brand.category_name;
    const keywords = productMap[categoryName];

    // productMap에 없는 카테고리가 혹시 있다면 skip
    if (!keywords) {
      console.warn(`⚠ No product keywords for category: ${categoryName}`);
      continue;
    }

    const productKeyword = faker.helpers.arrayElement(keywords);
    const productName = `${brand.brand_name} ${productKeyword}`;
    const productId = faker.string.uuid();

    // ───────── stock_quantity (외생) ─────────
    const stock_quantity = faker.number.int({ min: 10, max: 300 });

    const thumbnail_url = faker.image.urlPicsumPhotos();
    const description = faker.commerce.productDescription();

    // ───────── created_at = brand.created_at + U(0, 7 days) ─────────
    const brandCreated = new Date(brand.brand_created_at);
    const offsetDays = faker.number.int({ min: 0, max: 7 });
    let created_at = new Date(brandCreated.getTime() + offsetDays * DAY_MS);
    if (created_at > today) created_at = today;

    // ───────── is_active: a* = -1 + 1.5*I(stock>0) - 0.01*days_since_created + ε_a ─────────
    const days_since_created = (today.getTime() - created_at.getTime()) / DAY_MS;
    const I_stock = stock_quantity > 0 ? 1 : 0;
    const epsilonA = faker.number.float({ mean: 0, stddev: 1 });

    const aStar = -1 + 1.5 * I_stock - 0.01 * days_since_created + epsilonA;
    const pActive = sigmoid(aStar);
    const is_active = Math.random() < pActive;

    // ───────── updated_at = created_at (단순) ─────────
    const updated_at = created_at;

    try {
      await client.query(
        `
        INSERT INTO products (
          product_id,
          category_id,
          product_name,
          description,
          stock_quantity,
          thumbnail_url,
          is_active,
          created_at,
          updated_at
        ) VALUES (
          $1, $2, $3, $4, $5, $6, $7, $8, $9
        )
      `,
        [
          productId,
          brand.category_id,
          productName,
          description,
          stock_quantity,
          thumbnail_url,
          is_active,
          created_at,
          updated_at,
        ]
      );
    } catch (err) {
      console.error(`Error at row ${i}: ${err.message}`);
    }

    if (i % 200 === 0) console.log(`Inserted ${i} products...`);
  }

  console.log("All products inserted!");
  await client.end();
};