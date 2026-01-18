const { faker } = require('@faker-js/faker');
const getClient = require('./db');
const { v4: uuidv4 } = require('uuid');

const ORDER_COUNT = 10000;
const DAY_MS = 24 * 60 * 60 * 1000;


function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log('Connected. Seeding orders + order_items + payment + shipping...');

  // ───────────────── 1. 필요 데이터 로딩 ─────────────────

  // users: 결제 score, 포인트, 주문 생성 시점 등에 사용
  const userRes = await client.query(`
    SELECT user_id, age, gender, is_active, created_at, point_balance
    FROM users
  `);
  const users = userRes.rows;
  if (users.length === 0) {
    console.error('🚨 No users found');
    await client.end();
    return;
  }

  // sku + product 생성일: 주문 가능한 시점 계산용
  const skuRes = await client.query(`
    SELECT s.sku_id, s.price, p.created_at AS product_created_at
    FROM sku s
    JOIN products p ON s.product_id = p.product_id
  `);
  const skus = skuRes.rows;
  if (skus.length === 0) {
    console.error('🚨 No skus found');
    await client.end();
    return;
  }

  // user_coupons + coupon: 할인 금액/비율, 강도, 유효 기간
  const userCouponRes = await client.query(`
    SELECT
      uc.user_id,
      uc.coupon_id,
      uc.assigned_at,
      uc.is_used,
      c.discount_amount,
      c.discount_rate,
      c.discount_strength,
      c.start_date,
      c.expiration_date
    FROM user_coupons uc
    JOIN coupon c ON uc.coupon_id = c.coupon_id
  `);

  const couponsByUser = {};
  for (const row of userCouponRes.rows) {
    if (!couponsByUser[row.user_id]) couponsByUser[row.user_id] = [];
    couponsByUser[row.user_id].push(row);
  }

  // ───────────────── 2. 주문 생성 루프 ─────────────────

  for (let i = 0; i < ORDER_COUNT; i++) {
    const user = faker.helpers.arrayElement(users);
    const userId = user.user_id;

    // 2-1. 주문에 들어갈 SKU 샘플링
    const itemCount = faker.number.int({ min: 1, max: 5 });
    const selectedSkus = faker.helpers.arrayElements(skus, itemCount);

    // 주문 생성 가능 최소 시점 = max(유저 가입일, 해당 상품들 생성일)
    const earliestProductCreated = selectedSkus.reduce((acc, sku) => {
      const d = new Date(sku.product_created_at);
      return d > acc ? d : acc;
    }, new Date(selectedSkus[0].product_created_at));

    const baseCreated = new Date(
      Math.max(earliestProductCreated.getTime(), new Date(user.created_at).getTime())
    );
    const plusDays = faker.number.int({ min: 0, max: 300 });
    const orderCreatedAt = new Date(baseCreated.getTime() + plusDays * DAY_MS);

    // 2-2. order_items 생성 + subtotal 계산
    const orderId = uuidv4();
    const orderItems = [];
    let subtotal = 0;

    for (const sku of selectedSkus) {
      const quantity = faker.number.int({ min: 1, max: 3 });
      const unitPrice = Number(sku.price);        // unit_price = sku.price
      const totalPrice = unitPrice * quantity;    // total_price = quantity * unit_price

      subtotal += totalPrice;

      orderItems.push({
        order_item_id: uuidv4(),
        sku_id: sku.sku_id,
        quantity,
        unit_price: unitPrice,
        total_price: totalPrice
      });
    }

    // 2-3. 쿠폰/포인트 적용 → discount_amount, point_used, total_amount
    let coupon_used = null;
    let discount_amount = 0;

    const userCoupons = couponsByUser[userId] || [];
    const eligibleCoupons = userCoupons.filter(c =>
      c.is_used &&
      new Date(c.assigned_at) <= orderCreatedAt &&
      orderCreatedAt <= new Date(c.expiration_date)
    );

    if (eligibleCoupons.length > 0 && Math.random() < 0.7) {
      // 쿠폰이 있고, 70% 확률로 사용
      const coupon = faker.helpers.arrayElement(eligibleCoupons);
      coupon_used = coupon.coupon_id;

      if (coupon.discount_amount && coupon.discount_amount > 0) {
        discount_amount = Number(coupon.discount_amount);
      } else if (coupon.discount_rate && coupon.discount_rate > 0) {
        // coupon.discount_rate는 0.05~0.30 비율이라고 가정
        discount_amount = subtotal * Number(coupon.discount_rate);
      }
      if (discount_amount > subtotal) discount_amount = subtotal;
    }

    // point_used = user.point_balance * U(0,1), 단 subtotal 안 넘도록
    let point_used = 0;
    const maxPointUse = Math.min(Number(user.point_balance || 0), subtotal - discount_amount);
    if (maxPointUse > 0) {
      point_used = Math.floor(maxPointUse * faker.number.float({ min: 0, max: 1 }));
    }

    let total_amount = subtotal - discount_amount - point_used;
    if (total_amount < 0) total_amount = 0;

    // ───────────────── 3. 결제 생성 (payment) ─────────────────

    // score_card = α0 + α1*log1p(total_amount) + α2*I(age>=40) + ε
    // const ageOver40 = user.age >= 40 ? 1 : 0;
    // const epsM = faker.number.float({ mean: 0, stddev: 1 });

    // const scoreCard = -0.5 + 0.4 * Math.log1p(total_amount) + 0.3 * ageOver40 + epsM;
    // const pSuccess = sigmoid(scoreCard);

    // let paymentStatus;
    // const r = Math.random();
    // if (r < pSuccess * 0.85) {
    //   paymentStatus = 'COMPLETED';
    // } else if (r < pSuccess) {
    //   paymentStatus = 'PENDING';
    // } else {
    //   paymentStatus = 'FAILED';
    // }

    // payment_date: 주문일 이후 0~3일
    let paymentDate = null;
    const attempt = Math.random() < 0.9;
    if (attempt) {
      const payOffsetDays = faker.number.int({ min: 0, max: 3 });
      paymentDate = new Date(orderCreatedAt.getTime() + payOffsetDays * DAY_MS);
    }

    //payment_status

    const now = new Date();

    let paymentStatus;

    if (paymentDate === null) {
      const elapsedFromOrder = (now - orderCreatedAt) / DAY_MS;  // 주문 시점 기준 경과일
    
      if (elapsedFromOrder < 2) {
        // 아직 시간 충분히 안 지남 → 입금대기(PENDING)
        paymentStatus = 'PENDING';
      } else {
        // 오래됐는데도 아직 결제 X → 실패
        paymentStatus = 'FAILED';
      }
    } else {
      paymentStatus = 'COMPLETED';
    }
    

    const paymentMethod = faker.helpers.weightedArrayElement([
      { value: 'CARD', weight: 0.5 },
      { value: 'KAKAO', weight: 0.25 },
      { value: 'NAVER', weight: 0.15 },
      { value: 'BANK', weight: 0.10 }
    ]);

    // ───────────────── 4. 주문 상태(order_status) 결정 ─────────────────
    let orderStatus = 'PENDING';
    if (paymentStatus === 'FAILED') {
      orderStatus = 'CANCELLED';
    } else if (paymentStatus === 'COMPLETED') {
      orderStatus = 'PAID';
    }

    // ───────────────── 5. 배송(shipping) 생성 ─────────────────
    let shippingRow = null;

    if (paymentStatus === 'COMPLETED') {
      const shippingId = uuidv4();
      const carrier = faker.helpers.arrayElement(['CJ대한통운', '한진택배', '롯데택배', '우체국택배']);

      // shipped_at = payment_date + U(1,7 days)
      const shipOffset = faker.number.int({ min: 1, max: 7 });
      const shippedAt = new Date(paymentDate.getTime() + shipOffset * DAY_MS);

      // delivered_at = shipped_at + U(1,3 days)
      const delvOffset = faker.number.int({ min: 1, max: 3 });
      const deliveredAt = new Date(shippedAt.getTime() + delvOffset * DAY_MS);

      const status = 'DELIVERED'; // 대부분 배송 완료 상태로 둠

      shippingRow = {
        shipping_id: shippingId,
        order_id: orderId,
        tracking_number: faker.string.alphanumeric({ length: 12 }).toUpperCase(),
        carrier,
        status,
        shipped_at: shippedAt,
        delivered_at: deliveredAt
      };

      // 배송까지 끝났다면 주문 상태 = COMPLETED
      orderStatus = 'COMPLETED';
    }

    // ───────────────── 6. 실제 DB insert (트랜잭션) ─────────────────
    try {
      await client.query('BEGIN');

      // orders
      await client.query(
        `
        INSERT INTO orders (
          order_id,
          user_id,
          order_status,
          subtotal_amount,
          total_amount,
          discount_amount,
          coupon_used,
          point_used,
          created_at,
          updated_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$9)
      `,
        [
          orderId,
          userId,
          orderStatus,
          subtotal,
          total_amount,
          discount_amount,
          coupon_used,
          point_used,
          orderCreatedAt
        ]
      );

      // order_items
      for (const item of orderItems) {
        await client.query(
          `
          INSERT INTO order_items (
            order_item_id,
            order_id,
            sku_id,
            quantity,
            unit_price,
            total_price,
            created_at,
            updated_at
          ) VALUES ($1,$2,$3,$4,$5,$6,$7,$7)
        `,
          [
            item.order_item_id,
            orderId,
            item.sku_id,
            item.quantity,
            item.unit_price,
            item.total_price,
            orderCreatedAt
          ]
        );
      }

      // payment
      await client.query(
        `
        INSERT INTO payment (
          payment_id,
          order_id,
          payment_method,
          payment_status,
          amount,
          payment_date
        ) VALUES ($1,$2,$3,$4,$5,$6)
      `,
        [
          uuidv4(),
          orderId,
          paymentMethod,
          paymentStatus,
          total_amount,
          paymentDate
        ]
      );

      // shipping (있을 경우만)
      if (shippingRow) {
        await client.query(
          `
          INSERT INTO shipping (
            shipping_id,
            order_id,
            tracking_number,
            carrier,
            status,
            shipped_at,
            delivered_at
          ) VALUES ($1,$2,$3,$4,$5,$6,$7)
        `,
          [
            shippingRow.shipping_id,
            shippingRow.order_id,
            shippingRow.tracking_number,
            shippingRow.carrier,
            shippingRow.status,
            shippingRow.shipped_at,
            shippingRow.delivered_at
          ]
        );
      }

      await client.query('COMMIT');
    } catch (err) {
      await client.query('ROLLBACK');
      console.error(`❌ Error at order ${i}: ${err.message}`);
    }

    if (i > 0 && i % 500 === 0) {
      console.log(`Inserted ${i} orders...`);
    }
  }

  console.log('✅ All orders + order_items + payment + shipping inserted!');
  await client.end();
};