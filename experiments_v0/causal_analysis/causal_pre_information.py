DEFAULT_EXPRESSION_DICT = {
    # ─── Users & basic demographics ────────────────────────────────────────────
    "signup_days_ago": "DATE_PART('day', CURRENT_DATE - users.created_at)",
    "is_active":        "users.is_active",
    "age":              "DATE_PART('year', AGE(users.birth))",
    "gender":           "users.gender",
    "point_balance":    "users.point_balance",

    # ─── Order-level amounts & quantities ──────────────────────────────────────
    "unit_price":   "order_items.unit_price",
    "quantity":     "order_items.quantity",
    "order_total":  "order_items.total_price",        # generated column
    "paid_amount":  "(orders.total_amount - orders.discount_amount)",

    # ─── Coupon effects (requires coupon join, see notes) ──────────────────────
    "discount_amount": "coupon.discount_amount",
    "discount_rate":   "coupon.discount_rate",
    "used_coupon":
        "COALESCE((SELECT TRUE "
        "           FROM coupon_usage "
        "           WHERE coupon_usage.order_id = orders.order_id "
        "           LIMIT 1), FALSE)",

    # ─── Loyalty points (1 % of amount actually paid) ─────────────────────────
    "point_earned": "((orders.total_amount - orders.discount_amount) * 0.01)",

    # ─── Reviews ───────────────────────────────────────────────────────────────
    "review_score": "review.score"
}

BASE_SQL_QUERY = """
WITH orders_with_items AS (
    SELECT
        o.order_id,
        o.user_id,
        oi.unit_price,
        oi.quantity,
        oi.total_price AS order_total,
        o.total_amount,
        o.discount_amount
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
),
payments AS (
    SELECT
        *,
        (total_amount - discount_amount) AS paid_amount
    FROM orders_with_items
),
user_coupon_flag AS (
    SELECT
        cu.order_id,
        TRUE AS used_coupon
    FROM coupon_usage cu
),
base AS (
    SELECT
        u.user_id,
        DATE_PART('day', CURRENT_DATE - u.created_at)     AS signup_days_ago,
        DATE_PART('year', AGE(u.birth))                   AS age,
        u.gender,
        u.is_active,
        p.order_id,
        p.unit_price,
        p.quantity,
        p.order_total,
        p.paid_amount,
        c.discount_amount,
        c.discount_rate,
        COALESCE(uc.used_coupon, FALSE)                   AS used_coupon,
        (p.paid_amount * 0.01)                            AS point_earned,
        u.point_balance,
        r.score                                           AS review_score
    FROM payments               p
    JOIN users                  u  ON p.user_id = u.user_id
    LEFT JOIN user_coupon_flag  uc ON p.order_id = uc.order_id
    LEFT JOIN coupon_usage      cu ON p.order_id = cu.order_id
    LEFT JOIN coupon            c  ON cu.coupon_id = c.coupon_id
    LEFT JOIN order_items       oi ON p.order_id = oi.order_id
    LEFT JOIN sku               s  ON oi.sku_id  = s.sku_id
    LEFT JOIN products          pr ON s.product_id = pr.product_id
    LEFT JOIN review            r  ON r.user_id = u.user_id
                                    AND r.product_id = pr.product_id
)
SELECT *
FROM base;
"""