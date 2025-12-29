--
-- PostgreSQL database dump
--

-- Dumped from database version 17.4 (Postgres.app)
-- Dumped by pg_dump version 17.4 (Postgres.app)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    user_id uuid NOT NULL,
    username character varying(255),
    password character varying(255) NOT NULL,
    name character varying(255),
    email character varying(255),
    phone character varying(255),

    -- [추가] SCM 루트 변수
    age integer,

    birth date,
    gender character(10),
    address character varying(255),

    -- [추가] 파생 변수들
    avg_browsing_time numeric(10,2),
    is_active_score numeric(10,2),

    -- [타입 변경] float -> boolean (Bernoulli 결과)
    is_active boolean NOT NULL,

    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    point_balance numeric(10,2) DEFAULT 0.00 NOT NULL
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (user_id);

--
-- Name: cart; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cart (
    cart_id uuid NOT NULL,
    user_id uuid NOT NULL,
    quantity integer DEFAULT 1 NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    sku_id uuid NOT NULL

);


ALTER TABLE public.cart OWNER TO postgres;

--
-- Name: categories; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.categories (
    category_id uuid PRIMARY KEY,
    parent_id uuid,
    name character varying(100) NOT NULL,
    description text,

    -- optional: 계층 정보 (쓰고 싶으면 유지, 필요 없으면 지워도 됨)
    -- category_level integer,

    -- ★ seeding 코드와 맞추기: N(0,1)에서 뽑는 연속값
    category_popularity_score double precision,

    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,

    CONSTRAINT fk_categories_parent
        FOREIGN KEY(parent_id)
        REFERENCES public.categories(category_id)
        ON DELETE SET NULL
);

CREATE INDEX idx_categories_parent_id
    ON public.categories(parent_id);

--
-- Name: brands; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.brands (
    brand_id uuid PRIMARY KEY,
    category_id uuid NOT NULL,

    brand_name character varying(100) NOT NULL,

    created_at timestamp without time zone NOT NULL DEFAULT now(),
    updated_at timestamp without time zone NOT NULL DEFAULT now(),

    -- SCM: brand_strength_score = 0.8 * category_popularity_score + ε_B
    brand_strength_score double precision NOT NULL,

    CONSTRAINT fk_brands_category
        FOREIGN KEY(category_id)
        REFERENCES public.categories(category_id)
        ON DELETE CASCADE
);

ALTER TABLE public.brands OWNER TO postgres;

-- PROMOTION TABLE
CREATE TABLE public.promotion (
    promo_id uuid NOT NULL,
    name character,
    discount_value numeric(10,2) DEFAULT 0.00 NOT NULL,
    discount_type character varying(20) DEFAULT 'PERCENTAGE'::character varying NOT NULL,
    start_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    end_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE public.promotion OWNER TO postgres;

-- promotion id pk로

ALTER TABLE ONLY public.promotion
    ADD CONSTRAINT promotion_pkey PRIMARY KEY (promo_id);

--
-- Name: coupon; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.coupon (
    coupon_id uuid PRIMARY KEY,

    code varchar(50) NOT NULL,            -- 나중에 UNIQUE 주는 것 권장
    description text,

    discount_amount numeric(10,2) NOT NULL DEFAULT 0.00,
    discount_rate numeric(5,2) NOT NULL DEFAULT 0.00,

    -- ★ seeding 코드에서 계산함
    discount_strength numeric(10,2) NOT NULL,

    min_order_amount numeric(10,2) NOT NULL DEFAULT 0.00,

    -- ★ seeding 코드에서 사용하는 실제 시작일
    start_date timestamp NOT NULL,

    expiration_date timestamp NOT NULL,

    is_active boolean NOT NULL DEFAULT true,

    promo_id uuid NOT NULL,

    CONSTRAINT fk_coupon_promotion
        FOREIGN KEY (promo_id)
        REFERENCES public.promotion(promo_id)
        ON DELETE CASCADE
);

-- 필요하면 UNIQUE 설정
CREATE UNIQUE INDEX idx_coupon_code ON public.coupon(code);


ALTER TABLE public.coupon OWNER TO postgres;

--
-- Name: coupon coupon_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.coupon
    ADD CONSTRAINT coupon_pkey PRIMARY KEY (coupon_id);

--
-- Name: products; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.products (
    product_id uuid PRIMARY KEY NOT NULL,
    category_id uuid NOT NULL,
    product_name character varying(100) NOT NULL,
    description text,
    stock_quantity integer DEFAULT 0 NOT NULL,
    thumbnail_url character varying(255),
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.products OWNER TO postgres;

--
-- Name: inventory; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.inventory (
    inventory_id uuid NOT NULL,
    sku_id uuid NOT NULL,
    quantity integer DEFAULT 0 NOT NULL,
    last_updated timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.inventory OWNER TO postgres;

-- SKU TABLE

CREATE TABLE public.sku (
    sku_id uuid PRIMARY KEY,
    product_id uuid NOT NULL,

    sku_code varchar(150) NOT NULL,
    variant_name varchar(150),
    color varchar(100),
    variant_option varchar(100),   -- ← option 이름 변경 (option is reserved keyword)

    is_active boolean NOT NULL DEFAULT true,
    price numeric(10,2) NOT NULL DEFAULT 0.00,

    created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    available_from timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,

    discontinued_at timestamp NULL,  -- 기본값 NULL

    CONSTRAINT fk_sku_product
        FOREIGN KEY(product_id)
        REFERENCES public.products(product_id)
        ON DELETE CASCADE
);

ALTER TABLE public.sku OWNER TO postgres;

--
-- Name: orders; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.orders (
    order_id uuid PRIMARY KEY,
    user_id uuid NOT NULL,

    order_status varchar(20) NOT NULL DEFAULT 'PLACED',

    -- ★ seeding 코드에서 추가로 쓰는 컬럼
    subtotal_amount numeric(10,2) NOT NULL DEFAULT 0.00,
    total_amount    numeric(10,2) NOT NULL DEFAULT 0.00,
    discount_amount numeric(10,2) NOT NULL DEFAULT 0.00,
    coupon_used     uuid,                      -- 쿠폰 없을 수 있으니 NULL 허용
    point_used      numeric(10,2) NOT NULL DEFAULT 0.00,

    created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_orders_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(user_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_orders_coupon
        FOREIGN KEY (coupon_used)
        REFERENCES public.coupon(coupon_id)
        ON DELETE SET NULL
);

ALTER TABLE public.orders OWNER TO postgres;


--
-- Name: order_items; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.order_items (
    order_item_id uuid PRIMARY KEY,
    order_id uuid NOT NULL,
    sku_id uuid NOT NULL,

    quantity   integer      NOT NULL DEFAULT 1,
    unit_price numeric(10,2) NOT NULL,

    -- ★ 더 이상 GENERATED ALWAYS 아님 (코드에서 직접 total_price 넣음)
    total_price numeric(10,2) NOT NULL,

    created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_order_items_order
        FOREIGN KEY (order_id)
        REFERENCES public.orders(order_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_order_items_sku
        FOREIGN KEY (sku_id)
        REFERENCES public.sku(sku_id)
        ON DELETE RESTRICT
);

ALTER TABLE public.order_items OWNER TO postgres;

--
-- Name: payment; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.payment (
    payment_id uuid PRIMARY KEY,
    order_id   uuid NOT NULL,

    payment_method varchar(20) NOT NULL DEFAULT 'CARD',
    payment_status varchar(20) NOT NULL DEFAULT 'PENDING',

    amount numeric(10,2) NOT NULL DEFAULT 0.00,
    payment_date timestamp without time zone,

    CONSTRAINT fk_payment_order
        FOREIGN KEY (order_id)
        REFERENCES public.orders(order_id)
        ON DELETE CASCADE
);

ALTER TABLE public.payment OWNER TO postgres;

--
-- Name: point_transaction; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.point_transaction (
    transaction_id uuid NOT NULL,
    user_id uuid NOT NULL,
    point_change numeric(10,2) NOT NULL,
    reason character varying(100),
    transaction_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    type character varying(20) DEFAULT 'earn'::character varying
);


ALTER TABLE public.point_transaction OWNER TO postgres;

--
-- Name: coupon_usage; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.coupon_usage (
    usage_id uuid PRIMARY KEY,

    coupon_id uuid NOT NULL,
    user_id uuid NOT NULL,
    order_id uuid NOT NULL,

    used_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- FK: coupon_usage.coupon_id → coupon.coupon_id
    CONSTRAINT fk_coupon_usage_coupon
        FOREIGN KEY (coupon_id)
        REFERENCES public.coupon(coupon_id)
        ON DELETE CASCADE,

    -- FK: coupon_usage.user_id → users.user_id
    CONSTRAINT fk_coupon_usage_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(user_id)
        ON DELETE CASCADE,

    -- FK: coupon_usage.order_id → orders.order_id
    CONSTRAINT fk_coupon_usage_order
        FOREIGN KEY (order_id)
        REFERENCES public.orders(order_id)
        ON DELETE CASCADE
);

ALTER TABLE public.coupon_usage OWNER TO postgres;

--
-- Name: review; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.review (
    review_id uuid NOT NULL,
    order_id uuid NOT NULL,
    product_id uuid NOT NULL,
    user_id uuid NOT NULL,
    title character varying(100) NOT NULL,
    content text,
    score integer NOT NULL,
    score_cont float NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.review OWNER TO postgres;

--
-- Name: shipping; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.shipping (
    shipping_id uuid NOT NULL,
    order_id uuid NOT NULL,
    tracking_number character varying(50),
    carrier character varying(50) NOT NULL,
    status character varying(20) DEFAULT 'PENDING'::character varying NOT NULL,
    shipped_at timestamp without time zone,
    delivered_at timestamp without time zone
);


ALTER TABLE public.shipping OWNER TO postgres;

--
-- Name: user_coupons; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_coupons (
    id uuid NOT NULL,
    user_id uuid NOT NULL,
    coupon_id uuid NOT NULL,
    assigned_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    is_used_score float NOT NULL,
    is_used bool DEFAULT false NOT NULL
);


ALTER TABLE public.user_coupons OWNER TO postgres;

-- SKU PRICE HISTORY TABLE
CREATE TABLE public.sku_price_history (
    promo_id uuid NOT NULL,
    sku_id uuid NOT NULL,
    price numeric(10,2) DEFAULT 0.00 NOT NULL,
    discount_price numeric(10,2) DEFAULT 0.00 NOT NULL,
    start_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    end_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    is_stackable boolean DEFAULT false NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);

ALTER TABLE public.sku_price_history OWNER TO postgres;


-- -- sku_id pk로

-- ALTER TABLE ONLY public.sku
--     ADD CONSTRAINT sku_pkey PRIMARY KEY (sku_id);


--
-- Name: brands brands_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

-- ALTER TABLE ONLY public.brands
--     ADD CONSTRAINT brands_pkey PRIMARY KEY (brand_id);


--
-- Name: cart cart_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cart
    ADD CONSTRAINT cart_pkey PRIMARY KEY (cart_id);


--
-- Name: categories categories_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

-- ALTER TABLE ONLY public.categories
--     ADD CONSTRAINT categories_pkey PRIMARY KEY (category_id);


--
-- Name: coupon coupon_code_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.coupon
    ADD CONSTRAINT coupon_code_key UNIQUE (code);



--
-- Name: coupon_usage coupon_usage_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

-- ALTER TABLE ONLY public.coupon_usage
--     ADD CONSTRAINT coupon_usage_pkey PRIMARY KEY (usage_id);


--
-- Name: inventory inventory_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.inventory
    ADD CONSTRAINT inventory_pkey PRIMARY KEY (inventory_id);


--
-- Name: order_items order_items_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.order_items
    ADD CONSTRAINT order_items_pkey PRIMARY KEY (order_item_id);


--
-- Name: orders orders_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.orders
    ADD CONSTRAINT orders_pkey PRIMARY KEY (order_id);


--
-- Name: payment payment_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.payment
    ADD CONSTRAINT payment_pkey PRIMARY KEY (payment_id);


--
-- Name: point_transaction point_transaction_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.point_transaction
    ADD CONSTRAINT point_transaction_pkey PRIMARY KEY (transaction_id);


-- --
-- -- Name: products products_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
-- --

-- ALTER TABLE ONLY public.products
--     ADD CONSTRAINT products_pkey PRIMARY KEY (product_id);


--
-- Name: review review_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.review
    ADD CONSTRAINT review_pkey PRIMARY KEY (review_id);


--
-- Name: shipping shipping_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.shipping
    ADD CONSTRAINT shipping_pkey PRIMARY KEY (shipping_id);


--
-- Name: user_coupons unique_user_coupon; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_coupons
    ADD CONSTRAINT unique_user_coupon UNIQUE (user_id, coupon_id);


--
-- Name: review unique_user_product_review; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.review
    ADD CONSTRAINT unique_user_product_review UNIQUE (user_id, product_id);


--
-- Name: user_coupons user_coupons_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_coupons
    ADD CONSTRAINT user_coupons_pkey PRIMARY KEY (id);



-- promo fk

ALTER TABLE ONLY public.coupon
    ADD CONSTRAINT fk_coupon_promo FOREIGN KEY (promo_id) REFERENCES public.promotion(promo_id);

ALTER TABLE ONLY public.sku_price_history
    ADD CONSTRAINT fk_sku_his_promo FOREIGN KEY (promo_id) REFERENCES public.promotion(promo_id);

-- sku fk

ALTER TABLE ONLY public.sku_price_history
    ADD CONSTRAINT fk_sku_his_sku FOREIGN KEY (sku_id) REFERENCES public.sku(sku_id);

-- sku.product id fk로
-- ALTER TABLE ONLY public.sku
--     ADD CONSTRAINT fk_sku_product FOREIGN KEY (product_id) REFERENCES public.products(product_id);


--
-- Name: brands brands_category_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.brands
    ADD CONSTRAINT brands_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.categories(category_id);


--
-- Name: cart fk_cart_product; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cart
    ADD CONSTRAINT fk_cart_sku FOREIGN KEY (sku_id) REFERENCES public.sku(sku_id);


--
-- Name: cart fk_cart_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cart
    ADD CONSTRAINT fk_cart_user FOREIGN KEY (user_id) REFERENCES public.users(user_id);


--
-- Name: products fk_category; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.products
    ADD CONSTRAINT fk_category FOREIGN KEY (category_id) REFERENCES public.categories(category_id);


--
-- Name: coupon_usage fk_coupon_usage_coupon; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

-- ALTER TABLE ONLY public.coupon_usage
--     ADD CONSTRAINT fk_coupon_usage_coupon FOREIGN KEY (coupon_id) REFERENCES public.coupon(coupon_id);


--
-- Name: coupon_usage fk_coupon_usage_order; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

-- ALTER TABLE ONLY public.coupon_usage
--     ADD CONSTRAINT fk_coupon_usage_order FOREIGN KEY (order_id) REFERENCES public.orders(order_id);


--
-- Name: coupon_usage fk_coupon_usage_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

-- ALTER TABLE ONLY public.coupon_usage
--     ADD CONSTRAINT fk_coupon_usage_user FOREIGN KEY (user_id) REFERENCES public.users(user_id);


--
-- Name: inventory fk_inventory_product; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.inventory
    ADD CONSTRAINT fk_inventory_sku FOREIGN KEY (sku_id) REFERENCES public.sku(sku_id);


--
-- Name: order_items fk_order_item_order; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.order_items
    ADD CONSTRAINT fk_order_item_order FOREIGN KEY (order_id) REFERENCES public.orders(order_id);


--
-- Name: order_items fk_order_item_product; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.order_items
    ADD CONSTRAINT fk_order_item_sku FOREIGN KEY (sku_id) REFERENCES public.sku(sku_id);


--
-- Name: orders fk_order_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.orders
    ADD CONSTRAINT fk_order_user FOREIGN KEY (user_id) REFERENCES public.users(user_id);


--
-- Name: categories fk_parent_category; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.categories
    ADD CONSTRAINT fk_parent_category FOREIGN KEY (parent_id) REFERENCES public.categories(category_id);


--
-- Name: payment fk_payment_order; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

-- ALTER TABLE ONLY public.payment
--     ADD CONSTRAINT fk_payment_order FOREIGN KEY (order_id) REFERENCES public.orders(order_id);


--
-- Name: point_transaction fk_point_transaction_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.point_transaction
    ADD CONSTRAINT fk_point_transaction_user FOREIGN KEY (user_id) REFERENCES public.users(user_id);


--
-- Name: review fk_review_product; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.review
    ADD CONSTRAINT fk_review_product FOREIGN KEY (product_id) REFERENCES public.products(product_id);


--
-- Name: review fk_review_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.review
    ADD CONSTRAINT fk_review_user FOREIGN KEY (user_id) REFERENCES public.users(user_id);


--
-- Name: shipping fk_shipping_order; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.shipping
    ADD CONSTRAINT fk_shipping_order FOREIGN KEY (order_id) REFERENCES public.orders(order_id);


--
-- Name: user_coupons fk_user_coupon_coupon; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_coupons
    ADD CONSTRAINT fk_user_coupon_coupon FOREIGN KEY (coupon_id) REFERENCES public.coupon(coupon_id);


--
-- Name: user_coupons fk_user_coupon_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_coupons
    ADD CONSTRAINT fk_user_coupon_user FOREIGN KEY (user_id) REFERENCES public.users(user_id);

ALTER TABLE public.sku
ALTER COLUMN sku_code TYPE VARCHAR(50),
ALTER COLUMN variant_name TYPE VARCHAR(100),
ALTER COLUMN color TYPE VARCHAR(50),
ALTER COLUMN variant_option TYPE VARCHAR(50);

ALTER TABLE public.promotion
ALTER COLUMN name TYPE VARCHAR(100);

--
-- PostgreSQL database dump complete
--

