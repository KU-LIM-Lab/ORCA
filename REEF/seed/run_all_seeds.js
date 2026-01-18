const initSeed = require("./seed");
initSeed(process.env.SEED ?? 2);

const seedUsers = require("./seed_users");
const seedCategories = require("./seed_categories");
const seedBrands = require("./seed_brands");
const seedCart = require("./seed_carts");
const seedProducts = require("./seed_products");
const seedInventory = require("./seed_inventory");
const seedSku = require("./seed_sku");
const seedPromo = require("./seed_promo");
const seedSkuPriceHistory = require("./seed_sku_price_history");
const seedCoupon = require("./seed_coupon");
const seedUserCoupons = require("./seed_user_coupons");
const seedOrders = require("./seed_order_process");
const seedCouponUsage = require("./seed_coupon_usage");
const seedPointTransactions = require("./seed_point_transaction");
const seedReviews = require("./seed_reviews");
// const updateAvgRatings = require("./update_avg_ratings");

(async () => {
  try {
    console.log("🌱 Starting data seeding...");

    console.log("👥 Seeding users...");
    await seedUsers();

    console.log("📂 Seeding categories...");
    await seedCategories();

    console.log("🏷️ Seeding brands...");
    await seedBrands();

    console.log("📦 Seeding products...");
    await seedProducts();

    console.log("📦 Seeding inventory...");
    await seedInventory();

    console.log("🔢 Seeding SKUs...");
    await seedSku();

    console.log("🛒 Seeding cart...");
    await seedCart();
    
    console.log("🎯 Seeding promotions...");
    await seedPromo();

    console.log("Seeding sku price history ...")
    await seedSkuPriceHistory()

    console.log("🎫 Seeding coupons...");
    await seedCoupon();

    console.log("🎫 Seeding user coupons...");
    await seedUserCoupons();

    console.log("🛒 Seeding orders...");
    await seedOrders();

    console.log("Seeding coupon usage ...")
    await seedCouponUsage()

    console.log("💰 Seeding point transactions...");
    await seedPointTransactions();

    console.log("⭐ Seeding reviews...");
    await seedReviews();

    console.log("✅ All seeding completed successfully!");
  } catch (error) {
    console.error("❌ Error during seeding:", error);
  }
})();
