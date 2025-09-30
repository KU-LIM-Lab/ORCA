const seedUsers = require("./seed_users");
const seedCategories = require("./seed_categories");
const seedProducts = require("./seed_products");
const seedSku = require("./seed_sku");
const seedBrands = require("./seed_brands");
const seedPromo = require("./seed_promo");
const seedCoupon = require("./seed_coupon");
const seedInventory = require("./seed_inventory");
const seedOrders = require("./seed_orders_with_items");
const seedPayments = require("./seed_payments_and_shipping");
const seedReviews = require("./seed_reviews");
const seedUserCoupons = require("./seed_user_coupons");
const seedPointTransactions = require("./seed_point_transaction");
const seedCart = require("./seed_cart");
const updateAvgRatings = require("./update_avg_ratings");

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

    console.log("🔢 Seeding SKUs...");
    await seedSku();

    console.log("🎯 Seeding promotions...");
    await seedPromo();

    console.log("🎫 Seeding coupons...");
    await seedCoupon();

    console.log("📦 Seeding inventory...");
    await seedInventory();

    console.log("🛒 Seeding orders...");
    await seedOrders();

    console.log("💳 Seeding payments and shipping...");
    await seedPayments();

    console.log("⭐ Seeding reviews...");
    await seedReviews();

    console.log("🎫 Seeding user coupons...");
    await seedUserCoupons();

    console.log("💰 Seeding point transactions...");
    await seedPointTransactions();

    console.log("🛒 Seeding cart...");
    await seedCart();

    console.log("📊 Updating average ratings...");
    await updateAvgRatings();

    console.log("✅ All seeding completed successfully!");
  } catch (error) {
    console.error("❌ Error during seeding:", error);
  }
})();
