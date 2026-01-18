const { faker } = require('@faker-js/faker');
const getClient = require('./db');

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log("Connected. Seeding inventory...");

  // Get sku and corresponding product's stock_quantity, sku.created_at together
  const result = await client.query(`
    SELECT
      s.sku_id,
      s.created_at AS sku_created_at,
      p.stock_quantity
    FROM sku s
    JOIN products p ON s.product_id = p.product_id
  `);

  const rows = result.rows;

  for (const row of rows) {
    const inventory_id = faker.string.uuid();

    // SCM 1) quantity = products.stock_quantity
    const quantity = row.stock_quantity ?? 0;

    // SCM 2) last_updated = sku.created_at  (can add some noise here if needed)
    const lastUpdated = row.sku_created_at;

    await client.query(
      `
      INSERT INTO inventory (inventory_id, sku_id, quantity, last_updated)
      VALUES ($1, $2, $3, $4)
    `,
      [inventory_id, row.sku_id, quantity, lastUpdated]
    );
  }

  console.log("✅ Inventory seeded successfully.");
  await client.end();
};