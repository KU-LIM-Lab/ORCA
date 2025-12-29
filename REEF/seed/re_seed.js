const getClient = require("./db");

(async () => {
  const client = getClient();
  await client.connect();
  console.log("Syncing sku.created_at with product.created_at...");

  await client.query(`
    UPDATE sku
    SET
    created_at = p.created_at,
    available_from = p.created_at + INTERVAL '3 days',
    discontinued_at = '9999-12-31'
    FROM products p
    WHERE sku.product_id = p.product_id
  `);

  console.log("✅ sku.created_at 동기화 완료!");
  await client.end();
})();
