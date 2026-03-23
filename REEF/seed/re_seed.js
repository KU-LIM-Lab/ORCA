const getClient = require('./db');

module.exports = async function () {
  const client = getClient();
  await client.connect();
  console.log('Connected. Updating users.point_balance from point_transaction...');

  try {
    await client.query('BEGIN');

    await client.query(`
      UPDATE users u
      SET point_balance = COALESCE(pt.total_points, 0)
      FROM (
        SELECT user_id, SUM(point_change) AS total_points
        FROM point_transaction
        GROUP BY user_id
      ) AS pt
      WHERE u.user_id = pt.user_id
    `);

    await client.query(`
      UPDATE users
      SET point_balance = 0
      WHERE point_balance IS NULL
    `);

    await client.query('COMMIT');
    console.log('✅ users.point_balance updated based on point_transaction.');
  } catch (err) {
    await client.query('ROLLBACK');
    console.error('❌ Error updating point_balance:', err.message);
  } finally {
    await client.end();
  }
};