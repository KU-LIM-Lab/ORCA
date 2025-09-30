const { Client } = require("pg");
require("dotenv").config();

module.exports = function getClient() {
  return new Client({
    user: process.env.PGUSER,
    host: process.env.PGHOST,
    database: process.env.PGDATABASE,
    password: process.env.PGPASSWORD,
    port: process.env.PGPORT,
  });
};
