const seedrandom = require("seedrandom");
const { faker } = require("@faker-js/faker");

module.exports = function initSeed(seed = 42) {
  seedrandom(String(seed), { global: true }); // Fix Math.random
  faker.seed(seed);                           // Fix faker
  console.log(`[Seed initialized] seed=${seed}`);
};