const seedrandom = require("seedrandom");
const { faker } = require("@faker-js/faker");

module.exports = function initSeed(seed = 42) {
  seedrandom(String(seed), { global: true }); // Math.random 고정
  faker.seed(seed);                           // faker 고정
  console.log(`[Seed initialized] seed=${seed}`);
};