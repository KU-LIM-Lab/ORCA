import redis
from .settings import REDIS_CONFIG

# Override decode_responses for binary data support
redis_config = REDIS_CONFIG.copy()
redis_config['decode_responses'] = False

redis_client = redis.Redis(**redis_config)
async_redis_client = redis.asyncio.Redis(**redis_config)