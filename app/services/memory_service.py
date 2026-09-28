import json
from typing import List, Dict

import redis

from app.errors import unavailable_on

REDIS_ERRORS = (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError)


class RedisMemoryService:
    def __init__(self, host: str, port: int, db: int = 0, timeout: int = 5) -> None:
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        key = f"chat:{session_id}"
        message = {"role": role, "content": content}
        with unavailable_on(REDIS_ERRORS, "Redis"):
            self.client.rpush(key, json.dumps(message))

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        key = f"chat:{session_id}"
        with unavailable_on(REDIS_ERRORS, "Redis"):
            messages = self.client.lrange(key, 0, -1)
        return [json.loads(message) for message in messages]

    def clear_history(self, session_id: str) -> None:
        key = f"chat:{session_id}"
        with unavailable_on(REDIS_ERRORS, "Redis"):
            self.client.delete(key)

    def ping(self) -> bool:
        with unavailable_on(REDIS_ERRORS, "Redis"):
            return bool(self.client.ping())
