import json
from typing import Dict, List, Optional

import redis

from app.errors import unavailable_on

REDIS_ERRORS = (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError)


class RedisMemoryService:
    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        timeout: int = 5,
        ttl_seconds: int = 60 * 60 * 24,
        max_messages: int = 50,
    ) -> None:
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        )
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages

    def add_message(self, session_id: str, role: str, content: str) -> None:
        key = f"chat:{session_id}"
        message = {"role": role, "content": content}
        with unavailable_on(REDIS_ERRORS, "Redis"):
            pipeline = self.client.pipeline()
            pipeline.rpush(key, json.dumps(message))
            # Keep only the newest messages and expire idle sessions.
            pipeline.ltrim(key, -self.max_messages, -1)
            pipeline.expire(key, self.ttl_seconds)
            pipeline.execute()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        key = f"chat:{session_id}"
        with unavailable_on(REDIS_ERRORS, "Redis"):
            messages = self.client.lrange(key, 0, -1)
        return [json.loads(message) for message in messages]

    def clear_history(self, session_id: str) -> None:
        key = f"chat:{session_id}"
        with unavailable_on(REDIS_ERRORS, "Redis"):
            self.client.delete(key)

    def get_state(self, session_id: str, name: str) -> Optional[Dict]:
        """Read a small JSON state object stored for a session (e.g. a pending booking)."""
        with unavailable_on(REDIS_ERRORS, "Redis"):
            value = self.client.get(f"{name}:{session_id}")
        return json.loads(value) if value is not None else None

    def set_state(self, session_id: str, name: str, value: Dict, ttl_seconds: int) -> None:
        with unavailable_on(REDIS_ERRORS, "Redis"):
            self.client.set(f"{name}:{session_id}", json.dumps(value), ex=ttl_seconds)

    def clear_state(self, session_id: str, name: str) -> None:
        with unavailable_on(REDIS_ERRORS, "Redis"):
            self.client.delete(f"{name}:{session_id}")

    def ping(self) -> bool:
        with unavailable_on(REDIS_ERRORS, "Redis"):
            return bool(self.client.ping())
