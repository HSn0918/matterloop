"""使用 Redis 不可变版本和 CAS latest 指针保存模型上下文。"""

from __future__ import annotations

import hashlib

from matterloop_runtime import (
    ContextConflictError,
    ContextRetentionPolicy,
    ContextSnapshot,
    ContextSnapshotCodec,
    ContextSnapshotRef,
)

from matterloop_integration_redis.client import AsyncRedisClient, RedisConfig
from matterloop_integration_redis.errors import RedisPayloadError

_SAVE_SCRIPT = """
-- matterloop:context-save
local expected = tonumber(ARGV[1])
local revision = tonumber(ARGV[2])
local ttl = tonumber(ARGV[4])
if not expected or not revision or not ttl or ttl < 1 or revision ~= expected + 1 then
  return -2
end
local current = redis.call('GET', KEYS[1])
if not current then
  if expected ~= 0 then
    return 0
  end
else
  local current_revision = tonumber(current)
  if not current_revision then
    return -2
  end
  if current_revision ~= expected then
    return 0
  end
end
redis.call('SET', KEYS[2], ARGV[3], 'EX', ttl)
redis.call('SET', KEYS[1], tostring(revision), 'EX', ttl)
return revision
"""


class RedisContextStore:
    """在宿主注入的 Redis 中保存可恢复 ContextSnapshot。"""

    def __init__(
        self,
        client: AsyncRedisClient,
        retention: ContextRetentionPolicy,
        config: RedisConfig | None = None,
        *,
        codec: ContextSnapshotCodec | None = None,
    ) -> None:
        self._client = client
        self._retention = retention
        self._config = config or RedisConfig()
        self._codec = codec or ContextSnapshotCodec()

    async def load(
        self,
        key: str,
        revision: int | None = None,
    ) -> ContextSnapshot | None:
        """读取 latest 或调用方指定的精确不可变版本。"""
        if not key.strip():
            raise ValueError("context key must not be empty")
        selected = revision
        if selected is None:
            raw_latest = await self._client.get(self._latest_key(key))
            if raw_latest is None:
                return None
            selected = _integer(raw_latest, "context latest revision")
        if selected < 1:
            raise ValueError("context revision must be positive")
        raw = await self._client.get(self._version_key(key, selected))
        if raw is None:
            return None
        payload = _text(raw, "context snapshot")
        try:
            snapshot = self._codec.loads(payload)
        except Exception as exc:
            raise RedisPayloadError(f"Redis context snapshot is invalid: {exc}") from exc
        if snapshot.key != key or snapshot.revision != selected:
            raise RedisPayloadError("Redis context snapshot does not match its key")
        return snapshot

    async def save(
        self,
        snapshot: ContextSnapshot,
        *,
        expected_revision: int,
    ) -> ContextSnapshotRef:
        """原子写入新版本并推进 latest 指针。"""
        if expected_revision < 0:
            raise ValueError("expected context revision must not be negative")
        if snapshot.revision != expected_revision + 1:
            raise ValueError("context snapshot revision must equal expected revision plus one")
        try:
            payload = self._codec.dumps(snapshot)
        except Exception as exc:
            raise RedisPayloadError(f"context snapshot is invalid: {exc}") from exc
        result = await self._client.eval(
            _SAVE_SCRIPT,
            2,
            self._latest_key(snapshot.key),
            self._version_key(snapshot.key, snapshot.revision),
            str(expected_revision),
            str(snapshot.revision),
            payload,
            str(self._retention.snapshot_ttl_seconds),
        )
        saved_revision = _integer(result, "context CAS response")
        if saved_revision == 0:
            raise ContextConflictError(
                f"context revision conflict for {snapshot.key}: expected {expected_revision}"
            )
        if saved_revision == -2:
            raise RedisPayloadError("Redis context state is corrupted or violates CAS schema")
        if saved_revision != snapshot.revision:
            raise RedisPayloadError("Redis returned an invalid context revision")
        return ContextSnapshotRef(
            key=snapshot.key,
            revision=snapshot.revision,
            checksum=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            schema_version=self._codec.schema_version,
        )

    def _latest_key(self, key: str) -> str:
        return f"{self._config.prefix}:contexts:{key}:latest"

    def _version_key(self, key: str, revision: int) -> str:
        return f"{self._config.prefix}:contexts:{key}:versions:{revision}"


def _text(value: object, purpose: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RedisPayloadError(f"Redis {purpose} is not valid UTF-8") from exc
    if isinstance(value, str):
        return value
    raise RedisPayloadError(f"Redis {purpose} must be text or bytes")


def _integer(value: object, purpose: str) -> int:
    decoded = _text(value, purpose) if isinstance(value, (str, bytes)) else value
    if isinstance(decoded, str):
        try:
            decoded = int(decoded)
        except ValueError as exc:
            raise RedisPayloadError(f"Redis {purpose} is not an integer") from exc
    if not isinstance(decoded, int) or isinstance(decoded, bool):
        raise RedisPayloadError(f"Redis {purpose} is not an integer")
    return decoded


__all__ = ["RedisContextStore"]
