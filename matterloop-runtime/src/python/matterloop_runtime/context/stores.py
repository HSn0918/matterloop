"""Context 快照编解码、内存存储与本地 Blob Store。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from matterloop_models import (
    MessageRole,
    ModelCompactionItem,
    ModelContextScope,
    ModelInputItem,
    ModelItemCategory,
    ModelItemRetention,
    ModelMessageItem,
    ModelToolCallItem,
    ModelToolOutputItem,
)

from matterloop_runtime.context.errors import ContextConflictError, ContextSnapshotError
from matterloop_runtime.context.models import (
    ContextBlobRef,
    ContextSnapshot,
    ContextSnapshotRef,
    ContextTokenState,
)


class ContextSnapshotCodec:
    """在严格 JSON 数据和 ContextSnapshot 之间转换。"""

    schema_version = 1

    def dumps(self, snapshot: ContextSnapshot) -> str:
        """编码为稳定 JSON，编码失败时不降级为字符串。"""
        try:
            return json.dumps(
                self.encode(snapshot),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ContextSnapshotError(f"context snapshot is not JSON serializable: {exc}") from exc

    def loads(self, payload: str) -> ContextSnapshot:
        """解码并校验完整快照。"""
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContextSnapshotError("context snapshot is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ContextSnapshotError("context snapshot root must be an object")
        return self.decode(cast(Mapping[str, object], value))

    def checksum(self, snapshot: ContextSnapshot) -> str:
        """返回规范编码的 SHA-256。"""
        return hashlib.sha256(self.dumps(snapshot).encode("utf-8")).hexdigest()

    def encode(self, snapshot: ContextSnapshot) -> dict[str, object]:
        """编码快照及全部规范输入项。"""
        return {
            "schema_version": self.schema_version,
            "key": snapshot.key,
            "scope": {
                "tenant_id": snapshot.scope.tenant_id,
                "run_id": snapshot.scope.run_id,
                "participant": snapshot.scope.participant,
                "task_id": snapshot.scope.task_id,
                "invocation_id": snapshot.scope.invocation_id,
            },
            "revision": snapshot.revision,
            "input_items": encode_input_items(snapshot.input_items),
            "token_state": {
                "current_tokens": snapshot.token_state.current_tokens,
                "projected_tokens": snapshot.token_state.projected_tokens,
                "limit_tokens": snapshot.token_state.limit_tokens,
            },
            "provider": snapshot.provider,
            "model": snapshot.model,
            "compaction_count": snapshot.compaction_count,
            "archive_uris": list(snapshot.archive_uris),
            "metadata": dict(snapshot.metadata),
            "created_at": snapshot.created_at.isoformat(),
            "updated_at": snapshot.updated_at.isoformat(),
        }

    def decode(self, value: Mapping[str, object]) -> ContextSnapshot:
        """从 JSON 映射恢复值对象。"""
        if value.get("schema_version") != self.schema_version:
            raise ContextSnapshotError("unsupported context snapshot schema version")
        try:
            scope_value = _mapping(value["scope"], "scope")
            token_value = _mapping(value["token_state"], "token_state")
            scope = ModelContextScope(
                tenant_id=_text(scope_value.get("tenant_id"), "scope.tenant_id"),
                run_id=_text(scope_value.get("run_id"), "scope.run_id"),
                participant=_text(scope_value.get("participant"), "scope.participant"),
                task_id=_optional_text(scope_value.get("task_id"), "scope.task_id"),
                invocation_id=_optional_text(
                    scope_value.get("invocation_id"), "scope.invocation_id"
                ),
            )
            raw_items = value["input_items"]
            if not isinstance(raw_items, list):
                raise ContextSnapshotError("input_items must be an array")
            raw_archives = value["archive_uris"]
            if not isinstance(raw_archives, list):
                raise ContextSnapshotError("archive_uris must be an array")
            return ContextSnapshot(
                key=_text(value.get("key"), "key"),
                scope=scope,
                revision=_integer(value.get("revision"), "revision"),
                input_items=decode_input_items(raw_items),
                token_state=ContextTokenState(
                    current_tokens=_integer(
                        token_value.get("current_tokens"), "token_state.current_tokens"
                    ),
                    projected_tokens=_integer(
                        token_value.get("projected_tokens"), "token_state.projected_tokens"
                    ),
                    limit_tokens=_integer(
                        token_value.get("limit_tokens"), "token_state.limit_tokens"
                    ),
                ),
                provider=_optional_text(value.get("provider"), "provider"),
                model=_optional_text(value.get("model"), "model"),
                compaction_count=_integer(value.get("compaction_count"), "compaction_count"),
                archive_uris=tuple(_text(item, "archive URI") for item in raw_archives),
                metadata=_mapping(value.get("metadata"), "metadata"),
                created_at=_datetime(value.get("created_at"), "created_at"),
                updated_at=_datetime(value.get("updated_at"), "updated_at"),
            )
        except KeyError as exc:
            raise ContextSnapshotError(f"context snapshot is missing {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ContextSnapshotError):
                raise
            raise ContextSnapshotError(f"context snapshot is invalid: {exc}") from exc


class InMemoryContextStore:
    """使用进程内锁实现版本 CAS 的 ContextStore。"""

    def __init__(self, *, codec: ContextSnapshotCodec | None = None) -> None:
        self._codec = codec or ContextSnapshotCodec()
        self._values: dict[str, dict[int, str]] = {}
        self._latest: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def load(self, key: str, revision: int | None = None) -> ContextSnapshot | None:
        """读取并重新解码，避免调用方修改内部状态。"""
        if not key.strip():
            raise ValueError("context key must not be empty")
        async with self._lock:
            selected = self._latest.get(key) if revision is None else revision
            payload = None if selected is None else self._values.get(key, {}).get(selected)
        return None if payload is None else self._codec.loads(payload)

    async def save(
        self,
        snapshot: ContextSnapshot,
        *,
        expected_revision: int,
    ) -> ContextSnapshotRef:
        """仅在 latest 与预期一致时保存下一个不可变版本。"""
        if expected_revision < 0:
            raise ValueError("expected context revision must not be negative")
        if snapshot.revision != expected_revision + 1:
            raise ValueError("context snapshot revision must equal expected revision plus one")
        payload = self._codec.dumps(snapshot)
        async with self._lock:
            current = self._latest.get(snapshot.key, 0)
            if current != expected_revision:
                raise ContextConflictError(
                    f"context revision conflict for {snapshot.key}: expected "
                    f"{expected_revision}, found {current}"
                )
            self._values.setdefault(snapshot.key, {})[snapshot.revision] = payload
            self._latest[snapshot.key] = snapshot.revision
        return ContextSnapshotRef(
            key=snapshot.key,
            revision=snapshot.revision,
            checksum=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            schema_version=self._codec.schema_version,
        )


class InMemoryContextBlobStore:
    """按内容哈希去重的进程内 Blob Store。"""

    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    async def put(
        self,
        content: bytes,
        *,
        media_type: str,
        purpose: str,
        ttl_seconds: int | None = None,
    ) -> ContextBlobRef:
        """复制保存内容并返回不可变引用。"""
        _validate_blob_arguments(media_type, purpose)
        _validate_blob_ttl(ttl_seconds)
        digest = hashlib.sha256(content).hexdigest()
        uri = f"context-memory://{purpose}/{digest}"
        async with self._lock:
            self._values[uri] = bytes(content)
        return ContextBlobRef(uri, digest, len(content), media_type)

    async def get(self, uri: str) -> bytes:
        """读取内容副本。"""
        async with self._lock:
            try:
                return bytes(self._values[uri])
            except KeyError as exc:
                raise FileNotFoundError(uri) from exc


class FilesystemContextBlobStore:
    """在显式根目录下按 SHA-256 保存本地不可变 Blob。"""

    def __init__(self, root: str | Path) -> None:
        path = Path(root).resolve()
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError("context blob root must be a directory")
        self._root = path

    async def put(
        self,
        content: bytes,
        *,
        media_type: str,
        purpose: str,
        ttl_seconds: int | None = None,
    ) -> ContextBlobRef:
        """使用原子替换写入内容哈希文件。"""
        _validate_blob_arguments(media_type, purpose)
        _validate_blob_ttl(ttl_seconds)
        digest = hashlib.sha256(content).hexdigest()
        directory = self._root / purpose
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / digest
        if not target.exists():
            temporary = directory / f".{digest}.{os.getpid()}.tmp"
            temporary.write_bytes(content)
            os.replace(temporary, target)
        uri = f"context-file://{purpose}/{digest}"
        return ContextBlobRef(uri, digest, len(content), media_type)

    async def get(self, uri: str) -> bytes:
        """校验 URI 结构、路径边界和内容哈希后读取。"""
        prefix = "context-file://"
        if not uri.startswith(prefix):
            raise ValueError("unsupported context file URI")
        relative = uri[len(prefix) :]
        parts = relative.split("/")
        if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
            raise ValueError("invalid context file URI")
        target = (self._root / parts[0] / parts[1]).resolve()
        if self._root not in target.parents:
            raise ValueError("context file URI escapes its root")
        content = target.read_bytes()
        if hashlib.sha256(content).hexdigest() != parts[1]:
            raise ContextSnapshotError("context blob checksum does not match its URI")
        return content


def encode_input_items(items: Sequence[ModelInputItem]) -> list[dict[str, object]]:
    """把规范输入编码为 JSON 标准类型。"""
    encoded: list[dict[str, object]] = []
    for item in items:
        if item.category is None or item.retention is None:
            raise ValueError("model input item classification was not normalized")
        base: dict[str, object] = {
            "item_id": item.item_id,
            "category": item.category.value,
            "retention": item.retention.value,
            "metadata": dict(item.metadata),
        }
        if isinstance(item, ModelMessageItem):
            base.update(
                {
                    "type": "message",
                    "role": item.role.value,
                    "content": item.content,
                    "name": item.name,
                }
            )
        elif isinstance(item, ModelToolCallItem):
            base.update(
                {
                    "type": "tool_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": dict(item.arguments),
                }
            )
        elif isinstance(item, ModelToolOutputItem):
            base.update(
                {
                    "type": "tool_output",
                    "call_id": item.call_id,
                    "output": item.output,
                    "is_error": item.is_error,
                    "artifact_uri": item.artifact_uri,
                }
            )
        elif isinstance(item, ModelCompactionItem):
            base.update(
                {
                    "type": "compaction",
                    "payload": item.payload,
                    "provider": item.provider,
                    "model": item.model,
                    "native": item.native,
                }
            )
        else:
            raise TypeError(f"unsupported model input item: {type(item).__name__}")
        encoded.append(base)
    return encoded


def decode_input_items(values: Sequence[object]) -> tuple[ModelInputItem, ...]:
    """从 JSON 数组恢复规范输入项。"""
    items: list[ModelInputItem] = []
    for raw in values:
        value = _mapping(raw, "input item")
        item_id = _text(value.get("item_id"), "input item id")
        category = ModelItemCategory(_text(value.get("category"), "item category"))
        retention = ModelItemRetention(_text(value.get("retention"), "item retention"))
        metadata = _mapping(value.get("metadata"), "item metadata")
        item_type = value.get("type")
        if item_type == "message":
            items.append(
                ModelMessageItem(
                    role=MessageRole(_text(value.get("role"), "message role")),
                    content=_text(value.get("content"), "message content"),
                    name=_optional_text(value.get("name"), "message name"),
                    item_id=item_id,
                    category=category,
                    retention=retention,
                    metadata=metadata,
                )
            )
        elif item_type == "tool_call":
            items.append(
                ModelToolCallItem(
                    call_id=_text(value.get("call_id"), "tool call id"),
                    name=_text(value.get("name"), "tool call name"),
                    arguments=_mapping(value.get("arguments"), "tool call arguments"),
                    item_id=item_id,
                    category=category,
                    retention=retention,
                    metadata=metadata,
                )
            )
        elif item_type == "tool_output":
            is_error = value.get("is_error")
            if not isinstance(is_error, bool):
                raise ContextSnapshotError("tool output is_error must be boolean")
            items.append(
                ModelToolOutputItem(
                    call_id=_text(value.get("call_id"), "tool output call id"),
                    output=_string(value.get("output"), "tool output"),
                    is_error=is_error,
                    artifact_uri=_optional_text(
                        value.get("artifact_uri"), "tool output artifact URI"
                    ),
                    item_id=item_id,
                    category=category,
                    retention=retention,
                    metadata=metadata,
                )
            )
        elif item_type == "compaction":
            native = value.get("native")
            if not isinstance(native, bool):
                raise ContextSnapshotError("compaction native must be boolean")
            items.append(
                ModelCompactionItem(
                    payload=_text(value.get("payload"), "compaction payload"),
                    provider=_text(value.get("provider"), "compaction provider"),
                    model=_text(value.get("model"), "compaction model"),
                    native=native,
                    item_id=item_id,
                    category=category,
                    retention=retention,
                    metadata=metadata,
                )
            )
        else:
            raise ContextSnapshotError(f"unsupported context input item type: {item_type!r}")
    return tuple(items)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ContextSnapshotError(f"{path} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextSnapshotError(f"{path} must be non-empty text")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ContextSnapshotError(f"{path} must be text")
    return value


def _optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path)


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContextSnapshotError(f"{path} must be an integer")
    return value


def _datetime(value: object, path: str) -> datetime:
    text = _text(value, path)
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContextSnapshotError(f"{path} must be an ISO datetime") from exc


def _validate_blob_arguments(media_type: str, purpose: str) -> None:
    if not media_type.strip():
        raise ValueError("context blob media type must not be empty")
    if not purpose or not purpose.replace("-", "").replace("_", "").isalnum():
        raise ValueError("context blob purpose must be a safe non-empty identifier")


def _validate_blob_ttl(ttl_seconds: int | None) -> None:
    if ttl_seconds is not None and (
        not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds < 1
    ):
        raise ValueError("context blob TTL must be a positive integer")


__all__ = [
    "ContextSnapshotCodec",
    "FilesystemContextBlobStore",
    "InMemoryContextBlobStore",
    "InMemoryContextStore",
    "decode_input_items",
    "encode_input_items",
]
