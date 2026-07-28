"""Context Lifecycle Engine 的状态、引用和扩展协议。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from matterloop_models import ModelContextScope, ModelInputItem


class ContextPressure(str, Enum):
    """表示一次模型调用的上下文预算压力。"""

    NORMAL = "normal"
    SOFT = "soft"
    HARD = "hard"


@dataclass(frozen=True, slots=True)
class ContextTokenState:
    """记录当前、预测和最大 Token 数。"""

    current_tokens: int
    projected_tokens: int
    limit_tokens: int

    def __post_init__(self) -> None:
        """拒绝负数和小于当前用量的预测值。"""
        if min(self.current_tokens, self.projected_tokens) < 0:
            raise ValueError("context token counts must not be negative")
        if self.limit_tokens < 1:
            raise ValueError("context token limit must be positive")
        if self.projected_tokens < self.current_tokens:
            raise ValueError("projected tokens must not be below current tokens")

    @property
    def usage_ratio(self) -> float:
        """返回预测用量占上下文窗口的比例。"""
        return self.projected_tokens / self.limit_tokens


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """保存一个模型参与者可跨进程恢复的规范上下文。"""

    key: str
    scope: ModelContextScope
    revision: int
    input_items: tuple[ModelInputItem, ...] = field(repr=False)
    token_state: ContextTokenState
    provider: str | None = None
    model: str | None = None
    compaction_count: int = 0
    archive_uris: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """校验快照标识、版本和模型亲和信息。"""
        if not self.key.strip() or self.key != self.scope.key:
            raise ValueError("context snapshot key must match its scope")
        if self.revision < 1:
            raise ValueError("context snapshot revision must be at least 1")
        if self.compaction_count < 0:
            raise ValueError("context compaction count must not be negative")
        if self.provider is not None and not self.provider.strip():
            raise ValueError("context snapshot provider must not be empty")
        if self.model is not None and not self.model.strip():
            raise ValueError("context snapshot model must not be empty")
        if any(not uri.strip() for uri in self.archive_uris):
            raise ValueError("context archive URI must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ContextSnapshotRef:
    """引用 ContextStore 中一个精确、可校验的不可变版本。"""

    key: str
    revision: int
    checksum: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        """保证引用能够定位并校验一个确定版本。"""
        if not self.key.strip() or not self.checksum.strip():
            raise ValueError("context snapshot reference fields must not be empty")
        if self.revision < 1 or self.schema_version < 1:
            raise ValueError("context snapshot reference versions must be positive")


@dataclass(frozen=True, slots=True)
class ContextBlobRef:
    """引用外置的历史或工具原始结果。"""

    uri: str
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        """校验制品引用的完整性元数据。"""
        if not self.uri.strip() or not self.sha256.strip() or not self.media_type.strip():
            raise ValueError("context blob reference fields must not be empty")
        if self.size_bytes < 0:
            raise ValueError("context blob size must not be negative")


@runtime_checkable
class ContextStore(Protocol):
    """持久化不可变 ContextSnapshot 版本并维护 latest 指针。"""

    async def load(self, key: str, revision: int | None = None) -> ContextSnapshot | None:
        """读取 latest 或指定精确版本。"""
        ...

    async def save(
        self,
        snapshot: ContextSnapshot,
        *,
        expected_revision: int,
    ) -> ContextSnapshotRef:
        """使用 CAS 保存新版本并返回校验引用。"""
        ...


@runtime_checkable
class ContextBlobStore(Protocol):
    """外置保存不应直接进入模型窗口的大对象。"""

    async def put(
        self,
        content: bytes,
        *,
        media_type: str,
        purpose: str,
        ttl_seconds: int | None = None,
    ) -> ContextBlobRef:
        """保存不可变内容并返回校验引用。"""
        ...

    async def get(self, uri: str) -> bytes:
        """读取此前保存的原始内容。"""
        ...


@runtime_checkable
class ContextMemorySink(Protocol):
    """接收经过宿主准入策略批准的长期记忆候选。"""

    async def remember(
        self,
        scope: ModelContextScope,
        memories: Sequence[str],
        *,
        source_item_ids: Sequence[str],
    ) -> None:
        """保存带来源的稳定记忆。"""
        ...


@runtime_checkable
class MemoryAdmissionPolicy(Protocol):
    """过滤可从压缩摘要进入长期记忆的内容。"""

    def admit(self, candidates: Sequence[str]) -> tuple[str, ...]:
        """返回允许持久化的候选。"""
        ...


__all__ = [
    "ContextBlobRef",
    "ContextBlobStore",
    "ContextMemorySink",
    "ContextPressure",
    "ContextSnapshot",
    "ContextSnapshotRef",
    "ContextStore",
    "ContextTokenState",
    "MemoryAdmissionPolicy",
]
