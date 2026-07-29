"""Context 生命周期的无敏感内容事件。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable


class ContextEventType(str, Enum):
    """列出可审计的上下文生命周期变化。"""

    PRESSURE = "pressure"
    COMPACTION_STARTED = "compaction_started"
    COMPACTION_COMPLETED = "compaction_completed"
    COMPACTION_FAILED = "compaction_failed"
    TOOL_RESULT_EXTERNALIZED = "tool_result_externalized"
    SNAPSHOT_SAVED = "snapshot_saved"
    SNAPSHOT_RESTORED = "snapshot_restored"


@dataclass(frozen=True, slots=True)
class ContextEvent:
    """保存不含原始 Prompt、结果或压缩载荷的观测数据。"""

    event_type: ContextEventType
    context_key: str
    revision: int = 0
    current_tokens: int = 0
    projected_tokens: int = 0
    limit_tokens: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """校验事件字段并冻结安全元数据。"""
        if not self.context_key.strip():
            raise ValueError("context event key must not be empty")
        if (
            min(
                self.revision,
                self.current_tokens,
                self.projected_tokens,
                self.limit_tokens,
            )
            < 0
        ):
            raise ValueError("context event counters must not be negative")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class ContextEventPublisher(Protocol):
    """发布 Context 生命周期事件。"""

    async def publish(self, event: ContextEvent) -> None:
        """发布一条不含原始上下文的事件。"""
        ...


class LocalContextEventPublisher:
    """在内存中保存事件，适用于本地运行和测试。"""

    def __init__(self) -> None:
        self._events: list[ContextEvent] = []

    @property
    def events(self) -> tuple[ContextEvent, ...]:
        """返回隔离的不可变事件序列。"""
        return tuple(self._events)

    async def publish(self, event: ContextEvent) -> None:
        """按发生顺序保存事件。"""
        self._events.append(event)


class NullContextEventPublisher:
    """显式丢弃本地不需要观察的 Context 事件。"""

    async def publish(self, event: ContextEvent) -> None:
        """消费事件但不保留内容。"""
        del event


__all__ = [
    "ContextEvent",
    "ContextEventPublisher",
    "ContextEventType",
    "LocalContextEventPublisher",
    "NullContextEventPublisher",
]
