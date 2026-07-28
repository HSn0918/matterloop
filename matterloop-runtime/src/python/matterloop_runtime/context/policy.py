"""Context Lifecycle Engine 的不可变策略配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """配置上下文预算、压缩、工具外置和记忆提取边界。"""

    max_context_tokens: int | None = None
    soft_threshold: float = 0.70
    hard_threshold: float = 0.85
    target_threshold: float = 0.55
    recent_turns: int = 10
    tool_result_inline_tokens: int = 5_000
    reserved_output_tokens: int = 4_096
    estimate_safety_margin: float = 0.15
    max_compaction_passes: int = 2
    default_tool_output_tokens: int = 5_000
    summary_max_output_tokens: int = 2_048
    memory_extraction_enabled: bool = False
    allow_cross_provider_summary: bool = False

    def __post_init__(self) -> None:
        """拒绝会让预算判断失去确定性的策略。"""
        if self.max_context_tokens is not None and self.max_context_tokens < 1:
            raise ValueError("max context tokens must be at least 1")
        thresholds = (self.target_threshold, self.soft_threshold, self.hard_threshold)
        if any(not math.isfinite(value) for value in thresholds):
            raise ValueError("context thresholds must be finite")
        if not 0 < self.target_threshold < self.soft_threshold < self.hard_threshold < 1:
            raise ValueError("context thresholds must satisfy 0 < target < soft < hard < 1")
        integer_limits = (
            self.recent_turns,
            self.tool_result_inline_tokens,
            self.reserved_output_tokens,
            self.max_compaction_passes,
            self.default_tool_output_tokens,
            self.summary_max_output_tokens,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in integer_limits
        ):
            raise ValueError("context integer limits must be positive integers")
        if not math.isfinite(self.estimate_safety_margin) or self.estimate_safety_margin < 0:
            raise ValueError("estimate safety margin must be finite and non-negative")

    def resolve_limit(self, descriptor_limit: int | None) -> int:
        """按策略覆盖、模型描述的顺序解析上下文窗口。"""
        limit = self.max_context_tokens or descriptor_limit
        if limit is None:
            raise ValueError(
                "context lifecycle requires policy.max_context_tokens "
                "or model descriptor.context_window_tokens"
            )
        return limit


@dataclass(frozen=True, slots=True)
class ContextRetentionPolicy:
    """声明生产快照、归档和工具结果的显式保留期。"""

    snapshot_ttl_seconds: int
    archive_ttl_seconds: int
    tool_result_ttl_seconds: int

    def __post_init__(self) -> None:
        """保留期必须显式且为正，避免生产数据永久滞留。"""
        values = (
            self.snapshot_ttl_seconds,
            self.archive_ttl_seconds,
            self.tool_result_ttl_seconds,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values
        ):
            raise ValueError("context retention TTL values must be positive integers")


__all__ = ["ContextPolicy", "ContextRetentionPolicy"]
