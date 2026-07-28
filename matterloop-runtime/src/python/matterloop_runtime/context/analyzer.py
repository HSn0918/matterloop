"""按保留策略划分可压缩与受保护 Context 输入。"""

from __future__ import annotations

from dataclasses import dataclass

from matterloop_models import (
    MessageRole,
    ModelCompactionItem,
    ModelInputItem,
    ModelItemRetention,
    ModelMessageItem,
)


@dataclass(frozen=True, slots=True)
class ContextAnalysis:
    """保存一次分析得到的原始顺序和可压缩输入。"""

    original: tuple[ModelInputItem, ...]
    summarizable: tuple[ModelInputItem, ...]

    def rebuild(self, compacted: tuple[ModelInputItem, ...]) -> tuple[ModelInputItem, ...]:
        """在第一条被替换项的位置插入压缩结果，并保留其他项顺序。"""
        removable = {item.item_id for item in self.summarizable}
        if not removable:
            return self.original
        rebuilt: list[ModelInputItem] = []
        inserted = False
        for item in self.original:
            if item.item_id in removable:
                if not inserted:
                    rebuilt.extend(compacted)
                    inserted = True
                continue
            rebuilt.append(item)
        return tuple(rebuilt)


class ContextAnalyzer:
    """保护固定项和最近 N 轮，仅选择足够旧的显式可摘要项。"""

    def analyze(
        self,
        items: tuple[ModelInputItem, ...],
        *,
        recent_turns: int,
    ) -> ContextAnalysis:
        """返回不会包含固定项和最近对话的压缩候选。"""
        user_indexes = [
            index
            for index, item in enumerate(items)
            if isinstance(item, ModelMessageItem) and item.role is MessageRole.USER
        ]
        recent_start = user_indexes[-recent_turns] if len(user_indexes) >= recent_turns else 0
        summarizable = tuple(
            item
            for index, item in enumerate(items)
            if (
                index < recent_start
                or item.metadata.get("historical_payload") is True
                or (isinstance(item, ModelCompactionItem) and item.native)
            )
            and item.retention is ModelItemRetention.SUMMARIZABLE
        )
        return ContextAnalysis(original=items, summarizable=summarizable)


__all__ = ["ContextAnalysis", "ContextAnalyzer"]
