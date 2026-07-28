"""把通用 ContextMemorySink 适配到 MatterLoop MemoryStore。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from matterloop_models import ModelContextScope

from matterloop_memory.base import MemoryKind, MemoryRecord, MemoryStore


class MemoryContextSink:
    """将获准的上下文事实写为可去重的语义记忆。"""

    def __init__(self, store: MemoryStore, *, namespace: str = "context") -> None:
        if not namespace.strip():
            raise ValueError("context memory namespace must not be empty")
        self._store = store
        self._namespace = namespace

    async def remember(
        self,
        scope: ModelContextScope,
        memories: Sequence[str],
        *,
        source_item_ids: Sequence[str],
    ) -> None:
        """使用内容和作用域哈希生成稳定 ID，避免重复压缩重复写入。"""
        sources = ",".join(source_item_ids)
        for content in memories:
            normalized = content.strip()
            if not normalized:
                continue
            digest = hashlib.sha256(
                f"{scope.tenant_id}\0{scope.run_id}\0{normalized}".encode()
            ).hexdigest()
            await self._store.put(
                MemoryRecord(
                    namespace=self._namespace,
                    kind=MemoryKind.SEMANTIC,
                    content=normalized,
                    record_id=digest,
                    metadata={
                        "tenant_id": scope.tenant_id,
                        "run_id": scope.run_id,
                        "participant": scope.participant,
                        "source_item_ids": sources,
                    },
                )
            )


__all__ = ["MemoryContextSink"]
