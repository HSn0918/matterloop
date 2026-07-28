"""内存记忆与检查点存储测试。"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from matterloop_core import CheckpointConflictError, LoopContext, LoopRequest
from matterloop_memory import (
    InMemoryCheckpointStore,
    InMemoryMemoryStore,
    MemoryContextSink,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    NullMemoryStore,
)
from matterloop_models import ModelContextScope


def test_memory_store_filters_and_scores_records() -> None:
    """内存实现应按命名空间和词项相关度返回结果。"""

    async def scenario() -> None:
        store = InMemoryMemoryStore()
        await store.put(MemoryRecord("project", MemoryKind.SEMANTIC, "python loop agent"))
        await store.put(MemoryRecord("other", MemoryKind.SEMANTIC, "python loop agent"))

        matches = await store.search(MemoryQuery("project", text="python agent"))

        assert len(matches) == 1
        assert matches[0].score == 1
        assert await store.clear("project") == 1

    asyncio.run(scenario())


def test_context_memory_sink_deduplicates_admitted_facts() -> None:
    """重复压缩产生的同一事实必须覆盖同一稳定语义记忆。"""

    async def scenario() -> None:
        store = InMemoryMemoryStore()
        sink = MemoryContextSink(store, namespace="context-facts")
        scope = ModelContextScope("run-memory", "planner", tenant_id="tenant-a")

        await sink.remember(scope, ("stable fact",), source_item_ids=("a", "b"))
        await sink.remember(scope, ("stable fact",), source_item_ids=("a", "b"))
        matches = await store.search(MemoryQuery("context-facts"))

        assert len(matches) == 1
        assert matches[0].record.content == "stable fact"
        assert matches[0].record.metadata["tenant_id"] == "tenant-a"

    asyncio.run(scenario())


def test_checkpoint_store_returns_isolated_snapshots() -> None:
    """调用方修改加载结果时不能污染已保存检查点。"""

    async def scenario() -> None:
        store = InMemoryCheckpointStore()
        context = LoopContext(LoopRequest("保存检查点"))
        context.feedback = "saved"
        await store.save(context)

        loaded = await store.load(context.run_id)
        assert loaded is not None
        loaded.feedback = "changed"

        reloaded = await store.load(context.run_id)
        assert reloaded is not None
        assert reloaded.feedback == "saved"
        assert await store.list_run_ids() == (context.run_id,)
        assert await store.delete(context.run_id)
        assert await store.load(context.run_id) is None

    asyncio.run(scenario())


def test_checkpoint_store_uses_atomic_revision_comparison() -> None:
    """两个调用方基于同一 revision 写入时只能有一个成功。"""

    async def scenario() -> None:
        store = InMemoryCheckpointStore()
        initial = LoopContext(LoopRequest("CAS"), run_id="cas-run")
        revision = await store.save(initial, expected_revision=0)
        assert revision == 1

        first = await store.load(initial.run_id)
        stale = await store.load(initial.run_id)
        assert first is not None
        assert stale is not None
        first.feedback = "winner"
        committed = await store.save(first, expected_revision=first.revision)
        assert committed == 2

        stale.feedback = "loser"
        with pytest.raises(CheckpointConflictError):
            await store.save(stale, expected_revision=stale.revision)

        loaded = await store.load(initial.run_id)
        assert loaded is not None
        assert loaded.feedback == "winner"
        assert loaded.revision == 2

    asyncio.run(scenario())


def test_memory_store_hides_expired_records() -> None:
    """过期长期记忆不能被读取或检索。"""

    async def scenario() -> None:
        store = InMemoryMemoryStore()
        record = MemoryRecord(
            "project",
            MemoryKind.EPISODIC,
            "已经过期",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        await store.put(record)

        assert await store.get(record.record_id) is None
        assert await store.search(MemoryQuery("project")) == ()

    asyncio.run(scenario())


def test_null_memory_store_is_a_complete_noop() -> None:
    """禁用记忆时所有协议操作均应提供确定性空结果。"""

    async def scenario() -> None:
        store = NullMemoryStore()
        record = MemoryRecord("project", MemoryKind.PROCEDURAL, "无需持久化")

        await store.put(record)
        assert await store.get(record.record_id) is None
        assert await store.search(MemoryQuery("project")) == ()
        assert not await store.delete(record.record_id)
        assert await store.clear("project") == 0

    asyncio.run(scenario())
