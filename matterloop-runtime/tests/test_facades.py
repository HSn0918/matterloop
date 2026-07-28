"""同步和异步运行门面的行为测试。"""

from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest
from matterloop_core import (
    HumanAction,
    HumanResponse,
    LoopRequest,
    LoopResult,
    LoopStatus,
    ResumeMode,
)
from matterloop_runtime import AsyncRuntime, LocalRuntime


def _result(run_id: str) -> LoopResult:
    return LoopResult(
        run_id=run_id,
        status=LoopStatus.COMPLETED,
        output="done",
        cycles=1,
        total_attempts=1,
        completed_steps=1,
        records=(),
        stop_reason=None,
    )


class FakeLoopEngine:
    """记录门面委托参数的测试内核。"""

    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.resume_mode: ResumeMode | None = None
        self.human_responses: list[tuple[str, HumanResponse]] = []

    async def run(self, request: LoopRequest, *, run_id: str | None = None) -> LoopResult:
        del request
        return _result(run_id or "generated")

    async def resume(
        self,
        run_id: str,
        *,
        mode: ResumeMode = ResumeMode.CONTINUE,
    ) -> LoopResult:
        self.resume_mode = mode
        return _result(run_id)

    async def submit_human_response(
        self,
        run_id: str,
        response: HumanResponse,
    ) -> LoopResult:
        self.human_responses.append((run_id, response))
        return _result(run_id)

    def cancel(self, run_id: str) -> bool:
        self.cancelled.append(run_id)
        return True

    def create_run_id(self) -> str:
        return "new-id"


class Resource:
    """记录异步关闭的测试资源。"""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def test_async_runtime_delegates_all_operations() -> None:
    engine = FakeLoopEngine()
    runtime = AsyncRuntime(engine)

    assert runtime.create_run_id() == "new-id"
    assert (await runtime.run(LoopRequest("goal"), run_id="run-1")).run_id == "run-1"
    assert (await runtime.resume("run-1", mode=ResumeMode.REPLAN)).run_id == "run-1"
    assert engine.resume_mode is ResumeMode.REPLAN
    response = HumanResponse("interaction", HumanAction.APPROVE)
    assert (await runtime.submit_human_response("run-1", response)).run_id == "run-1"
    assert engine.human_responses == [("run-1", response)]
    assert await runtime.cancel("run-1")
    assert engine.cancelled == ["run-1"]


async def test_async_runtime_close_waits_for_inflight_run_before_closing_resources() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingLoopEngine(FakeLoopEngine):
        async def run(self, request: LoopRequest, *, run_id: str | None = None) -> LoopResult:
            del request
            started.set()
            await release.wait()
            return _result(run_id or "generated")

    engine = BlockingLoopEngine()
    resource = Resource()
    runtime = AsyncRuntime(engine, resources=[resource])
    running = asyncio.create_task(runtime.run(LoopRequest("goal"), run_id="in-flight"))
    await started.wait()
    closing = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0)

    assert not closing.done()
    assert not resource.closed

    release.set()
    assert (await running).run_id == "in-flight"
    await closing
    assert resource.closed


async def test_async_runtime_close_waits_for_inflight_cancel_before_closing_resources() -> None:
    """异步 cancel 也必须纳入运行时 drain，避免终态事件与 exporter 关闭竞态。"""
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingCancelEngine(FakeLoopEngine):
        async def cancel(self, run_id: str) -> bool:
            self.cancelled.append(run_id)
            started.set()
            await release.wait()
            return True

    engine = BlockingCancelEngine()
    resource = Resource()
    runtime = AsyncRuntime(engine, resources=[resource])
    cancelling = asyncio.create_task(runtime.cancel("in-flight"))
    await started.wait()
    closing = asyncio.create_task(runtime.aclose())
    for _ in range(3):
        await asyncio.sleep(0)

    assert not closing.done()
    assert not resource.closed

    release.set()
    assert await cancelling
    await closing
    assert resource.closed


async def test_async_runtime_cancelled_close_waiter_does_not_abandon_resource_close() -> None:
    """取消首个 aclose 调用方只能取消等待，不能伪造关闭完成或遗留资源。"""
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    class BlockingResource(Resource):
        async def aclose(self) -> None:
            close_started.set()
            await release_close.wait()
            self.closed = True

    resource = BlockingResource()
    runtime = AsyncRuntime(FakeLoopEngine(), resources=[resource])
    first_waiter = asyncio.create_task(runtime.aclose())
    await close_started.wait()
    first_waiter.cancel()
    with suppress(asyncio.CancelledError):
        await first_waiter

    second_waiter = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0)
    assert not second_waiter.done()
    assert not resource.closed

    release_close.set()
    await second_waiter
    assert resource.closed


async def test_async_runtime_context_exit_preserves_block_error_when_close_fails() -> None:
    """资源关闭失败不得掩盖 async with 块内原始异常。"""

    class FailingResource:
        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    runtime = AsyncRuntime(FakeLoopEngine(), resources=[FailingResource()])

    with pytest.raises(ValueError, match="block failed"):
        async with runtime:
            raise ValueError("block failed")


def test_local_runtime_uses_background_event_loop() -> None:
    engine = FakeLoopEngine()
    resource = Resource()

    with LocalRuntime(AsyncRuntime(engine, resources=[resource])) as runtime:
        assert runtime.create_run_id() == "new-id"
        assert runtime.run(LoopRequest("goal"), run_id="run-2").output == "done"
        assert runtime.resume("run-2").run_id == "run-2"
        response = HumanResponse("interaction", HumanAction.APPROVE)
        assert runtime.submit_human_response("run-2", response).run_id == "run-2"
        assert runtime.cancel("run-2")
    assert resource.closed
