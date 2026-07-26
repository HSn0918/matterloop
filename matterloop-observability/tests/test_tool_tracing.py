"""实时工具 Span、载荷采集与并发上下文测试。"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import pytest
from matterloop_core import LoopContext, LoopEvent, LoopEventType, LoopRequest, Plan, PlanStep
from matterloop_observability import (
    OpenTelemetryToolMiddleware,
    OpenTelemetryTracePublisher,
)
from matterloop_tools import (
    PermissionDecision,
    ToolContext,
    ToolNotFoundError,
    ToolPermissionDeniedError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


def _provider_and_exporter() -> tuple[Any, Any]:
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
    in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    provider = sdk_trace.TracerProvider()
    exporter = in_memory.InMemorySpanExporter()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    return provider, exporter


class ResultTool:
    """返回预设内容和元数据的测试工具。"""

    def __init__(
        self,
        name: str = "result",
        *,
        content: str = "result-secret",
        is_error: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._spec = ToolSpec(name, "测试工具", {"type": "object"})
        self._content = content
        self._is_error = is_error
        self._metadata = metadata or {}

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        del arguments, context
        return ToolResult(self._content, is_error=self._is_error, metadata=self._metadata)


class RaisingTool:
    spec = ToolSpec("raising", "抛出异常", {"type": "object"})

    async def invoke(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        del arguments, context
        raise RuntimeError("Authorization: Bearer exception-secret")


class CountingTool:
    spec = ToolSpec("counting", "记录执行次数", {"type": "object"})

    def __init__(self) -> None:
        self.invocations = 0

    async def invoke(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        del arguments, context
        self.invocations += 1
        return ToolResult("ok")


class DenyAuthorizer:
    async def authorize(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> PermissionDecision:
        del tool_name, arguments, context
        return PermissionDecision.DENY


async def test_tool_span_inherits_parent_and_excludes_raw_payloads_by_default() -> None:
    provider, exporter = _provider_and_exporter()
    registry = ToolRegistry(
        [ResultTool()],
        middleware=OpenTelemetryToolMiddleware(provider),
    )

    with provider.get_tracer("test").start_as_current_span("executor-parent") as parent:
        result = await registry.invoke(
            "result",
            {"password": "input-secret", "value": "visible"},
            context=ToolContext(
                "run-tool",
                "step-tool",
                metadata={"tool_call_id": "model-call-1", "executor": "worker"},
            ),
        )

    assert result.content == "result-secret"
    span = next(span for span in exporter.get_finished_spans() if span.name == "matterloop.tool")
    assert span.parent is not None
    assert span.parent.span_id == parent.get_span_context().span_id
    assert span.attributes["matterloop.run_id"] == "run-tool"
    assert span.attributes["matterloop.step_id"] == "step-tool"
    assert span.attributes["matterloop.executor"] == "worker"
    assert span.attributes["matterloop.tool.name"] == "result"
    assert span.attributes["matterloop.tool_call_id"] == "model-call-1"
    assert span.attributes["matterloop.tool.kind"] == "tool"
    assert span.attributes["matterloop.tool.outcome"] == "success"
    assert "matterloop.tool.arguments" not in span.attributes
    assert "matterloop.tool.result" not in span.attributes
    assert "matterloop.tool.arguments_truncated" not in span.attributes
    assert "matterloop.tool.result_truncated" not in span.attributes
    arguments = '{"password":"input-secret","value":"visible"}'
    assert span.attributes["matterloop.tool.arguments_bytes"] == len(arguments.encode())
    assert (
        span.attributes["matterloop.tool.arguments_sha256"]
        == hashlib.sha256(arguments.encode()).hexdigest()
    )
    assert span.attributes["matterloop.tool.result_bytes"] == len(b"result-secret")
    assert (
        span.attributes["matterloop.tool.result_sha256"]
        == hashlib.sha256(b"result-secret").hexdigest()
    )


async def test_non_json_arguments_keep_correlation_and_payload_attributes() -> None:
    """bytes/datetime 等参数必须稳定序列化，且不能抹掉工具关联标识。"""
    provider, exporter = _provider_and_exporter()
    middleware = OpenTelemetryToolMiddleware(provider)
    context = ToolContext(
        "run-non-json",
        "step-non-json",
        metadata={"tool_call_id": "call-non-json", "executor": "worker"},
    )

    result = await middleware.invoke(
        "result",
        {
            "blob": b"\x00\xff",
            "occurred_at": datetime(2026, 7, 28, tzinfo=timezone.utc),
        },
        context,
        lambda: ResultTool().invoke({}, context),
    )

    assert result.content == "result-secret"
    span = next(span for span in exporter.get_finished_spans() if span.name == "matterloop.tool")
    assert span.attributes["matterloop.run_id"] == "run-non-json"
    assert span.attributes["matterloop.step_id"] == "step-non-json"
    assert span.attributes["matterloop.tool.name"] == "result"
    assert span.attributes["matterloop.tool_call_id"] == "call-non-json"
    assert span.attributes["matterloop.tool.outcome"] == "success"
    assert span.attributes["matterloop.tool.arguments_bytes"] > 0
    assert len(span.attributes["matterloop.tool.arguments_sha256"]) == 64


async def test_tool_payload_capture_keeps_raw_values_and_truncates() -> None:
    provider, exporter = _provider_and_exporter()
    middleware = OpenTelemetryToolMiddleware(
        provider,
        capture_tool_payloads=True,
        capture_max_body_bytes=48,
    )
    registry = ToolRegistry(
        [
            ResultTool(
                "generic",
                content='{"password":"result-secret","text":"结果内容很长结果内容很长"}',
            ),
            ResultTool(
                "skill_reference",
                content="untrusted skill body secret",
                metadata={
                    "skill_name": "demo",
                    "skill_version": "1.2",
                    "skill_operation": "get",
                    "sha256": "a" * 64,
                    "trust": "untrusted_reference",
                },
            ),
            ResultTool("plain", content="Authorization: Bearer plain-result-secret"),
        ],
        middleware=middleware,
    )

    await registry.invoke(
        "generic",
        {"password": "argument-secret", "text": "很多很多文本"},
        context=ToolContext("run-generic"),
    )
    await registry.invoke(
        "skill_reference",
        {"operation": "get", "name": "demo"},
        context=ToolContext("run-skill"),
    )
    await registry.invoke("plain", {}, context=ToolContext("run-plain"))

    spans = {
        span.attributes["matterloop.run_id"]: span
        for span in exporter.get_finished_spans()
        if span.name == "matterloop.tool"
    }
    generic = spans["run-generic"]
    assert "argument-secret" in generic.attributes["matterloop.tool.arguments"]
    assert "result-secret" in generic.attributes["matterloop.tool.result"]
    assert "[REDACTED]" not in generic.attributes["matterloop.tool.arguments"]
    assert "[REDACTED]" not in generic.attributes["matterloop.tool.result"]
    assert len(generic.attributes["matterloop.tool.arguments"].encode("utf-8")) <= 48
    assert len(generic.attributes["matterloop.tool.result"].encode("utf-8")) <= 48
    assert generic.attributes["matterloop.tool.arguments_truncated"] is True
    assert generic.attributes["matterloop.tool.result_truncated"] is True

    skill = spans["run-skill"]
    assert skill.attributes["matterloop.tool.kind"] == "skill"
    assert skill.attributes["matterloop.skill.name"] == "demo"
    assert skill.attributes["matterloop.skill.version"] == "1.2"
    assert skill.attributes["matterloop.skill.operation"] == "get"
    assert skill.attributes["matterloop.skill.sha256"] == "a" * 64
    assert skill.attributes["matterloop.tool.result"] == "untrusted skill body secret"
    assert skill.attributes["matterloop.tool.result_truncated"] is False

    plain = spans["run-plain"]
    assert plain.attributes["matterloop.tool.result"] == "Authorization: Bearer plain-result-secret"
    assert plain.attributes["matterloop.tool.result_truncated"] is False


async def test_default_body_limit_is_4096_utf8_bytes_per_payload() -> None:
    provider, exporter = _provider_and_exporter()
    argument_value = "x" * 5000
    full_arguments = f'{{"text":"{argument_value}"}}'
    full_result = "界" * 1366
    registry = ToolRegistry(
        [ResultTool(content=full_result)],
        middleware=OpenTelemetryToolMiddleware(provider, capture_tool_payloads=True),
    )

    await registry.invoke(
        "result",
        {"text": argument_value},
        context=ToolContext("run-default-limit"),
    )

    span = next(span for span in exporter.get_finished_spans() if span.name == "matterloop.tool")
    arguments = span.attributes["matterloop.tool.arguments"]
    result = span.attributes["matterloop.tool.result"]
    assert len(arguments.encode("utf-8")) == 4096
    assert len(result.encode("utf-8")) <= 4096
    assert span.attributes["matterloop.tool.arguments_truncated"] is True
    assert span.attributes["matterloop.tool.result_truncated"] is True
    assert span.attributes["matterloop.tool.arguments_bytes"] == len(full_arguments.encode())
    assert span.attributes["matterloop.tool.result_bytes"] == len(full_result.encode())
    assert (
        span.attributes["matterloop.tool.arguments_sha256"]
        == hashlib.sha256(full_arguments.encode()).hexdigest()
    )
    assert (
        span.attributes["matterloop.tool.result_sha256"]
        == hashlib.sha256(full_result.encode()).hexdigest()
    )


async def test_tool_errors_record_stable_outcomes_without_exception_messages() -> None:
    provider, exporter = _provider_and_exporter()
    middleware = OpenTelemetryToolMiddleware(provider)
    raising = ToolRegistry([RaisingTool()], middleware=middleware)
    denied = ToolRegistry(
        [ResultTool("denied")],
        authorizer=DenyAuthorizer(),
        middleware=middleware,
    )
    missing = ToolRegistry(middleware=middleware)

    with pytest.raises(RuntimeError, match="exception-secret"):
        await raising.invoke("raising", {}, context=ToolContext("run-exception"))
    with pytest.raises(ToolPermissionDeniedError):
        await denied.invoke("denied", {}, context=ToolContext("run-denied"))
    with pytest.raises(ToolNotFoundError):
        await missing.invoke("missing", {}, context=ToolContext("run-missing"))

    spans = {
        span.attributes["matterloop.run_id"]: span
        for span in exporter.get_finished_spans()
        if span.name == "matterloop.tool"
    }
    assert spans["run-exception"].attributes["matterloop.tool.outcome"] == "exception"
    assert spans["run-denied"].attributes["matterloop.tool.outcome"] == "denied"
    assert spans["run-missing"].attributes["matterloop.tool.outcome"] == "not_found"
    assert all(span.status.status_code.name == "ERROR" for span in spans.values())
    assert "exception-secret" not in repr(
        [(span.attributes, span.status.description, tuple(span.events)) for span in spans.values()]
    )


async def test_mcp_metadata_maps_to_allowlisted_span_attributes() -> None:
    provider, exporter = _provider_and_exporter()
    registry = ToolRegistry(
        [
            ResultTool(
                "mcp__search",
                metadata={
                    "mcp_server": "docs",
                    "mcp_tool": "search",
                    "content_blocks": 3,
                    "truncated": True,
                    "private": "must-not-be-exported",
                },
            )
        ],
        middleware=OpenTelemetryToolMiddleware(provider),
    )

    await registry.invoke("mcp__search", {}, context=ToolContext("run-mcp"))

    span = next(span for span in exporter.get_finished_spans() if span.name == "matterloop.tool")
    assert span.attributes["matterloop.tool.kind"] == "mcp"
    assert span.attributes["matterloop.mcp.server"] == "docs"
    assert span.attributes["matterloop.mcp.tool"] == "search"
    assert span.attributes["matterloop.mcp.content_blocks"] == 3
    assert span.attributes["matterloop.mcp.truncated"] is True
    generated_call_id = span.attributes["matterloop.tool_call_id"]
    assert isinstance(generated_call_id, str) and len(generated_call_id) == 32
    int(generated_call_id, 16)
    assert "must-not-be-exported" not in repr(span.attributes)


async def test_regular_tool_metadata_never_populates_mcp_attributes() -> None:
    provider, exporter = _provider_and_exporter()
    registry = ToolRegistry(
        [
            ResultTool(
                "regular",
                metadata={
                    "truncated": True,
                    "content_blocks": 5,
                },
            )
        ],
        middleware=OpenTelemetryToolMiddleware(provider),
    )

    await registry.invoke("regular", {}, context=ToolContext("run-regular"))

    span = next(span for span in exporter.get_finished_spans() if span.name == "matterloop.tool")
    assert span.attributes["matterloop.tool.kind"] == "tool"
    assert "matterloop.mcp.truncated" not in span.attributes
    assert "matterloop.mcp.content_blocks" not in span.attributes


async def test_otel_span_creation_failure_does_not_change_tool_invocation() -> None:
    class BrokenTracer:
        def start_span(self, name: str) -> object:
            del name
            raise RuntimeError("otel unavailable")

    class BrokenProvider:
        def get_tracer(self, name: str) -> BrokenTracer:
            del name
            return BrokenTracer()

    tool = CountingTool()
    registry = ToolRegistry(
        [tool],
        middleware=OpenTelemetryToolMiddleware(BrokenProvider()),
    )

    result = await registry.invoke("counting", {}, context=ToolContext("run-broken-otel"))

    assert result.content == "ok"
    assert tool.invocations == 1


@pytest.mark.parametrize(
    "limit",
    [0, -1, True, 1.5],
)
def test_tool_middleware_rejects_invalid_body_limits(limit: object) -> None:
    provider, _ = _provider_and_exporter()
    with pytest.raises(ValueError):
        OpenTelemetryToolMiddleware(
            provider,
            capture_max_body_bytes=limit,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("capture_tool_payloads", [0, 1, "true", None])
def test_tool_middleware_rejects_non_boolean_payload_capture(
    capture_tool_payloads: object,
) -> None:
    provider, _ = _provider_and_exporter()
    with pytest.raises(ValueError, match="capture_tool_payloads"):
        OpenTelemetryToolMiddleware(
            provider,
            capture_tool_payloads=capture_tool_payloads,  # type: ignore[arg-type]
        )


class InterleavingTool:
    spec = ToolSpec("interleave", "强制两个调用交错", {"type": "object"})

    def __init__(self) -> None:
        self._started = 0
        self._release = asyncio.Event()

    async def invoke(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        del arguments
        self._started += 1
        if self._started == 2:
            self._release.set()
        await self._release.wait()
        return ToolResult(context.run_id)


def _loop_context(run_id: str) -> LoopContext:
    context = LoopContext(LoopRequest(f"run {run_id}"), run_id=run_id)
    context.current_plan = Plan((PlanStep("invoke", executor="worker", step_id=f"step-{run_id}"),))
    return context


async def test_parallel_runs_keep_root_executor_and_tool_context_isolated() -> None:
    provider, exporter = _provider_and_exporter()
    publisher = OpenTelemetryTracePublisher(provider)
    registry = ToolRegistry(
        [InterleavingTool()],
        middleware=OpenTelemetryToolMiddleware(provider),
    )

    async def run(run_id: str) -> None:
        context = _loop_context(run_id)
        await publisher.publish(LoopEvent(LoopEventType.LOOP_STARTED, context))
        await publisher.publish(LoopEvent(LoopEventType.EXECUTION_DISPATCHED, context))
        await registry.invoke(
            "interleave",
            {},
            context=ToolContext(
                run_id,
                f"step-{run_id}",
                metadata={"tool_call_id": f"call-{run_id}", "executor": "worker"},
            ),
        )
        await publisher.publish(LoopEvent(LoopEventType.EXECUTION_COMPLETED, context))
        await publisher.publish(LoopEvent(LoopEventType.LOOP_COMPLETED, context))

    await asyncio.gather(
        asyncio.create_task(run("a")),
        asyncio.create_task(run("b")),
    )

    finished = exporter.get_finished_spans()
    roots = {
        span.attributes["matterloop.run_id"]: span
        for span in finished
        if span.name == "matterloop.run"
    }
    executors = {
        span.attributes["matterloop.run_id"]: span
        for span in finished
        if span.name == "matterloop.executor"
    }
    tools = {
        span.attributes["matterloop.run_id"]: span
        for span in finished
        if span.name == "matterloop.tool"
    }
    assert set(roots) == set(executors) == set(tools) == {"a", "b"}
    assert roots["a"].get_span_context().trace_id != roots["b"].get_span_context().trace_id
    for run_id in ("a", "b"):
        root_context = roots[run_id].get_span_context()
        executor_context = executors[run_id].get_span_context()
        assert executors[run_id].parent is not None
        assert executors[run_id].parent.span_id == root_context.span_id
        assert tools[run_id].parent is not None
        assert tools[run_id].parent.span_id == executor_context.span_id
        assert tools[run_id].get_span_context().trace_id == root_context.trace_id
