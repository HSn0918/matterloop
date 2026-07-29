"""Team/子 Agent 的实时 OpenTelemetry 拓扑与 W3C 传播测试。"""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

import matterloop_observability.team_tracing as team_tracing_module
import pytest
from matterloop_agents.collaboration import (
    AgentDirectory,
    AgentSpec,
    AgentTaskContext,
    AlwaysApproveTeamGate,
    ConcatenateResultAggregator,
    InMemoryTeamRepository,
    LeastBusyScheduler,
    LocalTeamEventPublisher,
    LoopAgentEndpoint,
    ResultSuccessVerifier,
    StaticTeamPlanner,
    TaskSpec,
    TeamEvent,
    TeamEventType,
    TeamLimits,
    TeamOrchestrator,
    TeamOrchestratorComponents,
    TeamRequest,
    TeamSnapshot,
    TeamStatus,
)
from matterloop_core import (
    LoopContext,
    LoopEvent,
    LoopEventType,
    LoopResult,
    LoopStatus,
    StopReason,
)
from matterloop_observability import (
    OpenTelemetryTeamInstrumentation,
    OpenTelemetryTeamTaskMiddleware,
    OpenTelemetryTeamTracePublisher,
    OpenTelemetryTracePublisher,
)


def _provider_and_exporter() -> tuple[Any, Any]:
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
    in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    provider = sdk_trace.TracerProvider()
    exporter = in_memory.InMemorySpanExporter()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    return provider, exporter


class _TracedChildRuntime:
    """用真实 Loop trace publisher 模拟接收 Team 请求的独立子运行时。"""

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._events = OpenTelemetryTracePublisher(provider)
        self.team_instrumentation = OpenTelemetryTeamInstrumentation(provider)
        self.requests: list[Any] = []

    async def run(self, request: Any, *, run_id: str | None = None) -> LoopResult:
        assert run_id is not None
        self.requests.append(request)
        context = LoopContext(request, run_id=run_id, status=LoopStatus.EXECUTING)
        otel_context = importlib.import_module("opentelemetry.context")
        # 模拟远端 worker：丢弃本地 ContextVar，只允许 request metadata 的 W3C 载体建立父节点。
        token = otel_context.attach(otel_context.Context())
        try:
            await self._events.publish(LoopEvent(LoopEventType.LOOP_STARTED, context))
            with self._provider.get_tracer("child-runtime").start_as_current_span(
                "child.operation"
            ):
                pass
            context.status = LoopStatus.COMPLETED
            context.stop_reason = StopReason.COMPLETED
            await self._events.publish(LoopEvent(LoopEventType.LOOP_COMPLETED, context))
        finally:
            otel_context.detach(token)
        return LoopResult(
            run_id=run_id,
            status=LoopStatus.COMPLETED,
            output="child completed",
            cycles=1,
            total_attempts=1,
            completed_steps=1,
            records=(),
            stop_reason=StopReason.COMPLETED,
        )


async def test_team_and_child_loop_spans_form_one_trace_with_w3c_metadata() -> None:
    """并发子 Agent、子 Loop 与 Team 根 Span 必须保持正确父子关系。"""
    provider, exporter = _provider_and_exporter()
    events = LocalTeamEventPublisher()
    child_runtime = _TracedChildRuntime(provider)
    directory = AgentDirectory()
    directory.register(
        LoopAgentEndpoint(
            AgentSpec("worker", frozenset({"analysis"}), max_concurrency=2),
            child_runtime,
        )
    )
    orchestrator = TeamOrchestrator(
        TeamOrchestratorComponents(
            planner=StaticTeamPlanner(
                (
                    TaskSpec("facts", "收集事实", "analysis"),
                    TaskSpec("risks", "识别风险", "analysis"),
                )
            ),
            agents=directory,
            selection_policy=LeastBusyScheduler(),
            verifier=ResultSuccessVerifier(),
            approval_gate=AlwaysApproveTeamGate(),
            repository=InMemoryTeamRepository(),
            events=events,
            aggregator=ConcatenateResultAggregator(),
        )
    )

    result = await orchestrator.run(
        TeamRequest(
            "完成敏感团队目标",
            limits=TeamLimits(timeout_seconds=5),
            metadata={"propagation_context": {"traceparent": "untrusted"}},
        ),
        run_id="team-traced",
    )

    assert result.status is TeamStatus.COMPLETED
    spans = exporter.get_finished_spans()
    team = next(span for span in spans if span.name == "matterloop.team")
    agents = [span for span in spans if span.name == "matterloop.team.agent"]
    runs = [span for span in spans if span.name == "matterloop.run"]
    operations = [span for span in spans if span.name == "child.operation"]

    assert team.attributes["matterloop.team.run_id"] == "team-traced"
    assert team.attributes["matterloop.team.status"] == "completed"
    assert "matterloop.goal" not in team.attributes
    assert len(agents) == len(runs) == len(operations) == 2
    for agent in agents:
        assert agent.parent is not None
        assert agent.parent.span_id == team.get_span_context().span_id
        assert agent.attributes["matterloop.team.run_id"] == "team-traced"
        assert agent.attributes["matterloop.team.task.outcome"] == "success"
    for run in runs:
        assert run.parent is not None
        assert run.parent.span_id in {agent.get_span_context().span_id for agent in agents}
    for operation in operations:
        assert operation.parent is not None
        assert operation.parent.span_id in {run.get_span_context().span_id for run in runs}
    for request in child_runtime.requests:
        carrier = request.metadata["propagation_context"]
        assert set(carrier) <= {"traceparent", "tracestate"}
        assert carrier["traceparent"].startswith("00-")
        assert carrier["traceparent"] != "untrusted"

    tracer = provider.get_tracer("after-team")
    after = tracer.start_span("after.team")
    after.end()
    finished_after = next(
        span for span in exporter.get_finished_spans() if span.name == "after.team"
    )
    assert finished_after.parent is None


async def test_team_pause_resume_persists_parent_across_publisher_instances() -> None:
    """暂停保存的 W3C carrier 必须让新进程中的 Team segment 成为真实子 Span。"""
    provider, exporter = _provider_and_exporter()
    first = OpenTelemetryTeamTracePublisher(provider)
    repository = InMemoryTeamRepository()
    snapshot = TeamSnapshot(
        TeamRequest("跨进程恢复"),
        (),
        run_id="team-resume",
        status=TeamStatus.PLANNING,
    )
    await repository.create(snapshot)
    first.handle(TeamEvent(TeamEventType.TEAM_STARTED, snapshot))
    snapshot = await first.prepare_snapshot(snapshot)
    snapshot = await repository.save(snapshot, 0)
    paused = replace(snapshot, status=TeamStatus.PAUSED)
    paused = await first.prepare_snapshot(paused)
    paused = await repository.save(paused, snapshot.version)
    first.handle(TeamEvent(TeamEventType.TEAM_PAUSED, paused))

    restored = await repository.require("team-resume")
    assert restored.propagation_context["traceparent"].startswith("00-")

    resumed_publisher = OpenTelemetryTeamTracePublisher(provider)
    resumed = replace(restored, status=TeamStatus.RUNNING)
    resumed = await resumed_publisher.prepare_snapshot(resumed)
    resumed_publisher.handle(TeamEvent(TeamEventType.TEAM_RESUMED, resumed))
    completed = replace(resumed, status=TeamStatus.COMPLETED)
    completed = await resumed_publisher.prepare_snapshot(completed)
    resumed_publisher.handle(TeamEvent(TeamEventType.TEAM_COMPLETED, completed))

    teams = [span for span in exporter.get_finished_spans() if span.name == "matterloop.team"]
    assert len(teams) == 2
    first_segment, resumed_segment = teams
    assert resumed_segment.parent is not None
    assert resumed_segment.parent.span_id == first_segment.get_span_context().span_id
    assert resumed_segment.get_span_context().trace_id == first_segment.get_span_context().trace_id
    assert completed.propagation_context == {}


async def test_team_task_attribute_failures_do_not_change_endpoint_result(monkeypatch) -> None:
    """请求或结果属性写入失败时仍必须执行 Endpoint 并返回原始结果。"""
    provider, exporter = _provider_and_exporter()
    middleware = OpenTelemetryTeamTaskMiddleware(provider)
    context = AgentTaskContext(
        team_run_id="team-attributes",
        request=TeamRequest("属性隔离"),
        task=TaskSpec("task", "执行任务", "analysis"),
        agent_id="worker",
        attempt=1,
    )
    expected = object()
    calls = 0

    def fail_attributes(span: Any, attributes: dict[str, object]) -> None:
        del span, attributes
        raise RuntimeError("attribute backend failed")

    async def call_next(received: AgentTaskContext) -> object:
        nonlocal calls
        calls += 1
        assert received.team_run_id == context.team_run_id
        return expected

    monkeypatch.setattr(team_tracing_module, "_set_attributes", fail_attributes)

    result = await middleware.invoke(context, call_next)

    assert result is expected
    assert calls == 1
    assert [span.name for span in exporter.get_finished_spans()] == ["matterloop.team.agent"]
