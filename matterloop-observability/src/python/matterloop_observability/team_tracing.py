"""在中立 Team 事件和任务调用协议上记录实时 OpenTelemetry Span。"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from matterloop_observability._semantic_conventions import (
    ATTR_AGENT,
    ATTR_ATTEMPT,
    ATTR_EXCEPTION_TYPE,
    ATTR_TEAM_RUN_ID,
    ATTR_TEAM_STATUS,
    ATTR_TEAM_STOP_REASON,
    ATTR_TEAM_TASK_ID,
    ATTR_TEAM_TASK_OUTCOME,
    INSTRUMENTATION_SCOPE,
    TEAM_AGENT_SPAN_NAME,
    TEAM_SPAN_NAME,
)

logger = logging.getLogger(__name__)

_CLOSE_EVENTS = {
    "team.paused",
    "team.blocked",
    "team.completed",
    "team.cancelled",
    "team.timed_out",
    "team.failed",
}

_OPEN_EVENTS = {"team.started", "team.resumed"}
_ACTIVE_STATUSES = {"created", "planning", "running", "waiting_approval"}


@dataclass(slots=True)
class _LiveTeam:
    """保存不依赖任何 asyncio Task ContextVar 的活动 Team Span。"""

    span: Any


class OpenTelemetryTeamTracePublisher:
    """将 Team 生命周期表示为当前调用上下文下的 ``matterloop.team`` Span。

    实现只使用 Team 事件的结构属性，不导入 ``matterloop_agents``，从而保持
    observability 到 agents 的单向依赖。根 Span 仅记录稳定运行状态和关联标识，不记录
    Team request、任务输出、人工反馈或任意 metadata。并发任务由
    ``OpenTelemetryTeamTaskMiddleware`` 建立各自的子 Span。
    """

    def __init__(self, tracer_provider: Any) -> None:
        try:
            self._propagate = importlib.import_module("opentelemetry.propagate")
            self._trace = importlib.import_module("opentelemetry.trace")
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetryTeamTracePublisher 需要 OpenTelemetry API，请安装 "
                "matterloop-observability[otel]"
            ) from exc
        self._tracer = tracer_provider.get_tracer(INSTRUMENTATION_SCOPE)
        self._teams: dict[str, _LiveTeam] = {}

    async def publish(self, event: Any) -> None:
        """消费 Team 生命周期事件，观测故障不会中断团队执行。"""
        try:
            self.handle(event)
        except Exception:
            logger.exception("实时 Team OTel 事件处理失败")

    async def prepare_snapshot(self, snapshot: Any) -> Any:
        """在 Team 快照 CAS 保存前持久化当前根 Span 的有限 W3C 载体。"""
        try:
            return self._prepare_snapshot(snapshot)
        except Exception:
            logger.exception(
                "实时 Team OTel propagation context 持久化准备失败",
                extra={"run_id": getattr(snapshot, "run_id", "")},
            )
            return snapshot

    def handle(self, event: Any) -> None:
        """同步维护一次团队运行的根 Span，不修改当前 Task 的 OTel Context。"""
        snapshot = getattr(event, "snapshot", None)
        run_id = getattr(snapshot, "run_id", None)
        if not isinstance(run_id, str) or not run_id.strip():
            return
        event_type = _event_type(event)
        state = self._teams.get(run_id)
        if state is None and event_type in _OPEN_EVENTS:
            state = self._open_team(snapshot, run_id, getattr(event, "occurred_at", None))
            self._teams[run_id] = state
        if state is not None and event_type in _CLOSE_EVENTS:
            self._close_team(run_id, state, event, failed=event_type == "team.failed")

    def _prepare_snapshot(self, snapshot: Any) -> Any:
        run_id = getattr(snapshot, "run_id", None)
        if not isinstance(run_id, str) or not run_id.strip():
            return snapshot
        status = _status_value(snapshot)
        if _is_terminal_status(snapshot):
            return replace(snapshot, propagation_context={})
        state = self._teams.get(run_id)
        if state is None and status in _ACTIVE_STATUSES:
            state = self._open_team(snapshot, run_id, getattr(snapshot, "updated_at", None))
            self._teams[run_id] = state
        if state is None:
            return snapshot
        carrier: dict[str, str] = {}
        parent_context = self._trace.set_span_in_context(state.span)
        self._propagate.inject(carrier, context=parent_context)
        persisted = _w3c_carrier(carrier)
        if "traceparent" not in persisted:
            raise RuntimeError("OTel propagator did not inject traceparent")
        return replace(snapshot, propagation_context=persisted)

    def _open_team(self, snapshot: Any, run_id: str, occurred_at: Any) -> _LiveTeam:
        options: dict[str, Any] = {}
        if isinstance(occurred_at, datetime):
            options["start_time"] = _nanoseconds(occurred_at)
        parent_context = self._restored_parent_context(snapshot)
        if parent_context is not None:
            options["context"] = parent_context
        span = self._tracer.start_span(TEAM_SPAN_NAME, **options)
        try:
            _set_attributes(span, _team_attributes(snapshot, run_id))
        except Exception:
            logger.exception("实时 Team OTel 初始属性记录失败")
        return _LiveTeam(span=span)

    def _restored_parent_context(self, snapshot: Any) -> Any | None:
        carrier = getattr(snapshot, "propagation_context", None)
        if not isinstance(carrier, Mapping) or not carrier:
            request = getattr(snapshot, "request", None)
            metadata = getattr(request, "metadata", None)
            candidate = (
                metadata.get("propagation_context") if isinstance(metadata, Mapping) else None
            )
            carrier = candidate if isinstance(candidate, Mapping) else None
        persisted = _w3c_carrier(carrier)
        if not persisted:
            return None
        try:
            restored = self._propagate.extract(persisted)
            span_context = self._trace.get_current_span(restored).get_span_context()
            if not span_context.is_valid:
                logger.warning("Team 上游 OTel propagation context 无效，创建新的根 Span")
                return None
            return restored
        except Exception:
            logger.exception("Team 上游 OTel propagation context 恢复失败")
            return None

    def _close_team(self, run_id: str, state: _LiveTeam, event: Any, *, failed: bool) -> None:
        try:
            try:
                _set_attributes(
                    state.span,
                    _team_attributes(getattr(event, "snapshot", None), run_id),
                )
            except Exception:
                logger.exception("实时 Team OTel 结束属性记录失败")
            if failed:
                try:
                    state.span.set_status(
                        self._trace.Status(self._trace.StatusCode.ERROR, "team execution failed")
                    )
                except Exception:
                    logger.exception("实时 Team OTel 失败状态记录失败")
        finally:
            try:
                occurred_at = getattr(event, "occurred_at", None)
                options = (
                    {"end_time": _nanoseconds(occurred_at)}
                    if isinstance(occurred_at, datetime)
                    else {}
                )
                state.span.end(**options)
            except Exception:
                logger.exception("实时 Team OTel Span 结束失败")
            finally:
                self._teams.pop(run_id, None)


class OpenTelemetryTeamTaskMiddleware:
    """将每个 Endpoint 调用记录为 Team 根 Span 下的独立子 Agent Span。

    中间件通过 Team 的中立 ``TeamTaskInvocationMiddleware`` 协议工作。它把当前 Agent
    Span 的 W3C ``traceparent``/``tracestate`` 复制到传给 Endpoint 的上下文副本；
    ``LoopAgentEndpoint`` 会把该载体传给子 Loop。相同进程中的 asyncio 任务自然继承
    ContextVar，远端运行时只需原样转发该 metadata 即可在其 ``OpenTelemetryTracePublisher``
    中恢复真实父节点。
    """

    def __init__(self, tracer_provider: Any) -> None:
        try:
            self._context = importlib.import_module("opentelemetry.context")
            self._propagate = importlib.import_module("opentelemetry.propagate")
            self._trace = importlib.import_module("opentelemetry.trace")
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetryTeamTaskMiddleware 需要 OpenTelemetry API，请安装 "
                "matterloop-observability[otel]"
            ) from exc
        self._tracer = tracer_provider.get_tracer(INSTRUMENTATION_SCOPE)

    async def invoke(self, context: Any, call_next: Any) -> Any:
        """记录一次 Endpoint 调用，并把有限 W3C 载体传给下一层。"""
        span: Any | None = None
        token: Any | None = None
        try:
            options: dict[str, Any] = {}
            parent_context = self._restored_parent_context(context)
            if parent_context is not None:
                options["context"] = parent_context
            span = self._tracer.start_span(TEAM_AGENT_SPAN_NAME, **options)
            token = self._context.attach(self._trace.set_span_in_context(span))
        except Exception:
            logger.exception("实时 Team Agent Span 创建失败，改为直接执行 Endpoint")
            if span is not None:
                self._end_span(span)
            return await call_next(context)

        try:
            try:
                _set_attributes(span, _task_attributes(context))
            except Exception:
                logger.exception("实时 Team Agent 请求属性记录失败")
            enriched_context = self._with_propagation_context(context, span)
            try:
                result = await call_next(enriched_context)
            except asyncio.CancelledError as exc:
                self._set_failure(span, exc, outcome="cancelled")
                raise
            except Exception as exc:
                self._set_failure(span, exc, outcome="exception")
                raise
            outcome = "success" if getattr(result, "success", False) is True else "failed"
            try:
                _set_attributes(span, {ATTR_TEAM_TASK_OUTCOME: outcome})
                if outcome == "failed":
                    span.set_status(
                        self._trace.Status(
                            self._trace.StatusCode.ERROR,
                            "team task returned an unsuccessful result",
                        )
                    )
            except Exception:
                logger.exception("实时 Team Agent 结果属性记录失败")
            return result
        finally:
            try:
                if token is not None:
                    self._context.detach(token)
            except Exception:
                logger.exception("实时 Team Agent OTel 上下文解绑失败")
            finally:
                self._end_span(span)

    def _with_propagation_context(self, context: Any, span: Any) -> Any:
        carrier: dict[str, str] = {}
        try:
            parent_context = self._trace.set_span_in_context(span)
            self._propagate.inject(carrier, context=parent_context)
            carrier = _w3c_carrier(carrier)
            if "traceparent" not in carrier:
                raise RuntimeError("OTel propagator did not inject traceparent")
            return replace(context, propagation_context=carrier)
        except Exception:
            logger.exception("Team Agent W3C 上下文注入失败，继续执行但不传递远端父节点")
            return context

    def _restored_parent_context(self, context: Any) -> Any | None:
        carrier = getattr(context, "propagation_context", None)
        persisted = _w3c_carrier(carrier)
        if not persisted:
            return None
        try:
            restored = self._propagate.extract(persisted)
            span_context = self._trace.get_current_span(restored).get_span_context()
            if not span_context.is_valid:
                logger.warning("Team Agent OTel propagation context 无效，使用当前上下文")
                return None
            return restored
        except Exception:
            logger.exception("Team Agent OTel propagation context 恢复失败")
            return None

    def _set_failure(self, span: Any, exc: BaseException, *, outcome: str) -> None:
        try:
            _set_attributes(
                span,
                {
                    ATTR_TEAM_TASK_OUTCOME: outcome,
                    ATTR_EXCEPTION_TYPE: _qualified_type(exc),
                },
            )
            span.set_status(
                self._trace.Status(self._trace.StatusCode.ERROR, "team task invocation failed")
            )
        except Exception:
            logger.exception("实时 Team Agent 异常状态记录失败")

    @staticmethod
    def _end_span(span: Any | None) -> None:
        if span is None:
            return
        try:
            span.end()
        except Exception:
            logger.exception("实时 Team Agent Span 结束失败")


@dataclass(frozen=True, slots=True)
class OpenTelemetryTeamInstrumentation:
    """把同一 Provider 的 Team 事件与任务中间件作为可自动发现的能力包。"""

    event_publisher: OpenTelemetryTeamTracePublisher
    task_middleware: OpenTelemetryTeamTaskMiddleware

    def __init__(self, tracer_provider: Any) -> None:
        object.__setattr__(
            self,
            "event_publisher",
            OpenTelemetryTeamTracePublisher(tracer_provider),
        )
        object.__setattr__(
            self,
            "task_middleware",
            OpenTelemetryTeamTaskMiddleware(tracer_provider),
        )


def _event_type(event: Any) -> str:
    value = getattr(event, "event_type", "")
    raw = getattr(value, "value", value)
    return raw if isinstance(raw, str) else ""


def _status_value(snapshot: Any) -> str:
    status = getattr(snapshot, "status", None)
    raw = getattr(status, "value", status)
    return raw if isinstance(raw, str) else ""


def _is_terminal_status(snapshot: Any) -> bool:
    status = getattr(snapshot, "status", None)
    terminal = getattr(status, "is_terminal", False)
    return terminal is True


def _w3c_carrier(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        header: item
        for header, item in value.items()
        if header in {"traceparent", "tracestate"}
        and isinstance(item, str)
        and (bool(item) or header == "tracestate")
    }


def _team_attributes(snapshot: Any, run_id: str) -> dict[str, object]:
    attributes: dict[str, object] = {ATTR_TEAM_RUN_ID: run_id}
    status = getattr(snapshot, "status", None)
    status_value = getattr(status, "value", status)
    if isinstance(status_value, str) and status_value:
        attributes[ATTR_TEAM_STATUS] = status_value
    stop_reason = getattr(snapshot, "stop_reason", None)
    reason_value = getattr(stop_reason, "value", stop_reason)
    if isinstance(reason_value, str) and reason_value:
        attributes[ATTR_TEAM_STOP_REASON] = reason_value
    return attributes


def _task_attributes(context: Any) -> dict[str, object]:
    attributes: dict[str, object] = {}
    team_run_id = getattr(context, "team_run_id", None)
    if isinstance(team_run_id, str) and team_run_id:
        attributes[ATTR_TEAM_RUN_ID] = team_run_id
    task = getattr(context, "task", None)
    task_id = getattr(task, "task_id", None)
    if isinstance(task_id, str) and task_id:
        attributes[ATTR_TEAM_TASK_ID] = task_id
    agent_id = getattr(context, "agent_id", None)
    if isinstance(agent_id, str) and agent_id:
        attributes[ATTR_AGENT] = agent_id
    attempt = getattr(context, "attempt", None)
    if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt > 0:
        attributes[ATTR_ATTEMPT] = attempt
    return attributes


def _qualified_type(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def _set_attributes(span: Any, attributes: dict[str, object]) -> None:
    for key, value in attributes.items():
        span.set_attribute(key, value)


def _nanoseconds(moment: datetime) -> int:
    return int(moment.timestamp() * 1_000_000_000)


__all__ = [
    "OpenTelemetryTeamInstrumentation",
    "OpenTelemetryTeamTaskMiddleware",
    "OpenTelemetryTeamTracePublisher",
]
