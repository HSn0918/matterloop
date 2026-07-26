"""在中立工具中间件协议上记录实时 OpenTelemetry Span。"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from uuid import uuid4

from matterloop_observability._semantic_conventions import (
    ATTR_EXCEPTION_TYPE,
    ATTR_EXECUTOR,
    ATTR_MCP_CONTENT_BLOCKS,
    ATTR_MCP_SERVER,
    ATTR_MCP_TOOL,
    ATTR_MCP_TRUNCATED,
    ATTR_RUN_ID,
    ATTR_SKILL_NAME,
    ATTR_SKILL_OPERATION,
    ATTR_SKILL_SHA256,
    ATTR_SKILL_TRUST,
    ATTR_SKILL_VERSION,
    ATTR_STEP_ID,
    ATTR_TOOL_ARGUMENTS,
    ATTR_TOOL_ARGUMENTS_BYTES,
    ATTR_TOOL_ARGUMENTS_SHA256,
    ATTR_TOOL_ARGUMENTS_TRUNCATED,
    ATTR_TOOL_CALL_ID,
    ATTR_TOOL_KIND,
    ATTR_TOOL_NAME,
    ATTR_TOOL_OUTCOME,
    ATTR_TOOL_RESULT,
    ATTR_TOOL_RESULT_BYTES,
    ATTR_TOOL_RESULT_SHA256,
    ATTR_TOOL_RESULT_TRUNCATED,
    INSTRUMENTATION_SCOPE,
    TOOL_SPAN_NAME,
)

logger = logging.getLogger(__name__)


class OpenTelemetryToolMiddleware:
    """把工具调用记录为当前 executor 下的实时 OTel 子 Span。

    本实现只依赖工具对象的结构属性，不导入 ``matterloop_tools``，从而保持 observability
    的单向依赖边界。默认只记录参数和结果的字节数与 SHA-256，不记录原文；调用方显式
    选择后才会记录有界原文预览。所有追踪故障均被隔离；真实工具结果和异常会原样返回
    或抛出。
    """

    DEFAULT_CAPTURE_MAX_BODY_BYTES = 4096

    def __init__(
        self,
        tracer_provider: Any,
        *,
        capture_tool_payloads: bool = False,
        capture_max_body_bytes: int = DEFAULT_CAPTURE_MAX_BODY_BYTES,
    ) -> None:
        """创建工具追踪中间件，并按需启用有界 arguments/result 原文预览。"""
        if type(capture_tool_payloads) is not bool:
            raise ValueError("capture_tool_payloads must be a boolean")
        if type(capture_max_body_bytes) is not int or capture_max_body_bytes < 1:
            raise ValueError("capture_max_body_bytes must be a positive integer")
        try:
            self._context = importlib.import_module("opentelemetry.context")
            self._trace = importlib.import_module("opentelemetry.trace")
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetryToolMiddleware 需要 OpenTelemetry API，请安装 "
                "matterloop-observability[otel]"
            ) from exc
        self._tracer = tracer_provider.get_tracer(INSTRUMENTATION_SCOPE)
        self._capture_tool_payloads = capture_tool_payloads
        self._capture_max_body_bytes = capture_max_body_bytes

    async def invoke(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
        context: Any,
        call_next: Callable[[], Awaitable[Any]],
    ) -> Any:
        """执行下一层工具调用，并记录有界、低基数的调用结果。"""
        span: Any | None = None
        token: Any | None = None
        try:
            span = self._tracer.start_span(TOOL_SPAN_NAME)
            token = self._context.attach(self._trace.set_span_in_context(span))
        except Exception:
            logger.exception("实时 tool Span 创建失败，改为直接执行工具调用")
            if span is not None:
                self._end_span(span)
            return await call_next()

        try:
            try:
                await self._set_request_attributes(span, tool_name, arguments, context)
            except Exception:
                logger.exception("实时 tool 请求属性记录失败")
            try:
                result = await call_next()
            except asyncio.CancelledError as exc:
                self._set_failure(span, exc, outcome="cancelled")
                raise
            except Exception as exc:
                self._set_failure(span, exc, outcome=_exception_outcome(exc))
                raise
            try:
                await self._set_result_attributes(span, result)
            except Exception:
                logger.exception("实时 tool 结果属性记录失败")
            return result
        finally:
            try:
                if token is not None:
                    self._context.detach(token)
            except Exception:
                logger.exception("实时 tool OTel 上下文解绑失败")
            finally:
                self._end_span(span)

    async def _set_request_attributes(
        self,
        span: Any,
        tool_name: str,
        arguments: Mapping[str, object],
        context: Any,
    ) -> None:
        metadata = getattr(context, "metadata", None)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        tool_call_id = metadata.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            tool_call_id = uuid4().hex
        attributes: dict[str, object] = {
            ATTR_TOOL_NAME: tool_name,
            ATTR_TOOL_CALL_ID: tool_call_id,
            ATTR_TOOL_KIND: "tool",
        }
        run_id = getattr(context, "run_id", None)
        if isinstance(run_id, str) and run_id.strip():
            attributes[ATTR_RUN_ID] = run_id
        step_id = getattr(context, "step_id", None)
        if isinstance(step_id, str) and step_id.strip():
            attributes[ATTR_STEP_ID] = step_id
        executor = metadata.get("executor")
        if isinstance(executor, str) and executor.strip():
            attributes[ATTR_EXECUTOR] = executor

        # 关联标识先独立写入；即使未来的载荷序列化失败，也不能丢失整个 Span 的身份。
        _set_attributes(span, attributes)
        payload_attributes = await asyncio.to_thread(
            _argument_payload_attributes,
            arguments,
            content_key=ATTR_TOOL_ARGUMENTS,
            bytes_key=ATTR_TOOL_ARGUMENTS_BYTES,
            sha256_key=ATTR_TOOL_ARGUMENTS_SHA256,
            truncated_key=ATTR_TOOL_ARGUMENTS_TRUNCATED,
            capture_payload=self._capture_tool_payloads,
            max_bytes=self._capture_max_body_bytes,
        )
        _set_attributes(span, payload_attributes)

    async def _set_result_attributes(self, span: Any, result: Any) -> None:
        metadata = getattr(result, "metadata", None)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        kind = _tool_kind(metadata)
        is_error = getattr(result, "is_error", False) is True
        attributes: dict[str, object] = {
            ATTR_TOOL_KIND: kind,
            ATTR_TOOL_OUTCOME: "error" if is_error else "success",
        }
        if kind == "skill":
            attributes.update(_skill_attributes(metadata))
        elif kind == "mcp":
            attributes.update(_mcp_attributes(metadata))

        content = getattr(result, "content", "")
        if isinstance(content, str):
            payload_attributes = await asyncio.to_thread(
                _payload_attributes,
                content,
                content_key=ATTR_TOOL_RESULT,
                bytes_key=ATTR_TOOL_RESULT_BYTES,
                sha256_key=ATTR_TOOL_RESULT_SHA256,
                truncated_key=ATTR_TOOL_RESULT_TRUNCATED,
                capture_payload=self._capture_tool_payloads,
                max_bytes=self._capture_max_body_bytes,
            )
            attributes.update(payload_attributes)
        _set_attributes(span, attributes)
        if is_error:
            span.set_status(
                self._trace.Status(self._trace.StatusCode.ERROR, "tool returned an error")
            )

    def _set_failure(self, span: Any, exc: BaseException, *, outcome: str) -> None:
        """只记录异常类型和稳定 outcome，绝不把可能含凭据的异常消息写入 Span。"""
        try:
            _set_attributes(
                span,
                {
                    ATTR_TOOL_OUTCOME: outcome,
                    ATTR_EXCEPTION_TYPE: _qualified_type(exc),
                },
            )
            span.set_status(
                self._trace.Status(
                    self._trace.StatusCode.ERROR,
                    {
                        "cancelled": "tool invocation cancelled",
                        "denied": "tool permission denied",
                        "not_found": "tool not found",
                    }.get(outcome, "tool invocation failed"),
                )
            )
        except Exception:
            logger.exception("实时 tool 异常状态记录失败")

    @staticmethod
    def _end_span(span: Any) -> None:
        try:
            span.end()
        except Exception:
            logger.exception("实时 tool Span 结束失败")


def _tool_kind(metadata: Mapping[str, object]) -> str:
    if "skill_operation" in metadata or "skill_name" in metadata:
        return "skill"
    if "mcp_server" in metadata or "mcp_tool" in metadata:
        return "mcp"
    return "tool"


def _skill_attributes(metadata: Mapping[str, object]) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for source, target in (
        ("skill_name", ATTR_SKILL_NAME),
        ("skill_version", ATTR_SKILL_VERSION),
        ("skill_operation", ATTR_SKILL_OPERATION),
        ("sha256", ATTR_SKILL_SHA256),
        ("trust", ATTR_SKILL_TRUST),
    ):
        value = metadata.get(source)
        if isinstance(value, str) and value:
            attributes[target] = value
    return attributes


def _mcp_attributes(metadata: Mapping[str, object]) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for source, target in (
        ("mcp_server", ATTR_MCP_SERVER),
        ("mcp_tool", ATTR_MCP_TOOL),
    ):
        value = metadata.get(source)
        if isinstance(value, str) and value:
            attributes[target] = value
    blocks = metadata.get("content_blocks")
    if isinstance(blocks, int) and not isinstance(blocks, bool):
        attributes[ATTR_MCP_CONTENT_BLOCKS] = blocks
    truncated = metadata.get("truncated")
    if isinstance(truncated, bool):
        attributes[ATTR_MCP_TRUNCATED] = truncated
    return attributes


def _payload_attributes(
    value: str,
    *,
    content_key: str,
    bytes_key: str,
    sha256_key: str,
    truncated_key: str,
    capture_payload: bool,
    max_bytes: int,
) -> dict[str, object]:
    encoded = value.encode("utf-8")
    attributes: dict[str, object] = {
        bytes_key: len(encoded),
        sha256_key: hashlib.sha256(encoded).hexdigest(),
    }
    if capture_payload:
        preview, truncated = _truncate_utf8(encoded, max_bytes)
        attributes[content_key] = preview
        attributes[truncated_key] = truncated
    return attributes


def _argument_payload_attributes(
    value: object,
    *,
    content_key: str,
    bytes_key: str,
    sha256_key: str,
    truncated_key: str,
    capture_payload: bool,
    max_bytes: int,
) -> dict[str, object]:
    """在线程中完成可能较大的参数复制、序列化、编码和摘要计算。"""
    return _payload_attributes(
        _canonical_json(_copy_payload(value)),
        content_key=content_key,
        bytes_key=bytes_key,
        sha256_key=sha256_key,
        truncated_key=truncated_key,
        capture_payload=capture_payload,
        max_bytes=max_bytes,
    )


def _truncate_utf8(encoded: bytes, max_bytes: int) -> tuple[str, bool]:
    if len(encoded) <= max_bytes:
        return encoded.decode("utf-8"), False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _copy_payload(value: object) -> object:
    """把冻结的工具参数复制为可 JSON 序列化的原始结构，不修改或脱敏值。"""
    if isinstance(value, Mapping):
        return {str(key): _copy_payload(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_payload(item) for item in value]
    return value


def _exception_outcome(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "ToolPermissionDeniedError":
        return "denied"
    if name == "ToolNotFoundError":
        return "not_found"
    return "exception"


def _qualified_type(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def _set_attributes(span: Any, attributes: Mapping[str, object]) -> None:
    for key, value in attributes.items():
        span.set_attribute(key, value)


__all__ = ["OpenTelemetryToolMiddleware"]
