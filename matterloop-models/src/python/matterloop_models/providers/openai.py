"""将 OpenAI 异步 Responses API 适配为 MatterLoop 模型协议。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from matterloop_models.base import (
    MessageRole,
    ModelCompactionItem,
    ModelInputItem,
    ModelItemCategory,
    ModelItemRetention,
    ModelMessageItem,
    ModelRequest,
    ModelResponse,
    ModelToolCallItem,
    ModelToolOutputItem,
    TokenUsage,
    ToolCall,
)
from matterloop_models.capabilities import ModelCapabilities, ModelDescriptor, ModelFeature
from matterloop_models.errors import (
    ModelAuthenticationError,
    ModelCapabilityError,
    ModelInvocationError,
    ModelPaymentRequiredError,
    ModelRateLimitError,
    ModelResponseParseError,
    ModelServiceError,
)
from matterloop_models.providers.compatible import SupportsAsyncClose


class OpenAIResponsesResource(Protocol):
    """调用方注入客户端需要暴露的 Responses API 资源。"""

    @property
    def create(self) -> Callable[..., Awaitable[object]]:
        """返回可接受供应商关键字参数的异步请求函数。"""
        ...


class OpenAIResponsesClient(Protocol):
    """OpenAI 适配器依赖的最小异步客户端协议。"""

    @property
    def responses(self) -> OpenAIResponsesResource:
        """返回异步 Responses API 资源。"""
        ...


@dataclass(frozen=True, slots=True)
class OpenAIModelConfig:
    """配置 OpenAI Responses API 适配器。

    Args:
        model: 每次请求使用的明确模型标识；适配器不提供隐式默认值。

    Notes:
        SDK 端点、凭据、组织、项目、超时和重试属于客户端构造职责，不进入适配器配置。
    """

    model: str

    def __post_init__(self) -> None:
        """校验适配器使用的模型标识。"""
        if not self.model.strip():
            raise ValueError("OpenAI model must not be empty")


class OpenAIModelClient:
    """使用官方异步 SDK 调用 Responses API。

    调用方必须注入已经构造好的异步客户端，使凭据加载、代理、端点和连接池生命周期完全
    由应用层控制。适配器不导入供应商 SDK，也不读取任何进程环境。

    Args:
        config: 适配器的非敏感模型配置。
        client: 调用方构造且满足 ``OpenAIResponsesClient`` 的异步客户端。
        owns_client: 是否把客户端关闭责任交给适配器，默认由调用方继续管理。
    """

    def __init__(
        self,
        config: OpenAIModelConfig,
        *,
        client: OpenAIResponsesClient,
        owns_client: bool = False,
    ) -> None:
        if owns_client and not isinstance(client, SupportsAsyncClose):
            raise TypeError("owned OpenAI client must provide async close()")
        self._config = config
        self._client = client
        self._closer = cast(SupportsAsyncClose, client) if owns_client else None

    @property
    def descriptor(self) -> ModelDescriptor:
        """返回 OpenAI Responses 适配器的非敏感能力描述。"""
        return ModelDescriptor(
            provider="openai",
            model=self._config.model,
            capabilities=ModelCapabilities(
                supported=frozenset(
                    {
                        ModelFeature.TEXT_GENERATION,
                        ModelFeature.DEVELOPER_MESSAGES,
                        ModelFeature.TOOL_CALLING,
                        ModelFeature.JSON_OBJECT_OUTPUT,
                        ModelFeature.JSON_SCHEMA_OUTPUT,
                        ModelFeature.RESPONSE_ID_CONTINUATION,
                        ModelFeature.EXACT_TOKEN_COUNT,
                        ModelFeature.NATIVE_COMPACTION,
                    }
                ),
                unsupported=frozenset({ModelFeature.OPAQUE_CONTINUATION}),
            ),
            metadata={"api": "responses"},
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """调用 Responses API 并归一化文本、工具调用与用量。

        Args:
            request: 通用模型请求。

        Returns:
            不包含 SDK 原始对象和凭据的通用响应。

        Raises:
            ModelInvocationError: SDK 调用失败。
            ModelResponseParseError: 响应中的工具参数不是合法 JSON 对象。
        """
        parameters = self._build_parameters(request)
        try:
            response = await self._client.responses.create(**parameters)
        except Exception as exc:
            raise self._safe_invocation_error(exc) from None
        try:
            return self._parse_response(response)
        except ModelResponseParseError as exc:
            if exc.usage is None:
                exc.usage = self._parse_usage(response)
            raise

    async def count_input_tokens(self, request: ModelRequest) -> int:
        """调用 Responses Input Tokens API 精确计算完整请求输入。"""
        resource = getattr(self._client.responses, "input_tokens", None)
        count = getattr(resource, "count", None)
        if not callable(count):
            raise ModelCapabilityError("OpenAI client does not expose responses.input_tokens.count")
        parameters = self._build_parameters(request)
        allowed = {"model", "input", "tools"}
        try:
            result = await count(
                **{key: value for key, value in parameters.items() if key in allowed}
            )
        except Exception as exc:
            raise self._safe_invocation_error(exc) from None
        tokens = self._read(result, "input_tokens", self._read(result, "total_tokens", None))
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise ModelResponseParseError("OpenAI token count is invalid")
        return tokens

    async def compact_input(self, request: ModelRequest) -> tuple[ModelInputItem, ...]:
        """调用独立 Responses Compact API 并原样封装其规范下一窗口。"""
        compact = getattr(self._client.responses, "compact", None)
        if not callable(compact):
            raise ModelCapabilityError("OpenAI client does not expose responses.compact")
        parameters = self._build_parameters(request)
        try:
            result = await compact(
                model=self._config.model,
                input=parameters["input"],
            )
        except Exception as exc:
            raise self._safe_invocation_error(exc) from None
        output = self._read(result, "output", ())
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes, bytearray)):
            raise ModelResponseParseError("OpenAI compact output must be an array")
        items = tuple(self._parse_compact_item(item) for item in output)
        if not items:
            raise ModelResponseParseError("OpenAI compact output is empty")
        return items

    async def aclose(self) -> None:
        """仅在适配器持有客户端所有权时关闭其连接池。"""
        if self._closer is not None:
            await self._closer.close()

    def _build_parameters(self, request: ModelRequest) -> dict[str, object]:
        if request.continuation is not None:
            raise ValueError("OpenAI Responses API does not accept chat continuation state")
        parameters: dict[str, object] = {"model": self._config.model}
        if request.input_items:
            parameters["input"] = [self._map_input_item(item) for item in request.input_items]
        elif request.tool_outputs:
            parameters["input"] = [
                {
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": self._format_tool_output(output.output, output.is_error),
                }
                for output in request.tool_outputs
            ]
        else:
            parameters["input"] = [
                {
                    "role": message.role.value,
                    "content": message.content,
                    **({"name": message.name} if message.name is not None else {}),
                }
                for message in request.messages
            ]
        if request.previous_response_id is not None:
            parameters["previous_response_id"] = request.previous_response_id
        if request.tools:
            parameters["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                    "strict": tool.strict,
                }
                for tool in request.tools
            ]
        if request.tool_choice is not None:
            parameters["tool_choice"] = request.tool_choice.value
        if request.response_schema is not None:
            parameters["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.response_schema_name,
                    "schema": dict(request.response_schema),
                    "strict": True,
                }
            }
        if request.max_output_tokens is not None:
            parameters["max_output_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            parameters["temperature"] = request.temperature
        return parameters

    @classmethod
    def _parse_response(cls, response: object) -> ModelResponse:
        tool_calls: list[ToolCall] = []
        normalized_items: list[ModelInputItem] = []
        output = cls._read(response, "output", ())
        if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
            for item in output:
                item_type = cls._read(item, "type", "")
                if item_type != "function_call":
                    if item_type != "message":
                        raw_item = cls._plain_compact_item(item)
                        response_model = cls._read(response, "model", "unknown")
                        normalized_items.append(
                            ModelCompactionItem(
                                payload=json.dumps(
                                    raw_item,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                                provider="openai",
                                model=(
                                    response_model
                                    if isinstance(response_model, str) and response_model.strip()
                                    else "unknown"
                                ),
                                native=True,
                                category=ModelItemCategory.AGENT_STATE,
                                retention=ModelItemRetention.PINNED,
                                metadata={
                                    "provider_output_item": True,
                                    "provider_item_type": (
                                        item_type if isinstance(item_type, str) else "unknown"
                                    ),
                                },
                            )
                        )
                    continue
                arguments = cls._parse_arguments(cls._read(item, "arguments", "{}"))
                call_id = cls._read(item, "call_id", cls._read(item, "id", ""))
                name = cls._read(item, "name", "")
                if (
                    not isinstance(call_id, str)
                    or not call_id.strip()
                    or not isinstance(name, str)
                    or not name.strip()
                ):
                    raise ModelResponseParseError("OpenAI function call identifiers are invalid")
                tool_call = ToolCall(call_id=call_id, name=name, arguments=arguments)
                tool_calls.append(tool_call)
                normalized_items.append(
                    ModelToolCallItem(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                )

        token_usage = cls._parse_usage(response)
        response_id = cls._read(response, "id", None)
        output_text = cls._read(response, "output_text", "")
        status = cls._read(response, "status", None)
        if isinstance(output_text, str) and output_text.strip():
            normalized_items.append(
                ModelMessageItem(role=MessageRole.ASSISTANT, content=output_text)
            )
        return ModelResponse(
            output_text=output_text if isinstance(output_text, str) else "",
            tool_calls=tuple(tool_calls),
            output_items=tuple(normalized_items),
            usage=token_usage,
            response_id=response_id if isinstance(response_id, str) else None,
            metadata={"provider": "openai", "status": status},
        )

    @classmethod
    def _parse_usage(cls, response: object) -> TokenUsage:
        usage = cls._read(response, "usage", None)
        input_tokens = cls._read_int(usage, "input_tokens")
        output_tokens = cls._read_int(usage, "output_tokens")
        total_tokens = cls._read_int(usage, "total_tokens") or input_tokens + output_tokens
        input_details = cls._read(usage, "input_tokens_details", None)
        output_details = cls._read(usage, "output_tokens_details", None)
        cache_hit_tokens = cls._read_int(input_details, "cached_tokens")
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=max(input_tokens - cache_hit_tokens, 0),
            reasoning_tokens=cls._read_int(output_details, "reasoning_tokens"),
        )

    @staticmethod
    def _read(value: object, name: str, default: object) -> object:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _read_int(cls, value: object, name: str) -> int:
        result = cls._read(value, name, 0)
        return result if isinstance(result, int) and not isinstance(result, bool) else 0

    @staticmethod
    def _parse_arguments(value: object) -> Mapping[str, object]:
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
        if not isinstance(value, str):
            raise ModelResponseParseError("OpenAI function arguments must be a JSON object")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ModelResponseParseError("OpenAI function arguments contain invalid JSON") from exc
        if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
            raise ModelResponseParseError("OpenAI function arguments must decode to an object")
        return cast(dict[str, object], decoded)

    @staticmethod
    def _format_tool_output(output: str, is_error: bool) -> str:
        if not is_error:
            return output
        return json.dumps({"is_error": True, "content": output}, ensure_ascii=False)

    def _map_input_item(self, item: ModelInputItem) -> dict[str, object]:
        if isinstance(item, ModelMessageItem):
            return {
                "role": item.role.value,
                "content": item.content,
                **({"name": item.name} if item.name is not None else {}),
            }
        if isinstance(item, ModelToolCallItem):
            return {
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": json.dumps(
                    dict(item.arguments),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        if isinstance(item, ModelToolOutputItem):
            return {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": self._format_tool_output(item.output, item.is_error),
            }
        if isinstance(item, ModelCompactionItem):
            if item.native:
                if item.provider != "openai":
                    raise ModelCapabilityError("native compaction item belongs to another provider")
                try:
                    raw_item = json.loads(item.payload)
                except json.JSONDecodeError:
                    # 兼容 0.2.0 之前仅保存 encrypted_content 的本地开发快照。
                    return {
                        "type": "compaction",
                        "encrypted_content": item.payload,
                    }
                if not isinstance(raw_item, dict) or not all(
                    isinstance(key, str) for key in raw_item
                ):
                    raise ModelCapabilityError(
                        "OpenAI native compaction payload must encode an input item"
                    )
                return cast(dict[str, object], raw_item)
            return {
                "role": "developer",
                "content": f"Previous context summary:\n{item.payload}",
            }
        raise TypeError(f"unsupported OpenAI input item: {type(item).__name__}")

    def _parse_compact_item(self, item: object) -> ModelInputItem:
        raw_item = self._plain_compact_item(item)
        return ModelCompactionItem(
            payload=json.dumps(
                raw_item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            provider="openai",
            model=self._config.model,
            native=True,
        )

    @staticmethod
    def _plain_compact_item(item: object) -> dict[str, object]:
        """把 SDK 模型转换为可持久化 JSON，同时保留 Compact 返回的完整项。"""
        raw: object
        if isinstance(item, Mapping):
            raw = dict(item)
        else:
            model_dump = getattr(item, "model_dump", None)
            if callable(model_dump):
                raw = model_dump(mode="json", exclude_none=True)
            else:
                raw = vars(item) if hasattr(item, "__dict__") else None
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ModelResponseParseError("OpenAI compact output item is invalid")
        try:
            json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ModelResponseParseError(
                "OpenAI compact output item is not JSON serializable"
            ) from exc
        return cast(dict[str, object], raw)

    @classmethod
    def _safe_invocation_error(cls, error: Exception) -> ModelInvocationError:
        status_code = cls._status_code(error)
        if status_code in {401, 403}:
            return ModelAuthenticationError(f"OpenAI authentication failed (HTTP {status_code})")
        if status_code == 402:
            return ModelPaymentRequiredError("OpenAI payment is required (HTTP 402)")
        if status_code == 429:
            return ModelRateLimitError("OpenAI rate limit was exceeded (HTTP 429)")
        if status_code is not None and 500 <= status_code < 600:
            return ModelServiceError(f"OpenAI service failed (HTTP {status_code})")
        return ModelInvocationError(f"OpenAI Responses API call failed ({type(error).__name__})")

    @classmethod
    def _status_code(cls, error: Exception) -> int | None:
        status_code = cls._read(error, "status_code", None)
        if isinstance(status_code, int) and not isinstance(status_code, bool):
            return status_code
        response = cls._read(error, "response", None)
        response_status = cls._read(response, "status_code", None)
        if isinstance(response_status, int) and not isinstance(response_status, bool):
            return response_status
        return None
