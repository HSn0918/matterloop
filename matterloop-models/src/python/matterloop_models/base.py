"""定义与模型供应商无关的请求、响应和调用协议。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable
from urllib.parse import quote
from uuid import uuid4


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """复制并冻结顶层映射，防止调用期间被外部代码修改。"""
    return MappingProxyType(dict(value))


class MessageRole(str, Enum):
    """模型消息支持的通用角色。"""

    DEVELOPER = "developer"
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ToolChoice(str, Enum):
    """控制模型是否可以或必须调用已声明的工具。

    ``None`` 表示交给供应商采用其默认行为；显式枚举值可避免不同供应商对默认值的
    解释差异。
    """

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class ContextInputMode(str, Enum):
    """声明带作用域的模型输入如何更新已保存上下文。"""

    REPLACE = "replace"
    APPEND = "append"


class ModelItemCategory(str, Enum):
    """描述规范模型输入在 Agent 上下文中的语义类别。"""

    SYSTEM = "system"
    GOAL = "goal"
    WORKING_MEMORY = "working_memory"
    HISTORY = "history"
    TOOL_RESULT = "tool_result"
    AGENT_STATE = "agent_state"


class ModelItemRetention(str, Enum):
    """描述 Context Lifecycle Engine 对输入项的保留要求。"""

    PINNED = "pinned"
    RECENT = "recent"
    SUMMARIZABLE = "summarizable"
    EXTERNALIZED = "externalized"


@dataclass(frozen=True, slots=True)
class ModelContextScope:
    """标识一次模型调用所属的可恢复上下文。"""

    run_id: str
    participant: str
    tenant_id: str = "default"
    task_id: str | None = None
    invocation_id: str | None = None

    def __post_init__(self) -> None:
        """规范作用域文本，避免不同进程构造出不一致的存储键。"""
        for name in ("run_id", "participant", "tenant_id"):
            value = getattr(self, name).strip()
            if not value:
                raise ValueError(f"context scope {name} must not be empty")
            object.__setattr__(self, name, value)
        for name in ("task_id", "invocation_id"):
            value = getattr(self, name)
            if value is not None:
                normalized = value.strip()
                if not normalized:
                    raise ValueError(f"context scope {name} must not be empty")
                object.__setattr__(self, name, normalized)

    @property
    def key(self) -> str:
        """返回不包含调用序号的稳定上下文键。"""
        task = self.task_id or "-"
        return ":".join(
            quote(value, safe="-._~")
            for value in (self.tenant_id, self.run_id, self.participant, task)
        )


@runtime_checkable
class ModelContinuation(Protocol):
    """表示只能原样交还给对应模型适配器的不透明续轮状态。

    continuation 可能包含供应商要求回传、但不应该进入日志或检查点的私有状态。
    上层只能读取 ``provider`` 来做诊断，并应在一次模型事务内原样传递该对象。
    """

    @property
    def provider(self) -> str:
        """返回创建该续轮状态的供应商适配器标识。"""
        ...


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """表示一条与供应商无关的文本消息。

    Args:
        role: 消息发送方角色。
        content: 消息的纯文本内容。
        name: 可选的稳定发送方名称。
    """

    role: MessageRole
    content: str = field(repr=False)
    name: str | None = None

    def __post_init__(self) -> None:
        """拒绝无意义的空消息和空名称。"""
        if not self.content.strip():
            raise ValueError("message content must not be empty")
        if self.name is not None and not self.name.strip():
            raise ValueError("message name must not be empty")


@dataclass(frozen=True, slots=True)
class ModelMessageItem:
    """表示可持久化、可分类的规范消息输入。"""

    role: MessageRole
    content: str = field(repr=False)
    name: str | None = None
    category: ModelItemCategory | None = None
    retention: ModelItemRetention | None = None
    item_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """校验消息并为系统指令选择安全的默认保留策略。"""
        if not self.content.strip():
            raise ValueError("message item content must not be empty")
        if self.name is not None and not self.name.strip():
            raise ValueError("message item name must not be empty")
        if not self.item_id.strip():
            raise ValueError("message item id must not be empty")
        protected = self.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}
        if self.category is None:
            object.__setattr__(
                self,
                "category",
                ModelItemCategory.SYSTEM if protected else ModelItemCategory.HISTORY,
            )
        if self.retention is None:
            object.__setattr__(
                self,
                "retention",
                ModelItemRetention.PINNED if protected else ModelItemRetention.SUMMARIZABLE,
            )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ModelToolCallItem:
    """表示可跨模型轮次保存的规范工具调用。"""

    call_id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict, repr=False)
    category: ModelItemCategory = ModelItemCategory.AGENT_STATE
    retention: ModelItemRetention = ModelItemRetention.PINNED
    item_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """保证调用可关联并冻结参数与元数据。"""
        if not self.call_id.strip() or not self.name.strip() or not self.item_id.strip():
            raise ValueError("tool call item identifiers must not be empty")
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ModelToolOutputItem:
    """表示可外置并可关联回工具调用的规范结果。"""

    call_id: str
    output: str = field(repr=False)
    is_error: bool = False
    artifact_uri: str | None = None
    category: ModelItemCategory = ModelItemCategory.TOOL_RESULT
    retention: ModelItemRetention = ModelItemRetention.PINNED
    item_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """校验调用关联和可选制品引用。"""
        if not self.call_id.strip() or not self.item_id.strip():
            raise ValueError("tool output item identifiers must not be empty")
        if self.artifact_uri is not None and not self.artifact_uri.strip():
            raise ValueError("tool output artifact URI must not be empty")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ModelCompactionItem:
    """保存供应商原生或框架语义压缩产生的规范输入项。"""

    payload: str = field(repr=False)
    provider: str
    model: str
    native: bool = True
    category: ModelItemCategory = ModelItemCategory.HISTORY
    retention: ModelItemRetention = ModelItemRetention.SUMMARIZABLE
    item_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """拒绝无法恢复或无法路由的压缩载荷。"""
        if not self.payload or not self.provider.strip() or not self.model.strip():
            raise ValueError("compaction payload, provider, and model must not be empty")
        if not self.item_id.strip():
            raise ValueError("compaction item id must not be empty")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


ModelInputItem: TypeAlias = (
    ModelMessageItem | ModelToolCallItem | ModelToolOutputItem | ModelCompactionItem
)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """描述模型可以选择调用的一个函数工具。

    Args:
        name: 工具稳定名称。
        description: 面向模型的能力说明。
        parameters: 工具输入的 JSON Schema。
        strict: 是否要求模型严格遵循输入 Schema。
    """

    name: str
    description: str
    parameters: Mapping[str, object]
    strict: bool = True

    def __post_init__(self) -> None:
        """校验工具标识并冻结输入 Schema。"""
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True, slots=True)
class ToolCall:
    """记录模型请求执行的一个工具调用。

    Args:
        call_id: 关联后续工具输出的稳定标识。
        name: 模型选择的工具名称。
        arguments: 已解析的工具参数对象。
    """

    call_id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """保证工具调用可关联并冻结参数。"""
        if not self.call_id.strip():
            raise ValueError("tool call id must not be empty")
        if not self.name.strip():
            raise ValueError("tool call name must not be empty")
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """把本地工具执行结果关联回模型产生的调用。

    Args:
        call_id: 对应工具调用的稳定标识。
        output: 返回给模型的文本结果。
        is_error: 本地工具是否执行失败。
    """

    call_id: str
    output: str = field(repr=False)
    is_error: bool = False

    def __post_init__(self) -> None:
        """保证输出可以关联到一个明确调用。"""
        if not self.call_id.strip():
            raise ValueError("tool output call id must not be empty")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """保存一次模型调用可计量的 Token 用量。

    Args:
        input_tokens: 输入侧 Token 数。
        output_tokens: 输出侧 Token 数。
        total_tokens: 供应商报告或适配器计算的总 Token 数。
        cache_hit_tokens: 输入中命中供应商上下文缓存的 Token 数。
        cache_miss_tokens: 输入中未命中供应商上下文缓存的 Token 数。
        reasoning_tokens: 输出中由供应商归类为推理过程的 Token 数。
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        """拒绝供应商返回的非法负数用量。"""
        values = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.cache_hit_tokens,
            self.cache_miss_tokens,
            self.reasoning_tokens,
        )
        if min(values) < 0:
            raise ValueError("token usage must not be negative")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """描述一次通用模型调用。

    Args:
        messages: 本轮发送的上下文消息。
        tools: 可由模型选择的函数工具。
        tool_outputs: 上一响应中工具调用的本地执行结果。
        previous_response_id: 支持供应商保持连续响应的可选标识。
        response_schema: 要求结构化文本输出时使用的 JSON Schema。
        response_schema_name: 结构化输出 Schema 的稳定名称。
        max_output_tokens: 可选的最大输出 Token 数。
        temperature: 可选采样温度；不支持时由供应商适配器报错。
        tool_choice: 是否允许或强制模型调用工具；``None`` 使用供应商默认值。
        continuation: 上一响应返回的供应商不透明续轮状态。
        usage_scopes: 本次用量需要同时归集到的稳定额度作用域。
        metadata: 只在 MatterLoop 内部传播的关联信息。
    """

    messages: tuple[ModelMessage, ...] = ()
    input_items: tuple[ModelInputItem, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()
    tool_outputs: tuple[ToolOutput, ...] = ()
    previous_response_id: str | None = None
    response_schema: Mapping[str, object] | None = None
    response_schema_name: str = "matterloop_response"
    max_output_tokens: int | None = None
    temperature: float | None = None
    tool_choice: ToolChoice | None = None
    continuation: ModelContinuation | None = field(default=None, repr=False)
    usage_scopes: tuple[str, ...] = ()
    context_scope: ModelContextScope | None = None
    context_mode: ContextInputMode = ContextInputMode.REPLACE
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """校验调用边界并冻结映射字段。"""
        if (
            not self.messages
            and not self.input_items
            and not self.tool_outputs
            and self.continuation is None
        ):
            raise ValueError(
                "model request requires messages, input items, tool outputs, or continuation"
            )
        if self.input_items and (
            self.messages
            or self.tool_outputs
            or self.previous_response_id is not None
            or self.continuation is not None
        ):
            raise ValueError(
                "canonical input items must not be mixed with legacy conversation state"
            )
        item_types = (
            ModelMessageItem,
            ModelToolCallItem,
            ModelToolOutputItem,
            ModelCompactionItem,
        )
        if any(not isinstance(item, item_types) for item in self.input_items):
            raise TypeError("input items must be canonical model input items")
        if self.previous_response_id is not None and not self.previous_response_id.strip():
            raise ValueError("previous response id must not be empty")
        if not self.response_schema_name.strip():
            raise ValueError("response schema name must not be empty")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max output tokens must be at least 1")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.tool_choice is not None and not isinstance(self.tool_choice, ToolChoice):
            raise TypeError("tool choice must be a ToolChoice")
        if self.continuation is not None and not isinstance(self.continuation, ModelContinuation):
            raise TypeError("continuation must implement ModelContinuation")
        if self.context_scope is not None and not isinstance(self.context_scope, ModelContextScope):
            raise TypeError("context scope must be a ModelContextScope")
        if not isinstance(self.context_mode, ContextInputMode):
            raise TypeError("context mode must be a ContextInputMode")
        if self.context_mode is ContextInputMode.APPEND and self.context_scope is None:
            raise ValueError("append context mode requires a context scope")
        if any(not isinstance(scope, str) for scope in self.usage_scopes):
            raise TypeError("usage scope must be text")
        normalized_scopes = tuple(scope.strip() for scope in self.usage_scopes)
        if any(not scope for scope in normalized_scopes):
            raise ValueError("usage scope must not be empty")
        if len(set(normalized_scopes)) != len(normalized_scopes):
            raise ValueError("usage scopes must not contain duplicates")
        object.__setattr__(self, "usage_scopes", normalized_scopes)
        if self.response_schema is not None:
            object.__setattr__(self, "response_schema", _freeze_mapping(self.response_schema))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """保存供应商响应中供上层稳定消费的数据。

    Args:
        output_text: 聚合后的文本输出。
        tool_calls: 模型请求执行的函数工具调用。
        usage: 本次调用的归一化 Token 用量。
        response_id: 继续供应商会话时使用的可选标识。
        continuation: 只能原样传回对应适配器的私有续轮状态，不参与对象 repr。
        metadata: 不含原始 SDK 对象和凭据的供应商元数据。
    """

    output_text: str = field(default="", repr=False)
    tool_calls: tuple[ToolCall, ...] = ()
    output_items: tuple[ModelInputItem, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)
    response_id: str | None = None
    continuation: ModelContinuation | None = field(default=None, repr=False)
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """冻结元数据并拒绝空响应标识。"""
        if self.response_id is not None and not self.response_id.strip():
            raise ValueError("response id must not be empty")
        if self.continuation is not None and not isinstance(self.continuation, ModelContinuation):
            raise TypeError("continuation must implement ModelContinuation")
        item_types = (
            ModelMessageItem,
            ModelToolCallItem,
            ModelToolOutputItem,
            ModelCompactionItem,
        )
        if any(not isinstance(item, item_types) for item in self.output_items):
            raise TypeError("output items must be canonical model input items")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@runtime_checkable
class ModelClient(Protocol):
    """所有同步无关模型适配器必须实现的异步协议。"""

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """执行一次模型调用。

        Args:
            request: 与供应商无关的模型请求。

        Returns:
            归一化后的模型响应。
        """
        ...


@runtime_checkable
class ExactTokenCountingClient(Protocol):
    """可选的供应商精确输入 Token 计数能力。"""

    async def count_input_tokens(self, request: ModelRequest) -> int:
        """返回完整请求的输入 Token 数。"""
        ...


@runtime_checkable
class NativeCompactingClient(Protocol):
    """可选的供应商原生上下文压缩能力。"""

    async def compact_input(self, request: ModelRequest) -> tuple[ModelInputItem, ...]:
        """返回可作为后续输入原样续传的压缩窗口。"""
        ...
