"""OpenAI Responses API 适配器的离线映射测试。"""

from __future__ import annotations

import asyncio
from dataclasses import fields
from types import SimpleNamespace

import pytest
from matterloop_models import (
    MessageRole,
    ModelCompactionItem,
    ModelInvocationError,
    ModelMessage,
    ModelMessageItem,
    ModelRequest,
    ToolDefinition,
    ToolOutput,
)
from matterloop_models.providers import OpenAIModelClient, OpenAIModelConfig


class StubResponses:
    """记录调用参数并返回预设响应。"""

    def __init__(self, response: object) -> None:
        self.response = response
        self.parameters: dict[str, object] = {}

    async def create(self, **kwargs: object) -> object:
        """模拟 SDK 的异步 create 方法。"""
        self.parameters = kwargs
        return self.response


class FailingResponses:
    """抛出包含敏感示例文本的供应商异常。"""

    async def create(self, **kwargs: object) -> object:
        """模拟供应商调用失败。"""
        del kwargs
        raise RuntimeError("authorization sk-secret-value")


class StubInputTokens:
    """记录精确计数请求。"""

    def __init__(self) -> None:
        self.parameters: dict[str, object] = {}

    async def count(self, **kwargs: object) -> object:
        self.parameters = kwargs
        return SimpleNamespace(input_tokens=37)


class ContextResponses(StubResponses):
    """同时模拟 create、input_tokens.count 和 compact。"""

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(output_text="ok", output=[], usage=None))
        self.input_tokens = StubInputTokens()
        self.compact_parameters: dict[str, object] = {}

    async def compact(self, **kwargs: object) -> object:
        self.compact_parameters = kwargs
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    role="user",
                    content=[{"type": "input_text", "text": "retained"}],
                ),
                SimpleNamespace(type="compaction", encrypted_content="encrypted-window"),
            ]
        )


class ContextOpenAIClient:
    """暴露完整上下文能力资源。"""

    def __init__(self) -> None:
        self.responses = ContextResponses()


class StubOpenAIClient:
    """暴露与官方 SDK 相同的 responses 属性。"""

    def __init__(self, response: object) -> None:
        self.responses = StubResponses(response)
        self.closed = False

    async def close(self) -> None:
        """记录客户端资源已关闭。"""
        self.closed = True


class FailingOpenAIClient:
    """暴露失败的 Responses 资源。"""

    def __init__(self) -> None:
        self.responses = FailingResponses()

    async def close(self) -> None:
        """模拟关闭失败客户端。"""


def test_openai_config_contains_only_adapter_state() -> None:
    """SDK 连接、凭据和传输参数不得进入适配器配置。"""
    config = OpenAIModelConfig(model="configured-model")

    assert {item.name for item in fields(config)} == {"model"}


def test_openai_adapter_maps_structured_output_and_tool_call() -> None:
    asyncio.run(_structured_output_scenario())


async def _structured_output_scenario() -> None:
    sdk_response = SimpleNamespace(
        id="resp_1",
        status="completed",
        output_text="",
        output=[
            SimpleNamespace(type="reasoning", id="reasoning_1", summary=[]),
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="lookup",
                arguments='{"query":"matter"}',
            ),
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
    )
    sdk_client = StubOpenAIClient(sdk_response)
    client = OpenAIModelClient(
        OpenAIModelConfig(model="configured-model"),
        client=sdk_client,
    )
    request = ModelRequest(
        messages=(ModelMessage(MessageRole.USER, "search"),),
        tools=(
            ToolDefinition(
                name="lookup",
                description="查询数据",
                parameters={"type": "object", "properties": {}},
            ),
        ),
        response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    response = await client.generate(request)

    assert sdk_client.responses.parameters["model"] == "configured-model"
    assert sdk_client.responses.parameters["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "查询数据",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        }
    ]
    assert response.tool_calls[0].arguments == {"query": "matter"}
    assert isinstance(response.output_items[0], ModelCompactionItem)
    assert '"type":"reasoning"' in response.output_items[0].payload
    assert response.usage.total_tokens == 16


def test_openai_adapter_maps_function_call_outputs() -> None:
    asyncio.run(_function_output_scenario())


async def _function_output_scenario() -> None:
    sdk_client = StubOpenAIClient(
        SimpleNamespace(id="resp_2", status="completed", output_text="done", output=[], usage=None)
    )
    client = OpenAIModelClient(
        OpenAIModelConfig(model="configured-model"),
        client=sdk_client,
    )
    request = ModelRequest(
        messages=(),
        tool_outputs=(ToolOutput(call_id="call_1", output="result"),),
        previous_response_id="resp_1",
    )

    response = await client.generate(request)

    assert response.output_text == "done"
    assert sdk_client.responses.parameters["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "result"}
    ]
    assert sdk_client.responses.parameters["previous_response_id"] == "resp_1"

    await client.aclose()
    assert not sdk_client.closed


def test_openai_adapter_can_take_ownership_of_injected_client() -> None:
    sdk_client = StubOpenAIClient(SimpleNamespace())
    client = OpenAIModelClient(
        OpenAIModelConfig(model="configured-model"),
        client=sdk_client,
        owns_client=True,
    )

    asyncio.run(client.aclose())

    assert sdk_client.closed


def test_openai_adapter_requires_explicit_client() -> None:
    with pytest.raises(TypeError, match="client"):
        OpenAIModelClient(OpenAIModelConfig(model="configured-model"))


def test_openai_adapter_does_not_expose_supplier_error_text() -> None:
    async def scenario() -> None:
        client = OpenAIModelClient(
            OpenAIModelConfig(model="configured-model"),
            client=FailingOpenAIClient(),
        )
        request = ModelRequest(messages=(ModelMessage(MessageRole.USER, "hello"),))

        with pytest.raises(ModelInvocationError) as captured:
            await client.generate(request)

        assert "sk-secret-value" not in str(captured.value)
        assert captured.value.__suppress_context__

    asyncio.run(scenario())


def test_openai_adapter_counts_and_compacts_canonical_input() -> None:
    async def scenario() -> None:
        sdk_client = ContextOpenAIClient()
        client = OpenAIModelClient(
            OpenAIModelConfig(model="configured-model"),
            client=sdk_client,
        )
        request = ModelRequest(input_items=(ModelMessageItem(MessageRole.USER, "hello"),))

        assert await client.count_input_tokens(request) == 37
        compacted = await client.compact_input(request)

        assert len(compacted) == 2
        assert isinstance(compacted[0], ModelCompactionItem)
        assert '"type":"message"' in compacted[0].payload
        assert compacted[0].provider == "openai"
        follow_up = ModelRequest(input_items=compacted)
        await client.generate(follow_up)
        assert sdk_client.responses.parameters["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "retained"}],
            },
            {"type": "compaction", "encrypted_content": "encrypted-window"},
        ]
        assert sdk_client.responses.compact_parameters["input"] == [
            {"role": "user", "content": "hello"}
        ]

    asyncio.run(scenario())
