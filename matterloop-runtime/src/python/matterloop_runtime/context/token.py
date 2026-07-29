"""Context 输入的精确计数路由与保守估算。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from matterloop_models import (
    ModelCompactionItem,
    ModelMessageItem,
    ModelRequest,
    ModelToolCallItem,
    ModelToolOutputItem,
)


@runtime_checkable
class TokenCounter(Protocol):
    """计算完整模型请求的输入 Token 数。"""

    async def count(self, request: ModelRequest) -> int:
        """返回非负输入 Token 数。"""
        ...


class ApproximateTokenCounter:
    """按规范 JSON 的 UTF-8 大小进行带安全余量的保守估算。"""

    def __init__(self, safety_margin: float = 0.15) -> None:
        if not math.isfinite(safety_margin) or safety_margin < 0:
            raise ValueError("token estimate safety margin must be finite and non-negative")
        self._safety_margin = safety_margin

    async def count(self, request: ModelRequest) -> int:
        """估算消息、工具定义、Schema 与控制字段的总输入大小。"""
        payload = {
            "input": [_item_payload(item) for item in request.input_items]
            if request.input_items
            else {
                "messages": [
                    {
                        "role": message.role.value,
                        "content": message.content,
                        "name": message.name,
                    }
                    for message in request.messages
                ],
                "tool_outputs": [
                    {
                        "call_id": output.call_id,
                        "output": output.output,
                        "is_error": output.is_error,
                    }
                    for output in request.tool_outputs
                ],
            },
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                }
                for tool in request.tools
            ],
            "response_schema": (
                dict(request.response_schema) if request.response_schema is not None else None
            ),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        base = math.ceil(len(encoded) / 3)
        return math.ceil(base * (1 + self._safety_margin))

    def count_text(self, value: str) -> int:
        """估算单段文本，用于在构造模型请求前判断是否外置。"""
        base = math.ceil(len(value.encode("utf-8")) / 3)
        return math.ceil(base * (1 + self._safety_margin))


def _item_payload(item: object) -> Mapping[str, object]:
    if isinstance(item, ModelMessageItem):
        return {
            "type": "message",
            "role": item.role.value,
            "content": item.content,
            "name": item.name,
            "metadata": dict(item.metadata),
        }
    if isinstance(item, ModelToolCallItem):
        return {
            "type": "tool_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": dict(item.arguments),
            "metadata": dict(item.metadata),
        }
    if isinstance(item, ModelToolOutputItem):
        return {
            "type": "tool_output",
            "call_id": item.call_id,
            "output": item.output,
            "is_error": item.is_error,
            "artifact_uri": item.artifact_uri,
            "metadata": dict(item.metadata),
        }
    if isinstance(item, ModelCompactionItem):
        return {
            "type": "compaction",
            "payload": item.payload,
            "provider": item.provider,
            "model": item.model,
            "native": item.native,
            "metadata": dict(item.metadata),
        }
    raise TypeError(f"unsupported model input item: {type(item).__name__}")


__all__ = ["ApproximateTokenCounter", "TokenCounter"]
