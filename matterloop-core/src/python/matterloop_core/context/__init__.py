"""Loop 上下文、过程值对象与检查点编解码器。"""

from matterloop_core.context.checkpoint import LoopCheckpointCodec
from matterloop_core.context.human import (
    HumanAction,
    HumanInteractionKind,
    HumanInteractionRecord,
    HumanInteractionRequest,
    HumanResponse,
)
from matterloop_core.context.models import (
    ArtifactRef,
    ExecutionResult,
    ExternalStateRef,
    IterationRecord,
    LoopContext,
    LoopLimits,
    LoopRequest,
    LoopResult,
    Plan,
    PlanStep,
    VerificationResult,
    result_from_context,
)

__all__ = [
    "ArtifactRef",
    "ExecutionResult",
    "ExternalStateRef",
    "HumanAction",
    "HumanInteractionKind",
    "HumanInteractionRecord",
    "HumanInteractionRequest",
    "HumanResponse",
    "IterationRecord",
    "LoopCheckpointCodec",
    "LoopContext",
    "LoopLimits",
    "LoopRequest",
    "LoopResult",
    "Plan",
    "PlanStep",
    "VerificationResult",
    "result_from_context",
]
