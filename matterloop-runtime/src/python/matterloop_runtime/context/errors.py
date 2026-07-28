"""Context Lifecycle Engine 的安全边界异常。"""


class ContextLifecycleError(RuntimeError):
    """所有 Context 生命周期失败的基类。"""


class ContextBudgetExceededError(ContextLifecycleError):
    """上下文超过硬阈值且无法安全压缩。"""


class ContextConflictError(ContextLifecycleError):
    """ContextStore 中的版本与调用方预期不一致。"""


class ContextSnapshotError(ContextLifecycleError):
    """持久化快照损坏、缺失或无法校验。"""


class IncompatibleContextSnapshotError(ContextLifecycleError):
    """快照包含不能交给当前模型的供应商绑定状态。"""


__all__ = [
    "ContextBudgetExceededError",
    "ContextConflictError",
    "ContextLifecycleError",
    "ContextSnapshotError",
    "IncompatibleContextSnapshotError",
]
