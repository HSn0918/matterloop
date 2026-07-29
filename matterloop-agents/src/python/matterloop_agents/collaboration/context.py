"""把 Context 精确版本与 TeamRepository 快照一致保存。"""

from __future__ import annotations

from dataclasses import replace

from matterloop_runtime import ContextLifecycleManager

from matterloop_agents.collaboration.models import TeamSnapshot
from matterloop_agents.collaboration.protocols import TeamRepository


class ContextAwareTeamRepository:
    """在团队仓储读写边界附加或恢复模型 Context 引用。"""

    def __init__(
        self,
        delegate: TeamRepository,
        manager: ContextLifecycleManager,
    ) -> None:
        self._delegate = delegate
        self._manager = manager

    async def create(self, snapshot: TeamSnapshot) -> None:
        """创建初始快照；此时通常尚无模型上下文版本。"""
        await self._delegate.create(self._with_latest_references(snapshot))

    async def load(self, run_id: str) -> TeamSnapshot | None:
        """读取快照并注册其中的精确恢复版本。"""
        snapshot = await self._delegate.load(run_id)
        if snapshot is not None:
            self._manager.register_external_state_references(snapshot.external_state_refs)
        return snapshot

    async def save(
        self,
        snapshot: TeamSnapshot,
        expected_version: int,
    ) -> TeamSnapshot:
        """先写入当前 Context 引用，再执行底层 Team CAS。"""
        return await self._delegate.save(
            self._with_latest_references(snapshot),
            expected_version,
        )

    async def list(self) -> tuple[TeamSnapshot, ...]:
        """列出快照并注册其中的恢复引用。"""
        snapshots = await self._delegate.list()
        for snapshot in snapshots:
            self._manager.register_external_state_references(snapshot.external_state_refs)
        return snapshots

    async def acquire_lease(self, run_id: str, owner_id: str) -> bool:
        """透传团队运行租约获取。"""
        return await self._delegate.acquire_lease(run_id, owner_id)

    async def release_lease(self, run_id: str, owner_id: str) -> None:
        """透传团队运行租约释放。"""
        await self._delegate.release_lease(run_id, owner_id)

    def _with_latest_references(self, snapshot: TeamSnapshot) -> TeamSnapshot:
        retained = tuple(
            reference
            for reference in snapshot.external_state_refs
            if reference.kind != "model_context"
        )
        return replace(
            snapshot,
            external_state_refs=(
                *retained,
                *self._manager.external_state_references(snapshot.run_id),
            ),
        )


__all__ = ["ContextAwareTeamRepository"]
