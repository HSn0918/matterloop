"""Context 生命周期编排器与 ModelClient 包装器。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from urllib.parse import quote

from matterloop_core import (
    CheckpointPreparer,
    EventPublisher,
    ExternalStateRef,
    LoopContext,
    LoopEvent,
    LoopEventType,
)
from matterloop_models import (
    ContextInputMode,
    ExactTokenCountingClient,
    MessageRole,
    ModelClient,
    ModelCompactionItem,
    ModelContextScope,
    ModelInputItem,
    ModelItemCategory,
    ModelItemRetention,
    ModelMessageItem,
    ModelRequest,
    ModelResponse,
    ModelToolCallItem,
    ModelToolOutputItem,
    NativeCompactingClient,
)

from matterloop_runtime.context.analyzer import ContextAnalyzer
from matterloop_runtime.context.compression import (
    ContextCompactor,
    DefaultToolResultReducer,
    ToolResultReducer,
)
from matterloop_runtime.context.errors import (
    ContextBudgetExceededError,
    ContextSnapshotError,
    IncompatibleContextSnapshotError,
)
from matterloop_runtime.context.events import (
    ContextEvent,
    ContextEventPublisher,
    ContextEventType,
    NullContextEventPublisher,
)
from matterloop_runtime.context.models import (
    ContextBlobStore,
    ContextMemorySink,
    ContextPressure,
    ContextSnapshot,
    ContextSnapshotRef,
    ContextStore,
    ContextTokenState,
    MemoryAdmissionPolicy,
)
from matterloop_runtime.context.policy import ContextPolicy, ContextRetentionPolicy
from matterloop_runtime.context.stores import (
    ContextSnapshotCodec,
    decode_input_items,
    encode_input_items,
)
from matterloop_runtime.context.token import ApproximateTokenCounter, TokenCounter


@dataclass(frozen=True, slots=True)
class PreparedContextRequest:
    """关联发送给供应商的规范请求和已持久化快照。"""

    request: ModelRequest
    snapshot: ContextSnapshot
    snapshot_ref: ContextSnapshotRef


class ContextLifecycleManager:
    """在每次模型调用前后执行检测、外置、压缩和持久化。"""

    def __init__(
        self,
        policy: ContextPolicy,
        store: ContextStore,
        blob_store: ContextBlobStore,
        *,
        semantic_compactor: ContextCompactor | None = None,
        token_counter: TokenCounter | None = None,
        analyzer: ContextAnalyzer | None = None,
        events: ContextEventPublisher | None = None,
        tool_result_reducer: ToolResultReducer | None = None,
        memory_sink: ContextMemorySink | None = None,
        memory_admission: MemoryAdmissionPolicy | None = None,
        retention: ContextRetentionPolicy | None = None,
    ) -> None:
        if policy.memory_extraction_enabled and (memory_sink is None or memory_admission is None):
            raise ValueError(
                "enabled context memory extraction requires a sink and admission policy"
            )
        self._policy = policy
        self._store = store
        self._blob_store = blob_store
        self._semantic_compactor = semantic_compactor
        self._counter = token_counter or ApproximateTokenCounter(policy.estimate_safety_margin)
        self._estimate_counter = ApproximateTokenCounter(policy.estimate_safety_margin)
        self._analyzer = analyzer or ContextAnalyzer()
        self._events = events or NullContextEventPublisher()
        self._tool_result_reducer = tool_result_reducer or DefaultToolResultReducer()
        self._codec = ContextSnapshotCodec()
        self._memory_sink = memory_sink
        self._memory_admission = memory_admission
        self._retention = retention
        self._latest_refs: dict[str, ContextSnapshotRef] = {}
        self._recovery_refs: dict[str, ContextSnapshotRef] = {}

    async def prepare(
        self,
        request: ModelRequest,
        model: ModelClient,
    ) -> PreparedContextRequest:
        """构造并持久化保证在预算内的模型请求。"""
        scope = request.context_scope
        if scope is None:
            raise ValueError("managed model request requires a context scope")
        descriptor = getattr(model, "descriptor", None)
        provider = getattr(descriptor, "provider", type(model).__name__)
        model_name = getattr(descriptor, "model", type(model).__name__)
        descriptor_limit = getattr(descriptor, "context_window_tokens", None)
        limit = self._policy.resolve_limit(descriptor_limit)
        recovery_ref = self._recovery_refs.pop(scope.key, None)
        latest = await self._store.load(scope.key)
        existing = await self.restore(recovery_ref) if recovery_ref is not None else latest
        store_expected_revision = 0 if latest is None else latest.revision
        if existing is not None:
            await self._publish(
                ContextEventType.SNAPSHOT_RESTORED,
                scope.key,
                revision=existing.revision,
                token_state=existing.token_state,
            )

        incoming = self._canonical_items(request)
        incoming = await self._externalize_tool_results(incoming, scope.key)
        if request.context_mode is ContextInputMode.APPEND:
            if existing is None:
                raise ContextSnapshotError(f"cannot append to missing context snapshot {scope.key}")
            existing_items = await self._compatible_existing_items(
                existing,
                provider=provider,
                model_name=model_name,
                scope=scope,
                limit=limit,
            )
            items = (*existing_items, *incoming)
        else:
            items = incoming

        canonical = self._canonical_request(request, items)
        token_state = await self._token_state(canonical, model, limit)
        await self._publish(
            ContextEventType.PRESSURE,
            scope.key,
            revision=0 if existing is None else existing.revision,
            token_state=token_state,
            metadata={"pressure": self._pressure(token_state).value},
        )

        archive_uris = existing.archive_uris if existing is not None else ()
        compaction_count = existing.compaction_count if existing is not None else 0
        if token_state.usage_ratio >= self._policy.soft_threshold:
            (
                items,
                canonical,
                token_state,
                new_archives,
                passes,
            ) = await self._compact_to_budget(
                canonical,
                items,
                model,
                scope,
                token_state,
            )
            archive_uris = (*archive_uris, *new_archives)
            compaction_count += passes

        if token_state.usage_ratio >= self._policy.hard_threshold:
            raise ContextBudgetExceededError(
                f"context {scope.key} uses {token_state.projected_tokens}/"
                f"{token_state.limit_tokens} projected tokens"
            )

        expected = store_expected_revision
        now = datetime.now(timezone.utc)
        snapshot = ContextSnapshot(
            key=scope.key,
            scope=scope,
            revision=expected + 1,
            input_items=items,
            token_state=token_state,
            provider=provider,
            model=model_name,
            compaction_count=compaction_count,
            archive_uris=archive_uris,
            metadata={"last_invocation_id": scope.invocation_id},
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        snapshot_ref = await self._store.save(snapshot, expected_revision=expected)
        self._latest_refs[snapshot.key] = snapshot_ref
        await self._publish(
            ContextEventType.SNAPSHOT_SAVED,
            scope.key,
            revision=snapshot.revision,
            token_state=token_state,
        )
        return PreparedContextRequest(canonical, snapshot, snapshot_ref)

    async def record_response(
        self,
        prepared: PreparedContextRequest,
        response: ModelResponse,
    ) -> ModelResponse:
        """追加模型输出并保存新的不可变快照版本。"""
        output_items = response.output_items or self._response_items(response)
        released = self._release_completed_tool_pairs(prepared.snapshot.input_items)
        items = (*released, *output_items)
        request = replace(prepared.request, input_items=items)
        current = await self._counter.count(request)
        pending = self._pending_tool_count(items) * self._policy.default_tool_output_tokens
        limit = prepared.snapshot.token_state.limit_tokens
        state = ContextTokenState(
            current_tokens=current,
            projected_tokens=current + self._policy.reserved_output_tokens + pending,
            limit_tokens=limit,
        )
        snapshot = replace(
            prepared.snapshot,
            revision=prepared.snapshot.revision + 1,
            input_items=items,
            token_state=state,
            metadata={
                **prepared.snapshot.metadata,
                "last_model_usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cache_hit_tokens": response.usage.cache_hit_tokens,
                    "cache_miss_tokens": response.usage.cache_miss_tokens,
                    "reasoning_tokens": response.usage.reasoning_tokens,
                },
            },
            updated_at=datetime.now(timezone.utc),
        )
        snapshot_ref = await self._store.save(
            snapshot,
            expected_revision=prepared.snapshot.revision,
        )
        self._latest_refs[snapshot.key] = snapshot_ref
        await self._publish(
            ContextEventType.SNAPSHOT_SAVED,
            snapshot.key,
            revision=snapshot.revision,
            token_state=state,
        )
        metadata = dict(response.metadata)
        metadata.update(
            {
                "context_key": snapshot_ref.key,
                "context_revision": snapshot_ref.revision,
                "context_checksum": snapshot_ref.checksum,
            }
        )
        return replace(response, output_items=output_items, metadata=metadata)

    def register_recovery_references(
        self,
        references: Sequence[ContextSnapshotRef],
    ) -> None:
        """注册 Checkpoint 明确指向的精确恢复版本。"""
        for reference in references:
            self._recovery_refs[reference.key] = reference

    def register_external_state_references(
        self,
        references: Sequence[ExternalStateRef],
    ) -> None:
        """把 Core 或 Team Checkpoint 中的通用引用注册为恢复版本。"""
        self.register_recovery_references(
            tuple(
                ContextSnapshotRef(
                    key=reference.key,
                    revision=reference.revision,
                    checksum=reference.checksum,
                    schema_version=reference.schema_version,
                )
                for reference in references
                if reference.kind == "model_context"
            )
        )

    def external_state_references(self, run_id: str) -> tuple[ExternalStateRef, ...]:
        """返回指定运行当前已提交的全部模型上下文引用。"""
        prefix = f":{quote(run_id, safe='-._~')}:"
        references = {**self._latest_refs, **self._recovery_refs}
        return tuple(
            ExternalStateRef(
                kind="model_context",
                key=reference.key,
                revision=reference.revision,
                checksum=reference.checksum,
                schema_version=reference.schema_version,
            )
            for key, reference in sorted(references.items())
            if prefix in key
        )

    async def restore(self, reference: ContextSnapshotRef) -> ContextSnapshot:
        """读取并校验 Checkpoint 指向的精确 Context 版本。"""
        if reference.schema_version != self._codec.schema_version:
            raise ContextSnapshotError("context snapshot reference schema version is unsupported")
        snapshot = await self._store.load(reference.key, reference.revision)
        if snapshot is None:
            raise ContextSnapshotError(
                f"context snapshot {reference.key}@{reference.revision} is missing"
            )
        if self._codec.checksum(snapshot) != reference.checksum:
            raise ContextSnapshotError("context snapshot checksum mismatch")
        return snapshot

    async def _compact_to_budget(
        self,
        request: ModelRequest,
        items: tuple[ModelInputItem, ...],
        model: ModelClient,
        scope: ModelContextScope,
        token_state: ContextTokenState,
    ) -> tuple[
        tuple[ModelInputItem, ...],
        ModelRequest,
        ContextTokenState,
        tuple[str, ...],
        int,
    ]:
        archives: list[str] = []
        passes = 0
        for _ in range(self._policy.max_compaction_passes):
            if token_state.usage_ratio < self._policy.target_threshold:
                break
            before_tokens = token_state.current_tokens
            analysis = self._analyzer.analyze(
                items,
                recent_turns=self._policy.recent_turns,
            )
            if not analysis.summarizable:
                break
            await self._publish(
                ContextEventType.COMPACTION_STARTED,
                scope.key,
                token_state=token_state,
                metadata={"source_items": len(analysis.summarizable)},
            )
            try:
                archive = await self._archive_items(analysis.summarizable)
                compacted = await self._compact_items(
                    analysis.summarizable,
                    request,
                    model,
                    scope,
                    target_tokens=max(
                        1,
                        int(token_state.limit_tokens * self._policy.target_threshold),
                    ),
                )
                candidate = analysis.rebuild(compacted)
                candidate_request = replace(request, input_items=candidate)
                candidate_state = await self._token_state(
                    candidate_request,
                    model,
                    token_state.limit_tokens,
                )
                if candidate_state.current_tokens >= token_state.current_tokens:
                    raise ValueError("compaction did not reduce context tokens")
                await self._extract_memories(compacted, scope)
            except Exception as exc:
                await self._publish(
                    ContextEventType.COMPACTION_FAILED,
                    scope.key,
                    token_state=token_state,
                    metadata={"error_type": type(exc).__name__},
                )
                if token_state.usage_ratio >= self._policy.hard_threshold:
                    raise ContextBudgetExceededError(
                        f"context compaction failed at hard threshold ({type(exc).__name__})"
                    ) from exc
                break
            items = candidate
            request = candidate_request
            token_state = candidate_state
            archives.append(archive)
            passes += 1
            await self._publish(
                ContextEventType.COMPACTION_COMPLETED,
                scope.key,
                token_state=token_state,
                metadata={
                    "pass": passes,
                    "strategy": (
                        "native"
                        if any(
                            isinstance(item, ModelCompactionItem) and item.native
                            for item in compacted
                        )
                        else "semantic"
                    ),
                    "before_tokens": before_tokens,
                    "after_tokens": token_state.current_tokens,
                },
            )
        return items, request, token_state, tuple(archives), passes

    async def _extract_memories(
        self,
        compacted: tuple[ModelInputItem, ...],
        scope: ModelContextScope,
    ) -> None:
        if (
            not self._policy.memory_extraction_enabled
            or self._memory_sink is None
            or self._memory_admission is None
        ):
            return
        for item in compacted:
            if not isinstance(item, ModelCompactionItem) or item.native:
                continue
            try:
                summary = json.loads(item.payload)
            except json.JSONDecodeError:
                continue
            facts = summary.get("facts") if isinstance(summary, dict) else None
            if not isinstance(facts, list) or not all(isinstance(fact, str) for fact in facts):
                continue
            admitted = self._memory_admission.admit(facts)
            if admitted:
                source_ids = item.metadata.get("source_item_ids", ())
                await self._memory_sink.remember(
                    scope,
                    admitted,
                    source_item_ids=tuple(value for value in source_ids if isinstance(value, str))
                    if isinstance(source_ids, (tuple, list))
                    else (),
                )

    async def _compact_items(
        self,
        items: tuple[ModelInputItem, ...],
        request: ModelRequest,
        model: ModelClient,
        scope: ModelContextScope,
        *,
        target_tokens: int,
    ) -> tuple[ModelInputItem, ...]:
        native_error: Exception | None = None
        if isinstance(model, NativeCompactingClient):
            native_request = replace(
                request,
                input_items=items,
                tools=(),
                response_schema=None,
                max_output_tokens=None,
                temperature=None,
                tool_choice=None,
                context_scope=None,
                context_mode=ContextInputMode.REPLACE,
            )
            try:
                result = await model.compact_input(native_request)
                if result:
                    return result
            except Exception as exc:
                native_error = exc
        if self._semantic_compactor is not None:
            self._ensure_semantic_provider(model)
            return await self._semantic_compactor.compact(
                items,
                scope=scope,
                target_tokens=target_tokens,
            )
        if native_error is not None:
            raise native_error
        raise ContextBudgetExceededError("no context compactor is available")

    async def _archive_items(self, items: tuple[ModelInputItem, ...]) -> str:
        payload = json.dumps(
            encode_input_items(items),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        reference = await self._blob_store.put(
            payload,
            media_type="application/json",
            purpose="context-archive",
            ttl_seconds=(None if self._retention is None else self._retention.archive_ttl_seconds),
        )
        return reference.uri

    async def _externalize_tool_results(
        self,
        items: tuple[ModelInputItem, ...],
        context_key: str,
    ) -> tuple[ModelInputItem, ...]:
        reduced: list[ModelInputItem] = []
        for item in items:
            if (
                not isinstance(item, ModelToolOutputItem)
                or item.artifact_uri is not None
                or self._estimate_counter.count_text(item.output)
                <= self._policy.tool_result_inline_tokens
            ):
                reduced.append(item)
                continue
            content = item.output.encode("utf-8")
            try:
                reference = await self._blob_store.put(
                    content,
                    media_type="text/plain; charset=utf-8",
                    purpose="tool-result",
                    ttl_seconds=(
                        None if self._retention is None else self._retention.tool_result_ttl_seconds
                    ),
                )
            except Exception as exc:
                await self._publish(
                    ContextEventType.COMPACTION_FAILED,
                    context_key,
                    metadata={
                        "stage": "tool_result_externalization",
                        "error_type": type(exc).__name__,
                    },
                )
                reduced.append(item)
                continue
            output = self._tool_result_reducer.reduce(
                item.output,
                artifact_uri=reference.uri,
                sha256=reference.sha256,
                size_bytes=reference.size_bytes,
                is_error=item.is_error,
            )
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "externalized": True,
                    "sha256": reference.sha256,
                    "size_bytes": reference.size_bytes,
                }
            )
            reduced.append(
                replace(
                    item,
                    output=output,
                    artifact_uri=reference.uri,
                    retention=ModelItemRetention.EXTERNALIZED,
                    metadata=metadata,
                )
            )
            await self._publish(
                ContextEventType.TOOL_RESULT_EXTERNALIZED,
                context_key,
                metadata={"size_bytes": reference.size_bytes},
            )
        return tuple(reduced)

    async def _token_state(
        self,
        request: ModelRequest,
        model: ModelClient,
        limit: int,
    ) -> ContextTokenState:
        if isinstance(model, ExactTokenCountingClient):
            try:
                current = await model.count_input_tokens(request)
            except Exception:
                current = await self._counter.count(request)
        else:
            current = await self._counter.count(request)
        if current < 0:
            raise ValueError("token counter returned a negative value")
        reserved = request.max_output_tokens or self._policy.reserved_output_tokens
        pending = (
            self._pending_tool_count(request.input_items) * self._policy.default_tool_output_tokens
        )
        return ContextTokenState(
            current_tokens=current,
            projected_tokens=current + reserved + pending,
            limit_tokens=limit,
        )

    def _canonical_items(self, request: ModelRequest) -> tuple[ModelInputItem, ...]:
        if request.input_items:
            return request.input_items
        items: list[ModelInputItem] = []
        first_user = True
        for message in request.messages:
            is_goal = first_user and message.role is MessageRole.USER
            if is_goal:
                items.extend(self._split_goal_message(message.content, message.name))
            else:
                items.append(
                    ModelMessageItem(
                        role=message.role,
                        content=message.content,
                        name=message.name,
                    )
                )
            if message.role is MessageRole.USER:
                first_user = False
        items.extend(
            ModelToolOutputItem(
                call_id=output.call_id,
                output=output.output,
                is_error=output.is_error,
            )
            for output in request.tool_outputs
        )
        return tuple(items)

    @staticmethod
    def _split_goal_message(
        content: str,
        name: str | None,
    ) -> tuple[ModelMessageItem, ...]:
        """把内建 Agent 的 JSON 目标与大历史负载拆成不同保留等级。"""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            return (
                ModelMessageItem(
                    role=MessageRole.USER,
                    content=content,
                    name=name,
                    category=ModelItemCategory.GOAL,
                    retention=ModelItemRetention.PINNED,
                ),
            )
        protected_names = {
            "goal",
            "team_goal",
            "acceptance_criteria",
            "constraints",
            "current_plan",
            "current_state",
            "pending",
            "step",
            "step_id",
            "task",
            "execution",
            "execution_output",
            "draft_output",
            "task_results",
            "dependency_results",
            "artifacts",
        }
        protected = {key: value for key, value in payload.items() if key in protected_names}
        historical = {key: value for key, value in payload.items() if key not in protected_names}
        items = [
            ModelMessageItem(
                role=MessageRole.USER,
                content=json.dumps(protected or payload, ensure_ascii=False, sort_keys=True),
                name=name,
                category=ModelItemCategory.GOAL,
                retention=ModelItemRetention.PINNED,
            )
        ]
        if protected and historical:
            items.append(
                ModelMessageItem(
                    role=MessageRole.USER,
                    content=json.dumps(historical, ensure_ascii=False, sort_keys=True),
                    name=name,
                    category=ModelItemCategory.WORKING_MEMORY,
                    retention=ModelItemRetention.SUMMARIZABLE,
                    metadata={"historical_payload": True},
                )
            )
        return tuple(items)

    @staticmethod
    def _canonical_request(
        request: ModelRequest,
        items: tuple[ModelInputItem, ...],
    ) -> ModelRequest:
        return replace(
            request,
            messages=(),
            input_items=items,
            tool_outputs=(),
            previous_response_id=None,
            continuation=None,
        )

    @staticmethod
    def _response_items(response: ModelResponse) -> tuple[ModelInputItem, ...]:
        items: list[ModelInputItem] = []
        if response.output_text.strip():
            items.append(
                ModelMessageItem(
                    role=MessageRole.ASSISTANT,
                    content=response.output_text,
                    retention=ModelItemRetention.RECENT,
                )
            )
        items.extend(
            ModelToolCallItem(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
            )
            for call in response.tool_calls
        )
        return tuple(items)

    @staticmethod
    def _release_completed_tool_pairs(
        items: tuple[ModelInputItem, ...],
    ) -> tuple[ModelInputItem, ...]:
        output_ids = {item.call_id for item in items if isinstance(item, ModelToolOutputItem)}
        released: list[ModelInputItem] = []
        for item in items:
            completed_tool_item = (
                isinstance(item, (ModelToolCallItem, ModelToolOutputItem))
                and item.call_id in output_ids
            )
            completed_provider_state = (
                isinstance(item, ModelCompactionItem)
                and item.metadata.get("provider_output_item") is True
                and bool(output_ids)
            )
            if (
                completed_tool_item or completed_provider_state
            ) and item.retention is ModelItemRetention.PINNED:
                released.append(replace(item, retention=ModelItemRetention.SUMMARIZABLE))
            else:
                released.append(item)
        return tuple(released)

    @staticmethod
    def _pending_tool_count(items: tuple[ModelInputItem, ...]) -> int:
        calls = {item.call_id for item in items if isinstance(item, ModelToolCallItem)}
        outputs = {item.call_id for item in items if isinstance(item, ModelToolOutputItem)}
        return len(calls - outputs)

    async def _compatible_existing_items(
        self,
        snapshot: ContextSnapshot,
        *,
        provider: str,
        model_name: str,
        scope: ModelContextScope,
        limit: int,
    ) -> tuple[ModelInputItem, ...]:
        native_items = tuple(
            item
            for item in snapshot.input_items
            if (
                isinstance(item, ModelCompactionItem)
                and item.native
                and item.metadata.get("provider_output_item") is not True
            )
        )
        model_matches = snapshot.provider == provider and snapshot.model == model_name
        if model_matches:
            return snapshot.input_items
        portable_items: list[ModelInputItem] = []
        private_keys = {
            "provider_private_state",
            "provider",
            "model",
            "assistant_private_fields",
        }
        for item in snapshot.input_items:
            if (
                isinstance(item, ModelCompactionItem)
                and item.metadata.get("provider_output_item") is True
            ):
                continue
            if (
                isinstance(item, ModelToolCallItem)
                and item.metadata.get("provider_private_state") is True
            ):
                portable_items.append(
                    replace(
                        item,
                        metadata={
                            key: value
                            for key, value in item.metadata.items()
                            if key not in private_keys
                        },
                    )
                )
            else:
                portable_items.append(item)
        without_provider_state = tuple(portable_items)
        if not native_items:
            return without_provider_state
        if self._semantic_compactor is None or not snapshot.archive_uris:
            raise IncompatibleContextSnapshotError(
                f"context {snapshot.key} contains native compaction state for "
                f"{snapshot.provider}/{snapshot.model}; archived history and a semantic "
                "compactor are required to switch models"
            )
        self._ensure_semantic_provider_name(provider)
        archived: list[ModelInputItem] = []
        seen: set[str] = set()
        try:
            for uri in snapshot.archive_uris:
                payload = await self._blob_store.get(uri)
                value = json.loads(payload)
                if not isinstance(value, list):
                    raise ValueError("context archive root must be an array")
                for item in decode_input_items(value):
                    if item.item_id not in seen:
                        archived.append(item)
                        seen.add(item.item_id)
        except Exception as exc:
            raise IncompatibleContextSnapshotError(
                f"context {snapshot.key} archived history cannot be restored"
            ) from exc
        if not archived:
            raise IncompatibleContextSnapshotError(
                f"context {snapshot.key} archived history is empty"
            )
        compacted = await self._semantic_compactor.compact(
            tuple(archived),
            scope=scope,
            target_tokens=max(1, int(limit * self._policy.target_threshold)),
        )
        native_ids = {item.item_id for item in native_items}
        rebuilt: list[ModelInputItem] = []
        inserted = False
        for item in without_provider_state:
            if item.item_id in native_ids:
                if not inserted:
                    rebuilt.extend(compacted)
                    inserted = True
                continue
            rebuilt.append(item)
        return tuple(rebuilt)

    def _ensure_semantic_provider(self, model: ModelClient) -> None:
        descriptor = getattr(model, "descriptor", None)
        provider = getattr(descriptor, "provider", type(model).__name__)
        self._ensure_semantic_provider_name(provider)

    def _ensure_semantic_provider_name(self, provider: object) -> None:
        summary_provider = getattr(self._semantic_compactor, "provider", None)
        if (
            not self._policy.allow_cross_provider_summary
            and isinstance(provider, str)
            and isinstance(summary_provider, str)
            and provider != summary_provider
        ):
            raise IncompatibleContextSnapshotError(
                "semantic compactor provider differs from the primary model; "
                "set allow_cross_provider_summary=True to opt in"
            )

    def _pressure(self, state: ContextTokenState) -> ContextPressure:
        if state.usage_ratio >= self._policy.hard_threshold:
            return ContextPressure.HARD
        if state.usage_ratio >= self._policy.soft_threshold:
            return ContextPressure.SOFT
        return ContextPressure.NORMAL

    async def _publish(
        self,
        event_type: ContextEventType,
        key: str,
        *,
        revision: int = 0,
        token_state: ContextTokenState | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        state = token_state or ContextTokenState(0, 0, 1)
        await self._events.publish(
            ContextEvent(
                event_type=event_type,
                context_key=key,
                revision=revision,
                current_tokens=state.current_tokens,
                projected_tokens=state.projected_tokens,
                limit_tokens=state.limit_tokens,
                metadata=metadata or {},
            )
        )


class ContextManagedModelClient:
    """只管理带 Context Scope 的请求，其余请求透明转发。"""

    def __init__(self, client: ModelClient, manager: ContextLifecycleManager) -> None:
        self._client = client
        self._manager = manager

    @property
    def descriptor(self) -> object:
        """透传底层客户端的非敏感模型描述。"""
        return getattr(self._client, "descriptor", None)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """在底层模型调用前后维护持久化上下文。"""
        if request.context_scope is None or request.metadata.get("context_lifecycle_internal"):
            return await self._client.generate(request)
        prepared = await self._manager.prepare(request, self._client)
        response = await self._client.generate(prepared.request)
        return await self._manager.record_response(prepared, response)

    async def aclose(self) -> None:
        """底层客户端支持关闭时透传资源生命周期。"""
        closer = getattr(self._client, "aclose", None)
        if callable(closer):
            await closer()


class ContextCheckpointEventPublisher:
    """把 Context 精确版本加入 Core Checkpoint，同时透传生命周期事件。"""

    def __init__(
        self,
        delegate: EventPublisher,
        manager: ContextLifecycleManager,
    ) -> None:
        self._delegate = delegate
        self._manager = manager
        self._registered_runs: set[str] = set()

    async def publish(self, event: LoopEvent) -> None:
        """在恢复事件首次出现时注册 Checkpoint 中的 Context 引用。"""
        if event.context.run_id not in self._registered_runs:
            self._manager.register_external_state_references(event.context.external_state_refs)
            self._registered_runs.add(event.context.run_id)
        await self._delegate.publish(event)

    async def prepare_checkpoint(
        self,
        context: LoopContext,
        event_types: tuple[LoopEventType, ...],
    ) -> None:
        """先保留下游准备逻辑，再写入当前已提交的 Context 引用。"""
        if context.run_id not in self._registered_runs:
            self._manager.register_external_state_references(context.external_state_refs)
            self._registered_runs.add(context.run_id)
        if isinstance(self._delegate, CheckpointPreparer):
            await self._delegate.prepare_checkpoint(context, event_types)
        retained = [
            reference
            for reference in context.external_state_refs
            if reference.kind != "model_context"
        ]
        context.external_state_refs[:] = [
            *retained,
            *self._manager.external_state_references(context.run_id),
        ]


__all__ = [
    "ContextCheckpointEventPublisher",
    "ContextLifecycleManager",
    "ContextManagedModelClient",
    "PreparedContextRequest",
]
