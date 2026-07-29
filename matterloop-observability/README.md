简体中文 | [English](https://github.com/huleidada/matterloop/blob/main/matterloop-observability/README.en.md)

# matterloop-observability

MatterLoop 的事件是业务事实，不是日志字符串。`matterloop-observability` 把 Core `LoopEvent`
接到日志、指标、树形 Trace 和评分，同时把日志后端与 OpenTelemetry 的进程级配置留给宿主应用。

```bash
pip install matterloop-observability
# 需要 OpenTelemetry 导出时（含 SDK 与 OTLP/HTTP Exporter）
pip install "matterloop-observability[otel]"
```

## 一次合理的装配

```python
import logging

from matterloop_observability import (
    CompositeEventPublisher,
    HandlerEventPublisher,
    MetricsHandler,
    PublisherFailureMode,
    Redactor,
    StructuredLoggingHandler,
)

redactor = Redactor(extra_fields=("tenant_secret", "session_credential"))
metrics = MetricsHandler()

events = CompositeEventPublisher(
    publishers=(
        HandlerEventPublisher(
            StructuredLoggingHandler(
                logger=logging.getLogger("app.matterloop.audit"),
                redactor=redactor,
            )
        ),
        HandlerEventPublisher(metrics),
    ),
    failure_mode=PublisherFailureMode.RAISE,
)
```

将 `events` 注入 `AgentLoop(events=...)`。处理器按顺序执行；同步处理器不创建后台队列，也不接管
Logger 或其关闭流程。唯一的例外是下文用于 Trace 导出的 `BatchingPipeline`：它持有后台守护线程，
由调用方在应用退出前 `shutdown()`。

## 失败策略要显式选择

`CompositeEventPublisher(publishers, failure_mode)` 支持两种策略：

- `LOG_AND_CONTINUE` 是默认值，适合可丢失的遥测。单个发布器失败会记录异常，并继续发布后续事件。
- `RAISE` 在第一个失败处停止，适合审计不可缺失的场景。代价是可观测性故障可能中断业务闭环。

如果要求“状态提交与审计记录同时成功”，顺序调用几个 Publisher 并不能提供事务保证。应使用
Outbox、持久化事件表或消息系统完成原子交接。

## 日志里有什么

`StructuredLoggingHandler(logger, redactor)` 输出单行 JSON，包含事件类型、`run_id`、Loop 状态、
发生时间、事件说明和请求 metadata。默认 Logger 名称是 `matterloop.events`；日志格式、轮转、
保留期和访问控制仍由应用配置。

`Redactor(extra_fields)` 会递归检查映射键，默认识别 `token`、`authorization`、`cookie`、
`api_key`、`password` 和 `secret`，也能命中 `access_token` 之类的前后缀名称。它不会扫描自由文本：
提示词、模型输出、URL 查询参数和异常堆栈里的秘密仍可能泄漏。不要把凭据放进 `goal`、`detail`
或任意字符串 metadata。

## 指标与 Trace

- `MetricsHandler` 保存当前进程内的事件计数，适合测试和轻量诊断。
- `OpenTelemetryMetricsHandler` 写入 `matterloop.loop.events`，只附带事件类型和 Loop 状态。
- `TracingHandler` 已废弃：它为每个事件创建孤立的短 Span，无法还原父子关系，请改用下文的
  `TraceBuilder`，它会在后续版本移除。

`OpenTelemetryMetricsHandler` 与 `TracingHandler` 只使用 API，宿主必须先配置 SDK、Exporter、采样
和资源属性，构造时缺少依赖会立即抛出 `RuntimeError`。`OtelExporter` 例外：它自带 SDK 与 OTLP/HTTP
Exporter（由 `[otel]` extra 提供），缺少依赖时构造抛出 `ImportError`。

## 树形 Trace 与评分

`TraceBuilder(pipeline)` 实现 Core `EventPublisher` 协议，把生命周期事件流重建为树形跨度结构：
根跨度覆盖整个运行，执行、验证、迭代快照和整体完成度验收各成跨度；验证跨度关闭时会把
`VerificationResult.score`（0–100）归一提取为 `Score`。已关闭的跨度和评分进入
`BatchingPipeline(exporter, flush_at, flush_interval)`，由后台守护线程聚批后交给 `SpanExporter`。

```python
from matterloop_observability import (
    BatchingPipeline,
    CompositeEventPublisher,
    JsonlExporter,
    PublisherFailureMode,
    TraceBuilder,
)

pipeline = BatchingPipeline(
    JsonlExporter("traces.jsonl"),
    flush_at=50,
    flush_interval=5.0,
)
trace_builder = TraceBuilder(pipeline)
events = CompositeEventPublisher(
    publishers=(audit_publisher, trace_builder),
    failure_mode=PublisherFailureMode.RAISE,
)
# 应用退出前：pipeline.shutdown()
```

`JsonlExporter(path)` 每行追加一个带 `type` 字段的 JSON 记录，零额外依赖。`OtelExporter(endpoint)`
按原父子关系和起止时间把跨度重建到 OTLP/HTTP 后端，评分导出为同一 trace 下名为 `score:<name>`
的瞬时子跨度。实际 OTel trace/span ID 由 SDK 生成，MatterLoop 的 `run_id`、`span_id` 和父标识分别
保存在 `matterloop.trace_id`、`matterloop.span_id`、`matterloop.parent_span_id` 属性中。流水线队列有界
（默认 10000），满时丢新并告警；OTel 需等待根跨度到达以建立公开 API 的父子 context，单运行暂存同样
默认最多 10000 条，超过后丢新并告警。导出失败重试一次后丢弃，任何异常都不会抛回 Loop 主流程。

`SpanRecord` 是不可变跨度记录：`trace_id`（即产生跨度的 `run_id`）、`span_id`、`parent_span_id`、
`name`、`observation_type`、`started_at`、`ended_at`、`attributes`、`level` 和 `status_message`。
`Score` 是不可变评分：`name`、`value`（NUMERIC 归一到 0–1）、`data_type`、`source`、`run_id`、
`step_id`、`comment`、`evidence` 和 `timestamp`。`score_from_verification` 完成验证结论到 NUMERIC
评分的映射；`score_from_review` 接受具备 `score`/`summary`/`evidence` 属性的鸭子类型审查结论，
不要求安装 agents 组件。

## 生产环境：与数据库共用一条实时 OTel Trace

如果应用还要为 SQLAlchemy、HTTP 客户端或消息队列做 OTel 自动埋点，最佳实践是**由应用只创建一个
`TracerProvider`**：先把它设置为全局 Provider，再把同一实例传给 `OtelExporter`。production preset
识别到 `OtelExporter` 后，会在 Loop 执行时实时创建 `matterloop.run`、Planner/Executor/Verifier 和
`matterloop.generation` Span；工具调用还会创建 `matterloop.tool`。自动 instrumentation 产生的
数据库/HTTP Span 会继承当前阶段，成为同一 Trace 的子节点。
阻塞或暂停前，实时发布器会把当前 `matterloop.run` 的标准 W3C `traceparent`/`tracestate` 写入同一次
checkpoint CAS。恢复时从这个上下文创建新的真实子 Span：人工等待不计入执行时长，跨进程恢复仍在同一条
Trace 中，并且后端能找到已导出的真实父节点。checkpoint 只保存 `traceparent`/`tracestate`，不会持久化
W3C baggage，避免业务元数据被带入存储。`run_id` 只作为业务关联标识和查询属性，不决定 OTel Trace ID。

```bash
pip install "matterloop-observability[otel]" opentelemetry-instrumentation-sqlalchemy
```

```python
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from matterloop_observability import OtelExporter
from matterloop_presets import build_production_runtime

provider = TracerProvider(Resource.create({"service.name": "my-agent-service"}))
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]))
)
# 一个进程只能设置一次；必须在初始化框架和自动 instrumentation 前完成。
trace.set_tracer_provider(provider)
SQLAlchemyInstrumentor().instrument(engine=engine)

runtime = build_production_runtime(
    model=model_client,
    config=production_config,
    queue_backend=queue_backend,
    run_repository=run_repository,
    checkpoint_store=checkpoint_store,
    audit_publisher=audit_publisher,
    trace_exporter=OtelExporter(tracer_provider=provider),
    tools=production_tools,
)
```

`await runtime.aclose()` 会对共享 Provider 执行 `force_flush()`，但不会关闭它；随后由应用对自己拥有的
Provider 执行 `provider.shutdown()`。不要把 `OtelExporter(endpoint=...)` 自行创建的内部 Provider 用于
数据库自动埋点：
它没有注册成全局 Provider，数据库 Span 会落到另一条 Trace（或被默认 no-op Provider 丢弃）；production
preset 会为这种配置记录警告，并在 runtime 关闭时 flush 和 shutdown 这个内部 Provider。

## 工具调用跨度

`OpenTelemetryToolMiddleware(provider)` 通过 `ToolRegistry(middleware=...)`
记录工具查找、权限判断和真实执行，因此成功、`ToolResult.is_error`、权限拒绝、工具不存在和异常都有
`matterloop.tool` Span。Worker 会把模型已有的 `ToolCall.call_id` 透传为
`matterloop.tool_call_id`；直接调用缺少 ID 时中间件才生成新的 UUID。

默认不会把参数、自由文本结果或 Skill 正文写入 Trace；arguments/result 仅记录 UTF-8 字节数与 SHA-256。
显式设置 `capture_tool_payloads=True` 后，才会按原文记录每项最多 4096 个 UTF-8 字节的预览，并可用
`capture_max_body_bytes` 调整。该开关会把凭据、PII 或不可信内容交给 Trace 后端，必须仅在访问控制和
保留策略已就绪时开启。Skill 的 name/version/operation/sha256/trust 和 MCP 的
server/tool/content_blocks/truncated 来自白名单元数据；不会把任意 `ToolResult.metadata` 整体写入 Span，
也不会把普通 Tool 的 `truncated` 写成 MCP 属性。production preset 传入 `OtelExporter` 时自动安装该
middleware，并通过 `tools=` 显式声明默认执行器的工具 allowlist。runtime 关闭会先等待在途 Loop 和工具调用
结束，再 flush Provider。

## 模型调用跨度

```python
from matterloop_observability import wrap_model_client

client = wrap_model_client(model_client, trace_builder)
```

`TracedModelClient(client, trace_builder, pipeline)` 可包装任意 `ModelClient`：请求 metadata 含
`run_id` 时记录一个固定名为 `matterloop.generation` 的跨度，内容包含脱敏后的输入消息、采样参数、
输出文本和六项 Token
用量，父跨度由 `trace_builder` 按 `run_id`/`step_id` 解析，解析不到时挂到运行根跨度；metadata
缺少 `run_id` 时直接透传，观测永远不会阻断调用。模型异常会记录 ERROR 跨度并原样继续抛出。
agents 组件的 Planner、Worker、Verifier 和 Reviewer 已在请求 metadata 中写入 `run_id`、`step_id`
和 `agent`，包装注册进 `ModelRegistry` 的客户端即可自动获得模型跨度；production preset 可通过
`trace_exporter` 参数一键完成这套装配。传入普通 `SpanExporter` 时使用离线 `TracedModelClient`；传入
共享 Provider 的 `OtelExporter` 时使用实时 `OpenTelemetryModelClient`，generation 会嵌套在对应阶段下。
见 [matterloop-presets](../matterloop-presets/README.md)。

## 扩展方式

同步或异步 callable 可用 `HandlerEventPublisher(handler)` 接入。跨度与评分的批量、重试和背压已由
`BatchingPipeline` 提供；需要自定义事件去向时，直接实现 Core `EventPublisher.publish(event)`，
在实现内部管理有界队列和关闭流程。

## TeamLoop 与子 Agent Span

当 `LoopAgentEndpoint` 使用由 `build_production_runtime(..., trace_exporter=OtelExporter(...))`
创建的 `worker_runtime` 时，Team tracing 会自动加载，无需再次订阅事件或配置 `task_middleware`。
Runtime 会复用同一个 `TracerProvider`，生成
`matterloop.team -> matterloop.team.agent -> matterloop.run`，再继续嵌套子 Loop 的
phase/generation/tool Span。

```python
from matterloop_agents.collaboration import LoopAgentEndpoint, TeamOrchestratorComponents

directory.register(LoopAgentEndpoint(agent_spec, production_runtime.worker_runtime))

components = TeamOrchestratorComponents(
    # planner, agents, selection_policy, verifier, approval_gate, repository, aggregator ...
    events=team_events,
)
```

自动发现发生在 `TeamOrchestrator` 构造时，因此应先注册 Endpoint、再构造控制器。多个子 Runtime
应共享应用的 Provider；若存在多套 Provider，控制器按 `agent_id` 稳定选择第一套记录 Team/Agent
Span，其余远端子 Runtime 仍通过 W3C 载体恢复父节点。自定义 Runtime 未暴露该能力时，仍可显式使用
`OpenTelemetryTeamTracePublisher` 和 `OpenTelemetryTeamTaskMiddleware`；显式
`task_middleware` 会覆盖自动配置，避免重复 Span。此时还应把同一个
`OpenTelemetryTeamTracePublisher` 传给 `snapshot_preparer`，确保 Team 根 Span 的 W3C carrier
与 Team 快照同次 CAS 保存。

Team 事件发布器不会把根 Span 长期 attach 到事件发布 task；每个子 Agent Span 都从快照中的 carrier
显式恢复父节点，因此 timeout 创建的 task、取消 shield 和远端执行不会跨 task 解绑 ContextVar token。
暂停或阻塞会先保存 carrier 再结束当前 Team segment；另一个进程 resume 时以该 carrier 创建真实子
segment，整次协作仍保持同一 Trace。

Team Span 默认只写 `team_run_id`、状态、停止原因、task/agent/attempt 与结果状态；不会写 Team
goal、任务描述、输出、人工反馈、异常消息或任意 metadata。中间件只向子 Endpoint 传递标准 W3C
`traceparent`/`tracestate`，不传递 baggage。`LoopAgentEndpoint` 将该载体置于子 `LoopRequest.metadata`
的 `propagation_context`；相同进程的 asyncio 调用天然继承当前上下文，远端 runtime 则须原样转发该
metadata，并在其子 Loop 中使用 `OpenTelemetryTracePublisher` 以恢复真实父节点。

`AsyncTeamRuntime.aclose()` 会先拒绝新 Team 调用并等待在途 `run`、`resume`、人工响应操作结束，随后
才按逆序关闭资源；因此把 child runtime/exporter 放入其 `resources` 后，不会在子 Agent Span 结束前
提前 flush/shutdown Provider。生产拓扑和关闭顺序见[企业集成指南](../docs/enterprise-integration.md)。
