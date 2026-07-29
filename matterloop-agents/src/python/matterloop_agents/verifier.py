"""实现按明确验收条件独立判定执行结果的模型验证器。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from matterloop_core import ExecutionResult, LoopContext, PlanStep, VerificationResult
from matterloop_models import (
    ContextInputMode,
    MessageRole,
    ModelContextScope,
    ModelMessage,
    ModelRegistry,
    ModelRequest,
)

from matterloop_agents._parsing import (
    context_tenant_id,
    parse_json_object,
    require_boolean,
    require_score,
    require_string,
    string_tuple,
)
from matterloop_agents.config import CriteriaVerifierConfig
from matterloop_agents.errors import AgentModelOutputError
from matterloop_agents.scoring import (
    CriterionAssessment,
    ModelReportedScorer,
    VerificationAssessment,
    VerificationScorer,
)

_VERIFICATION_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "passed": {"type": "boolean"},
        "score": {"type": "number", "minimum": 0, "maximum": 100},
        "feedback": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "failed_criteria": {"type": "array", "items": {"type": "string"}},
        "criterion_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "criterion": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "score": {"type": "number", "minimum": 0, "maximum": 100},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "feedback": {"type": "string"},
                },
                "required": ["criterion", "passed", "score", "evidence", "feedback"],
            },
        },
    },
    "required": [
        "passed",
        "score",
        "feedback",
        "evidence",
        "failed_criteria",
        "criterion_assessments",
    ],
}


class CriteriaVerifier:
    """使用独立模型逐条检查步骤验收条件。

    Args:
        models: 支持热替换的模型注册表。
        config: 验证分数阈值和输出预算配置。
    """

    def __init__(
        self,
        models: ModelRegistry,
        config: CriteriaVerifierConfig,
        scorer: VerificationScorer | None = None,
    ) -> None:
        self._models = models
        self._config = config
        self._scorer = scorer or ModelReportedScorer(pass_score=config.pass_score)

    async def verify(
        self,
        step: PlanStep,
        result: ExecutionResult,
        context: LoopContext,
    ) -> VerificationResult:
        """返回带分数、证据和失败条件的保守验证结论。

        Args:
            step: 正在验收的计划步骤。
            result: Worker 产生的输出和制品引用。
            context: 当前 Loop 运行上下文。

        Returns:
            只有模型声明通过、达到阈值且没有失败条件时才通过的结果。
        """
        request = ModelRequest(
            messages=(
                ModelMessage(
                    MessageRole.DEVELOPER,
                    "You are an independent verifier. Judge only from the provided "
                    "result, artifact references, and acceptance criteria. Never infer "
                    "success without evidence. Every criterion in failed_criteria and "
                    "criterion_assessments must exactly match a provided acceptance "
                    "criterion. Return one criterion_assessments entry per criterion "
                    "with passed, a score from 0 to 100, evidence, and feedback.",
                ),
                ModelMessage(MessageRole.USER, self._verification_payload(step, result, context)),
            ),
            response_schema=_VERIFICATION_SCHEMA,
            response_schema_name="matterloop_verification",
            max_output_tokens=self._config.max_output_tokens,
            usage_scopes=self._usage_scopes(context),
            context_scope=ModelContextScope(
                tenant_id=context_tenant_id(context.request.metadata),
                run_id=context.run_id,
                participant="verifier",
                task_id=step.step_id,
                invocation_id=f"attempt:{context.total_attempts + 1}",
            ),
            context_mode=ContextInputMode.REPLACE,
            metadata={"run_id": context.run_id, "step_id": step.step_id, "agent": "verifier"},
        )
        async with self._models.acquire(self._config.model) as model:
            response = await model.generate(request)
        value = parse_json_object(response.output_text, purpose="verifier")
        score = require_score(value, "score", purpose="verifier")
        failed_criteria = string_tuple(value, "failed_criteria", purpose="verifier")
        evidence = string_tuple(value, "evidence", purpose="verifier")
        criterion_assessments = self._criterion_assessments(value)
        model_passed = require_boolean(value, "passed", purpose="verifier")
        criteria = step.acceptance_criteria or context.request.acceptance_criteria
        scoring = self._scorer.score(
            VerificationAssessment(
                criteria=criteria,
                model_passed=model_passed,
                model_score=score,
                failed_criteria=failed_criteria,
                evidence=evidence,
                criterion_assessments=criterion_assessments,
            )
        )
        return VerificationResult(
            passed=scoring.passed,
            feedback=require_string(value, "feedback", purpose="verifier"),
            score=scoring.score,
            evidence=evidence,
            failed_criteria=scoring.failed_criteria,
        )

    @staticmethod
    def _criterion_assessments(
        value: Mapping[str, object],
    ) -> tuple[CriterionAssessment, ...]:
        """解析可选的逐条件评分，并兼容旧版验证器响应。"""
        raw_items = value.get("criterion_assessments")
        if raw_items is None:
            return ()
        if not isinstance(raw_items, list):
            raise AgentModelOutputError("verifier.criterion_assessments must be an array")
        assessments: list[CriterionAssessment] = []
        for index, raw_item in enumerate(raw_items):
            purpose = f"verifier.criterion_assessments[{index}]"
            if not isinstance(raw_item, Mapping):
                raise AgentModelOutputError(f"{purpose} must be an object")
            assessments.append(
                CriterionAssessment(
                    criterion=require_string(raw_item, "criterion", purpose=purpose),
                    passed=require_boolean(raw_item, "passed", purpose=purpose),
                    score=require_score(raw_item, "score", purpose=purpose),
                    evidence=string_tuple(raw_item, "evidence", purpose=purpose),
                    feedback=require_string(raw_item, "feedback", purpose=purpose),
                )
            )
        return tuple(assessments)

    @staticmethod
    def _usage_scopes(context: LoopContext) -> tuple[str, ...]:
        """读取由组合根显式注入的额度作用域。"""
        raw = context.request.metadata.get("usage_scopes", ())
        if not isinstance(raw, (tuple, list)):
            return ()
        return tuple(item for item in raw if isinstance(item, str) and item.strip())

    @staticmethod
    def _verification_payload(
        step: PlanStep,
        result: ExecutionResult,
        context: LoopContext,
    ) -> str:
        artifacts = [
            {
                "name": artifact.name,
                "uri": artifact.uri,
                "media_type": artifact.media_type,
            }
            for artifact in result.artifacts
        ]
        criteria = step.acceptance_criteria or context.request.acceptance_criteria
        return json.dumps(
            {
                "goal": context.request.goal,
                "step": step.description,
                "acceptance_criteria": list(criteria),
                "execution_output": result.output,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
        )
