"""定义验证评分的策略契约与内置实现。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Protocol


def _validate_score(value: float, field_name: str) -> None:
    """校验统一的零到一百分值。"""
    if not isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 100")


@dataclass(frozen=True, slots=True)
class CriterionAssessment:
    """保存验证模型对单个验收条件的结构化判断。

    Args:
        criterion: 必须逐字对应输入验收条件的文本。
        passed: 该条件是否达到最低验收要求。
        score: 该条件零到一百的质量分数。
        evidence: 支持判断的证据条目。
        feedback: 针对该条件的简短说明。
    """

    criterion: str
    passed: bool
    score: float
    evidence: tuple[str, ...] = ()
    feedback: str = ""

    def __post_init__(self) -> None:
        """校验单项评分和证据。"""
        if not self.criterion.strip():
            raise ValueError("criterion assessment name must not be empty")
        _validate_score(self.score, "criterion assessment score")
        if any(not item.strip() for item in self.evidence):
            raise ValueError("criterion assessment evidence must not contain empty values")


@dataclass(frozen=True, slots=True)
class VerificationAssessment:
    """封装验证模型提供的原始判断材料。

    Args:
        criteria: 本次需要逐项检查的验收条件。
        model_passed: 验证模型报告的整体通过标记。
        model_score: 验证模型报告的建议分数。
        failed_criteria: 验证模型明确报告的失败条件。
        evidence: 验证模型引用的证据。
        criterion_assessments: 验证模型提供的逐条件结构化判断。
    """

    criteria: tuple[str, ...]
    model_passed: bool
    model_score: float
    failed_criteria: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    criterion_assessments: tuple[CriterionAssessment, ...] = ()

    def __post_init__(self) -> None:
        """拒绝空条件、空证据和越界分数。"""
        if any(not criterion.strip() for criterion in self.criteria):
            raise ValueError("verification criteria must not contain empty values")
        if any(not criterion.strip() for criterion in self.failed_criteria):
            raise ValueError("failed criteria must not contain empty values")
        if any(not item.strip() for item in self.evidence):
            raise ValueError("verification evidence must not contain empty values")
        criterion_names = tuple(item.criterion for item in self.criterion_assessments)
        if len(set(criterion_names)) != len(criterion_names):
            raise ValueError("criterion assessments must not contain duplicate criteria")
        _validate_score(self.model_score, "model score")


@dataclass(frozen=True, slots=True)
class VerificationScore:
    """保存评分策略产生的最终可审计结论。

    Args:
        passed: 评分策略最终是否允许步骤通过。
        score: 零到一百之间的最终分数。
        failed_criteria: 最终保留的失败条件。
        detail: 可观测但不参与内核控制流的评分明细。
    """

    passed: bool
    score: float
    failed_criteria: tuple[str, ...] = ()
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验评分结果并冻结明细。"""
        _validate_score(self.score, "verification score")
        if any(not criterion.strip() for criterion in self.failed_criteria):
            raise ValueError("failed criteria must not contain empty values")
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))


class VerificationScorer(Protocol):
    """把验证模型的原始判断转换为最终分数和通过状态。"""

    def score(self, assessment: VerificationAssessment) -> VerificationScore:
        """根据明确策略计算最终验证分数。"""
        ...


class ModelReportedScorer:
    """保留 MatterLoop 既有的模型报告分数语义。

    Args:
        pass_score: 允许通过的最低分数。
    """

    def __init__(self, pass_score: float = 80.0) -> None:
        _validate_score(pass_score, "pass score")
        self._pass_score = pass_score

    def score(self, assessment: VerificationAssessment) -> VerificationScore:
        """应用模型标记、模型分数阈值和失败条件三重门禁。"""
        passed = (
            assessment.model_passed
            and assessment.model_score >= self._pass_score
            and not assessment.failed_criteria
        )
        return VerificationScore(
            passed=passed,
            score=assessment.model_score,
            failed_criteria=assessment.failed_criteria,
            detail={
                "strategy": "model_reported",
                "pass_score": self._pass_score,
            },
        )


class CriteriaCoverageScorer:
    """按验收条件权重计算可解释的覆盖率分数。

    模型只负责报告哪些条件失败；最终分数由本策略确定。没有验收条件时退回模型
    建议分数。出现不属于输入条件的失败项时采取保守策略，分数记为零。

    Args:
        pass_score: 允许通过的最低覆盖率分数。
        weights: 按验收条件原文配置的正权重；未配置条件使用权重一。
    """

    def __init__(
        self,
        pass_score: float = 80.0,
        *,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        _validate_score(pass_score, "pass score")
        normalized_weights = dict(weights or {})
        for criterion, weight in normalized_weights.items():
            if not criterion.strip():
                raise ValueError("criterion weight keys must not be empty")
            if not isfinite(weight) or weight <= 0:
                raise ValueError("criterion weights must be finite and greater than zero")
        self._pass_score = pass_score
        self._weights = MappingProxyType(normalized_weights)

    def score(self, assessment: VerificationAssessment) -> VerificationScore:
        """根据失败条件占用的权重计算最终覆盖率。"""
        failed_criteria = _effective_failed_criteria(assessment)
        if not assessment.criteria:
            passed = (
                assessment.model_passed
                and assessment.model_score >= self._pass_score
                and not failed_criteria
            )
            return VerificationScore(
                passed=passed,
                score=assessment.model_score,
                failed_criteria=failed_criteria,
                detail={
                    "strategy": "criteria_coverage",
                    "fallback": "model_reported",
                    "pass_score": self._pass_score,
                },
            )

        criterion_names = set(assessment.criteria)
        unknown_failures = tuple(
            criterion for criterion in failed_criteria if criterion not in criterion_names
        )
        total_weight = sum(self._weights.get(criterion, 1.0) for criterion in assessment.criteria)
        failed_weight = sum(
            self._weights.get(criterion, 1.0)
            for criterion in assessment.criteria
            if criterion in failed_criteria
        )
        coverage_score = 100.0 * (total_weight - failed_weight) / total_weight
        if unknown_failures:
            coverage_score = 0.0
        unexplained_failure = not assessment.model_passed and not failed_criteria
        if unexplained_failure:
            coverage_score = min(coverage_score, assessment.model_score)
        all_satisfied = not failed_criteria
        passed = (
            assessment.model_passed
            and coverage_score >= self._pass_score
            and not unknown_failures
            and all_satisfied
        )
        return VerificationScore(
            passed=passed,
            score=coverage_score,
            failed_criteria=failed_criteria,
            detail={
                "strategy": "criteria_coverage",
                "pass_score": self._pass_score,
                "total_weight": total_weight,
                "failed_weight": failed_weight,
                "unknown_failures": unknown_failures,
                "unexplained_failure": unexplained_failure,
            },
        )


class WeightedRubricScorer:
    """按逐条件质量分数计算语义任务的加权量表分。

    模型提供 ``criterion_assessments`` 时使用每项分数；旧模型没有逐项输出时，
    未失败条件退回整体模型分数，保证协议向后兼容。所有明确失败项仍是硬否决。

    Args:
        pass_score: 最终加权分的通过阈值。
        weights: 各验收条件的正权重。
        required_criteria: 必须达到单项最低分的关键条件。
        required_min_score: 关键条件的最低单项分数。
    """

    def __init__(
        self,
        pass_score: float = 80.0,
        *,
        weights: Mapping[str, float] | None = None,
        required_criteria: tuple[str, ...] = (),
        required_min_score: float = 70.0,
    ) -> None:
        _validate_score(pass_score, "pass score")
        _validate_score(required_min_score, "required criterion minimum score")
        normalized_weights = dict(weights or {})
        for criterion, weight in normalized_weights.items():
            if not criterion.strip():
                raise ValueError("rubric weight keys must not be empty")
            if not isfinite(weight) or weight <= 0:
                raise ValueError("rubric weights must be finite and greater than zero")
        if any(not criterion.strip() for criterion in required_criteria):
            raise ValueError("required rubric criteria must not contain empty values")
        self._pass_score = pass_score
        self._weights = MappingProxyType(normalized_weights)
        self._required_criteria = required_criteria
        self._required_min_score = required_min_score

    def score(self, assessment: VerificationAssessment) -> VerificationScore:
        """计算逐条件加权分，并应用关键条件门禁。"""
        if not assessment.criteria:
            return ModelReportedScorer(self._pass_score).score(assessment)

        assessment_by_criterion = {
            item.criterion: item for item in assessment.criterion_assessments
        }
        unknown_assessments = tuple(
            criterion
            for criterion in assessment_by_criterion
            if criterion not in assessment.criteria
        )
        failed_criteria = list(_effective_failed_criteria(assessment))
        criterion_scores: dict[str, float] = {}
        total_weight = 0.0
        weighted_score = 0.0
        for criterion in assessment.criteria:
            item = assessment_by_criterion.get(criterion)
            if item is not None:
                item_score = item.score
                if not item.passed and criterion not in failed_criteria:
                    failed_criteria.append(criterion)
            elif criterion in failed_criteria:
                item_score = 0.0
            else:
                item_score = assessment.model_score
            weight = self._weights.get(criterion, 1.0)
            criterion_scores[criterion] = item_score
            total_weight += weight
            weighted_score += item_score * weight

        score = weighted_score / total_weight
        required_failures = tuple(
            criterion
            for criterion in self._required_criteria
            if criterion not in criterion_scores
            or criterion_scores[criterion] < self._required_min_score
        )
        for criterion in required_failures:
            if criterion not in failed_criteria:
                failed_criteria.append(criterion)
        if unknown_assessments:
            score = 0.0
        if not assessment.model_passed and not failed_criteria:
            score = min(score, assessment.model_score)
        final_failures = tuple(failed_criteria)
        passed = (
            assessment.model_passed
            and score >= self._pass_score
            and not final_failures
            and not unknown_assessments
        )
        return VerificationScore(
            passed=passed,
            score=score,
            failed_criteria=final_failures,
            detail={
                "strategy": "weighted_rubric",
                "pass_score": self._pass_score,
                "criterion_scores": MappingProxyType(criterion_scores),
                "weights": self._weights,
                "required_criteria": self._required_criteria,
                "required_min_score": self._required_min_score,
                "unknown_assessments": unknown_assessments,
            },
        )


def _effective_failed_criteria(
    assessment: VerificationAssessment,
) -> tuple[str, ...]:
    """合并整体失败列表与逐条件失败标记，并保持原始顺序。"""
    failures = list(assessment.failed_criteria)
    for item in assessment.criterion_assessments:
        if not item.passed and item.criterion not in failures:
            failures.append(item.criterion)
    return tuple(failures)
