"""验收验证器、审查器和适配器测试。"""

from __future__ import annotations

import asyncio

from matterloop_agents import (
    CriteriaCoverageScorer,
    CriteriaVerifier,
    CriteriaVerifierConfig,
    ModelReviewer,
    ModelReviewerConfig,
    ReviewerVerifierAdapter,
    VerificationAssessment,
    WeightedRubricScorer,
)
from matterloop_core import ExecutionResult, LoopContext, LoopRequest, PlanStep
from matterloop_models import (
    FakeModelClient,
    ModelClient,
    ModelLease,
    ModelRegistry,
    ModelResponse,
)


class AcquireTrackingRegistry(ModelRegistry):
    """记录验证类 Agent 的模型租约，并禁止回退到直接查询。"""

    def __init__(self) -> None:
        super().__init__()
        self.acquired: list[str] = []

    def acquire(self, name: str) -> ModelLease:
        """记录并返回查询时刻固定的模型客户端。"""
        self.acquired.append(name)
        return super().acquire(name)

    def get(self, name: str) -> ModelClient:
        """禁止 Agent 绕过事务租约直接查询客户端。"""
        raise AssertionError(f"verification agent used ModelRegistry.get({name!r})")


def test_criteria_verifier_requires_score_and_no_failed_criteria() -> None:
    async def scenario() -> None:
        models = AcquireTrackingRegistry()
        models.register(
            "verifier",
            FakeModelClient(
                [
                    ModelResponse(
                        output_text=(
                            '{"passed":true,"score":95,"feedback":"Tests are missing",'
                            '"evidence":["Output exists"],'
                            '"failed_criteria":["Tests pass"]}'
                        )
                    )
                ]
            ),
        )
        verifier = CriteriaVerifier(models, CriteriaVerifierConfig(model="verifier"))

        verification = await verifier.verify(
            PlanStep(description="Implement the feature", acceptance_criteria=("Tests pass",)),
            ExecutionResult(output="Implementation completed"),
            LoopContext(LoopRequest(goal="Deliver the feature")),
        )

        assert not verification.passed
        assert verification.score == 95
        assert verification.failed_criteria == ("Tests pass",)
        assert models.acquired == ["verifier"]

    asyncio.run(scenario())


def test_criteria_verifier_can_compute_score_from_criterion_coverage() -> None:
    async def scenario() -> None:
        models = AcquireTrackingRegistry()
        models.register(
            "verifier",
            FakeModelClient(
                [
                    ModelResponse(
                        output_text=(
                            '{"passed":true,"score":95,"feedback":"Documentation is missing",'
                            '"evidence":["Core feature executed"],'
                            '"failed_criteria":["Documentation is complete"]}'
                        )
                    )
                ]
            ),
        )
        verifier = CriteriaVerifier(
            models,
            CriteriaVerifierConfig(model="verifier"),
            scorer=CriteriaCoverageScorer(
                pass_score=80,
                weights={"Core feature passes": 4, "Documentation is complete": 1},
            ),
        )

        verification = await verifier.verify(
            PlanStep(
                description="Deliver the feature",
                acceptance_criteria=("Core feature passes", "Documentation is complete"),
            ),
            ExecutionResult(output="Core feature implemented"),
            LoopContext(LoopRequest(goal="Deliver the feature")),
        )

        assert not verification.passed
        assert verification.score == 80
        assert verification.failed_criteria == ("Documentation is complete",)

    asyncio.run(scenario())


def test_criteria_coverage_scorer_caps_unexplained_model_failure() -> None:
    scorer = CriteriaCoverageScorer(pass_score=100)

    result = scorer.score(
        VerificationAssessment(
            criteria=("Answer is complete",),
            model_passed=False,
            model_score=20,
        )
    )

    assert not result.passed
    assert result.score == 20
    assert result.detail["unexplained_failure"] is True


def test_weighted_rubric_scorer_uses_per_criterion_scores() -> None:
    async def scenario() -> None:
        models = AcquireTrackingRegistry()
        models.register(
            "verifier",
            FakeModelClient(
                [
                    ModelResponse(
                        output_text=(
                            '{"passed":true,"score":90,"feedback":"Quality accepted",'
                            '"evidence":["Answer"],"failed_criteria":[],'
                            '"criterion_assessments":['
                            '{"criterion":"Facts are accurate","passed":true,"score":95,'
                            '"evidence":["Facts"],"feedback":"Accurate"},'
                            '{"criterion":"Coverage is complete","passed":true,"score":80,'
                            '"evidence":["Scope"],"feedback":"Mostly complete"}]}'
                        )
                    )
                ]
            ),
        )
        verifier = CriteriaVerifier(
            models,
            CriteriaVerifierConfig(model="verifier"),
            scorer=WeightedRubricScorer(
                pass_score=85,
                weights={"Facts are accurate": 2, "Coverage is complete": 1},
                required_criteria=("Facts are accurate",),
                required_min_score=90,
            ),
        )

        result = await verifier.verify(
            PlanStep(
                description="Answer the question",
                acceptance_criteria=("Facts are accurate", "Coverage is complete"),
            ),
            ExecutionResult(output="Answer"),
            LoopContext(LoopRequest(goal="Answer the question")),
        )

        assert result.passed
        assert result.score == 90
        assert result.failed_criteria == ()

    asyncio.run(scenario())


def test_reviewer_adapter_converts_issues_to_failed_verification() -> None:
    async def scenario() -> None:
        models = AcquireTrackingRegistry()
        models.register(
            "reviewer",
            FakeModelClient(
                [
                    ModelResponse(
                        output_text=(
                            '{"score":88,"summary":"Risk identified","evidence":["Logs"],'
                            '"issues":["Timeout is not handled"],'
                            '"recommendations":["Add timeout handling"]}'
                        )
                    )
                ]
            ),
        )
        adapter = ReviewerVerifierAdapter(
            ModelReviewer(models, ModelReviewerConfig(model="reviewer"))
        )

        verification = await adapter.verify(
            PlanStep(description="Implement the interface"),
            ExecutionResult(output="Completed"),
            LoopContext(LoopRequest(goal="Implement the interface")),
        )

        assert not verification.passed
        assert verification.failed_criteria == ("Timeout is not handled",)
        assert verification.evidence == ("Logs",)
        assert models.acquired == ["reviewer"]

    asyncio.run(scenario())
