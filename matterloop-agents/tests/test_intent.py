"""意图识别通用契约测试。"""

from __future__ import annotations

import pytest
from matterloop_agents import (
    ConfidenceIntentResolver,
    IntentCandidate,
    IntentDecision,
    IntentEffect,
    IntentRequest,
)


def test_intent_values_freeze_context_and_expose_side_effect_boundary() -> None:
    context = {"session_id": "s-1"}
    request = IntentRequest(text="run calculation", context=context)
    decision = IntentDecision(
        name="compute",
        effect=IntentEffect.WRITE,
        confidence=0.95,
        reason="Matched an explicit execution action",
        matched_signals=("run calculation",),
        attributes={"requires_receipt": True},
    )
    context["session_id"] = "changed"

    assert request.context["session_id"] == "s-1"
    assert decision.allows_side_effects
    assert decision.attributes["requires_receipt"] is True

    with pytest.raises(TypeError):
        request.context["new"] = "value"  # type: ignore[index]


def test_intent_decision_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        IntentDecision(
            name="answer",
            effect=IntentEffect.ANSWER,
            confidence=1.1,
            reason="invalid",
        )


def test_confidence_resolver_preserves_secondary_intents() -> None:
    resolver = ConfidenceIntentResolver()

    resolution = resolver.resolve(
        (
            IntentCandidate(
                name="capability",
                effect=IntentEffect.READ,
                confidence=0.95,
                priority=100,
                reason="Capability query",
                matched_signals=("supported capabilities",),
            ),
            IntentCandidate(
                name="compute",
                effect=IntentEffect.WRITE,
                confidence=0.85,
                priority=80,
                reason="Contains a compute-domain term",
                matched_signals=("compute",),
            ),
        )
    )

    assert resolution.primary.name == "capability"
    assert resolution.secondary[0].name == "compute"
    assert not resolution.ambiguous


def test_confidence_resolver_requires_clarification_for_tied_candidates() -> None:
    resolver = ConfidenceIntentResolver(ambiguity_margin=0.05)

    resolution = resolver.resolve(
        (
            IntentCandidate(
                name="read_result",
                effect=IntentEffect.READ,
                confidence=0.90,
                priority=100,
                reason="Read an existing result",
            ),
            IntentCandidate(
                name="submit",
                effect=IntentEffect.WRITE,
                confidence=0.88,
                priority=100,
                reason="Submit a calculation",
            ),
        )
    )

    assert resolution.primary.effect is IntentEffect.CLARIFY
    assert resolution.ambiguous
    assert {item.name for item in resolution.candidates} == {"read_result", "submit"}


def test_confidence_resolver_keeps_highest_ranked_duplicate() -> None:
    resolver = ConfidenceIntentResolver()

    resolution = resolver.resolve(
        (
            IntentCandidate(
                name="compute",
                effect=IntentEffect.WRITE,
                confidence=0.99,
                priority=80,
                reason="Lower-priority model candidate",
                source="model",
            ),
            IntentCandidate(
                name="compute",
                effect=IntentEffect.WRITE,
                confidence=0.90,
                priority=100,
                reason="Higher-priority rule candidate",
                source="rule",
            ),
            IntentCandidate(
                name="answer",
                effect=IntentEffect.ANSWER,
                confidence=0.70,
                priority=10,
                reason="General answer",
            ),
        )
    )

    assert len(resolution.candidates) == 2
    assert resolution.primary.name == "compute"
    assert resolution.primary.reason == "Higher-priority rule candidate"
