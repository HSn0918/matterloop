"""定义可插拔意图识别的通用契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Protocol


class IntentEffect(str, Enum):
    """描述意图允许产生的最大业务副作用。"""

    ANSWER = "answer"
    READ = "read"
    WRITE = "write"
    CLARIFY = "clarify"


@dataclass(frozen=True, slots=True)
class IntentRequest:
    """封装一次意图识别所需的输入。

    Args:
        text: 需要识别的原始用户文本。
        context: 领域分类器可选使用的只读会话上下文。
        candidates: 调用方允许返回的意图名称；空元组表示不限制。
        metadata: 不参与通用层解释的只读关联数据。
    """

    text: str
    context: Mapping[str, object] = field(default_factory=dict)
    candidates: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验输入并冻结映射，避免分类期间被调用方修改。"""
        if not self.text.strip():
            raise ValueError("intent request text must not be empty")
        if any(not candidate.strip() for candidate in self.candidates):
            raise ValueError("intent candidates must not contain empty values")
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """保存一次可审计的意图识别结论。

    Args:
        name: 由领域分类器定义的稳定意图名称。
        effect: 该意图允许产生的最大副作用级别。
        confidence: 零到一之间的分类置信度。
        reason: 面向日志和诊断的简短判定原因。
        matched_signals: 支持该结论的规则或模型信号。
        attributes: 领域可扩展的只读属性。
    """

    name: str
    effect: IntentEffect
    confidence: float
    reason: str
    matched_signals: tuple[str, ...] = ()
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """保证分类结论具备稳定标识和合法置信度。"""
        if not self.name.strip():
            raise ValueError("intent name must not be empty")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("intent confidence must be finite and between 0 and 1")
        if not self.reason.strip():
            raise ValueError("intent reason must not be empty")
        if any(not signal.strip() for signal in self.matched_signals):
            raise ValueError("intent matched signals must not contain empty values")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @property
    def allows_side_effects(self) -> bool:
        """返回该意图是否允许创建或修改外部状态。"""
        return self.effect is IntentEffect.WRITE


@dataclass(frozen=True, slots=True)
class IntentCandidate:
    """表示一个分类器产生的候选意图。

    Args:
        name: 候选意图稳定名称。
        effect: 候选意图的最大副作用级别。
        confidence: 零到一之间的候选置信度。
        reason: 产生该候选的原因。
        priority: 领域策略定义的整数优先级。
        source: 候选来源，例如 ``rule``、``model`` 或 ``context``。
        matched_signals: 支持候选的规则、上下文或模型信号。
        attributes: 领域可扩展的只读属性。
    """

    name: str
    effect: IntentEffect
    confidence: float
    reason: str
    priority: int = 0
    source: str = "unspecified"
    matched_signals: tuple[str, ...] = ()
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验候选意图并冻结扩展属性。"""
        if not self.name.strip():
            raise ValueError("intent candidate name must not be empty")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("intent candidate confidence must be finite and between 0 and 1")
        if not self.reason.strip():
            raise ValueError("intent candidate reason must not be empty")
        if not self.source.strip():
            raise ValueError("intent candidate source must not be empty")
        if any(not signal.strip() for signal in self.matched_signals):
            raise ValueError("intent candidate signals must not contain empty values")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    def to_decision(self) -> IntentDecision:
        """把候选转换为最终意图结论。"""
        return IntentDecision(
            name=self.name,
            effect=self.effect,
            confidence=self.confidence,
            reason=self.reason,
            matched_signals=self.matched_signals,
            attributes=self.attributes,
        )


@dataclass(frozen=True, slots=True)
class IntentResolution:
    """保存候选排序、冲突判断和最终意图。

    Args:
        primary: 最终采用的主意图。
        candidates: 按领域优先级与置信度排序的全部候选。
        ambiguous: 是否因为同级候选过于接近而需要澄清。
        reason: 冲突消解过程的简短说明。
    """

    primary: IntentDecision
    candidates: tuple[IntentCandidate, ...]
    ambiguous: bool
    reason: str

    def __post_init__(self) -> None:
        """保证冲突消解说明不为空。"""
        if not self.reason.strip():
            raise ValueError("intent resolution reason must not be empty")

    @property
    def secondary(self) -> tuple[IntentCandidate, ...]:
        """返回主意图之外保留的次级候选。"""
        return tuple(
            candidate for candidate in self.candidates if candidate.name != self.primary.name
        )

    def has_intent(self, name: str) -> bool:
        """返回候选集合中是否包含指定意图。"""
        return any(candidate.name == name for candidate in self.candidates)


class IntentClassifier(Protocol):
    """把用户请求解析为领域意图结论的同步分类器接口。"""

    def classify(self, request: IntentRequest) -> IntentDecision:
        """识别请求并返回带判定依据的意图结论。"""
        ...


class IntentResolver(Protocol):
    """把多个候选意图消解为一个可执行主意图。"""

    def resolve(self, candidates: tuple[IntentCandidate, ...]) -> IntentResolution:
        """排序候选、识别冲突并返回最终结论。"""
        ...


class ConfidenceIntentResolver:
    """按领域优先级、置信度和副作用等级消解意图冲突。

    Args:
        min_confidence: 低于该置信度时返回澄清意图。
        ambiguity_margin: 同优先级前两名的置信度差不大于该值时返回澄清意图。
        clarify_name: 澄清意图使用的稳定名称。
    """

    _EFFECT_PRIORITY = {
        IntentEffect.CLARIFY: 0,
        IntentEffect.ANSWER: 1,
        IntentEffect.READ: 2,
        IntentEffect.WRITE: 3,
    }

    def __init__(
        self,
        *,
        min_confidence: float = 0.5,
        ambiguity_margin: float = 0.05,
        clarify_name: str = "clarify",
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("minimum intent confidence must be between 0 and 1")
        if not 0.0 <= ambiguity_margin <= 1.0:
            raise ValueError("intent ambiguity margin must be between 0 and 1")
        if not clarify_name.strip():
            raise ValueError("clarify intent name must not be empty")
        self._min_confidence = min_confidence
        self._ambiguity_margin = ambiguity_margin
        self._clarify_name = clarify_name

    def resolve(self, candidates: tuple[IntentCandidate, ...]) -> IntentResolution:
        """按稳定排序规则选择主意图，并保留全部次级候选。"""
        ordered = tuple(
            sorted(
                self._deduplicate(candidates),
                key=lambda candidate: (
                    candidate.priority,
                    candidate.confidence,
                    self._EFFECT_PRIORITY[candidate.effect],
                    candidate.name,
                ),
                reverse=True,
            )
        )
        if not ordered:
            return self._clarification(
                ordered,
                confidence=0.0,
                reason="No intent candidates were produced",
            )
        top = ordered[0]
        if top.confidence < self._min_confidence:
            return self._clarification(
                ordered,
                confidence=top.confidence,
                reason="The top candidate confidence is below the execution threshold",
            )
        runner_up = next(
            (candidate for candidate in ordered[1:] if candidate.name != top.name),
            None,
        )
        if (
            runner_up is not None
            and runner_up.priority == top.priority
            and abs(top.confidence - runner_up.confidence) <= self._ambiguity_margin
        ):
            return self._clarification(
                ordered,
                confidence=top.confidence,
                reason=(
                    f"Candidates {top.name} and {runner_up.name} have equal priority "
                    "and confidence values within the ambiguity margin"
                ),
            )
        return IntentResolution(
            primary=top.to_decision(),
            candidates=ordered,
            ambiguous=False,
            reason=(
                f"Selected the highest-priority candidate with sufficient confidence: {top.name}"
            ),
        )

    @staticmethod
    def _deduplicate(
        candidates: tuple[IntentCandidate, ...],
    ) -> tuple[IntentCandidate, ...]:
        """同名候选只保留排序更高的一项，避免重复信号扭曲冲突判断。"""
        selected: dict[str, IntentCandidate] = {}
        for candidate in candidates:
            current = selected.get(candidate.name)
            if current is None or ConfidenceIntentResolver._has_higher_rank(
                candidate,
                current,
            ):
                selected[candidate.name] = candidate
        deduplicated: list[IntentCandidate] = list(selected.values())
        return tuple(deduplicated)

    @staticmethod
    def _has_higher_rank(
        candidate: IntentCandidate,
        current: IntentCandidate,
    ) -> bool:
        """按优先级和置信度判断同名候选是否应替换当前候选。"""
        if candidate.priority != current.priority:
            return candidate.priority > current.priority
        return candidate.confidence > current.confidence

    def _clarification(
        self,
        candidates: tuple[IntentCandidate, ...],
        *,
        confidence: float,
        reason: str,
    ) -> IntentResolution:
        """构造不允许副作用的澄清结论。"""
        signals = tuple(
            signal for candidate in candidates[:3] for signal in candidate.matched_signals[:2]
        )
        return IntentResolution(
            primary=IntentDecision(
                name=self._clarify_name,
                effect=IntentEffect.CLARIFY,
                confidence=confidence,
                reason=reason,
                matched_signals=signals,
                attributes={"candidate_names": tuple(item.name for item in candidates)},
            ),
            candidates=candidates,
            ambiguous=True,
            reason=reason,
        )
