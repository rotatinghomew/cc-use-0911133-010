"""主动选样策略：balanced / uncertainty / random。

公平比较的前提写死在代码里：
- 三个策略面对同一个可采购家族池（未采购、且具备 training 授权）、同一轮预算、
  同一个模型类与同一份已入账证据；
- 预算填充规则完全一致（按分数贪心、家族 ID 做确定性兜底），差异只在评分函数；
- 权重与常数是预注册的（见下方模块常量），CLI 不暴露任何可调参数，
  保留比较集无法反向影响这些选择，杜绝“在保留集上反复试参”。

策略构造函数只接收 :class:`~fashion_data_foundry.acquisition.protocol.InputCatalog`、
随机数生成器和可选的已训练模型；任何策略对象都不持有候选真标签或比较集标签。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from fashion_data_foundry.acquisition.model import GaussianNaiveBayes
from fashion_data_foundry.acquisition.protocol import (
    FamilyView,
    InputCatalog,
    LabeledFamily,
)

# ---- 预注册策略参数：修改即改变实验设计，不允许由调用方按保留集表现调整 ----
BALANCED_WEIGHTS = {
    "uncertainty": 0.4,
    "coverage_gap": 0.3,
    "supplier_diversity": 0.2,
    "cost_efficiency": 0.1,
}
STRATEGY_NAMES: tuple[str, ...] = ("balanced", "uncertainty", "random")


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    """选样时可见的证据画像：只有聚合计数，不含任何候选真标签。"""

    total: int
    label_counts: Mapping[str, int]
    supplier_counts: Mapping[str, int]

    @classmethod
    def from_evidence(cls, evidence: Sequence[LabeledFamily]) -> "EvidenceSummary":
        label_counts = {"soft": 0, "structured": 0}
        supplier_counts: dict[str, int] = {}
        total = 0
        for item in evidence:
            if not item.usable_for_training():
                continue
            total += 1
            label_counts[item.label] = label_counts.get(item.label, 0) + 1
            supplier_counts[item.supplier] = supplier_counts.get(item.supplier, 0) + 1
        return cls(total=total, label_counts=label_counts, supplier_counts=supplier_counts)

    def class_gap(self, predicted_class: str) -> float:
        """类别覆盖缺口：当前越少证据的预测类别，缺口越大。"""
        maximum = max(self.label_counts.values(), default=1)
        share = self.label_counts.get(predicted_class, 0) / max(maximum, 1)
        return 1.0 - share

    def supplier_gap(self, supplier: str) -> float:
        """来源多样性缺口：当前越少使用的供货商，缺口越大。"""
        if self.total == 0:
            return 1.0
        share = self.supplier_counts.get(supplier, 0) / self.total
        return 1.0 - share


@dataclass(frozen=True, slots=True)
class ScoredFamily:
    family: str
    score: float
    components: Mapping[str, float]
    unit_cost: float
    supplier: str
    training_eligible: bool


class AcquisitionStrategy:
    """策略基类：统一候选池、统一贪心预算填充。"""

    name = "base"

    def __init__(
        self,
        catalog: InputCatalog,
        rng: random.Random,
        model: GaussianNaiveBayes | None = None,
        evidence: EvidenceSummary | None = None,
    ) -> None:
        self.catalog = catalog
        self.rng = rng
        self.model = model
        self.evidence = evidence or EvidenceSummary(0, {"soft": 0, "structured": 0}, {})

    def _eligible(self, acquired_families: frozenset[str]) -> list[FamilyView]:
        return [
            view for family, view in sorted(self.catalog.candidates.items())
            if family not in acquired_families and view.training_eligible
        ]

    def score_family(self, view: FamilyView) -> ScoredFamily:
        raise NotImplementedError

    def propose(self, budget: float, acquired_families: frozenset[str]) -> list[ScoredFamily]:
        ranked = [self.score_family(view) for view in self._eligible(acquired_families)]
        # 分数降序、家族 ID 升序兜底，保证同种子下结果可复算。
        ranked.sort(key=lambda item: (-item.score, item.family))
        remaining = budget
        selected: list[ScoredFamily] = []
        for item in ranked:
            if item.unit_cost <= remaining + 1e-9:
                selected.append(item)
                remaining -= item.unit_cost
        return selected

    def ranked_catalog(self, acquired_families: frozenset[str]) -> list[ScoredFamily]:
        ranked = [self.score_family(view) for view in self._eligible(acquired_families)]
        ranked.sort(key=lambda item: (-item.score, item.family))
        return ranked


class RandomStrategy(AcquisitionStrategy):
    """随机采购基线：RGN 由活动种子派生，可复现。"""

    name = "random"

    def score_family(self, view: FamilyView) -> ScoredFamily:
        return ScoredFamily(
            family=view.family, score=self.rng.random(),
            components={"random_draw": 0.0}, unit_cost=view.unit_cost,
            supplier=view.supplier, training_eligible=view.training_eligible,
        )


class UncertaintyStrategy(AcquisitionStrategy):
    """纯不确定性基线：只买模型最拿不准的家族。"""

    name = "uncertainty"

    def score_family(self, view: FamilyView) -> ScoredFamily:
        uncertainty = (
            self.model.uncertainty(view.features) if self.model is not None else 1.0
        )
        return ScoredFamily(
            family=view.family, score=uncertainty,
            components={"uncertainty": round(uncertainty, 6)},
            unit_cost=view.unit_cost, supplier=view.supplier,
            training_eligible=view.training_eligible,
        )


class BalancedStrategy(AcquisitionStrategy):
    """平衡策略：不确定性 + 类别覆盖 + 来源多样性 + 成本效率。"""

    name = "balanced"

    def score_family(self, view: FamilyView) -> ScoredFamily:
        if self.model is not None:
            probabilities = self.model.predict_proba(view.features)
            uncertainty = self.model.uncertainty(view.features)
            predicted = max(probabilities, key=lambda key: (probabilities[key], key))
        else:
            uncertainty, predicted = 1.0, "soft"
        coverage_gap = self.evidence.class_gap(predicted)
        supplier_gap = self.evidence.supplier_gap(view.supplier)
        costs = [candidate.unit_cost for candidate in self.catalog.candidates.values()]
        spread = max(costs) - min(costs)
        cost_efficiency = (
            1.0 - (view.unit_cost - min(costs)) / spread if spread > 0 else 1.0
        )
        components = {
            "uncertainty": round(uncertainty, 6),
            "coverage_gap": round(coverage_gap, 6),
            "supplier_diversity": round(supplier_gap, 6),
            "cost_efficiency": round(cost_efficiency, 6),
        }
        score = sum(BALANCED_WEIGHTS[key] * components[key] for key in BALANCED_WEIGHTS)
        return ScoredFamily(
            family=view.family, score=round(score, 6), components=components,
            unit_cost=view.unit_cost, supplier=view.supplier,
            training_eligible=view.training_eligible,
        )


def build_strategy(
    name: str,
    catalog: InputCatalog,
    rng: random.Random,
    model: GaussianNaiveBayes | None,
    evidence: EvidenceSummary,
) -> AcquisitionStrategy:
    if name == "random":
        return RandomStrategy(catalog, rng, model, evidence)
    if name == "uncertainty":
        return UncertaintyStrategy(catalog, rng, model, evidence)
    if name == "balanced":
        return BalancedStrategy(catalog, rng, model, evidence)
    raise ValueError(f"未知策略: {name}")
