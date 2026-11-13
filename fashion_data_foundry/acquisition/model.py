"""纯标准库实现的高斯朴素贝叶斯分类器与训练快照。

选一个可解释、无第三方依赖、对小样本稳定的模型：每个类别维护特征均值/方差与先验，
预测概率可直接转成不确定性（熵）和边际，供主动选样与比较集评估共用。

模型只消费：
- 基线训练家族（含仅展示授权的过滤结果）；
- 已入账、未被更正作废、未被撤销授权、且具备 training 用途的采购家族。
回执未到达的候选家族不可能进入训练集——训练快照的构造函数只接受 :class:`LabeledFamily`。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from fashion_data_foundry.acquisition.protocol import LABEL_CLASSES, LabeledFamily

# 方差地板，避免零方差维度产生无穷对数概率。
VARIANCE_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class TrainingSnapshot:
    """一次训练所用证据的可追溯快照。"""

    usable_families: tuple[str, ...]
    excluded_families: tuple[dict[str, str], ...]
    label_counts: dict[str, int]
    supplier_counts: dict[str, int]
    evidence_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "usable_families": list(self.usable_families),
            "excluded_families": list(self.excluded_families),
            "label_counts": dict(self.label_counts),
            "supplier_counts": dict(self.supplier_counts),
            "evidence_fingerprint": self.evidence_fingerprint,
        }


class GaussianNaiveBayes:
    """对角协方差高斯朴素贝叶斯，输出两分类概率。"""

    def __init__(self) -> None:
        self._means: dict[str, list[float]] = {}
        self._variances: dict[str, list[float]] = {}
        self._priors: dict[str, float] = {}
        self._dimensions = 0
        self._classes: tuple[str, ...] = LABEL_CLASSES

    @classmethod
    def train(cls, evidence: Sequence[LabeledFamily]) -> "GaussianNaiveBayes":
        usable = [item for item in evidence if item.usable_for_training()]
        model = cls()
        if not usable:
            raise ValueError("没有可用于训练的家族证据")
        dimensions = {len(item.features) for item in usable}
        if len(dimensions) != 1:
            raise ValueError("训练特征维度不一致")
        model._dimensions = dimensions.pop()
        total = float(len(usable))
        for label in model._classes:
            members = [item for item in usable if item.label == label]
            if not members:
                # 类别缺证据时退回均匀先验与全局零均值，保证预测仍有定义。
                model._priors[label] = 1.0 / len(model._classes)
                model._means[label] = [0.0] * model._dimensions
                model._variances[label] = [1.0] * model._dimensions
                continue
            model._priors[label] = len(members) / total
            means: list[float] = []
            variances: list[float] = []
            for dim in range(model._dimensions):
                values = [item.features[dim] for item in members]
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                means.append(mean)
                variances.append(max(variance, VARIANCE_FLOOR))
            model._means[label] = means
            model._variances[label] = variances
        return model

    def predict_proba(self, features: Sequence[float]) -> dict[str, float]:
        if len(features) != self._dimensions:
            raise ValueError("预测特征维度与训练维度不一致")
        log_scores: dict[str, float] = {}
        for label in self._classes:
            score = math.log(self._priors[label])
            for dim, value in enumerate(features):
                variance = self._variances[label][dim]
                mean = self._means[label][dim]
                score += -0.5 * math.log(2.0 * math.pi * variance)
                score += -((float(value) - mean) ** 2) / (2.0 * variance)
            log_scores[label] = score
        ceiling = max(log_scores.values())
        exp_scores = {label: math.exp(score - ceiling) for label, score in log_scores.items()}
        normalizer = sum(exp_scores.values())
        return {label: exp_scores[label] / normalizer for label in self._classes}

    def predict(self, features: Sequence[float]) -> str:
        probabilities = self.predict_proba(features)
        return max(probabilities, key=lambda key: (probabilities[key], key))

    def uncertainty(self, features: Sequence[float]) -> float:
        """预测分布的归一化香农熵，取值 [0, 1]。"""
        probabilities = self.predict_proba(features)
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities.values() if probability > 0.0
        )
        return entropy / math.log(len(self._classes))


def build_snapshot(evidence: Sequence[LabeledFamily]) -> TrainingSnapshot:
    """把证据拆成可用/排除清单并留下指纹，供状态文件追溯。"""
    usable = [item for item in evidence if item.usable_for_training()]
    excluded: list[dict[str, str]] = []
    for item in evidence:
        if item.usable_for_training():
            continue
        reason = "withdrawn" if item.withdrawn else (
            "superseded" if item.superseded else "display_only"
        )
        excluded.append({"family": item.family, "reason": reason})
    label_counts = {label: 0 for label in LABEL_CLASSES}
    supplier_counts: dict[str, int] = {}
    for item in usable:
        label_counts[item.label] += 1
        supplier_counts[item.supplier] = supplier_counts.get(item.supplier, 0) + 1
    signature = [
        [item.family, item.label, item.receipt_id, item.revision, item.withdrawn, item.superseded]
        for item in evidence
    ]
    from fashion_data_foundry.governance.asset_ledger import fingerprint
    return TrainingSnapshot(
        usable_families=tuple(sorted(item.family for item in usable)),
        excluded_families=tuple(excluded),
        label_counts=label_counts, supplier_counts=supplier_counts,
        evidence_fingerprint=fingerprint(signature),
    )
