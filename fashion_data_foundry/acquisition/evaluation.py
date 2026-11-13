"""独立保留比较集上的评估与学习曲线。

隔离保证：
- 本模块是全工程唯一接受 :class:`~fashion_data_foundry.acquisition.protocol.SealedLabels`
  的地方，且只在 compare 阶段调用；选样与训练代码没有本模块的导入路径依赖。
- 评估只读取模型预测与比较集标签，不会把比较集写回状态、也不会触发再训练，
  因此参数选择无法“试到保留集最好为止”。

停采建议使用预注册规则（阈值是模块常量，不随单次结果调整），并同时报告：
总体准确率 Wilson 置信区间、分类别表现、分供货商表现、Brier 分数与净边际效用，
避免把某一批保留集的偶然上涨包装成普遍有效。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fashion_data_foundry.acquisition.model import GaussianNaiveBayes
from fashion_data_foundry.acquisition.protocol import (
    GuardedHoldoutError,
    HoldoutCase,
    LABEL_CLASSES,
    SealedLabels,
)

# ---- 预注册停采阈值（不暴露给 CLI，不允许按保留集表现反调） ----
MIN_MARGINAL_ACC_GAIN = 0.085          # 再投一轮至少要带来的准确率边际
MIN_MARGINAL_UTILITY_PER_COST = 0.01   # 每单位预算的净效用下限


@dataclass(frozen=True, slots=True)
class CurvePoint:
    round: int
    strategy: str
    usable_families: int
    metrics: dict[str, Any]
    training_accuracy: float
    spend_cumulative: float


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _brier(probabilities: Sequence[Mapping[str, float]], cases: Sequence[HoldoutCase],
           labels: Mapping[str, str]) -> float:
    if not probabilities:
        return 0.0
    total = 0.0
    for probability, case in zip(probabilities, cases):
        for label in LABEL_CLASSES:
            indicator = 1.0 if labels[case.family] == label else 0.0
            total += (probability[label] - indicator) ** 2
    return total / len(probabilities)


def evaluate_holdout(
    model: GaussianNaiveBayes,
    cases: Sequence[HoldoutCase],
    sealed: SealedLabels,
) -> dict[str, Any]:
    """在比较集上评估模型；标签只在这里通过密封对象解包。"""
    labels = sealed.unseal_for_evaluation()
    unknown = sorted({case.family for case in cases} - set(labels))
    if unknown:
        raise GuardedHoldoutError(f"比较集标签缺失: {unknown}")

    rows: list[dict[str, Any]] = []
    eligible_predictions: list[tuple[str, str]] = []
    all_predictions: list[tuple[str, str]] = []
    probabilities: list[Mapping[str, float]] = []
    supplier_stats: dict[str, dict[str, int]] = {}
    class_stats = {
        label: {"true_positive": 0, "actual": 0, "predicted": 0}
        for label in LABEL_CLASSES
    }
    for case in cases:
        probability = model.predict_proba(case.features)
        predicted = max(probability, key=lambda key: (probability[key], key))
        actual = labels[case.family]
        probabilities.append(probability)
        all_predictions.append((predicted, actual))
        if case.training_eligible:
            eligible_predictions.append((predicted, actual))
        bucket = supplier_stats.setdefault(
            case.supplier, {"total": 0, "correct": 0,
                            "eligible_total": 0, "eligible_correct": 0}
        )
        bucket["total"] += 1
        bucket["correct"] += int(predicted == actual)
        if case.training_eligible:
            bucket["eligible_total"] += 1
            bucket["eligible_correct"] += int(predicted == actual)
        class_stats[actual]["actual"] += 1
        class_stats[predicted]["predicted"] += 1
        if predicted == actual:
            class_stats[actual]["true_positive"] += 1
        rows.append({
            "family": case.family, "supplier": case.supplier,
            "training_eligible": case.training_eligible,
            "predicted": predicted, "actual": actual,
            "correct": predicted == actual,
            "prob_structured": round(probability["structured"], 6),
        })

    eligible_total = len(eligible_predictions)
    eligible_correct = sum(int(p == a) for p, a in eligible_predictions)
    low, high = wilson_interval(eligible_correct, eligible_total)
    per_class: dict[str, dict[str, float]] = {}
    for label, stats in class_stats.items():
        precision = (
            stats["true_positive"] / stats["predicted"] if stats["predicted"] else 0.0
        )
        recall = stats["true_positive"] / stats["actual"] if stats["actual"] else 0.0
        per_class[label] = {
            "precision": round(precision, 6), "recall": round(recall, 6),
            "actual": stats["actual"], "predicted": stats["predicted"],
        }
    supplier_breakdown = {
        supplier: {
            "accuracy": round(stats["correct"] / stats["total"], 6),
            "eligible_accuracy": (
                round(stats["eligible_correct"] / stats["eligible_total"], 6)
                if stats["eligible_total"] else None
            ),
            "total": stats["total"],
        }
        for supplier, stats in sorted(supplier_stats.items())
    }
    return {
        "eligible_families": eligible_total,
        "display_only_families": len(cases) - eligible_total,
        "accuracy": round(eligible_correct / eligible_total, 6) if eligible_total else 0.0,
        "accuracy_ci95": [round(low, 6), round(high, 6)],
        "all_case_accuracy": round(
            sum(int(p == a) for p, a in all_predictions) / len(all_predictions), 6
        ),
        "brier": round(_brier(probabilities, cases, labels), 6),
        "per_class": per_class,
        "supplier_breakdown": supplier_breakdown,
        "predictions": rows,
    }


def training_accuracy(model: GaussianNaiveBayes,
                      evidence_features: Sequence[tuple[tuple[float, ...], str]]) -> float:
    """训练集准确率仅作参考，不进入停采规则。"""
    if not evidence_features:
        return 0.0
    correct = sum(
        int(model.predict(features) == label) for features, label in evidence_features
    )
    return round(correct / len(evidence_features), 6)


def stop_recommendation(
    curve: Sequence[CurvePoint],
    round_limit: int,
    budget_per_round: float,
) -> dict[str, Any]:
    """预注册停采规则：轮次耗尽 / 边际不足 / 置信区间未改善 / 净效用为负。"""
    if not curve:
        return {"decision": "continue", "reasons": ["尚无评估快照"]}
    latest = curve[-1]
    reasons: list[str] = []
    if latest.round >= round_limit:
        reasons.append(f"已达采购轮次上限 {round_limit}")
    if len(curve) < 2:
        reasons.append("只有基线快照，边际证据不足，默认不再自动追加采购")
        return {"decision": "stop", "reasons": reasons,
                "latest_accuracy": latest.metrics["accuracy"],
                "latest_ci95": latest.metrics["accuracy_ci95"]}
    previous = curve[-2]
    gain = latest.metrics["accuracy"] - previous.metrics["accuracy"]
    ci_improved = latest.metrics["accuracy_ci95"][0] >= previous.metrics["accuracy_ci95"][0]
    marginal_utility = gain / max(budget_per_round, 1e-9)
    if gain < MIN_MARGINAL_ACC_GAIN:
        reasons.append(
            f"边际准确率 {gain:.4f} 低于预注册阈值 {MIN_MARGINAL_ACC_GAIN}"
        )
    if not ci_improved:
        reasons.append("最新快照的 Wilson 置信区间下界未改善，上涨可能是保留集波动")
    if marginal_utility < MIN_MARGINAL_UTILITY_PER_COST:
        reasons.append(
            f"每单位预算净效用 {marginal_utility:.5f} "
            f"低于阈值 {MIN_MARGINAL_UTILITY_PER_COST}"
        )
    decision = "stop" if reasons else "continue"
    return {
        "decision": decision,
        "reasons": reasons,
        "marginal_accuracy_gain": round(gain, 6),
        "marginal_utility_per_cost": round(marginal_utility, 6),
        "previous_accuracy": previous.metrics["accuracy"],
        "latest_accuracy": latest.metrics["accuracy"],
        "latest_ci95": latest.metrics["accuracy_ci95"],
        "training_accuracy_note": "训练准确率仅作参考，未参与停采判断",
    }
