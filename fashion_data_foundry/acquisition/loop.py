"""采购闭环编排服务：把目录、账本、策略、模型与评估串起来。

三个对外动作与 CLI 子命令一一对应：

- :func:`propose_round`：只读无标签候选视图与已入账证据，输出可审计的订单与评分拆解；
- :func:`ingest_round`：回执入账，支持部分回执、乱序、更正、撤销与重复调用；
- :func:`compare_strategies`：在独立比较集上重放各策略的学习曲线并给出停采建议。

任何函数签名都不接受比较集路径（compare 除外），比较集标签在 :mod:`evaluation`
内部解包，选样与训练代码无法触及。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from fashion_data_foundry.acquisition.catalog import (
    load_catalog,
    load_holdout,
    load_receipt_bundle,
    load_revision_examples,
)
from fashion_data_foundry.acquisition.evaluation import (
    CurvePoint,
    evaluate_holdout,
    stop_recommendation,
    training_accuracy,
)
from fashion_data_foundry.acquisition.ledger import (
    AcquisitionLedger,
    PHASE_INGESTED,
)
from fashion_data_foundry.acquisition.model import (
    GaussianNaiveBayes,
    build_snapshot,
)
from fashion_data_foundry.acquisition.protocol import (
    AcquisitionError,
    HoldoutCase,
    InputCatalog,
    ReceiptRecord,
    SealedLabels,
)
from fashion_data_foundry.acquisition.strategy import (
    BALANCED_WEIGHTS,
    STRATEGY_NAMES,
    EvidenceSummary,
    build_strategy,
)


def _rng_for(catalog: InputCatalog, strategy_name: str) -> random.Random:
    # 每个策略独立派生随机流，保证公平且可复现。
    return random.Random(f"{catalog.seed}:{strategy_name}")


def _model_for(evidence: Sequence) -> GaussianNaiveBayes | None:
    usable = [item for item in evidence if item.usable_for_training()]
    if not usable:
        return None
    return GaussianNaiveBayes.train(usable)


def _coverage_gaps(catalog: InputCatalog, evidence: Sequence) -> dict[str, Any]:
    summary = EvidenceSummary.from_evidence(evidence)
    suppliers = sorted({view.supplier for view in catalog.candidates.values()})
    labels = ("soft", "structured")
    return {
        "supplier_evidence": {
            supplier: summary.supplier_counts.get(supplier, 0) for supplier in suppliers
        },
        "supplier_without_evidence": sorted(
            supplier for supplier in suppliers
            if summary.supplier_counts.get(supplier, 0) == 0
        ),
        "class_evidence": {
            label: summary.label_counts.get(label, 0) for label in labels
        },
        "class_without_evidence": sorted(
            label for label in labels if summary.label_counts.get(label, 0) == 0
        ),
        "rights_blocked_candidate_families": sorted(
            view.family for view in catalog.candidates.values()
            if not view.training_eligible
        ),
        "rights_blocked_note": "仅展示授权的家族可采购展示，但永远不进入训练候选",
    }


def propose_round(
    input_path: str | Path,
    state_path: str | Path,
    strategy_name: str = "balanced",
) -> dict[str, Any]:
    """生成（或幂等回放）一轮家族采购订单。"""
    if strategy_name not in STRATEGY_NAMES:
        raise AcquisitionError(f"策略必须是 {STRATEGY_NAMES} 之一")
    catalog = load_catalog(input_path)
    ledger = AcquisitionLedger(state_path, catalog)
    round_number = ledger.current_round_number(strategy_name)
    if round_number > catalog.round_limit:
        raise AcquisitionError(
            f"已达轮次上限 {catalog.round_limit}，采购活动应进入 compare/停采"
        )

    evidence, evidence_note = ledger.build_evidence(strategy_name, round_number - 1)
    model = _model_for(evidence)
    summary = EvidenceSummary.from_evidence(evidence)
    rng = _rng_for(catalog, strategy_name)
    strategy = build_strategy(strategy_name, catalog, rng, model, summary)

    existing = ledger.latest_round(strategy_name)
    replay = (
        existing is not None
        and existing["round"] == round_number
        and existing["phase"] == "proposed"
    )
    if replay:
        scored = [
            _scaled_order_to_dict(order) for order in existing["orders"]
        ]
    else:
        selected = strategy.propose(
            catalog.budget_per_round, ledger.acquired_families(strategy_name)
        )
        orders = [_scored_to_order(item) for item in selected]
        record = ledger.record_proposal(strategy_name, round_number, orders)
        scored = record["orders"]
        ledger.save()

    ranked = strategy.ranked_catalog(ledger.acquired_families(strategy_name))
    return {
        "action": "propose",
        "strategy": strategy_name,
        "round": round_number,
        "replayed": bool(replay),
        "budget_per_round": catalog.budget_per_round,
        "orders": scored,
        "order_count": len(scored),
        "order_spend": round(sum(item["unit_cost"] for item in scored), 6),
        "budget_remaining": round(
            catalog.budget_per_round - sum(item["unit_cost"] for item in scored), 6
        ),
        "weights": dict(BALANCED_WEIGHTS) if strategy_name == "balanced" else None,
        "model_ready": model is not None,
        "evidence": evidence_note,
        "coverage_gaps": _coverage_gaps(catalog, evidence),
        "top_ranked_families": [
            {"family": item.family, "score": item.score, "components": dict(item.components)}
            for item in ranked[:10]
        ],
        "budget_report": ledger.budget_report(strategy_name),
        "state_path": str(state_path),
    }


def _scored_to_order(item: Any) -> dict[str, Any]:
    return {
        "family": item.family,
        "supplier": item.supplier,
        "unit_cost": item.unit_cost,
        "score": item.score,
        "score_components": dict(item.components),
        "training_eligible": item.training_eligible,
    }


def _scaled_order_to_dict(order: Mapping[str, Any]) -> dict[str, Any]:
    return {key: order[key] for key in (
        "family", "supplier", "unit_cost", "score", "score_components",
        "training_eligible", "status",
    ) if key in order}


def ingest_round(
    input_path: str | Path,
    state_path: str | Path,
    receipt_paths: Sequence[str | Path],
    strategy_name: str = "balanced",
    include_revision_examples: bool = True,
) -> dict[str, Any]:
    """把回执文件计入当前轮；重复调用安全，未回执订单保持 pending。"""
    catalog = load_catalog(input_path)
    ledger = AcquisitionLedger(state_path, catalog)
    latest = ledger.latest_round(strategy_name)
    if latest is None:
        raise AcquisitionError("还没有任何采购订单，请先执行 propose")
    round_number = latest["round"]
    if round_number > catalog.round_limit:
        raise AcquisitionError("活动已结束，不再接收回执")

    raw_rows: list[Mapping[str, Any]] = []
    sources: list[str] = []
    for path in receipt_paths:
        raw_rows.extend(load_receipt_bundle(path))
        sources.append(str(path))
    if include_revision_examples:
        examples = load_revision_examples(input_path)
        if examples:
            raw_rows.extend(examples)
            sources.append(f"{input_path}#receipt_revision_examples")

    receipts = [ReceiptRecord.from_mapping(row) for row in raw_rows]
    # 同文件乱序：先按 revision 排序，保证“先到的旧件、后到的更正”处理可复现；
    # 幂等与陈旧判断仍由 ledger 负责，排序只是让 ingest_log 可读。
    receipts.sort(key=lambda item: (item.family, item.revision, item.receipt_id))
    log = ledger.apply_receipts(strategy_name, round_number, receipts)
    ledger.save()

    evidence, evidence_note = ledger.build_evidence(strategy_name, round_number)
    snapshot = build_snapshot(evidence)
    return {
        "action": "ingest",
        "strategy": strategy_name,
        "round": round_number,
        "receipt_sources": sources,
        "receipts_offered": len(receipts),
        "ingest_log": log,
        "phase": PHASE_INGESTED,
        "training_snapshot": snapshot.to_dict(),
        "evidence": evidence_note,
        "budget_report": ledger.budget_report(strategy_name),
        "state_path": str(state_path),
    }


def _curve_for_strategy(
    strategy_name: str,
    ledger: AcquisitionLedger,
    catalog: InputCatalog,
    holdout_cases: Sequence[HoldoutCase],
    sealed: SealedLabels,
) -> list[CurvePoint]:
    """按轮次重放证据→训练→比较集评估，比较集只在本函数内解包标签。"""
    curve: list[CurvePoint] = []
    total_rounds = len(ledger.rounds(strategy_name))
    cumulative_spend = 0.0

    # 基线快照（round=0，零采购证据）
    baseline_evidence, _ = ledger.build_evidence(strategy_name, 0)
    baseline_model = _model_for(baseline_evidence)
    baseline_metrics = evaluate_holdout(baseline_model, holdout_cases, sealed)
    curve.append(CurvePoint(
        round=0, strategy=strategy_name,
        usable_families=len(build_snapshot(baseline_evidence).usable_families),
        metrics=baseline_metrics,
        training_accuracy=training_accuracy(
            baseline_model,
            [(item.features, item.label) for item in baseline_evidence
             if item.usable_for_training()],
        ),
        spend_cumulative=0.0,
    ))

    for round_number in range(1, total_rounds + 1):
        record = ledger.rounds(strategy_name)[round_number - 1]
        cumulative_spend += float(record["spend"])
        evidence, _ = ledger.build_evidence(strategy_name, round_number)
        snapshot = build_snapshot(evidence)
        model = _model_for(evidence)
        metrics = evaluate_holdout(model, holdout_cases, sealed)
        curve.append(CurvePoint(
            round=round_number, strategy=strategy_name,
            usable_families=len(snapshot.usable_families),
            metrics=metrics,
            training_accuracy=training_accuracy(
                model,
                [(item.features, item.label) for item in evidence
                 if item.usable_for_training()],
            ),
            spend_cumulative=round(cumulative_spend, 6),
        ))
    return curve


def _curve_to_dict(point: CurvePoint) -> dict[str, Any]:
    return {
        "round": point.round,
        "strategy": point.strategy,
        "usable_training_families": point.usable_families,
        "cumulative_spend": point.spend_cumulative,
        "holdout_accuracy": point.metrics["accuracy"],
        "holdout_accuracy_ci95": point.metrics["accuracy_ci95"],
        "holdout_all_case_accuracy": point.metrics["all_case_accuracy"],
        "brier": point.metrics["brier"],
        "training_accuracy_reference_only": point.training_accuracy,
        "per_class": point.metrics["per_class"],
        "supplier_breakdown": point.metrics["supplier_breakdown"],
    }


def compare_strategies(
    input_path: str | Path,
    state_path: str | Path,
    holdout_path: str | Path,
    strategies: Sequence[str] = STRATEGY_NAMES,
) -> dict[str, Any]:
    """在独立比较集上输出各策略学习曲线、公平性摘要与停采建议。"""
    catalog = load_catalog(input_path)
    ledger = AcquisitionLedger(state_path, catalog)
    holdout_cases, sealed = load_holdout(holdout_path)

    # 硬性家族隔离：比较集家族绝不能出现在训练/候选池里。
    holdout_families = {case.family for case in holdout_cases}
    train_families = {item.family for item in catalog.base_training}
    candidate_families = set(catalog.candidates)
    overlap = holdout_families & (train_families | candidate_families)
    if overlap:
        raise AcquisitionError(f"比较集与训练/候选家族重叠，隔离失败: {sorted(overlap)}")

    unknown = [name for name in strategies if name not in STRATEGY_NAMES]
    if unknown:
        raise AcquisitionError(f"未知策略: {unknown}")

    curves: dict[str, list[CurvePoint]] = {}
    fairness: dict[str, Any] = {}
    for name in strategies:
        curves[name] = _curve_for_strategy(
            name, ledger, catalog, holdout_cases, sealed
        )
        fairness[name] = {
            "rounds_executed": len(ledger.rounds(name)),
            "budget_per_round": catalog.budget_per_round,
            "cumulative_spend": ledger.budget_report(name)["cumulative_spend"],
            "model_class": "GaussianNaiveBayes",
            "eligible_pool_rule": "未采购且具备 training 授权的候选家族",
            "rng_stream": f"{catalog.seed}:{name}",
        }

    learning_curves = {
        name: [_curve_to_dict(point) for point in points]
        for name, points in curves.items()
    }
    # 最终快照横向对比（所有策略在同一份比较集、同一评估器下）
    final_comparison = {
        name: {
            "holdout_accuracy": points[-1].metrics["accuracy"],
            "holdout_accuracy_ci95": points[-1].metrics["accuracy_ci95"],
            "brier": points[-1].metrics["brier"],
            "supplier_breakdown": points[-1].metrics["supplier_breakdown"],
            "per_class": points[-1].metrics["per_class"],
            "training_accuracy_reference_only": points[-1].training_accuracy,
            "usable_training_families": points[-1].usable_families,
            "cumulative_spend": points[-1].spend_cumulative,
        }
        for name, points in curves.items()
    }
    recommendations = {
        name: stop_recommendation(
            points, catalog.round_limit, catalog.budget_per_round
        )
        for name, points in curves.items()
    }
    return {
        "action": "compare",
        "holdout_source": str(holdout_path),
        "holdout_families": len(holdout_families),
        "holdout_sealed_until_evaluation": True,
        "fairness": fairness,
        "learning_curves": learning_curves,
        "final_comparison": final_comparison,
        "stop_recommendations": recommendations,
        "pre_registered_thresholds": {
            "min_marginal_accuracy_gain": 0.085,
            "min_marginal_utility_per_cost": 0.01,
            "note": "阈值预注册，CLI 不暴露，未使用保留集反向调参",
        },
        "state_path": str(state_path),
    }


def export_state_summary(state_path: str | Path) -> dict[str, Any]:
    """读取状态文件用于人工审计（不触发任何写入）。"""
    path = Path(state_path)
    if not path.exists():
        raise AcquisitionError(f"状态文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
