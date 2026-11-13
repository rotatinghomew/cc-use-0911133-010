"""资料加载与字段白名单过滤。

``fixtures/label_acquisition.json`` 同时包含训练数据（带标签）和候选池（无标签），
还夹带了 ``receipt_revision_examples`` 这类模拟回执片段。加载器在这里做三件事：

1. 用 :data:`CANDIDATE_ALLOWED_KEYS` 白名单构造候选视图，即使输入文件被塞入候选 ``label``
   也会在加载阶段被剥离并报错，保证选样接口物理上看不到候选真标签；
2. 把模拟回执片段从选样输入里剔除，交给 ingest 阶段显式入账；
3. 保留比较集由 :func:`load_holdout` 单独加载，返回 :class:`SealedLabels`，
   propose 流程的任何对象都不持有它。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from fashion_data_foundry.acquisition.protocol import (
    CANDIDATE_ALLOWED_KEYS,
    AcquisitionError,
    FamilyView,
    HoldoutCase,
    InputCatalog,
    LabeledFamily,
    SealedLabels,
    TRAINING_PURPOSE,
    Withdrawal,
)
from fashion_data_foundry.governance.asset_ledger import fingerprint


def _read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise AcquisitionError(f"资料文件不存在: {file_path}")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AcquisitionError(f"资料文件不是合法 JSON: {file_path}") from exc
    if not isinstance(data, dict):
        raise AcquisitionError("采购资料根节点必须是对象")
    return data


def _features(row: Mapping[str, Any], context: str) -> tuple[float, ...]:
    raw = row.get("features")
    if not isinstance(raw, list) or not raw:
        raise AcquisitionError(f"{context} 缺少非空 features 列表")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise AcquisitionError(f"{context} 的 features 必须全部是数值") from exc
    return values


def _purposes(row: Mapping[str, Any]) -> frozenset[str]:
    raw = row.get("purposes")
    if not isinstance(raw, list) or not raw:
        raise AcquisitionError("purposes 必须是非空列表")
    return frozenset(str(item) for item in raw)


def _assert_no_candidate_label(row: Mapping[str, Any]) -> None:
    # 硬性隔离：候选行一旦带 label（或疑似标签键），直接拒绝而不是静默忽略，
    # 防止“看不见但其实在文件里”的隐性泄漏。
    leaked = [key for key in row if key == "label" or key.endswith("_label")]
    if leaked:
        raise AcquisitionError(
            f"候选资料 {row.get('asset_id', '?')} 携带了禁止暴露的标签字段 {leaked}；"
            "候选真标签只能通过回执或独立比较集进入系统"
        )


def _sanitized_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    _assert_no_candidate_label(row)
    return {key: row[key] for key in CANDIDATE_ALLOWED_KEYS if key in row}


def load_catalog(path: str | Path) -> InputCatalog:
    """加载采购输入；返回的 :class:`InputCatalog` 不含任何候选真标签。"""
    data = _read_json(path)
    if data.get("schema_version") != 1:
        raise AcquisitionError("仅支持 schema_version=1 的采购资料")

    candidates: dict[str, FamilyView] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for raw in data.get("candidate_pool", []):
        if not isinstance(raw, Mapping):
            raise AcquisitionError("candidate_pool 条目必须是对象")
        row = _sanitized_candidate_row(raw)
        family = str(row.get("family", "")).strip()
        if not family:
            raise AcquisitionError("候选条目缺少 family")
        grouped.setdefault(family, []).append(row)

    for family, rows in grouped.items():
        supplier = str(rows[0]["supplier"])
        unit_cost = float(rows[0]["unit_cost"])
        purposes = _purposes(rows[0])
        features = _features(rows[0], f"候选 {family}")
        asset_ids: list[str] = []
        for row in rows:
            # 同家族必须保持报价、供货商、特征、用途一致，否则“按家族采购”无法成立。
            if str(row["supplier"]) != supplier:
                raise AcquisitionError(f"家族 {family} 内部供货商不一致")
            if float(row["unit_cost"]) != unit_cost:
                raise AcquisitionError(f"家族 {family} 内部报价不一致，重复图片不能重复计价")
            if _features(row, f"候选 {family}") != features:
                raise AcquisitionError(f"家族 {family} 内部特征不一致")
            if _purposes(row) != purposes:
                raise AcquisitionError(f"家族 {family} 内部用途授权不一致")
            asset_ids.append(str(row["asset_id"]))
        if len(set(asset_ids)) != len(asset_ids):
            raise AcquisitionError(f"家族 {family} 内部 asset_id 重复")
        candidates[family] = FamilyView(
            family=family, supplier=supplier, unit_cost=unit_cost, features=features,
            asset_ids=tuple(asset_ids), purposes=purposes,
            training_eligible=TRAINING_PURPOSE in purposes,
        )

    base_training: list[LabeledFamily] = []
    base_families: set[str] = set()
    for raw in data.get("training", []):
        if not isinstance(raw, Mapping):
            raise AcquisitionError("training 条目必须是对象")
        family = str(raw.get("family", "")).strip()
        if not family:
            raise AcquisitionError("训练条目缺少 family")
        if family in candidates:
            raise AcquisitionError(f"家族 {family} 同时出现在训练集与候选池，隔离失败")
        if family in base_families:
            raise AcquisitionError(f"训练集家族 {family} 重复")
        base_families.add(family)
        label = raw.get("label")
        if label not in ("soft", "structured"):
            raise AcquisitionError(f"训练家族 {family} 标签非法: {label!r}")
        purposes = _purposes(raw)
        base_training.append(
            LabeledFamily(
                family=family, supplier=str(raw["supplier"]),
                unit_cost=float(raw["unit_cost"]), features=_features(raw, f"训练 {family}"),
                asset_ids=(str(raw["asset_id"]),), label=str(label), purposes=purposes,
                training_eligible=TRAINING_PURPOSE in purposes,
                receipt_id="baseline", revision=0, charged_cost=0.0,
                received_round=0,
            )
        )

    withdrawals: list[Withdrawal] = []
    for raw in data.get("withdrawals", []):
        if not isinstance(raw, Mapping) or "family" not in raw or "effective_round" not in raw:
            raise AcquisitionError("withdrawals 条目必须包含 family 与 effective_round")
        effective = int(raw["effective_round"])
        if effective < 1:
            raise AcquisitionError("撤销生效轮次必须 >= 1")
        withdrawals.append(
            Withdrawal(family=str(raw["family"]), purpose=str(raw.get("purpose", TRAINING_PURPOSE)),
                       effective_round=effective)
        )

    try:
        seed = int(data["seed"])
        budget = float(data["budget_per_round"])
        round_limit = int(data["round_limit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AcquisitionError("seed/budget_per_round/round_limit 字段非法") from exc
    if budget <= 0 or round_limit < 1:
        raise AcquisitionError("预算必须为正，轮次上限必须 >= 1")

    return InputCatalog(
        seed=seed, budget_per_round=budget, round_limit=round_limit,
        candidates=candidates, base_training=tuple(base_training),
        withdrawals=tuple(withdrawals), input_fingerprint=fingerprint(data),
    )


def load_receipt_bundle(path: str | Path) -> list[Mapping[str, Any]]:
    """加载模拟回执文件（含更正片段），不做选样可见性处理。"""
    data = _read_json(path)
    if str(data.get("allowed_use", "")) != "receipt-simulation-only":
        raise AcquisitionError("回执文件 allowed_use 必须是 receipt-simulation-only")
    receipts = data.get("receipts")
    if not isinstance(receipts, list):
        raise AcquisitionError("回执文件缺少 receipts 列表")
    return receipts


def load_revision_examples(path: str | Path) -> list[Mapping[str, Any]]:
    """从采购输入里取出夹带的更正回执片段；它们不属于选样输入。"""
    data = _read_json(path)
    examples = data.get("receipt_revision_examples", [])
    if not isinstance(examples, list):
        raise AcquisitionError("receipt_revision_examples 必须是列表")
    return examples


def load_holdout(path: str | Path) -> tuple[list[HoldoutCase], SealedLabels]:
    """加载独立保留比较集，返回评估用案例与密封标签。

    本函数只应由 compare/evaluation 调用；propose 服务不接收本文件路径。
    """
    data = _read_json(path)
    if str(data.get("allowed_use", "")) != "comparison-only":
        raise AcquisitionError("比较集文件 allowed_use 必须是 comparison-only")
    rows = data.get("holdout")
    if not isinstance(rows, list) or not rows:
        raise AcquisitionError("比较集缺少非空 holdout 列表")
    cases: list[HoldoutCase] = []
    sealed: dict[str, str] = {}
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise AcquisitionError("holdout 条目必须是对象")
        family = str(raw.get("family", "")).strip()
        if not family or family in seen:
            raise AcquisitionError(f"比较集家族缺失或重复: {family!r}")
        seen.add(family)
        label = raw.get("label")
        if label not in ("soft", "structured"):
            raise AcquisitionError(f"比较集家族 {family} 标签非法")
        purposes = _purposes(raw)
        cases.append(
            HoldoutCase(
                family=family, supplier=str(raw["supplier"]),
                features=_features(raw, f"比较集 {family}"), label=str(label),
                training_eligible=TRAINING_PURPOSE in purposes,
            )
        )
        sealed[family] = str(label)
    return cases, SealedLabels(sealed, source=str(path))
