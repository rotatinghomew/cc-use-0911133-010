"""采购闭环的领域协议：数据类、状态常量与受控访问包装。

设计要点：
- :class:`FamilyView` 是候选家族暴露给选样策略的唯一视图，字段白名单里没有 ``label``，
  真标签只存在于 :class:`LabeledFamily`（已入账回执）和独立加载的比较集中；
- :class:`SealedLabels` 把候选/比较集真标签包成显式密封对象，只有训练与评估组件能解包，
  选样策略的构造签名不接受它，从代码结构上隔离不可见标签，而不是靠调用者自觉。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1

TRAINING_PURPOSE = "training"
LABEL_CLASSES: tuple[str, ...] = ("soft", "structured")

# 候选资料允许进入选样视图的字段白名单：刻意不含 label。
CANDIDATE_ALLOWED_KEYS: tuple[str, ...] = (
    "asset_id", "family", "supplier", "features", "purposes", "unit_cost",
)
RECEIPT_ALLOWED_KEYS: tuple[str, ...] = (
    "receipt_id", "family", "revision", "status", "label", "charged_cost",
)


class AcquisitionError(ValueError):
    """采购流程的领域错误（输入不合法、状态机冲突等）。"""


class GuardedHoldoutError(AcquisitionError):
    """选样组件试图触碰保留比较集时抛出，属于硬性隔离违约。"""


@dataclass(frozen=True, slots=True)
class FamilyView:
    """候选家族的无标签视图。重复图片折叠为同一采购单位。"""

    family: str
    supplier: str
    unit_cost: float
    features: tuple[float, ...]
    asset_ids: tuple[str, ...]
    purposes: frozenset[str]
    training_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "supplier": self.supplier,
            "unit_cost": self.unit_cost,
            "features": list(self.features),
            "asset_ids": list(self.asset_ids),
            "purposes": sorted(self.purposes),
            "training_eligible": self.training_eligible,
        }


@dataclass(frozen=True, slots=True)
class LabeledFamily:
    """已通过回执入账的训练证据（标签在回执到达后才存在）。"""

    family: str
    supplier: str
    unit_cost: float
    features: tuple[float, ...]
    asset_ids: tuple[str, ...]
    label: str
    purposes: frozenset[str]
    training_eligible: bool
    receipt_id: str
    revision: int
    charged_cost: float
    received_round: int
    superseded: bool = False
    withdrawn: bool = False

    def usable_for_training(self) -> bool:
        return self.training_eligible and not self.superseded and not self.withdrawn

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "supplier": self.supplier,
            "unit_cost": self.unit_cost,
            "features": list(self.features),
            "asset_ids": list(self.asset_ids),
            "label": self.label,
            "purposes": sorted(self.purposes),
            "training_eligible": self.training_eligible,
            "receipt_id": self.receipt_id,
            "revision": self.revision,
            "charged_cost": self.charged_cost,
            "received_round": self.received_round,
            "superseded": self.superseded,
            "withdrawn": self.withdrawn,
            "usable_for_training": self.usable_for_training(),
        }


@dataclass(frozen=True, slots=True)
class ReceiptRecord:
    """供货商回执的规范化形式。revision 越大越新，更正不重复收费。"""

    receipt_id: str
    family: str
    revision: int
    status: str  # labelled | unjudgeable
    label: str | None
    charged_cost: float

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ReceiptRecord":
        missing = [key for key in ("receipt_id", "family", "revision", "status") if key not in row]
        if missing:
            raise AcquisitionError(f"回执缺少必填字段: {missing}")
        status = str(row["status"])
        if status not in ("labelled", "unjudgeable"):
            raise AcquisitionError(f"回执状态不支持: {status}")
        label = row.get("label")
        if status == "labelled":
            if label not in LABEL_CLASSES:
                raise AcquisitionError(f"回执标签必须是 {LABEL_CLASSES}，实际: {label!r}")
        elif label is not None:
            raise AcquisitionError("unjudgeable 回执不能携带标签")
        try:
            revision = int(row["revision"])
            charged = float(row.get("charged_cost", 0))
        except (TypeError, ValueError) as exc:
            raise AcquisitionError("回执 revision/charged_cost 必须是数值") from exc
        if revision < 1 or charged < 0:
            raise AcquisitionError("回执 revision 必须 >=1，charged_cost 不能为负")
        return cls(
            receipt_id=str(row["receipt_id"]), family=str(row["family"]),
            revision=revision, status=status,
            label=str(label) if status == "labelled" else None,
            charged_cost=charged,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id, "family": self.family,
            "revision": self.revision, "status": self.status,
            "label": self.label, "charged_cost": self.charged_cost,
        }


@dataclass(frozen=True, slots=True)
class HoldoutCase:
    """独立保留比较集中的一个家族（仅评估组件可见）。"""

    family: str
    supplier: str
    features: tuple[float, ...]
    label: str
    training_eligible: bool


class SealedLabels:
    """真标签密封容器。

    比较集/候选真标签通过本对象持有，选样策略拿不到本类型的实例；
    解包方法名显式表达“此处在消费不可见标签”，便于审计调用点。
    """

    __slots__ = ("_labels", "_source")

    def __init__(self, labels: Mapping[str, str], source: str) -> None:
        self._labels: dict[str, str] = dict(labels)
        self._source = source

    @property
    def source(self) -> str:
        return self._source

    def unseal_for_evaluation(self) -> dict[str, str]:
        return dict(self._labels)

    def __len__(self) -> int:
        return len(self._labels)


@dataclass(frozen=True, slots=True)
class Withdrawal:
    """授权撤销：某家族自指定轮次起不得再用于训练。"""

    family: str
    purpose: str
    effective_round: int


@dataclass(slots=True)
class InputCatalog:
    """一次采购活动的全部输入快照（候选视图无标签）。"""

    seed: int
    budget_per_round: float
    round_limit: int
    candidates: dict[str, FamilyView] = field(default_factory=dict)
    base_training: tuple[LabeledFamily, ...] = ()
    withdrawals: tuple[Withdrawal, ...] = ()
    input_fingerprint: str = ""

    def purchasable_families(self) -> list[str]:
        return sorted(self.candidates)

    def assert_family_is_purchasable(self, family: str) -> None:
        if family not in self.candidates:
            raise AcquisitionError(f"家族 {family} 不在候选采购池中")
