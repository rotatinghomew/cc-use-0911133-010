"""采购轮次状态机与 JSON 账本。

状态文件 (:file:`*.acquisition.json`) 是唯一事实来源，关键语义：

- **费用在 propose 时锁定**：采购单位是家族，订单落盘即按 ``unit_cost`` 记一次费，
  重复图片折叠在同一订单里；回执只改变证据，绝不再次计费。重复 ingest 同一回执
  走 ``duplicate_skipped``，中断恢复不可能重复付费。
- **轮次阶段**：``proposed -> ingested``。ingest 可多次调用（迟到回执、更正），
  但进入下一轮 propose 前，上一轮必须至少 ingest 过一次。
- **更正与乱序**：同家族以最大 revision 为准；更旧的回执 ``stale_ignored``；
  revision=1 的回执 charged_cost 必须等于报价，更高 revision 必须为 0（更正不收费）。
- **撤销**：授权撤销来自输入资料，生效轮次起该家族证据标记 withdrawn，
  历史训练快照保留、后续快照排除，失效素材不会继续进训练。
- 所有写操作先写临时文件再 ``os.replace``，崩溃不会留下半截状态。
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from fashion_data_foundry.acquisition.protocol import (
    SCHEMA_VERSION,
    AcquisitionError,
    FamilyView,
    InputCatalog,
    LabeledFamily,
    ReceiptRecord,
)
from fashion_data_foundry.governance.asset_ledger import utc_now

PHASE_PROPOSED = "proposed"
PHASE_INGESTED = "ingested"

# 订单/回执处理动作
ACTION_FIRST = "first_receipt"
ACTION_CORRECTION = "correction"
ACTION_DUPLICATE = "duplicate_skipped"
ACTION_STALE = "stale_ignored"


def init_state(catalog: InputCatalog) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "input_fingerprint": catalog.input_fingerprint,
        "seed": catalog.seed,
        "budget_per_round": catalog.budget_per_round,
        "round_limit": catalog.round_limit,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "strategies": {},
        "withdrawals": [
            {"family": item.family, "purpose": item.purpose,
             "effective_round": item.effective_round}
            for item in catalog.withdrawals
        ],
        "events": [],
    }


class AcquisitionLedger:
    """负责状态文件的加载、原子落盘与状态机校验。"""

    def __init__(self, path: str | Path, catalog: InputCatalog) -> None:
        self.path = Path(path)
        self.catalog = catalog
        self.state = self._load_or_init()

    # ------------------------------------------------------------------ IO
    def _load_or_init(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise AcquisitionError(f"状态文件损坏: {self.path}") from exc
            if state.get("input_fingerprint") != self.catalog.input_fingerprint:
                raise AcquisitionError(
                    "状态文件与输入资料指纹不一致，拒绝把旧采购活动接到新资料上"
                )
            if state.get("schema_version") != SCHEMA_VERSION:
                raise AcquisitionError("状态文件 schema_version 不受支持")
            return state
        return init_state(self.catalog)

    def _atomic_write(self) -> None:
        self.state["updated_at"] = utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def save(self) -> None:
        self._atomic_write()

    def emit(self, kind: str, **detail: Any) -> None:
        self.state["events"].append(
            {"at": utc_now(), "kind": kind, "detail": detail}
        )

    # -------------------------------------------------------------- strategy
    def ensure_strategy(self, strategy_name: str) -> dict[str, Any]:
        strategies = self.state["strategies"]
        if strategy_name not in strategies:
            strategies[strategy_name] = {"rounds": [], "events": []}
        return strategies[strategy_name]

    def rounds(self, strategy_name: str) -> list[dict[str, Any]]:
        return self.ensure_strategy(strategy_name)["rounds"]

    def current_round_number(self, strategy_name: str) -> int:
        """当前活动轮次号：末轮仍停在 proposed 时复用该轮，否则开启下一轮。"""
        rounds = self.rounds(strategy_name)
        if rounds and rounds[-1]["phase"] == PHASE_PROPOSED:
            return rounds[-1]["round"]
        return len(rounds) + 1

    def latest_round(self, strategy_name: str) -> dict[str, Any] | None:
        rounds = self.rounds(strategy_name)
        return rounds[-1] if rounds else None

    # ------------------------------------------------------------- proposals
    def record_proposal(
        self,
        strategy_name: str,
        round_number: int,
        orders: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """落盘一轮订单；重复 propose 同一轮直接回放已有订单（幂等）。"""
        rounds = self.rounds(strategy_name)
        existing = next((item for item in rounds if item["round"] == round_number), None)
        if existing is not None:
            if existing["phase"] != PHASE_PROPOSED:
                raise AcquisitionError(
                    f"策略 {strategy_name} 第 {round_number} 轮已入账，不能重新 propose"
                )
            return existing
        if round_number != len(rounds) + 1:
            raise AcquisitionError("轮次必须连续推进，不能跳轮 propose")
        if round_number > self.state["round_limit"]:
            raise AcquisitionError(
                f"已达到轮次上限 {self.state['round_limit']}，不能再开新轮"
            )
        if rounds and rounds[-1]["phase"] != PHASE_INGESTED:
            raise AcquisitionError(
                f"第 {round_number - 1} 轮尚未 ingest，回执未到齐不能提前开下一轮"
            )
        spend = round(sum(float(order["unit_cost"]) for order in orders), 6)
        if spend > self.state["budget_per_round"] + 1e-9:
            raise AcquisitionError(
                f"订单总价 {spend} 超过本轮预算 {self.state['budget_per_round']}"
            )
        families = [order["family"] for order in orders]
        if len(set(families)) != len(families):
            raise AcquisitionError("同一轮不能重复采购同一家族")
        round_record = {
            "round": round_number,
            "phase": PHASE_PROPOSED,
            "spend": spend,
            "budget": self.state["budget_per_round"],
            "proposed_at": utc_now(),
            "ingested_at": None,
            "orders": [dict(order, status="pending") for order in orders],
            "receipts": [],
            "ingest_log": [],
        }
        rounds.append(round_record)
        self.ensure_strategy(strategy_name)["events"].append(
            {"at": utc_now(), "kind": "proposal_recorded",
             "detail": {"round": round_number, "families": families, "spend": spend}}
        )
        self.emit("proposal_recorded", strategy=strategy_name, round=round_number,
                  family_count=len(families), spend=spend)
        return round_record

    # --------------------------------------------------------------- receipts
    def _all_order_families(self, strategy_name: str) -> dict[str, dict[str, Any]]:
        orders: dict[str, dict[str, Any]] = {}
        for round_record in self.rounds(strategy_name):
            for order in round_record["orders"]:
                orders[order["family"]] = order
        return orders

    def _applied_receipt_pairs(self, strategy_name: str) -> set[tuple[str, int]]:
        """已入账的 (family, revision) 对；更正回执复用 receipt_id 时靠 revision 区分。"""
        applied: set[tuple[str, int]] = set()
        for round_record in self.rounds(strategy_name):
            for entry in round_record["ingest_log"]:
                if entry["action"] != ACTION_STALE:
                    applied.add((entry["family"], int(entry["revision"])))
        return applied

    def _family_latest_revision(self, strategy_name: str, family: str) -> int:
        latest = 0
        for round_record in self.rounds(strategy_name):
            for receipt in round_record["receipts"]:
                if receipt["family"] == family:
                    latest = max(latest, int(receipt["revision"]))
        return latest

    def apply_receipts(
        self,
        strategy_name: str,
        round_number: int,
        receipts: Sequence[ReceiptRecord],
    ) -> list[dict[str, Any]]:
        """把一批回执计入指定轮次；支持部分回执、乱序、更正、重复与中断恢复。"""
        target = next(
            (item for item in self.rounds(strategy_name) if item["round"] == round_number),
            None,
        )
        if target is None:
            raise AcquisitionError(
                f"策略 {strategy_name} 第 {round_number} 轮还没有 propose，无法 ingest"
            )
        orders = self._all_order_families(strategy_name)
        applied_pairs = self._applied_receipt_pairs(strategy_name)
        log_entries: list[dict[str, Any]] = []
        for receipt in receipts:
            if receipt.family not in orders:
                # 全量回执文件里含未采购家族：忽略并留痕，费用与证据都不受影响。
                log_entries.append({
                    "at": utc_now(), "round": round_number,
                    "receipt_id": receipt.receipt_id, "family": receipt.family,
                    "revision": receipt.revision, "action": "not_ordered_ignored",
                    "charged_cost": receipt.charged_cost,
                    "note": "该家族未在任何轮次被采购，回执忽略",
                })
                continue
            latest_revision = self._family_latest_revision(strategy_name, receipt.family)
            pair = (receipt.family, receipt.revision)
            if pair in applied_pairs:
                # 同一 (family, revision) 再次投递才是重复；更正件 revision 不同，不会命中。
                entry = self._log(target, receipt, ACTION_DUPLICATE,
                                  note="同 (family, revision) 回执已处理过，跳过且不重复计费")
                log_entries.append(entry)
                continue
            if receipt.revision < latest_revision:
                entry = self._log(target, receipt, ACTION_STALE,
                                  note=f"已有更新 revision={latest_revision}，旧回执忽略")
                log_entries.append(entry)
                continue
            order = orders[receipt.family]
            if receipt.revision == 1:
                if abs(receipt.charged_cost - float(order["unit_cost"])) > 1e-9:
                    raise AcquisitionError(
                        f"回执 {receipt.receipt_id} 首单费用 {receipt.charged_cost} "
                        f"与订单报价 {order['unit_cost']} 不一致"
                    )
                action = ACTION_FIRST
                note = "首单回执，费用在 propose 时已锁定"
            else:
                if receipt.charged_cost != 0:
                    raise AcquisitionError(
                        f"更正回执 {receipt.receipt_id} 不得追加费用，"
                        f"charged_cost={receipt.charged_cost}"
                    )
                action = ACTION_CORRECTION
                note = f"更正至 revision={receipt.revision}，不重复收费"
            target["receipts"].append(receipt.to_dict())
            entry = self._log(target, receipt, action, note=note)
            log_entries.append(entry)
            applied_pairs.add(pair)
        self._refresh_order_statuses(strategy_name)
        target["phase"] = PHASE_INGESTED
        target["ingested_at"] = utc_now()
        self.ensure_strategy(strategy_name)["events"].append(
            {"at": utc_now(), "kind": "receipts_ingested",
             "detail": {"round": round_number, "receipt_count": len(receipts)}}
        )
        self.emit("receipts_ingested", strategy=strategy_name, round=round_number,
                  receipt_count=len(receipts))
        return log_entries

    def _log(
        self, round_record: dict[str, Any], receipt: ReceiptRecord,
        action: str, note: str,
    ) -> dict[str, Any]:
        entry = {
            "at": utc_now(), "round": round_record["round"],
            "receipt_id": receipt.receipt_id, "family": receipt.family,
            "revision": receipt.revision, "action": action,
            "charged_cost": receipt.charged_cost, "note": note,
        }
        round_record["ingest_log"].append(entry)
        return entry

    def _refresh_order_statuses(self, strategy_name: str) -> None:
        """根据各家族最新回执与撤销授权，重算订单状态。"""
        latest: dict[str, dict[str, Any]] = {}
        for round_record in self.rounds(strategy_name):
            for receipt in round_record["receipts"]:
                current = latest.get(receipt["family"])
                if current is None or receipt["revision"] > current["revision"]:
                    latest[receipt["family"]] = receipt
        withdrawn_by_round = self._withdrawn_families()
        for round_record in self.rounds(strategy_name):
            for order in round_record["orders"]:
                family = order["family"]
                effective = withdrawn_by_round.get(family)
                if effective is not None and effective <= round_record["round"]:
                    order["status"] = "withdrawn"
                    continue
                receipt = latest.get(family)
                if receipt is None:
                    order["status"] = "pending"
                elif receipt["status"] == "unjudgeable":
                    order["status"] = "unjudgeable"
                else:
                    order["status"] = "labelled"
                order["latest_revision"] = receipt["revision"] if receipt else 0
                if receipt and receipt["status"] == "labelled":
                    order["latest_label"] = receipt["label"]

    def _withdrawn_families(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in self.state["withdrawals"]:
            if item["purpose"] != "training":
                continue
            result[item["family"]] = min(
                result.get(item["family"], item["effective_round"]),
                item["effective_round"],
            )
        return result

    # -------------------------------------------------------------- evidence
    def build_evidence(
        self, strategy_name: str, as_of_round: int
    ) -> tuple[list[LabeledFamily], dict[str, Any]]:
        """组装截至某轮末尾的全部训练证据，并解释每条证据为何可用/排除。"""
        latest_receipts: dict[str, dict[str, Any]] = {}
        first_seen_round: dict[str, int] = {}
        for round_record in self.rounds(strategy_name):
            if round_record["round"] > as_of_round:
                continue
            for receipt in round_record["receipts"]:
                family = receipt["family"]
                if family not in first_seen_round:
                    first_seen_round[family] = round_record["round"]
                current = latest_receipts.get(family)
                if current is None or receipt["revision"] > current["revision"]:
                    latest_receipts[family] = receipt
        withdrawn = self._withdrawn_families()
        evidence: list[LabeledFamily] = []
        for item in self.catalog.base_training:
            effective = withdrawn.get(item.family)
            evidence.append(
                replace(item, withdrawn=bool(
                    effective is not None and effective <= as_of_round
                ))
            )
        for family, receipt in sorted(latest_receipts.items()):
            if receipt["status"] != "labelled":
                continue  # unjudgeable 不产生训练证据，但费用已花
            view: FamilyView = self.catalog.candidates[family]
            effective = withdrawn.get(family)
            evidence.append(
                LabeledFamily(
                    family=family, supplier=view.supplier, unit_cost=view.unit_cost,
                    features=view.features, asset_ids=view.asset_ids,
                    label=receipt["label"], purposes=view.purposes,
                    training_eligible=view.training_eligible,
                    receipt_id=receipt["receipt_id"], revision=receipt["revision"],
                    charged_cost=float(view.unit_cost),
                    received_round=first_seen_round[family],
                    withdrawn=effective is not None and effective <= as_of_round,
                )
            )
        explanation = self._evidence_explanation(strategy_name, as_of_round, evidence)
        return evidence, explanation

    def _evidence_explanation(
        self, strategy_name: str, as_of_round: int,
        evidence: Sequence[LabeledFamily],
    ) -> dict[str, Any]:
        pending: list[str] = []
        unjudgeable: list[str] = []
        for round_record in self.rounds(strategy_name):
            if round_record["round"] > as_of_round:
                continue
            for order in round_record["orders"]:
                if order["status"] == "pending":
                    pending.append(order["family"])
                elif order["status"] == "unjudgeable":
                    unjudgeable.append(order["family"])
        rights_blocked = sorted(
            view.family for view in self.catalog.candidates.values()
            if not view.training_eligible
        )
        return {
            "as_of_round": as_of_round,
            "pending_receipt_families": sorted(pending),
            "unjudgeable_families": sorted(unjudgeable),
            "rights_blocked_families": rights_blocked,
            "withdrawn_families": sorted(
                family for family, effective in self._withdrawn_families().items()
                if effective <= as_of_round
            ),
            "excluded_from_training": sorted(
                item.family for item in evidence if not item.usable_for_training()
            ),
        }

    def acquired_families(self, strategy_name: str) -> frozenset[str]:
        families: set[str] = set()
        for round_record in self.rounds(strategy_name):
            families.update(order["family"] for order in round_record["orders"])
        return frozenset(families)

    def budget_report(self, strategy_name: str) -> dict[str, Any]:
        rounds = []
        cumulative = 0.0
        for round_record in self.rounds(strategy_name):
            cumulative += float(round_record["spend"])
            orders = round_record["orders"]
            rounds.append({
                "round": round_record["round"],
                "phase": round_record["phase"],
                "budget": round_record["budget"],
                "spend": round_record["spend"],
                "remaining": round(round_record["budget"] - float(round_record["spend"]), 6),
                "families": len(orders),
                "labelled": sum(1 for o in orders if o["status"] == "labelled"),
                "unjudgeable": sum(1 for o in orders if o["status"] == "unjudgeable"),
                "pending": sum(1 for o in orders if o["status"] == "pending"),
                "withdrawn": sum(1 for o in orders if o["status"] == "withdrawn"),
            })
        return {
            "budget_per_round": self.state["budget_per_round"],
            "round_limit": self.state["round_limit"],
            "cumulative_spend": round(cumulative, 6),
            "rounds": rounds,
        }
