"""轮次状态机测试：预算锁定、幂等恢复、乱序/更正/撤销语义。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fashion_data_foundry.acquisition.catalog import load_catalog
from fashion_data_foundry.acquisition.ledger import (
    ACTION_CORRECTION,
    ACTION_DUPLICATE,
    ACTION_STALE,
    AcquisitionLedger,
)
from fashion_data_foundry.acquisition.loop import ingest_round, propose_round
from fashion_data_foundry.acquisition.protocol import (
    AcquisitionError,
    ReceiptRecord,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "fixtures" / "label_acquisition.json"
RECEIPTS = ROOT / "fixtures" / "acquisition_receipts.json"


def receipt(family: str, revision: int = 1, status: str = "labelled",
            label: str | None = "soft", charged: float | None = None,
            receipt_id: str | None = None) -> ReceiptRecord:
    catalog = load_catalog(INPUT)
    if charged is None:
        charged = catalog.candidates[family].unit_cost if revision == 1 else 0.0
    return ReceiptRecord(
        receipt_id=receipt_id or f"receipt-{family}",
        family=family, revision=revision, status=status, label=label,
        charged_cost=charged,
    )


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "state.json"
        self.catalog = load_catalog(INPUT)

    def test_proposal_locks_budget_per_family_without_double_counting_repeats(self):
        result = propose_round(INPUT, self.state, "balanced")
        families = [order["family"] for order in result["orders"]]
        self.assertEqual(len(families), len(set(families)))
        self.assertLessEqual(result["order_spend"], self.catalog.budget_per_round + 1e-9)
        # 含重复图片的家族若被选中，也只按一个 unit_cost 计费
        ledger = AcquisitionLedger(self.state, self.catalog)
        for round_record in ledger.rounds("balanced"):
            for order in round_record["orders"]:
                view = self.catalog.candidates[order["family"]]
                self.assertEqual(order["unit_cost"], view.unit_cost)
                self.assertGreaterEqual(len(view.asset_ids), 1)

    def test_propose_is_idempotent_and_does_not_reopen_ingested_round(self):
        first = propose_round(INPUT, self.state, "balanced")
        second = propose_round(INPUT, self.state, "balanced")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(
            [o["family"] for o in first["orders"]],
            [o["family"] for o in second["orders"]],
        )
        ledger = AcquisitionLedger(self.state, self.catalog)
        self.assertEqual(len(ledger.rounds("balanced")), 1)

    def test_ingest_before_propose_is_rejected(self):
        with self.assertRaisesRegex(AcquisitionError, "先执行 propose"):
            ingest_round(INPUT, self.state, [RECEIPTS], "balanced")

    def test_partial_receipts_keep_other_orders_pending(self):
        proposal = propose_round(INPUT, self.state, "balanced")
        target_family = proposal["orders"][0]["family"]
        payload = ingest_round(
            INPUT, self.state,
            [self._write_bundle([self._raw(target_family)])],
            "balanced", include_revision_examples=False,
        )
        pending = set(payload["evidence"]["pending_receipt_families"])
        self.assertNotIn(target_family, pending)
        self.assertGreater(len(pending), 0)

    def test_interrupted_re_ingest_never_charges_twice(self):
        propose_round(INPUT, self.state, "balanced")
        ingest_round(INPUT, self.state, [RECEIPTS], "balanced",
                     include_revision_examples=False)
        before = json.loads(self.state.read_text(encoding="utf-8"))
        before_spend = sum(r["spend"] for r in before["strategies"]["balanced"]["rounds"])
        payload = ingest_round(INPUT, self.state, [RECEIPTS], "balanced",
                               include_revision_examples=False)
        after = json.loads(self.state.read_text(encoding="utf-8"))
        after_spend = sum(r["spend"] for r in after["strategies"]["balanced"]["rounds"])
        self.assertEqual(before_spend, after_spend)
        ordered_actions = [
            entry["action"] for entry in payload["ingest_log"]
            if entry["family"] in {o["family"] for o in
                                   before["strategies"]["balanced"]["rounds"][-1]["orders"]}
        ]
        self.assertTrue(ordered_actions)
        self.assertTrue(all(action == ACTION_DUPLICATE for action in ordered_actions))

    def test_out_of_order_revisions_take_latest_and_ignore_stale(self):
        proposal = propose_round(INPUT, self.state, "balanced")
        family = proposal["orders"][0]["family"]
        ledger = AcquisitionLedger(self.state, self.catalog)
        round_number = 1
        # 先到更正件 revision=2，再补首单 revision=1（乱序）
        log = ledger.apply_receipts("balanced", round_number, [
            receipt(family, revision=2, label="structured", charged=0.0),
        ])
        self.assertEqual([entry["action"] for entry in log], [ACTION_CORRECTION])
        log = ledger.apply_receipts("balanced", round_number, [
            receipt(family, revision=1, label="soft"),
        ])
        self.assertEqual([entry["action"] for entry in log], [ACTION_STALE])
        ledger.save()
        evidence, _ = ledger.build_evidence("balanced", 1)
        labeled = next(item for item in evidence if item.family == family)
        self.assertEqual(labeled.label, "structured")
        self.assertEqual(labeled.revision, 2)
        # 更正不收费：花费仍等于 propose 锁定值
        self.assertEqual(
            sum(r["spend"] for r in ledger.rounds("balanced")),
            proposal["order_spend"],
        )

    def test_same_receipt_id_correction_is_not_treated_as_duplicate(self):
        proposal = propose_round(INPUT, self.state, "balanced")
        family = proposal["orders"][0]["family"]
        ledger = AcquisitionLedger(self.state, self.catalog)
        ledger.apply_receipts("balanced", 1, [
            receipt(family, revision=1, status="unjudgeable", label=None,
                    receipt_id="receipt-x"),
        ])
        log = ledger.apply_receipts("balanced", 1, [
            receipt(family, revision=2, status="labelled", label="structured",
                    charged=0.0, receipt_id="receipt-x"),
        ])
        self.assertEqual(log[0]["action"], ACTION_CORRECTION)

    def test_correction_with_extra_charge_is_rejected(self):
        proposal = propose_round(INPUT, self.state, "balanced")
        family = proposal["orders"][0]["family"]
        ledger = AcquisitionLedger(self.state, self.catalog)
        ledger.apply_receipts("balanced", 1, [receipt(family, revision=1)])
        with self.assertRaisesRegex(AcquisitionError, "不得追加费用"):
            ledger.apply_receipts("balanced", 1, [
                receipt(family, revision=2, charged=1.0),
            ])

    def test_first_receipt_cost_must_match_order_price(self):
        proposal = propose_round(INPUT, self.state, "balanced")
        family = proposal["orders"][0]["family"]
        ledger = AcquisitionLedger(self.state, self.catalog)
        with self.assertRaisesRegex(AcquisitionError, "首单费用"):
            ledger.apply_receipts("balanced", 1, [
                receipt(family, revision=1, charged=99.0),
            ])

    def test_unjudgeable_family_costs_money_but_yields_no_evidence(self):
        # 直接构造一个必定可采购的便宜家族场景
        proposal = propose_round(INPUT, self.state, "uncertainty")
        ledger = AcquisitionLedger(self.state, self.catalog)
        target_round = ledger.rounds("uncertainty")[0]
        family = target_round["orders"][0]["family"]
        ledger.apply_receipts("uncertainty", 1, [
            receipt(family, revision=1, status="unjudgeable", label=None),
        ])
        ledger.save()
        evidence, explanation = ledger.build_evidence("uncertainty", 1)
        self.assertNotIn(family, {item.family for item in evidence
                                  if item.usable_for_training()})
        self.assertIn(family, explanation["unjudgeable_families"])
        self.assertEqual(target_round["spend"], proposal["order_spend"])

    def test_withdrawal_removes_family_from_later_training_snapshot(self):
        # family-050 的撤销在第 2 轮生效；先保证它在第 1 轮被采购
        proposal = propose_round(INPUT, self.state, "balanced")
        families = [o["family"] for o in proposal["orders"]]
        if "family-050" not in families:
            self.skipTest("本次采样未选中 family-050")
        ingest_round(INPUT, self.state, [RECEIPTS], "balanced",
                     include_revision_examples=False)
        evidence_r1, _ = AcquisitionLedger(self.state, self.catalog).build_evidence(
            "balanced", 1)
        self.assertIn("family-050", {item.family for item in evidence_r1
                                     if item.usable_for_training()})
        propose_round(INPUT, self.state, "balanced")
        ingest_round(INPUT, self.state, [RECEIPTS], "balanced",
                     include_revision_examples=False)
        evidence_r2, explanation = AcquisitionLedger(
            self.state, self.catalog).build_evidence("balanced", 2)
        self.assertNotIn("family-050", {item.family for item in evidence_r2
                                        if item.usable_for_training()})
        self.assertIn("family-050", explanation["withdrawn_families"])

    def test_cannot_propose_next_round_before_ingest(self):
        propose_round(INPUT, self.state, "balanced")
        with self.assertRaisesRegex(AcquisitionError, "尚未 ingest"):
            ledger = AcquisitionLedger(self.state, self.catalog)
            ledger.record_proposal("balanced", 2, [])

    def test_state_rejects_fingerprint_mismatch(self):
        propose_round(INPUT, self.state, "balanced")
        other = load_catalog(INPUT)
        object.__setattr__(other, "input_fingerprint", "different")
        with self.assertRaisesRegex(AcquisitionError, "指纹不一致"):
            AcquisitionLedger(self.state, other)

    def test_round_limit_is_enforced(self):
        for _ in range(2):
            propose_round(INPUT, self.state, "balanced")
            ingest_round(INPUT, self.state, [RECEIPTS], "balanced",
                         include_revision_examples=False)
        with self.assertRaisesRegex(AcquisitionError, "轮次上限"):
            propose_round(INPUT, self.state, "balanced")

    # ------------------------------------------------------------------ helpers
    def _raw(self, family: str) -> dict:
        view = self.catalog.candidates[family]
        return {
            "receipt_id": f"receipt-{family}", "family": family, "revision": 1,
            "status": "labelled", "label": "soft",
            "charged_cost": view.unit_cost,
        }

    def _write_bundle(self, rows: list[dict]) -> Path:
        path = Path(self.temp.name) / f"bundle-{len(rows)}.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "allowed_use": "receipt-simulation-only",
            "receipts": rows,
        }), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
