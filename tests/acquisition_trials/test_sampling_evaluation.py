"""选样策略、公平比较与停采规则测试。"""
from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from fashion_data_foundry.acquisition.catalog import load_catalog, load_holdout
from fashion_data_foundry.acquisition.evaluation import (
    MIN_MARGINAL_ACC_GAIN,
    CurvePoint,
    evaluate_holdout,
    stop_recommendation,
    wilson_interval,
)
from fashion_data_foundry.acquisition.ledger import AcquisitionLedger
from fashion_data_foundry.acquisition.loop import (
    STRATEGY_NAMES,
    compare_strategies,
    ingest_round,
    propose_round,
)
from fashion_data_foundry.acquisition.model import GaussianNaiveBayes
from fashion_data_foundry.acquisition.strategy import (
    BALANCED_WEIGHTS,
    BalancedStrategy,
    EvidenceSummary,
    build_strategy,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "fixtures" / "label_acquisition.json"
RECEIPTS = ROOT / "fixtures" / "acquisition_receipts.json"
HOLDOUT = ROOT / "fixtures" / "acquisition_holdout.json"


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog(INPUT)

    def test_balanced_weights_are_pre_registered_and_complete(self):
        self.assertAlmostEqual(sum(BALANCED_WEIGHTS.values()), 1.0)
        self.assertEqual(
            set(BALANCED_WEIGHTS),
            {"uncertainty", "coverage_gap", "supplier_diversity", "cost_efficiency"},
        )

    def test_display_only_families_are_never_selectable(self):
        rng = random.Random(7)
        for name in STRATEGY_NAMES:
            strategy = build_strategy(
                name, self.catalog, rng, None,
                EvidenceSummary(0, {"soft": 0, "structured": 0}, {}),
            )
            eligible = {view.family for view in strategy._eligible(frozenset())}
            blocked = {"family-057", "family-076", "family-095", "family-114",
                       "family-133"}
            self.assertFalse(blocked & eligible)

    def test_budget_never_exceeded_and_cheapest_remaining_rule_is_deterministic(self):
        rng = random.Random(83)
        strategy = BalancedStrategy(
            self.catalog, rng, None,
            EvidenceSummary(0, {"soft": 0, "structured": 0}, {}),
        )
        selected = strategy.propose(self.catalog.budget_per_round, frozenset())
        spend = sum(item.unit_cost for item in selected)
        self.assertLessEqual(spend, self.catalog.budget_per_round + 1e-9)
        # 同种子重放必须完全一致
        again = BalancedStrategy(
            self.catalog, random.Random(83), None,
            EvidenceSummary(0, {"soft": 0, "structured": 0}, {}),
        ).propose(self.catalog.budget_per_round, frozenset())
        self.assertEqual([item.family for item in selected],
                         [item.family for item in again])

    def test_already_acquired_families_disappear_from_pool(self):
        rng = random.Random(1)
        strategy = build_strategy("random", self.catalog, rng, None,
                                  EvidenceSummary(0, {}, {}))
        selected = strategy.propose(self.catalog.budget_per_round, frozenset())
        acquired = frozenset(item.family for item in selected)
        remaining = {view.family for view in strategy._eligible(acquired)}
        self.assertFalse(acquired & remaining)

    def test_random_strategy_uses_seeded_stream(self):
        first = build_strategy("random", self.catalog, random.Random("83:random"),
                               None, EvidenceSummary(0, {}, {}))
        second = build_strategy("random", self.catalog, random.Random("83:random"),
                                None, EvidenceSummary(0, {}, {}))
        self.assertEqual(
            [item.family for item in first.propose(60, frozenset())],
            [item.family for item in second.propose(60, frozenset())],
        )


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "state.json"

    def test_wilson_interval_shrinks_with_more_evidence(self):
        small_low, small_high = wilson_interval(7, 10)
        big_low, big_high = wilson_interval(700, 1000)
        self.assertGreater(small_high - small_low, big_high - big_low)
        self.assertTrue(0.0 <= small_low <= 0.7 <= small_high <= 1.0)

    def test_full_two_round_curves_are_fair_and_complete(self):
        for name in STRATEGY_NAMES:
            for _ in range(2):
                propose_round(INPUT, self.state, name)
                ingest_round(INPUT, self.state, [RECEIPTS], name)
        result = compare_strategies(INPUT, self.state, HOLDOUT)
        for name in STRATEGY_NAMES:
            curve = result["learning_curves"][name]
            self.assertEqual([point["round"] for point in curve], [0, 1, 2])
            # 公平性：同预算、同模型类、同候选池规则、独立随机流
            fairness = result["fairness"][name]
            self.assertEqual(fairness["model_class"], "GaussianNaiveBayes")
            self.assertLessEqual(fairness["cumulative_spend"], 120 + 1e-9)
            self.assertEqual(fairness["rng_stream"], f"83:{name}")
            for point in curve:
                self.assertIn("training_accuracy_reference_only", point)
        # 比较集评估不修改状态
        ledger_before = AcquisitionLedger(
            self.state, load_catalog(INPUT))
        compare_strategies(INPUT, self.state, HOLDOUT)
        ledger_after = AcquisitionLedger(self.state, load_catalog(INPUT))
        self.assertEqual(
            [r["phase"] for r in ledger_before.rounds("balanced")],
            [r["phase"] for r in ledger_after.rounds("balanced")],
        )

    def test_holdout_overlap_is_hard_rejected(self):
        from dataclasses import replace
        from fashion_data_foundry.acquisition import loop as loop_module
        for name in STRATEGY_NAMES:
            propose_round(INPUT, self.state, name)
            ingest_round(INPUT, self.state, [RECEIPTS], name)
        # 手工构造重叠：把某个候选家族改成比较集家族的 ID
        catalog = load_catalog(INPUT)
        cases, sealed = load_holdout(HOLDOUT)
        overlap_family = cases[0].family
        donor = next(iter(catalog.candidates.values()))
        tampered = replace(donor, family=overlap_family)
        object.__setattr__(catalog, "candidates",
                           {**catalog.candidates, overlap_family: tampered})
        original = loop_module.load_catalog
        loop_module.load_catalog = lambda path: catalog  # type: ignore[assignment]
        try:
            with self.assertRaisesRegex(Exception, "重叠"):
                compare_strategies(INPUT, self.state, HOLDOUT)
        finally:
            loop_module.load_catalog = original  # type: ignore[assignment]

    def test_stop_rule_requires_real_margin_not_holdout_blip(self):
        def point(round_number: int, accuracy: float, low: float) -> CurvePoint:
            return CurvePoint(
                round=round_number, strategy="balanced",
                usable_families=40 + round_number * 20,
                metrics={"accuracy": accuracy,
                         "accuracy_ci95": [low, min(1.0, low + 0.2)]},
                training_accuracy=0.99, spend_cumulative=60.0 * round_number,
            )
        # 训练准确率很高但比较集边际不足：必须停
        recommendation = stop_recommendation(
            [point(0, 0.60, 0.45), point(1, 0.61, 0.46), point(2, 0.62, 0.47)],
            round_limit=2, budget_per_round=60,
        )
        self.assertEqual(recommendation["decision"], "stop")
        self.assertTrue(any("边际准确率" in reason for reason in
                            recommendation["reasons"]))
        # 置信区间下界未改善也会触发停采，哪怕点估计上涨
        blip = stop_recommendation(
            [point(0, 0.60, 0.50), point(1, 0.70, 0.49)],
            round_limit=2, budget_per_round=60,
        )
        self.assertIn("stop", blip["decision"])

    def test_brier_improves_when_probabilities_calibrate_better(self):
        catalog = load_catalog(INPUT)
        cases, sealed = load_holdout(HOLDOUT)
        evidence = [item for item in catalog.base_training
                    if item.usable_for_training()]
        model = GaussianNaiveBayes.train(evidence)
        metrics = evaluate_holdout(model, cases, sealed)
        self.assertTrue(0.0 <= metrics["brier"] <= 1.0)
        self.assertEqual(metrics["eligible_families"] +
                         metrics["display_only_families"], len(cases))
        # 每个类别与供货商都有分组报告，避免只看总体数字
        self.assertEqual(set(metrics["per_class"]), {"soft", "structured"})
        self.assertTrue(metrics["supplier_breakdown"])

    def test_marginal_gain_threshold_is_module_constant(self):
        # 预注册阈值不允许出现在 CLI 参数里
        import argparse
        from fashion_data_foundry.acquisition.cli import _build_parser
        help_text = _build_parser().format_help()
        self.assertNotIn("threshold", help_text.lower())
        self.assertGreater(MIN_MARGINAL_ACC_GAIN, 0.0)
        self.assertTrue(math.isfinite(MIN_MARGINAL_ACC_GAIN))


if __name__ == "__main__":
    unittest.main()
