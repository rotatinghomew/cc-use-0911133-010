"""标签隔离的硬性回归测试。

这些测试不是“提醒调用者别偷看标签”，而是从结构上验证：
- 候选视图的白名单里没有 label，输入文件被塞入候选标签会直接加载失败；
- 选样策略的构造与评分路径拿不到任何真标签或比较集对象；
- 比较集家族与训练/候选家族不相交，比较集只通过密封对象在评估阶段解包。
"""
from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from fashion_data_foundry.acquisition.catalog import (
    load_catalog,
    load_holdout,
)
from fashion_data_foundry.acquisition.protocol import (
    CANDIDATE_ALLOWED_KEYS,
    GuardedHoldoutError,
    SealedLabels,
)
from fashion_data_foundry.acquisition.strategy import (
    AcquisitionStrategy,
    BalancedStrategy,
    EvidenceSummary,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "fixtures" / "label_acquisition.json"
HOLDOUT = ROOT / "fixtures" / "acquisition_holdout.json"


class LabelIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.catalog = load_catalog(INPUT)

    def test_candidate_whitelist_has_no_label_key(self):
        self.assertNotIn("label", CANDIDATE_ALLOWED_KEYS)
        for view in self.catalog.candidates.values():
            payload = view.to_dict()
            self.assertNotIn("label", payload)
            self.assertNotIn("true_label", payload)

    def test_injected_candidate_label_is_rejected_at_load(self):
        import json
        data = json.loads(INPUT.read_text(encoding="utf-8"))
        data["candidate_pool"][0]["label"] = "soft"
        poisoned = Path(self.temp.name) / "poisoned.json"
        poisoned.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(Exception, "标签字段"):
            load_catalog(poisoned)

    def test_strategy_construction_cannot_reach_labels(self):
        import fashion_data_foundry.acquisition.strategy as strategy_module
        # 策略模块不允许导入任何密封标签/比较集类型。
        namespace = vars(strategy_module)
        self.assertNotIn("SealedLabels", namespace)
        self.assertNotIn("HoldoutCase", namespace)
        signature = inspect.signature(AcquisitionStrategy.__init__)
        parameter_names = set(signature.parameters)
        self.assertNotIn("labels", parameter_names)
        self.assertNotIn("holdout", parameter_names)
        self.assertNotIn("sealed", parameter_names)

    def test_scoring_path_only_uses_views_and_summary(self):
        rng = __import__("random").Random(1)
        strategy = BalancedStrategy(
            self.catalog, rng, None, EvidenceSummary(0, {"soft": 0, "structured": 0}, {})
        )
        view = next(iter(self.catalog.candidates.values()))
        scored = strategy.score_family(view)
        self.assertEqual(set(scored.components),
                         {"uncertainty", "coverage_gap", "supplier_diversity",
                          "cost_efficiency"})
        self.assertFalse(hasattr(scored, "label"))

    def test_holdout_families_are_disjoint_and_sealed(self):
        cases, sealed = load_holdout(HOLDOUT)
        holdout_families = {case.family for case in cases}
        self.assertFalse(holdout_families & set(self.catalog.candidates))
        self.assertFalse(
            holdout_families & {item.family for item in self.catalog.base_training}
        )
        self.assertIsInstance(sealed, SealedLabels)
        # 密封对象不暴露裸属性，只能显式解包
        self.assertFalse(hasattr(sealed, "labels"))
        unsealed = sealed.unseal_for_evaluation()
        self.assertEqual(len(unsealed), len(cases))

    def test_holdout_guard_rejects_missing_labels(self):
        cases, _ = load_holdout(HOLDOUT)
        sealed = SealedLabels({}, source="empty")
        from fashion_data_foundry.acquisition.evaluation import evaluate_holdout
        from fashion_data_foundry.acquisition.model import GaussianNaiveBayes
        evidence = [item for item in self.catalog.base_training
                    if item.usable_for_training()]
        model = GaussianNaiveBayes.train(evidence)
        with self.assertRaises(GuardedHoldoutError):
            evaluate_holdout(model, cases[:2], sealed)

    def test_catalog_is_immutable_snapshot(self):
        view = next(iter(self.catalog.candidates.values()))
        with self.assertRaises(Exception):
            view.family = "tampered"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
