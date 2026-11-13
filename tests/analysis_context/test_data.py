from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DataContextTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "fixtures/label_acquisition.json").read_text(encoding="utf-8"))

    def test_candidate_labels_and_holdout_families_are_separate(self):
        self.assertTrue(all("label" not in row for row in self.data["candidate_pool"]))
        holdout = json.loads((ROOT / "fixtures/acquisition_holdout.json").read_text())
        train = {row["family"] for row in self.data["training"]}
        pool = {row["family"] for row in self.data["candidate_pool"]}
        reserved = {row["family"] for row in holdout["holdout"]}
        self.assertFalse(train & pool or train & reserved or pool & reserved)

    def test_purchase_units_and_permissions_are_visible(self):
        rows = self.data["candidate_pool"]
        self.assertLess(len({row["family"] for row in rows}), len(rows))
        self.assertTrue(any("training" not in row["purposes"] for row in rows))
        prices = {}
        for row in rows:
            prices.setdefault(row["family"], row["unit_cost"])
            self.assertEqual(prices[row["family"]], row["unit_cost"])

