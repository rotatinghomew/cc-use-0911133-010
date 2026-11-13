"""本地可走完两轮的主动采样采购闭环演示。

运行方式（在仓库根目录）::

    python3 scripts/run_acquisition_demo.py

演示在临时目录里完成全部动作，不污染源码树；每个关键步骤后都有断言，
覆盖：预算锁定、部分回执、乱序回执、同 ID 更正、授权撤销、中断恢复不重复付费、
仅展示素材不进训练、三策略公平对比与预注册停采建议。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "fixtures" / "label_acquisition.json"
RECEIPTS = ROOT / "fixtures" / "acquisition_receipts.json"
HOLDOUT = ROOT / "fixtures" / "acquisition_holdout.json"
STRATEGIES = ("balanced", "uncertainty", "random")


def run_cli(*args: str) -> dict:
    command = [sys.executable, "-m", "fashion_data_foundry.acquisition", *args]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="acquisition-demo-") as tmp:
        work = Path(tmp)
        state = work / "acquisition-state.json"
        print("=" * 72)
        print("主动采样采购闭环演示（两轮预算，每轮 60，采购单位=家族）")
        print("状态文件:", state)
        print("=" * 72)

        # ------------------------------------------------------------ 第 1 轮
        r1_outputs: dict[str, Path] = {}
        for name in STRATEGIES:
            out = work / f"{name}-round1-propose.json"
            payload = run_cli(
                "propose", "--input", str(INPUT), "--state", str(state),
                "--output", str(out), "--strategy", name,
            )
            r1_outputs[name] = out
            assert payload["replayed"] is False
            assert payload["order_spend"] <= 60
            # 选样输出物理上不含任何标签键
            assert all("label" not in order for order in payload["orders"])
            # 重复图片折叠：同家族只出现一次
            families = [order["family"] for order in payload["orders"]]
            assert len(families) == len(set(families))
            # 仅展示授权家族不会进入训练候选订单
            assert all(order["training_eligible"] for order in payload["orders"])
            print(f"[第1轮 propose] {name:11s} 采购 {payload['order_count']:2d} 个家族，"
                  f"花费 {payload['order_spend']:5.1f}/60，剩余 {payload['budget_remaining']}")

        balanced_r1 = json.loads(r1_outputs["balanced"].read_text(encoding="utf-8"))
        print("\n本轮为什么买这些家族（balanced 前 5 个评分拆解）:")
        for item in balanced_r1["top_ranked_families"][:5]:
            print(" ", item["family"], "score=", item["score"], item["components"])
        print("仍缺证据的群体:", json.dumps(
            balanced_r1["coverage_gaps"], ensure_ascii=False))

        # 模拟“只回一部分”：取每个策略订单家族对应的一个真子集回执
        full_receipts = json.loads(RECEIPTS.read_text(encoding="utf-8"))["receipts"]
        for name in STRATEGIES:
            orders = json.loads(r1_outputs[name].read_text(encoding="utf-8"))["orders"]
            ordered_families = {order["family"] for order in orders}
            partial = [row for row in full_receipts
                       if row["family"] in ordered_families][: max(3, len(orders) // 2)]
            partial_path = write_json(
                work / f"{name}-round1-partial.json",
                {"schema_version": 1, "allowed_use": "receipt-simulation-only",
                 "receipts": partial},
            )
            payload = run_cli(
                "ingest", "--input", str(INPUT), "--state", str(state),
                "--output", str(work / f"{name}-round1-partial-ingest.json"),
                "--receipts", str(partial_path), "--strategy", name,
                "--no-revision-examples",
            )
            pending = payload["evidence"]["pending_receipt_families"]
            assert pending, "部分回执后必须仍有 pending 家族"
            print(f"[第1轮 部分回执] {name:11s} 入账 {len(partial):2d} 条，"
                  f"仍等待 {len(pending):2d} 个家族回执")

        # 模拟“中断恢复”：把剩余回执再喂一遍，并故意打乱顺序
        for name in STRATEGIES:
            orders = json.loads(r1_outputs[name].read_text(encoding="utf-8"))["orders"]
            ordered_families = {order["family"] for order in orders}
            remaining = [row for row in full_receipts
                         if row["family"] in ordered_families]
            shuffled = list(reversed(remaining))  # 乱序投递
            rest_path = write_json(
                work / f"{name}-round1-rest.json",
                {"schema_version": 1, "allowed_use": "receipt-simulation-only",
                 "receipts": shuffled},
            )
            before = json.loads(state.read_text(encoding="utf-8"))
            before_spend = sum(
                r["spend"] for r in before["strategies"][name]["rounds"]
            )
            run_cli(
                "ingest", "--input", str(INPUT), "--state", str(state),
                "--output", str(work / f"{name}-round1-rest-ingest.json"),
                "--receipts", str(rest_path), "--strategy", name,
                "--no-revision-examples",
            )
            after_spend = sum(
                r["spend"] for r in
                json.loads(state.read_text(encoding="utf-8"))["strategies"][name]["rounds"]
            )
            assert after_spend == before_spend, "回执补录不能再次收费"
            print(f"[第1轮 中断恢复] {name:11s} 乱序补录后累计花费仍为 {after_spend}")

        # 再投一次同一批回执：重复投递必须全部跳过，费用不变
        repeat = run_cli(
            "ingest", "--input", str(INPUT), "--state", str(state),
            "--output", str(work / "repeat-ingest.json"),
            "--receipts", str(work / "balanced-round1-rest.json"),
            "--strategy", "balanced", "--no-revision-examples",
        )
        assert all(entry["action"] == "duplicate_skipped"
                   for entry in repeat["ingest_log"]
                   if entry["family"] in {o["family"] for o in balanced_r1["orders"]})
        print("[幂等] 重复投递 balanced 第1轮回执，全部 duplicate_skipped，费用不变")

        # ------------------------------------------------------------ 第 2 轮
        r2_outputs: dict[str, Path] = {}
        for name in STRATEGIES:
            out = work / f"{name}-round2-propose.json"
            payload = run_cli(
                "propose", "--input", str(INPUT), "--state", str(state),
                "--output", str(out), "--strategy", name,
            )
            r2_outputs[name] = out
            assert payload["round"] == 2
            assert payload["order_spend"] <= 60
            prior = {order["family"] for order in
                     json.loads(r1_outputs[name].read_text(encoding="utf-8"))["orders"]}
            assert not ({o["family"] for o in payload["orders"]} & prior), \
                "已采购家族不能二次收费采购"
            print(f"[第2轮 propose] {name:11s} 采购 {payload['order_count']:2d} 个家族，"
                  f"花费 {payload['order_spend']:5.1f}/60")

        # 第2轮回执：全量文件 + 输入资料里的更正片段（family-051 revision=2，0 费用）
        for name in STRATEGIES:
            payload = run_cli(
                "ingest", "--input", str(INPUT), "--state", str(state),
                "--output", str(work / f"{name}-round2-ingest.json"),
                "--receipts", str(RECEIPTS), "--strategy", name,
            )
            corrections = [entry for entry in payload["ingest_log"]
                           if entry["action"] == "correction"]
            assert corrections, "family-051 的更正回执必须被识别"
            assert all(entry["charged_cost"] == 0 for entry in corrections)
            print(f"[第2轮 ingest] {name:11s} 更正 {len(corrections)} 条，"
                  f"更正零收费；快照排除 {payload['training_snapshot']['excluded_families']}")

        # ------------------------------------------ 撤销授权：family-050 第2轮起失效
        snapshot = json.loads((work / "balanced-round2-ingest.json").read_text(
            encoding="utf-8"))["training_snapshot"]
        excluded = {item["family"]: item["reason"]
                    for item in snapshot["excluded_families"]}
        assert excluded.get("family-050") == "withdrawn"
        assert "family-050" not in snapshot["usable_families"]
        # 仅展示授权的基线/候选家族永远不进训练
        for display_only in ("family-000", "family-019", "family-038"):
            assert excluded.get(display_only) == "display_only"
        print("[授权] family-050 撤销生效，已从训练证据剔除；仅展示家族全部排除")

        # 预算总账核对：两轮、三策略均不得超 60/轮
        state_payload = json.loads(state.read_text(encoding="utf-8"))
        for name in STRATEGIES:
            for round_record in state_payload["strategies"][name]["rounds"]:
                assert round_record["spend"] <= 60 + 1e-9
            total = sum(r["spend"] for r in state_payload["strategies"][name]["rounds"])
            assert total <= 120 + 1e-9
            print(f"[预算] {name:11s} 两轮累计花费 {total}（上限 120）")

        # ------------------------------------------------------------- compare
        compare_out = work / "compare.json"
        compare = run_cli(
            "compare", "--input", str(INPUT), "--state", str(state),
            "--output", str(compare_out), "--holdout", str(HOLDOUT),
        )
        assert compare["holdout_sealed_until_evaluation"] is True
        for name in STRATEGIES:
            curve = compare["learning_curves"][name]
            assert [point["round"] for point in curve] == [0, 1, 2]
            assert compare["fairness"][name]["cumulative_spend"] <= 120 + 1e-9
        print("\n学习曲线（同一份独立比较集，准确率 / Wilson95% / Brier）:")
        for name in STRATEGIES:
            for point in compare["learning_curves"][name]:
                print(f"  {name:11s} round={point['round']} "
                      f"acc={point['holdout_accuracy']:.4f} "
                      f"ci95={point['holdout_accuracy_ci95']} brier={point['brier']:.4f}")
        print("\n最终横向对比:")
        for name, summary in compare["final_comparison"].items():
            print(f"  {name:11s} acc={summary['holdout_accuracy']:.4f} "
                  f"brier={summary['brier']:.4f} spend={summary['cumulative_spend']}")
        print("\n预注册停采建议:")
        for name, recommendation in compare["stop_recommendations"].items():
            print(f"  {name:11s} -> {recommendation['decision']}；"
                  f"{'；'.join(recommendation['reasons'])}")

        print("\n全部断言通过。输出目录：", work)
        print("（临时目录将在退出后自动清理；状态文件结构可直接打开上述 JSON 审计）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
