"""命令行入口：``python3 -m fashion_data_foundry.acquisition``。

子命令：
- ``propose``  生成一轮家族采购订单（输入里没有比较集/回执参数，物理隔离）；
- ``ingest``   回执入账，可多次调用，支持部分回执、乱序与更正；
- ``compare``  读取独立比较集，输出学习曲线、横向对比与停采建议。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fashion_data_foundry.acquisition.loop import (
    STRATEGY_NAMES,
    compare_strategies,
    ingest_round,
    propose_round,
)
from fashion_data_foundry.acquisition.protocol import AcquisitionError


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fashion_data_foundry.acquisition",
        description="可追溯的主动标签采样采购闭环（propose / ingest / compare）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose = subparsers.add_parser("propose", help="生成一轮家族采购订单")
    propose.add_argument("--input", required=True, help="采购资料（无候选标签）")
    propose.add_argument("--state", required=True, help="采购状态文件（JSON）")
    propose.add_argument("--output", required=True, help="订单结果输出文件（JSON）")
    propose.add_argument("--strategy", choices=STRATEGY_NAMES, default="balanced")

    ingest = subparsers.add_parser("ingest", help="回执入账（可重复、可部分回执）")
    ingest.add_argument("--input", required=True, help="采购资料")
    ingest.add_argument("--state", required=True, help="采购状态文件（JSON）")
    ingest.add_argument("--output", required=True, help="入账结果输出文件（JSON）")
    ingest.add_argument(
        "--receipts", required=True, nargs="+",
        help="一个或多个回执文件；同文件内允许乱序，更正片段自动并入",
    )
    ingest.add_argument("--strategy", choices=STRATEGY_NAMES, default="balanced")
    ingest.add_argument(
        "--no-revision-examples", action="store_true",
        help="不并入输入资料中的 receipt_revision_examples 更正片段",
    )

    compare = subparsers.add_parser("compare", help="独立比较集上的学习曲线对比")
    compare.add_argument("--input", required=True, help="采购资料")
    compare.add_argument("--state", required=True, help="采购状态文件（JSON）")
    compare.add_argument("--output", required=True, help="比较结果输出文件（JSON）")
    compare.add_argument("--holdout", required=True, help="独立保留比较集文件")
    compare.add_argument(
        "--strategies", nargs="+", choices=STRATEGY_NAMES, default=list(STRATEGY_NAMES)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "propose":
            payload = propose_round(args.input, args.state, args.strategy)
        elif args.command == "ingest":
            payload = ingest_round(
                args.input, args.state, args.receipts, args.strategy,
                include_revision_examples=not args.no_revision_examples,
            )
        else:
            payload = compare_strategies(
                args.input, args.state, args.holdout, args.strategies
            )
    except AcquisitionError as exc:
        parser.exit(2, f"采购流程错误: {exc}\n")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
    return _print(payload)


if __name__ == "__main__":
    raise SystemExit(main())
