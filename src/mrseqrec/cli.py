"""命令行入口：python -m mrseqrec.cli <train|signal-check> [options]。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mrseqrec.data.io import load_interactions
from mrseqrec.data.signal import interval_stats, repurchase_check
from mrseqrec.pipeline import run_training
from mrseqrec.utils.config import load_config
from mrseqrec.utils.log import get_logger

logger = get_logger("cli")


def _cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = run_training(config, save_dir=Path(args.save_dir) if args.save_dir else None)
    logger.info("done: n_users=%d vocab=%d", result.n_users, result.item_vocab_size)


def _cmd_signal_check(args: argparse.Namespace) -> None:
    df = load_interactions(args.data)
    logger.info("loaded %d interactions, users=%d items=%d", len(df), df["user"].nunique(), df["item"].nunique())
    logger.info("cadence: %s", interval_stats(df).summary())
    logger.info("repurchase: %s", repurchase_check(df).summary())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrseqrec")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="训练并评估基线（SASRec）")
    p_train.add_argument("--config", default="configs/s1_default.yaml")
    p_train.add_argument("--save-dir", default=None, help="输出 checkpoint 与 metrics.json 的目录")
    p_train.set_defaults(func=_cmd_train)

    p_sig = sub.add_parser("signal-check", help="购买间隔/周期性信号检验（三审 3.2）")
    p_sig.add_argument("--data", required=True, help="交互文件：每行 `user item ts`")
    p_sig.set_defaults(func=_cmd_signal_check)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
