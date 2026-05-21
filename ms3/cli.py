"""ms3 CLI — hypothesis | chat | eval modes.

Examples:
  python -m ms3.cli --mode hypothesis --contract test:0 --retriever vector
  python -m ms3.cli --mode chat       --contract test:0 --retriever graph
  python -m ms3.cli --mode eval       --split test --limit 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ms3.agents.conversation import ConversationSession
from ms3.agents.orchestrator import run_hypothesis_analysis
from ms3.agents.tools import register_contract, set_runtrace
from ms3.config import RUNS_DIR
from ms3.contract_loader import load_contract
from ms3.runtrace.builder import Runtrace, write_runtrace


def _cmd_hypothesis(args: argparse.Namespace) -> int:
    contract, gold = load_contract(args.contract)
    rt = run_hypothesis_analysis(contract, gold, retriever_mode=args.retriever)
    out_path = Path(args.out) if args.out else RUNS_DIR / f"{contract['contract_id']}.runtrace.json"
    write_runtrace(rt, out_path)
    m = rt._metrics()
    print(
        f"[ms3] hypothesis run complete: contract_id={contract['contract_id']} "
        f"accuracy={m.get('contract_accuracy', 0):.3f} "
        f"groundedness={m.get('groundedness_rate', 0):.3f} "
        f"quote_integrity={m.get('quote_integrity_rate', 0):.3f}"
    )
    print(f"[ms3] runtrace -> {out_path}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    contract, _ = load_contract(args.contract)
    register_contract(contract)
    rt = Runtrace(mode="conversation", retriever_mode=args.retriever, contract=contract)
    token = set_runtrace(rt)
    try:
        # join all chunk text back together for the initial preview
        full_text = "".join(ch["text"] for ch in contract["chunks"])
        session = ConversationSession(contract_id=contract["contract_id"], contract_text_preview=full_text)
        if session.bootstrap_reply:
            rt.add_turn("assistant", session.bootstrap_reply, retrieval_mode=args.retriever)
            print(f"\n[assistant] {session.bootstrap_reply}\n")
        print(
            "Legal Assistant ready. Type your question, or :exit / :save / :tools. "
            "Press Ctrl-D to exit."
        )
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line == ":exit":
                break
            if line == ":save":
                tmp = RUNS_DIR / f"{contract['contract_id']}.chat.runtrace.json"
                write_runtrace(rt, tmp)
                print(f"[saved -> {tmp}]")
                continue
            if line == ":tools":
                total, by_agent = rt._summaries()
                print("  total:", total)
                print("  by_agent:", by_agent)
                continue
            rt.add_turn("user", line, retrieval_mode=args.retriever)
            reply = session.ask(line)
            rt.add_turn("assistant", reply, retrieval_mode=args.retriever)
            print(f"\n[assistant] {reply}\n")
        out_path = RUNS_DIR / f"{contract['contract_id']}.chat.runtrace.json"
        write_runtrace(rt, out_path)
        print(f"[ms3] chat runtrace -> {out_path}")
    finally:
        set_runtrace(None)
        _ = token
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from ms3.eval.run_eval import run_eval

    csv_path, zip_path = run_eval(
        split=args.split, limit=args.limit, out_dir=Path(args.out_dir) if args.out_dir else RUNS_DIR,
        retriever=args.retriever,
    )
    print(f"[ms3] eval csv -> {csv_path}")
    print(f"[ms3] runtraces zip -> {zip_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ms3")
    p.add_argument("--mode", required=True, choices=["hypothesis", "chat", "eval"])
    p.add_argument("--contract", help="test:N | dev:N | path/to/file")
    p.add_argument("--retriever", choices=["vector", "graph"], default="vector")
    p.add_argument("--out", help="Override runtrace output path (hypothesis mode)")
    p.add_argument("--split", choices=["test", "dev"], default="test")
    p.add_argument("--limit", type=int, default=None, help="eval: max #contracts")
    p.add_argument("--out-dir", help="eval: directory for per-contract runtraces")
    args = p.parse_args(argv)

    if args.mode in ("hypothesis", "chat") and not args.contract:
        p.error(f"--contract is required for --mode {args.mode}")

    if args.mode == "hypothesis":
        return _cmd_hypothesis(args)
    if args.mode == "chat":
        return _cmd_chat(args)
    if args.mode == "eval":
        return _cmd_eval(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
