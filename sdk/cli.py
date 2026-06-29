"""Command-line entry point for running a PhySeek policy."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .client import PolicyClient


def _load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("object spec must use 'module:ClassOrFactory'")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def _parse_kv(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        result[key] = parsed
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local policy as a PhySeek SEP policy.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="connect a local policy to SEP")
    run.add_argument("policy_class", nargs="?", help="module:ClassOrFactory")
    run.add_argument("--sep-url", "--sep", dest="sep_url", required=True)
    run.add_argument("--policy-id", "--id", required=True)
    run.add_argument("--policy-type", default="external")
    run.add_argument("--policy-class", dest="policy_class_flag", help="module:ClassOrFactory")
    run.add_argument(
        "--policy-arg",
        action="append",
        default=[],
        help="constructor kwarg as KEY=VALUE; VALUE may be JSON",
    )
    run.add_argument(
        "--policy-spec",
        action="append",
        default=[],
        help="policy_spec entry as KEY=VALUE; VALUE may be JSON",
    )
    run.add_argument("--insecure", action="store_true", help="disable TLS verification for wss://")
    run.add_argument("--log-level", default="INFO")
    return parser


def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    )
    policy_class = args.policy_class_flag or args.policy_class
    if not policy_class:
        raise SystemExit("missing policy class, e.g. physeek run my_module:MyPolicy")

    policy_factory = _load_object(policy_class)
    policy = policy_factory(**_parse_kv(args.policy_arg))
    client = PolicyClient(
        policy=policy,
        policy_id=args.policy_id,
        policy_type=args.policy_type,
        sep_url=args.sep_url,
        policy_spec=_parse_kv(args.policy_spec),
        insecure=args.insecure,
    )
    client.run()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
