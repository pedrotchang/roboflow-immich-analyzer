"""Entrypoint: ``immich-analyzer {run,report,check}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import httpx

from .config import Config
from .immich import ImmichClient
from .inference import InferenceClient
from .pipeline import Analyzer, Mode
from .store import Store

_RESERVED = set(vars(logging.makeLogRecord({})))


class JsonFormatter(logging.Formatter):
    """One JSON object per line; ``extra=`` fields become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def _setup_logging() -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[h], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _wait_ready(inference: InferenceClient, timeout_s: int = 600) -> None:
    log = logging.getLogger("immich_analyzer")
    deadline = time.monotonic() + timeout_s
    while not inference.ready():
        if time.monotonic() > deadline:
            raise SystemExit("inference server not ready")
        log.info("waiting for inference server")
        time.sleep(10)


def cmd_run(cfg: Config, mode: Mode) -> int:
    store = Store(cfg.state_path)
    inference = InferenceClient(cfg.inference_url, cfg.clip_version)
    _wait_ready(inference)
    analyzer = Analyzer(cfg, store, inference)
    failures = 0
    try:
        for key in cfg.immich_api_keys:
            immich = ImmichClient(cfg.immich_url, key)
            try:
                analyzer.run(immich, mode)
            except Exception:
                failures += 1
                logging.getLogger("immich_analyzer").exception("run failed for one user")
            finally:
                immich.close()
    finally:
        store.close()
        inference.close()
    return 1 if failures else 0


def cmd_report(cfg: Config) -> int:
    store = Store(cfg.state_path)
    report = {
        "counts": store.label_counts(),
        "samples": {
            label: [{"id": i, "reason": r} for i, r in store.samples(label, 5)]
            for label in ("receipt", "screenshot", "blurry", "duplicate")
        },
    }
    print(json.dumps(report, indent=2))
    store.close()
    return 0


def cmd_check(cfg: Config) -> int:
    inference = InferenceClient(cfg.inference_url, cfg.clip_version)
    ok = inference.ready()
    print(json.dumps({"inference_ready": ok}))
    for n, key in enumerate(cfg.immich_api_keys):
        immich = ImmichClient(cfg.immich_url, key)
        try:
            me = immich.whoami()
            print(json.dumps({"key": n, "immich_user": me.get("email"), "id": me.get("id")}))
        except httpx.HTTPError as e:
            ok = False
            print(json.dumps({"key": n, "error": f"{type(e).__name__}: {e}"}))
        finally:
            immich.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="immich-analyzer")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="analyse new assets and sync albums")
    run.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.AUTO.value)
    sub.add_parser("report", help="print label counts and samples from the state DB")
    sub.add_parser("check", help="verify connectivity to Immich and the inference server")
    args = p.parse_args(argv)

    _setup_logging()
    cfg = Config.from_env()
    if args.cmd == "run":
        return cmd_run(cfg, Mode(args.mode))
    if args.cmd == "report":
        return cmd_report(cfg)
    return cmd_check(cfg)


if __name__ == "__main__":
    sys.exit(main())
