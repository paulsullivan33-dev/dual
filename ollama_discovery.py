"""Scenario catalog and read-only model availability checks."""

import json
import sys
import textwrap
import urllib.request

from ollama_common import DEFAULT_HOST


def inspect_scenarios(paths, args, profile, loader, override, checker):
    """Validate settings and dispatch an offline catalog or preflight."""
    try:
        if args.num_ctx is not None and args.num_ctx < 1:
            raise ValueError("--num-ctx must be at least 1")
        if args.max_tokens is not None and args.max_tokens < 1:
            raise ValueError("--max-tokens must be at least 1")
        resolved = resolve_scenarios(paths, args, profile, loader, override)
        if args.list_scenarios:
            print_catalog(resolved)
        else:
            checker(resolved)
    except ValueError as exc:
        sys.exit(str(exc))


def resolve_scenarios(paths, args, profile, loader, override):
    """Apply the same profile and CLI model overrides used by actual runs."""
    resolved, errors = [], []
    for path in paths:
        try:
            cfg = loader(path, profile)
            override(cfg, args.model_a, args.model_b, args.num_ctx)
            cfg["host"] = args.host if args.host is not None else cfg.get("host") or DEFAULT_HOST
            for key in ("topic", "turns", "run_code"):
                value = getattr(args, key)
                if value is not None:
                    cfg[key] = value
            if not cfg.get("topic") or not cfg["topic"].strip():
                raise ValueError("a nonempty topic is required")
            if cfg.get("turns", 6) < 1:
                raise ValueError("turns must be at least 1")
            if args.judge and cfg.get("judge"):
                judge = cfg["judge"]
                judge["host"] = args.judge_host or judge.get("host") or cfg["host"]
            else:
                cfg["judge"] = None
            resolved.append((path, cfg))
        except (SystemExit, OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise ValueError("Scenario validation failed:\n" + "\n".join(errors))
    return resolved


def print_catalog(resolved):
    """List scenario topics and effective execution settings without requests."""
    for path, cfg in resolved:
        print(f"{path}\n  {textwrap.shorten(cfg['topic'], width=120)}")
        print(f"  Tags: {', '.join(cfg.get('tags') or []) or '-'} | "
              f"Turns: {cfg.get('turns', 6)} | "
              f"Code execution: {'ON' if cfg.get('run_code') else 'off'}")
        print("  Models: " + " vs ".join(m["model"] for m in cfg["models"]))
    print(f"{len(resolved)} scenario(s).")


def installed_models(host):
    """Query model inventory without loading, generating, or downloading."""
    with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=10) as response:
        data = json.load(response)
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ValueError("expected a models list from /api/tags")
    names = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("invalid model entry from /api/tags")
        name = model.get("name") or model.get("model")
        if not isinstance(name, str) or not name:
            raise ValueError("model entry has no name")
        names.add(name)
    return names


def preflight(resolved):
    """Check every distinct host/model pair and report all failures together."""
    required = {}
    for path, cfg in resolved:
        pairs = [(cfg["host"], m["model"]) for m in cfg["models"]]
        if cfg.get("judge"):
            pairs.append((cfg["judge"]["host"], cfg["judge"]["model"]))
        for host, model in pairs:
            required.setdefault(host.rstrip("/"), {}).setdefault(model, set()).add(path)
    errors = []
    for host, models in required.items():
        try:
            names = installed_models(host)
        except Exception as exc:
            errors.append(f"{host}: cannot read model inventory: {exc} "
                          f"(needed: {', '.join(sorted(models))})")
            continue
        for model, paths in models.items():
            # Ollama treats an omitted tag as :latest; namespaced names
            # and registries containing a port still need a tag on the last part.
            canonical = model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
            if model not in names and canonical not in names:
                errors.append(f"{host}: missing model {model} "
                              f"(used by {', '.join(sorted(paths))})")
    if errors:
        raise ValueError("Preflight failed; no scenarios started:\n" + "\n".join(errors))
    print(f"Preflight passed: {len(resolved)} scenario(s), {len(required)} server(s).")
