"""ollama_profiles.py -- machine profiles for ollama_duel.py.

A machine profile (profiles/<name>.json, chosen with --profile) adapts any
scenario to one machine: it swaps the two speakers' models by position and
forces generation settings, keeping the topic, roles and prompts. It can
also list the scenarios that make up that machine's set, which run as a
batch when no scenario is given.

A profile can build on another with "extends": it starts from the named
profile and overrides only what it states. "models", "scenarios" and
"description" replace the parent's; "settings" are merged key by key, and a
setting given as null removes the inherited value (e.g. "judge": null lets
each scenario's own judge apply).
"""

import json
import os
import sys

from ollama_common import check_unknown_keys, validate_field
from ollama_judge import normalize_judge

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROFILE_KEYS = {"description", "extends", "models", "settings", "scenarios"}
PROFILE_SETTING_KINDS = {
    "think": ("bool", None), "max_tokens": ("int", 1),
    "temperature": ("number", 0), "num_ctx": ("int", 1),
    "repeat_penalty": ("number", 1), "host": ("str", None),
    "timeout": ("number", 1), "history_turns": ("int", 1),
    "display": ("bool", None),
    "temp_guard": ("bool", None),
    "max_temp": ("number", 1), "resume_temp": ("number", 1),
    # "judge" is a model name or dict; validated by normalize_judge below,
    # not by the scalar loop.
    "judge": (None, None),
}


def profile_path(name):
    """--profile takes a file path, or a bare name looked up as
    profiles/<name>.json next to this script."""
    if name.endswith(".json") or "/" in name or os.sep in name:
        return name
    return os.path.join(SCRIPT_DIR, "profiles", name + ".json")


def _read_profile(name, base_dir=None, chain=()):
    """Read a profile file and resolve its "extends" chain into one dict.

    A relative path in "extends" is taken from the extending profile's
    folder; a bare name is looked up in profiles/ as usual.
    """
    path = profile_path(name)
    # profile_path returns a path argument unchanged; when that path is
    # relative and came from "extends", read it from the extender's folder.
    if base_dir and path == name and not os.path.isabs(name):
        path = os.path.join(base_dir, name)
    key = os.path.normcase(os.path.realpath(path))
    if key in chain:
        sys.exit(f'Profile "{name}" extends itself (through {len(chain)} '
                 f"profile(s)); remove the loop.")
    try:
        with open(path, encoding="utf-8") as f:
            profile = json.load(f)
    except FileNotFoundError:
        if chain:
            sys.exit(f'Profile extends a profile that was not found: {path}')
        sys.exit(f"Profile not found: {path}")
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        sys.exit(f"Invalid profile {path}: {e}")
    label = f'profile "{name}"'
    if not isinstance(profile, dict):
        sys.exit(f"{label} must be a JSON object.")
    check_unknown_keys(profile, PROFILE_KEYS, label)
    if not isinstance(profile.get("settings", {}), dict):
        sys.exit(f'"settings" in {label} must be a JSON object.')
    parent_name = profile.pop("extends", None)
    if parent_name is None:
        merged = profile
    else:
        if not isinstance(parent_name, str) or not parent_name.strip():
            sys.exit(f'"extends" in {label} must be a profile name.')
        merged = _read_profile(parent_name, os.path.dirname(path), chain + (key,))
        settings = dict(merged.get("settings", {}))
        settings.update(profile.pop("settings", {}))
        merged.update(profile)
        merged["settings"] = settings
    # A null setting means "not set": it removes an inherited value.
    if "settings" in merged:
        merged["settings"] = {k: v for k, v in merged["settings"].items()
                              if v is not None}
    return merged


def load_profile(name):
    """Read and validate a machine profile, exiting with a clear message
    on any problem -- before a duel starts, like load_config."""
    profile = _read_profile(name)
    label = f'profile "{name}"'
    models = profile.get("models")
    if models is not None and (
            not isinstance(models, list) or len(models) != 2
            or not all(isinstance(m, str) and m.strip() for m in models)):
        sys.exit(f'"models" in {label} must list exactly 2 model names.')
    settings = profile.get("settings", {})
    if not isinstance(settings, dict):
        sys.exit(f'"settings" in {label} must be a JSON object.')
    check_unknown_keys(settings, set(PROFILE_SETTING_KINDS), f"{label} settings")
    for key, (kind, minimum) in PROFILE_SETTING_KINDS.items():
        if kind is None:
            continue  # "judge": validated by normalize_judge, below
        validate_field(settings, key, kind, f"{label} settings", minimum=minimum)
    if "judge" in settings:
        # Full judge validation now, so a bad judge block fails at profile
        # load -- not halfway through a batch. Accepts the same string or
        # dict form as a scenario's "judge".
        settings["judge"] = normalize_judge(settings["judge"])
    scenarios = profile.get("scenarios")
    if scenarios is not None and (
            not isinstance(scenarios, list)
            or not all(isinstance(s, str) and s.strip() for s in scenarios)):
        sys.exit(f'"scenarios" in {label} must be a list of scenario file names.')
    return profile


def apply_profile(cfg, profile):
    """Adapt a scenario config to a machine profile, in place.

    Each forced setting replaces the top-level value and removes any
    per-model value, so it holds for both speakers. The profile's models
    replace the speakers' models by position (first speaker, second
    speaker). Topic, names, system prompts and turn prompts are untouched.
    """
    for key, value in profile.get("settings", {}).items():
        cfg[key] = value
        for m in cfg["models"]:
            m.pop(key, None)
    for m, model in zip(cfg["models"], profile.get("models") or []):
        m["model"] = model


def profile_scenarios(profile, name):
    """The scenario files a profile lists, resolved in scenarios/."""
    paths = [os.path.join(SCRIPT_DIR, "scenarios", s)
             for s in profile.get("scenarios") or []]
    if not paths:
        sys.exit(f'No scenario given, and profile "{name}" lists no "scenarios" '
                 f"to run. Pass a scenario file, directory or glob.")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        sys.exit(f'Profile "{name}" lists missing scenario file(s): '
                 + ", ".join(missing))
    return paths
