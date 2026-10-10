"""ollama_profiles.py -- machine profiles for ollama_duel.py.

A machine profile (profiles/<name>.json, chosen with --profile) adapts any
scenario to one machine: it swaps the two speakers' models by position and
forces generation settings, keeping the topic, roles and prompts. It can
also list the scenarios that make up that machine's set, which run as a
batch when no scenario is given.
"""

import json
import os
import sys

from ollama_common import check_unknown_keys, validate_field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROFILE_KEYS = {"description", "models", "settings", "scenarios"}
PROFILE_SETTING_KINDS = {
    "think": ("bool", None), "max_tokens": ("int", 1),
    "temperature": ("number", 0), "num_ctx": ("int", 1),
    "repeat_penalty": ("number", 1), "host": ("str", None),
    "timeout": ("number", 1), "history_turns": ("int", 1),
    "display": ("bool", None),
    "temp_guard": ("bool", None),
    "max_temp": ("number", 1), "resume_temp": ("number", 1),
}


def profile_path(name):
    """--profile takes a file path, or a bare name looked up as
    profiles/<name>.json next to this script."""
    if name.endswith(".json") or "/" in name or os.sep in name:
        return name
    return os.path.join(SCRIPT_DIR, "profiles", name + ".json")


def load_profile(name):
    """Read and validate a machine profile, exiting with a clear message
    on any problem -- before a duel starts, like load_config."""
    path = profile_path(name)
    try:
        with open(path, encoding="utf-8") as f:
            profile = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Profile not found: {path}")
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        sys.exit(f"Invalid profile {path}: {e}")
    label = f'profile "{name}"'
    if not isinstance(profile, dict):
        sys.exit(f"{label} must be a JSON object.")
    check_unknown_keys(profile, PROFILE_KEYS, label)
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
        validate_field(settings, key, kind, f"{label} settings", minimum=minimum)
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
