"""
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Model file names used by ColorMNet, loaded from the data file models.json
(stored next to this module).

The checkpoint file names are part of the contract with the model releases
(see the "Models Download" section of the README): keeping them in a data file
instead of hardcoding them in the code gives a single point of truth, easy to
inspect and to update.  models.json overrides the DEFAULTS defined below.

models.json is optional: if it is missing, unreadable or malformed, DEFAULTS
is used and a warning is logged via the standard `logging` module.
-------------------------------------------------------------------------------
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback values (same as models.json), used when models.json is missing or
# does not define the requested entry.  Keep in sync with models.json.
DEFAULTS = {
    "cmnet2": {
        "dinov3": {
            "checkpoint": "DINOv3FeatureV6_LocalAtten_p372402.pth",
            "weights_dir": "dinov3-vitb16",
            "enable_proximity_bias": False,
            "proximity_bias_alpha": 0.5,
        },
        "dinov2": {
            "checkpoint": "DINOv2FeatureV6_LocalAtten_s2_154000.pth",
        },
    },
}

_CONFIG_PATH = Path(__file__).with_name("models.json")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("the root of models.json must be an object")
        return _deep_merge(DEFAULTS, data)
    except (OSError, ValueError) as e:
        logger.warning("models_config: cannot read '%s' (%s: %s), "
                        "using the built-in defaults", _CONFIG_PATH, type(e).__name__, e)
        return _deep_merge(DEFAULTS, {})


MODELS = _load()


def get_cmnet2_model(backbone: str) -> dict:
    """Return the model entry for a CMNET2 backbone ('dinov2' or 'dinov3').

    The returned dict always contains 'checkpoint' (file name inside the
    weights directory) and may contain 'weights_dir' (directory holding the
    auxiliary backbone files, e.g. dinov3-vitb16).
    """
    try:
        entry = MODELS["cmnet2"][backbone]
    except KeyError:
        raise ValueError(f"models_config: no CMNET2 entry for backbone {backbone!r}")
    if not isinstance(entry, dict) or not entry.get("checkpoint"):
        raise ValueError(f"models_config: invalid CMNET2 entry for backbone {backbone!r}")
    return entry


def check_file(path: str, what: str = "model file") -> str:
    """Return path unchanged when it exists, otherwise raise a clear error.

    On failure the error also lists the files actually present in the
    directory, so a wrong/misspelled checkpoint name is immediately visible.
    """
    p = Path(path)
    if not p.is_file():
        listing = ""
        try:
            names = sorted(f.name for f in p.parent.iterdir() if f.is_file())
            listing = ("\n  files present in " + str(p.parent) + ": " + ", ".join(names)) \
                if names else ("\n  (no files in " + str(p.parent) + ")")
        except OSError:
            pass
        raise FileNotFoundError(f"{what} not found: {path}{listing}")
    return path
