"""Workflow configuration: load ``config.yaml``, apply overrides, open the platform."""

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

#: The default configuration shipped with the package.
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


#: Mappings that a config file replaces as a whole instead of merging into
#: (a new basin's region names must not inherit the Indus ones).
REPLACE = {"region_node_map", "groups", "relax"}


def _merge(base: dict, extra: dict) -> dict:
    out = deepcopy(base)
    for k, v in extra.items():
        merge = isinstance(v, dict) and isinstance(out.get(k), dict) and k not in REPLACE
        out[k] = _merge(out[k], v) if merge else v
    return out


def _parse_value(text: str) -> Any:
    """``--set`` values are YAML scalars: ``true``, ``700``, ``null``, ``[a, b]``."""
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | tuple[str, ...]) -> dict:
    """Apply ``dotted.key=value`` overrides, e.g. ``nexus.rcp=7p0``."""
    cfg = deepcopy(cfg)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        key, value = item.split("=", 1)
        node = cfg
        *path, last = key.strip().split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = _parse_value(value)
    return cfg


def load_config(path: str | Path | None = None, overrides=()) -> dict:
    """The default configuration, updated by the file at ``path`` (only the
    keys it sets) and then by ``overrides``."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    return apply_overrides(cfg, overrides)


def expand(path: str | Path, base: str | Path | None = None) -> Path:
    """``~`` expanded; relative paths resolved against ``base`` when given."""
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute() and base is not None:
        p = Path(os.path.expanduser(str(base))) / p
    return p


def setup_gams(cfg: dict) -> None:
    """Put GAMS on PATH before ixmp/message_ix solve anything."""
    gams = cfg["platform"].get("gams_dir")
    if gams and Path(gams).exists():
        os.environ["IXMP_GAMS_PATH"] = str(gams)
        if str(gams) not in os.environ["PATH"]:
            os.environ["PATH"] = str(gams) + os.pathsep + os.environ["PATH"]


def open_platform(cfg: dict):
    """The ixmp Platform described by ``cfg["platform"]``.

    HSQLDB allows one process at a time: a "lock acquisition failure" means
    another notebook or script still has the database open.
    """
    import ixmp

    setup_gams(cfg)
    p = cfg["platform"]
    kw = dict(backend=p.get("backend", "jdbc"))
    if kw["backend"] == "jdbc":
        kw.update(driver=p.get("driver", "hsqldb"), path=expand(p["path"]).as_posix(),
                  jvmargs=p.get("jvmargs", []))
    return ixmp.Platform(**kw)


def national_node(cfg: dict) -> str:
    """Node of the (first) national energy model."""
    return next(iter((cfg["basin"].get("region_node_map") or {"": "R12_PAK"}).values()))


def output_dir(cfg: dict, scenario: str | None = None) -> Path:
    d = expand(cfg["report"]["output_dir"])
    if scenario:
        d = d / scenario
    d.mkdir(parents=True, exist_ok=True)
    return d
