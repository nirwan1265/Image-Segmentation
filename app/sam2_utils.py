#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam2_utils.py
=============
SAM2 model loading, Hydra config resolution, and mask-generator factory.
No tkinter — these helpers work headlessly and are called by the GUI.

Public API:
  load_sam2_model(ckpt_path, cfg_field, device, apply_pp)
      -> (sam2_model, mask_generator)  or raises RuntimeError

  make_mask_generator(sam2_model, **kwargs)
      -> SAM2AutomaticMaskGenerator with sensible plant-phenotyping defaults

  load_sam2_bundle(bundle_path, device, apply_pp)
      -> (sam2_model, mask_generator, meta)  or raises RuntimeError
"""

import os
import logging
import tempfile
from pathlib import Path

import torch

# Hydra / OmegaConf
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

try:
    from omegaconf import DictConfig
except Exception:
    class DictConfig:   # minimal fallback
        pass

# SAM2 — optional; GUI shows a friendly error if missing
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    _SAM2_AVAILABLE = True
    _sam2_import_error = None
except Exception as _e:
    SAM2AutomaticMaskGenerator = None
    build_sam2 = None
    _SAM2_AVAILABLE = False
    _sam2_import_error = _e

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    SAM2ImagePredictor = None

log = logging.getLogger(__name__)


# =============================================================================
# Hydra helpers
# =============================================================================

def _hydra_reinit_to_dir(cfg_dir: str) -> None:
    """Clear any existing Hydra singleton and re-initialise to cfg_dir."""
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass
    initialize_config_dir(config_dir=cfg_dir, job_name="sam2_gui")


def _compose_from_yaml(yaml_path: str):
    """Load and compose a Hydra config from an explicit YAML file path."""
    conf_dir = str(Path(yaml_path).parent)
    conf_name = Path(yaml_path).stem
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_dir, job_name="sam2_gui",
                               version_base=None):
        return compose(config_name=conf_name)


def _resolve_sam2_cfg(
    cfg_field,
    ckpt_path: str | None = None,
    fallback_short: str = "sam2.1_hiera_l",
):
    """
    Resolve cfg_field to something build_sam2() accepts.

    Accepted inputs
    ---------------
    None / ""           → fallback_short string
    dict / DictConfig   → returned as OmegaConf DictConfig
    path to *.yaml      → composed with Hydra
    directory path      → searches for a hiera YAML inside it
    short string        → searched near checkpoint and $SAM2_CONFIG_DIR;
                          if not found, returned as-is (let build_sam2 resolve)
    """
    if cfg_field is None:
        return fallback_short
    if isinstance(cfg_field, DictConfig):
        return cfg_field
    if isinstance(cfg_field, dict):
        return OmegaConf.create(cfg_field)

    s = str(cfg_field).strip()
    if not s:
        return fallback_short

    # Explicit YAML file
    if Path(s).is_file() and s.lower().endswith((".yaml", ".yml")):
        return _compose_from_yaml(s)

    # Config directory — try to find the right YAML inside it
    if Path(s).is_dir():
        guess = Path(s) / "sam2.1_hiera_l.yaml"
        if guess.exists():
            return _compose_from_yaml(str(guess))
        for y in sorted(Path(s).glob("*.y*ml")):
            if "hiera" in y.stem:
                return _compose_from_yaml(str(y))
        cands = sorted(Path(s).glob("*.y*ml"))
        if cands:
            return _compose_from_yaml(str(cands[0]))

    # Short name — search near checkpoint and $SAM2_CONFIG_DIR
    guesses: list[str] = []
    env_dir = os.environ.get("SAM2_CONFIG_DIR")
    if env_dir:
        guesses += [str(Path(env_dir) / "sam2.1"), env_dir]
    if ckpt_path:
        ck = Path(ckpt_path)
        repo_root = ck.parent.parent if ck.suffix == ".pt" else ck.parent
        guesses += [
            str(repo_root / "configs" / "sam2.1"),
            str(repo_root / "configs"),
        ]
    for d in guesses:
        y = Path(d) / f"{s}.yaml"
        if y.exists():
            return _compose_from_yaml(str(y))

    # Give up — let build_sam2 resolve a package-level short name
    return s


# =============================================================================
# Mask-generator factory
# =============================================================================

def make_mask_generator(sam2_model, **overrides):
    """
    Build a SAM2AutomaticMaskGenerator with plant-phenotyping defaults.
    Any keyword arg accepted by SAM2AutomaticMaskGenerator can be overridden.
    """
    if not _SAM2_AVAILABLE:
        raise RuntimeError(
            f"SAM2 is not importable: {_sam2_import_error}"
        )
    defaults = dict(
        points_per_side=32,
        points_per_batch=32,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.80,
        crop_n_layers=1,
        crop_overlap_ratio=0.30,
        crop_n_points_downscale_factor=2,
        box_nms_thresh=0.6,
        min_mask_region_area=800,
        use_m2m=True,
        output_mode="binary_mask",
    )
    defaults.update(overrides)
    return SAM2AutomaticMaskGenerator(sam2_model, **defaults)


# =============================================================================
# Model loading
# =============================================================================

def load_sam2_model(
    ckpt_path: str,
    cfg_field=None,
    device: str = "cpu",
    apply_pp: bool = True,
):
    """
    Load a SAM2 model from a checkpoint file + config.

    Parameters
    ----------
    ckpt_path : str   Path to the .pt checkpoint file.
    cfg_field : str | dict | None
                      Short config name, YAML path, directory, or dict.
    device    : str   'cpu' | 'cuda' | 'mps'
    apply_pp  : bool  Whether to apply SAM2 post-processing.

    Returns
    -------
    (sam2_model, mask_generator)
    """
    if not _SAM2_AVAILABLE:
        raise RuntimeError(
            f"Cannot load SAM2 — import failed:\n{_sam2_import_error}"
        )
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path!r}")

    cfg_resolved = _resolve_sam2_cfg(cfg_field, ckpt_path=ckpt_path)
    log.info("Loading SAM2: ckpt=%s  cfg=%s  device=%s", ckpt_path, cfg_resolved, device)
    model = build_sam2(cfg_resolved, ckpt_path, device=device,
                       apply_postprocessing=apply_pp)
    gen = make_mask_generator(model)
    return model, gen


def load_sam2_bundle(
    bundle_path: str,
    device: str = "cpu",
    apply_pp: bool = True,
):
    """
    Load a SAM2 'bundle' .pt file that contains both checkpoint bytes and
    config inside a single torch-saved dict.

    Keys recognised in the bundle:
      ckpt_bytes / checkpoint_bytes  – raw bytes of the checkpoint
      ckpt_path                      – path string (used if bytes absent)
      state_dict                     – loaded directly if no checkpoint path
      cfg / cfg_short_name           – config (dict, DictConfig, or short name)
      meta                           – dict of extra metadata
      apply_postprocessing           – bool override

    Returns
    -------
    (sam2_model, mask_generator, meta_dict)
    """
    if not _SAM2_AVAILABLE:
        raise RuntimeError(
            f"Cannot load SAM2 bundle — import failed:\n{_sam2_import_error}"
        )

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    log.info("Bundle keys: %s", list(bundle.keys()))

    meta        = bundle.get("meta") or {}
    cfg_in      = bundle.get("cfg")
    ck_bytes    = bundle.get("ckpt_bytes") or bundle.get("checkpoint_bytes")
    ck_path_str = bundle.get("ckpt_path") if isinstance(
        bundle.get("ckpt_path"), str) else None
    apply_pp    = bool(bundle.get("apply_postprocessing", apply_pp))

    tmp_ckpt = None
    tmp_cfg_dir = None

    try:
        # ── Prepare checkpoint path ───────────────────────────────────────────
        if isinstance(ck_bytes, (bytes, bytearray)):
            fd, tmp_path = tempfile.mkstemp(suffix=".pt")
            with os.fdopen(fd, "wb") as fh:
                fh.write(ck_bytes)
            tmp_ckpt = tmp_path
            use_ckpt_path = tmp_ckpt
            log.info("Checkpoint from bytes -> %s", tmp_ckpt)
        elif ck_path_str and Path(ck_path_str).exists():
            use_ckpt_path = ck_path_str
            log.info("Checkpoint at %s", use_ckpt_path)
        elif "state_dict" in bundle:
            use_ckpt_path = None
        else:
            raise RuntimeError(
                "Bundle has neither 'ckpt_bytes', 'ckpt_path', nor 'state_dict'."
            )

        # ── Resolve config ────────────────────────────────────────────────────
        cfg_name_for_build = None

        if cfg_in is not None and not isinstance(cfg_in, str):
            # dict / DictConfig — dump to a temp YAML so build_sam2 gets a name
            cfg_dc = (OmegaConf.create(cfg_in)
                      if not isinstance(cfg_in, DictConfig) else cfg_in)
            tmp_cfg_dir = tempfile.mkdtemp(prefix="sam2cfg_")
            tmp_yaml = os.path.join(tmp_cfg_dir, "bundle_cfg.yaml")
            with open(tmp_yaml, "w") as f:
                f.write(OmegaConf.to_yaml(cfg_dc))
            _hydra_reinit_to_dir(tmp_cfg_dir)
            cfg_name_for_build = "bundle_cfg"
            log.info("Config from bundle dict -> %s", tmp_yaml)

        if cfg_name_for_build is None:
            cfg_short = (
                meta.get("config_name")
                or bundle.get("cfg_short_name")
                or (str(cfg_in).strip() if isinstance(cfg_in, str) else None)
                or "sam2.1_hiera_l"
            )
            cfg_dir_env = os.environ.get("SAM2_CONFIG_DIR")
            if cfg_dir_env and Path(cfg_dir_env).is_dir():
                _hydra_reinit_to_dir(cfg_dir_env)
            cfg_name_for_build = cfg_short
            log.info("Config short name: %s", cfg_name_for_build)

        # ── Build model ───────────────────────────────────────────────────────
        model = build_sam2(cfg_name_for_build, use_ckpt_path,
                           device=device, apply_postprocessing=apply_pp)

        if use_ckpt_path is None and "state_dict" in bundle:
            model.load_state_dict(bundle["state_dict"], strict=False)

        gen = make_mask_generator(model)
        return model, gen, meta

    finally:
        if tmp_ckpt and Path(tmp_ckpt).exists():
            try:
                os.remove(tmp_ckpt)
            except Exception:
                pass
