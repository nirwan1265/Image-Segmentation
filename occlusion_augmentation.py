#!/usr/bin/env python3
import os, sys, json, math, random, glob
from pathlib import Path
import tempfile
import shutil

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

# ----------------------------
# 1) Try SAM2 first, fallback to SAM(v1)
# ----------------------------
SAM2_AVAILABLE = True
try:
    from sam2.build_sam import build_sam2 as _build_sam
except Exception:
    SAM2_AVAILABLE = False
    try:
        from segment_anything import sam_model_registry as _sam_reg
    except Exception as e:
        raise RuntimeError(
            "Neither SAM2 nor SAM(v1) found. Install SAM2 repo (preferred) or segment-anything."
        ) from e


def _setup_hydra_for_sam2(config_dir: str = None):
    """Initialize Hydra to find SAM2 configs."""
    try:
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        # Try to find SAM2 config directory
        if config_dir and os.path.isdir(config_dir):
            cfg_dir = config_dir
        else:
            # Common locations
            candidates = [
                os.environ.get("SAM2_CONFIG_DIR", ""),
                os.path.expanduser("~/Documents/Github/sam2/configs/sam2.1"),
                "/Users/nirwantandukar/Documents/Github/sam2/configs/sam2.1",
                os.path.join(os.path.dirname(__file__), "../sam2/configs/sam2.1"),
            ]
            cfg_dir = None
            for c in candidates:
                if c and os.path.isdir(c):
                    cfg_dir = os.path.abspath(c)
                    break

        if cfg_dir:
            GlobalHydra.instance().clear()
            initialize_config_dir(config_dir=cfg_dir, job_name="occlusion_train")
            print(f"Hydra initialized with config dir: {cfg_dir}")
            return True
    except Exception as e:
        print(f"Hydra setup warning: {e}")
    return False


def _detect_model_size_from_state_dict(state_dict: dict) -> str:
    """Detect SAM2 model size (tiny/small/base_plus/large) from state dict keys."""
    # Count image encoder blocks to determine model size
    block_keys = [k for k in state_dict.keys() if "image_encoder.trunk.blocks." in k and ".norm1.weight" in k]
    num_blocks = len(block_keys)

    print(f"Detected {num_blocks} image encoder blocks")

    # SAM2.1 model sizes:
    # tiny: ~12 blocks, small: ~16 blocks, base_plus: ~24 blocks, large: ~48 blocks
    if num_blocks >= 40:
        return "sam2.1_hiera_l"  # large
    elif num_blocks >= 20:
        return "sam2.1_hiera_b+"  # base_plus
    elif num_blocks >= 14:
        return "sam2.1_hiera_s"  # small
    else:
        return "sam2.1_hiera_t"  # tiny


def build_model(checkpoint: str, config: str | None, device: str = "cpu"):
    """
    Returns (model, is_sam2: bool)
    model must expose .image_encoder, .prompt_encoder, .mask_decoder

    Handles both regular checkpoints and bundle format (sam2_bundle.pt).
    """
    if not SAM2_AVAILABLE:
        # Fallback to SAM v1
        name = "vit_h" if "sam_vit_h" in checkpoint.lower() else "vit_b"
        model = _sam_reg[name](checkpoint=checkpoint)
        return model, False

    # Load the checkpoint to inspect it
    print(f"Loading checkpoint: {checkpoint}")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Check if it's a bundle format
    is_bundle = isinstance(bundle, dict) and (
        "ckpt_bytes" in bundle or "checkpoint_bytes" in bundle or
        "state_dict" in bundle or "cfg" in bundle
    )

    if is_bundle:
        print("Detected bundle format")

        # Extract components from bundle
        ck_bytes = bundle.get("ckpt_bytes") or bundle.get("checkpoint_bytes")
        state_dict = bundle.get("state_dict")
        cfg_data = bundle.get("cfg")
        meta = bundle.get("meta", {})

        # Try to get config name from bundle
        bundle_config = meta.get("config_name") or bundle.get("cfg_short_name")
        if bundle_config:
            print(f"Bundle specifies config: {bundle_config}")
            config = bundle_config

        # Determine what we have for weights
        if ck_bytes:
            # Write checkpoint bytes to temp file
            fd, tmp_ckpt = tempfile.mkstemp(suffix=".pt")
            with os.fdopen(fd, "wb") as fh:
                fh.write(ck_bytes)
            print(f"Extracted checkpoint bytes to: {tmp_ckpt}")

            # Load the extracted checkpoint to get state dict
            inner_ckpt = torch.load(tmp_ckpt, map_location="cpu", weights_only=False)
            if isinstance(inner_ckpt, dict) and "model" in inner_ckpt:
                state_dict = inner_ckpt["model"]
            else:
                state_dict = inner_ckpt

            os.remove(tmp_ckpt)

        if state_dict is None:
            raise RuntimeError("Bundle has no weights (ckpt_bytes or state_dict)")

        # Auto-detect config from state dict if not provided
        if not config or config in ["sam2_hiera_s.yaml", "sam2_hiera_s"]:
            detected = _detect_model_size_from_state_dict(state_dict)
            print(f"Auto-detected model config: {detected}")
            config = detected

    else:
        # Regular checkpoint file
        if isinstance(bundle, dict) and "model" in bundle:
            state_dict = bundle["model"]
        else:
            state_dict = bundle

        # Auto-detect config if not valid
        if not config or "hiera_s" in (config or ""):
            detected = _detect_model_size_from_state_dict(state_dict)
            print(f"Auto-detected model config: {detected}")
            config = detected

    # Setup Hydra for SAM2 config resolution
    _setup_hydra_for_sam2()

    # Clean config name (remove .yaml if present)
    if config and config.endswith(".yaml"):
        config = config[:-5]

    print(f"Building SAM2 model with config: {config}")

    # Create a temp checkpoint file with proper format for build_sam2
    fd, tmp_ckpt = tempfile.mkstemp(suffix=".pt")
    torch.save({"model": state_dict}, tmp_ckpt)
    os.close(fd)

    try:
        model = _build_sam(config, tmp_ckpt, device=device, apply_postprocessing=False)
        print("Model built successfully!")
    finally:
        try:
            os.remove(tmp_ckpt)
        except Exception:
            pass

    return model, True


# ----------------------------
# 2) Dataset: read single-leaf crops + occlusion aug
# ----------------------------
def _read_img_and_mask(path: Path):
    """
    Accepts:
      - PNG with alpha: mask from alpha
      - <name>.jpg + <name>_mask.png  (or .png + _mask.png)
      - Otherwise, tries Otsu on grayscale as a fallback
    Returns RGB uint8 (H,W,3), mask uint8 {0,1}
    """
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(p)
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA -> RGB+alpha
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        a = img[..., 3]
        m = (a > 0).astype(np.uint8)
        return rgb, m

    # look for paired mask
    base = p.with_suffix('')
    for cand in [base.as_posix() + "_mask.png", base.as_posix() + "_m.png", base.with_suffix(".png").as_posix()]:
        if cand != str(p) and os.path.exists(cand):
            mask = cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                m = (mask > 0).astype(np.uint8)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return rgb, m

    # fallback: otsu (works if background is very different)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    m = (th > 0).astype(np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb, m


def _read_img_and_mask_from_dirs(img_path: Path, masks_dir: Path):
    """
    Read image from img_path and mask from masks_dir using same stem.
    Mask is expected as a binary or grayscale image; values > 0 are treated as mask.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks_dir = Path(masks_dir)
    stem = img_path.stem
    mask_path = None
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        cand = masks_dir / f"{stem}{ext}"
        if cand.exists():
            mask_path = cand
            break
    if mask_path is None:
        raise FileNotFoundError(f"No mask for {img_path.name} in {masks_dir}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    m = (mask > 0).astype(np.uint8)
    return rgb, m


def _build_mask_index(masks_dir: Path):
    """Index mask files by stem and common suffix patterns."""
    masks_dir = Path(masks_dir)
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    index = {}
    for p in masks_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        stem = p.stem
        bases = {stem}
        # common suffixes
        for suf in ("_mask", "_m"):
            if stem.endswith(suf):
                bases.add(stem[: -len(suf)])
        # instance masks like *_inst01
        if "_inst" in stem:
            bases.add(stem.split("_inst")[0])
        for base in bases:
            index.setdefault(base, []).append(p)
    return index


def _load_mask_union(mask_paths):
    """Load and union one or more mask files (grayscale or binary)."""
    if not mask_paths:
        return None
    union = None
    for mp in mask_paths:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 0).astype(np.uint8)
        if union is None:
            union = m
        else:
            # ensure same size
            if union.shape != m.shape:
                m = cv2.resize(m, (union.shape[1], union.shape[0]), interpolation=cv2.INTER_NEAREST)
            union = np.maximum(union, m)
    return union


def _candidate_stems_from_image(stem: str):
    """Generate possible mask stems from an image stem (handles _crop naming)."""
    stems = {stem}
    if "_crop_" in stem:
        stems.add(stem.replace("_crop_", "_"))
    if stem.endswith("_crop"):
        stems.add(stem[: -len("_crop")])
    if "_crop" in stem:
        stems.add(stem.replace("_crop", ""))
    return stems


def _load_rgb_and_mask_for_image(img_path: Path, mask_index: dict):
    """Load RGB image and unioned mask using a prebuilt mask index."""
    stems = _candidate_stems_from_image(img_path.stem)
    mask_paths = []
    for s in stems:
        if s in mask_index:
            mask_paths = mask_index.get(s, [])
            if mask_paths:
                break
    if not mask_paths:
        return None, None
    rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return None, None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    full_mask = _load_mask_union(mask_paths)
    if full_mask is None:
        return None, None
    if full_mask.shape[:2] != rgb.shape[:2]:
        full_mask = cv2.resize(full_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return rgb, full_mask


def _random_shape_mask(h, w, center=None):
    m = np.zeros((h, w), np.uint8)
    if h < 2 or w < 2:
        return m
    k = random.choice([1, 2, 3])
    for _ in range(k):
        kind = random.choice(["rect", "ellipse", "poly"])
        if center is None:
            cx, cy = random.randrange(0, w), random.randrange(0, h)
        else:
            cx, cy = int(center[0]), int(center[1])

        if kind == "rect":
            if center is None:
                x1 = random.randrange(0, w - 1)
                y1 = random.randrange(0, h - 1)
                x2 = random.randrange(x1 + 1, w)
                y2 = random.randrange(y1 + 1, h)
            else:
                min_dim = max(6, min(h, w) // 12)
                max_dim = max(min_dim + 2, min(h, w) // 3)
                rw = random.randrange(min_dim, max_dim)
                rh = random.randrange(min_dim, max_dim)
                x1 = max(0, cx - rw // 2)
                y1 = max(0, cy - rh // 2)
                x2 = min(w, x1 + rw)
                y2 = min(h, y1 + rh)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(m, (x1, y1), (x2, y2), 255, -1)
        elif kind == "ellipse":
            ax, ay = max(6, w//12), max(6, h//12)
            ang = random.randrange(0, 180)
            cv2.ellipse(m, (cx, cy), (ax, ay), ang, 0, 360, 255, -1)
        else:
            if center is None:
                pts = np.stack([np.random.randint(0, w, 6), np.random.randint(0, h, 6)], 1).astype(np.int32)
            else:
                min_dim = max(6, min(h, w) // 12)
                max_dim = max(min_dim + 2, min(h, w) // 3)
                angles = np.random.uniform(0, 2*np.pi, 6)
                radii = np.random.uniform(min_dim, max_dim, 6)
                xs = (cx + radii * np.cos(angles)).clip(0, w-1)
                ys = (cy + radii * np.sin(angles)).clip(0, h-1)
                pts = np.stack([xs, ys], 1).astype(np.int32)
            cv2.fillConvexPoly(m, pts, 255)
    return m


def _edge_points_from_mask(mask_u8: np.ndarray, band_px: int = 8):
    """Return a list of (x,y) points near the boundary inside the mask."""
    if band_px <= 0:
        return None
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    edge = (dist > 0) & (dist <= float(band_px))
    ys, xs = np.where(edge)
    if len(xs) == 0:
        return None
    return list(zip(xs.tolist(), ys.tolist()))


def _paste_rgba(src_rgba: np.ndarray, dst_rgb: np.ndarray, cx: int, cy: int):
    """Alpha paste src RGBA onto dst RGB with center (cx,cy). In-place on dst."""
    Hs, Ws = src_rgba.shape[:2]
    Hd, Wd = dst_rgb.shape[:2]
    x1 = int(round(cx - Ws/2)); y1 = int(round(cy - Hs/2))
    x2, y2 = x1 + Ws, y1 + Hs

    sx1 = max(0, -x1); sy1 = max(0, -y1)
    dx1 = max(0, x1);  dy1 = max(0, y1)
    sx2 = Ws - max(0, x2 - Wd); sy2 = Hs - max(0, y2 - Hd)
    dx2 = min(Wd, x2); dy2 = min(Hd, y2)

    if sx2 <= sx1 or sy2 <= sy1: 
        return

    roi_src = src_rgba[sy1:sy2, sx1:sx2]
    roi_dst = dst_rgb[dy1:dy2, dx1:dx2]

    alpha = roi_src[..., 3:4].astype(np.float32) / 255.0
    color = roi_src[..., :3].astype(np.float32)
    dst_rgb[dy1:dy2, dx1:dx2] = (alpha * color + (1 - alpha) * roi_dst.astype(np.float32)).astype(np.uint8)


def occlude_with_bank(
    base_rgb: np.ndarray,
    target_mask: np.ndarray,
    bank_rgba: list,
    min_occ: float = 0.15,
    max_occ: float = 0.50,
    occ_count_min: int = 1,
    occ_count_max: int = 3,
    use_geom: bool = True,
    edge_bias: float = 0.0,
    edge_band_px: int = 8,
    edge_only: bool = False,
):
    """
    Paste random RGBA leaves + (optionally) geometric shapes to occlude a random
    fraction in [min_occ, max_occ]. Returns (img_occ, visible_mask).
    """
    assert 0.0 <= min_occ <= max_occ < 0.95
    assert 0 <= occ_count_min <= occ_count_max

    H, W = target_mask.shape
    occ = np.zeros((H, W), np.uint8)
    img_occ = base_rgb.copy()

    # target occlusion fraction FOR THIS SAMPLE
    target_frac = random.uniform(min_occ, max_occ)

    # edge-biased centers (optional)
    edge_pts = None
    if edge_bias > 0:
        edge_pts = _edge_points_from_mask(target_mask, band_px=edge_band_px)

    def _pick_center():
        if edge_only and edge_pts:
            return random.choice(edge_pts)
        if edge_pts and random.random() < edge_bias:
            return random.choice(edge_pts)
        return (random.randrange(0, W), random.randrange(0, H))

    # 1) paste N leaf occluders
    n = random.randint(occ_count_min, occ_count_max)
    for _ in range(n):
        if not bank_rgba:
            break
        rgba = random.choice(bank_rgba)
        scale = 0.5 + 1.2 * random.random()
        rgba_res = cv2.resize(rgba, (int(rgba.shape[1]*scale), int(rgba.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
        cx, cy = _pick_center()
        _paste_rgba(rgba_res, img_occ, cx, cy)

        # update occ mask from alpha
        Hs, Ws = rgba_res.shape[:2]
        tmp = np.zeros((H, W), np.uint8)
        _paste_rgba(np.dstack([np.zeros_like(rgba_res[..., :3]), rgba_res[..., 3]]),
                    np.dstack([tmp, tmp, tmp]), cx, cy)
        occ = cv2.max(occ, (tmp > 0).astype(np.uint8) * 255)

    # 2) bump occlusion (geom or extra leaves) until we hit target_frac (limited tries)
    def _frac():
        inter = cv2.bitwise_and(occ, (target_mask > 0).astype(np.uint8) * 255)
        denom = max(1, int((target_mask > 0).sum()))
        return float((inter > 0).sum()) / float(denom)

    tries, f = 0, _frac()
    while f < target_frac and tries < 6:
        if use_geom:
            cx, cy = _pick_center()
            occ = cv2.bitwise_or(occ, _random_shape_mask(H, W, center=(cx, cy)))
        elif bank_rgba:
            rgba = random.choice(bank_rgba)
            scale = 0.6 + 1.4 * random.random()
            rgba_res = cv2.resize(rgba, (int(rgba.shape[1]*scale), int(rgba.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
            cx, cy = _pick_center()
            _paste_rgba(rgba_res, img_occ, cx, cy)
            tmp = np.zeros((H, W), np.uint8)
            _paste_rgba(np.dstack([np.zeros_like(rgba_res[..., :3]), rgba_res[..., 3]]),
                        np.dstack([tmp, tmp, tmp]), cx, cy)
            occ = cv2.max(occ, (tmp > 0).astype(np.uint8) * 255)
        f, tries = _frac(), tries + 1

    # mild dim on occluded pixels
    dim = (occ > 0)
    if dim.any():
        img_occ[dim] = (0.5 * img_occ[dim] + 0.5 * np.random.randint(150, 200)).astype(np.uint8)

    visible = np.logical_and(target_mask > 0, occ == 0).astype(np.uint8)
    return img_occ, visible




def _rand_point_in_mask(m_uint8: np.ndarray):
    ys, xs = np.where(m_uint8 > 0)
    if len(xs) == 0: return None
    i = np.random.randint(0, len(xs))
    return int(xs[i]), int(ys[i])

from pathlib import Path
from torch.utils.data import Dataset
import random, cv2, numpy as np

class LeafSegmentsDataset(Dataset):
    def __init__(
        self,
        root: str,
        resize_to: int = 512,
        val: bool = False,
        split: float = 0.1,
        occ_min: float = 0.15,
        occ_max: float = 0.50,
        occ_count_min: int = 1,
        occ_count_max: int = 3,
        use_geom: bool = True,
        edge_bias: float = 0.0,
        edge_band_px: int = 8,
        edge_only: bool = False,
        images_dir: str | None = None,
        masks_dir: str | None = None,
    ):
        self.root = Path(root)
        self.images_dir = Path(images_dir) if images_dir else None
        self.masks_dir = Path(masks_dir) if masks_dir else None

        if self.images_dir:
            self.paths = sorted([p for p in self.images_dir.iterdir()
                                 if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")])
        else:
            self.paths = sorted([p for p in self.root.rglob("*")
                                 if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        assert self.paths, f"No images found in {self.images_dir or root}"

        # If masks_dir provided, build an index to match common suffix patterns
        if self.masks_dir:
            mask_index = _build_mask_index(self.masks_dir)
            self.mask_index = mask_index
            paired = []
            for p in self.paths:
                stems = _candidate_stems_from_image(p.stem)
                if any(s in mask_index for s in stems):
                    paired.append(p)
            self.paths = paired
            assert self.paths, f"No image/mask pairs found in {self.images_dir} + {self.masks_dir}"
        else:
            self.mask_index = None

        # simple split
        random.seed(42)
        random.shuffle(self.paths)
        n_val = max(1, int(len(self.paths) * split))
        self.paths = self.paths[-n_val:] if val else self.paths[:-n_val]

        # occluder bank (RGBA) from a subset
        self.bank = []
        for p in self.paths[: min(200, len(self.paths))]:
            if self.masks_dir:
                rgb, m = _load_rgb_and_mask_for_image(p, self.mask_index or {})
                if rgb is None or m is None:
                    continue
            else:
                rgb, m = _read_img_and_mask(p)
            if m.sum() < 100:
                continue
            a = (m * 255).astype(np.uint8)
            rgba = np.dstack([rgb, a])
            self.bank.append(rgba)

        self.resize_to = resize_to
        self.occ_min = occ_min
        self.occ_max = occ_max
        self.occ_count_min = occ_count_min
        self.occ_count_max = occ_count_max
        self.use_geom = use_geom
        self.edge_bias = float(edge_bias)
        self.edge_band_px = int(edge_band_px)
        self.edge_only = bool(edge_only)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.masks_dir:
            # Support mask files with suffixes like _mask, _m, _instXX
            rgb, full_mask = _load_rgb_and_mask_for_image(self.paths[i], self.mask_index or {})
            if rgb is None or full_mask is None:
                raise FileNotFoundError(f"No mask for {self.paths[i].name} in {self.masks_dir}")
        else:
            rgb, full_mask = _read_img_and_mask(self.paths[i])

        # resize keeping aspect ratio → pad to square
        H, W = rgb.shape[:2]
        scale = self.resize_to / max(H, W)
        newW, newH = int(round(W * scale)), int(round(H * scale))
        rgb = cv2.resize(rgb, (newW, newH), interpolation=cv2.INTER_AREA)
        full_mask = cv2.resize(full_mask, (newW, newH), interpolation=cv2.INTER_NEAREST)

        padL = (self.resize_to - newW) // 2
        padT = (self.resize_to - newH) // 2
        rgb_sq = np.full((self.resize_to, self.resize_to, 3), 185, np.uint8)
        m_sq   = np.zeros((self.resize_to, self.resize_to), np.uint8)
        rgb_sq[padT:padT + newH, padL:padL + newW] = rgb
        m_sq  [padT:padT + newH, padL:padL + newW] = full_mask

        # build occluded view with your ranges
        img_occ, vis_mask = occlude_with_bank(
            rgb_sq, m_sq, self.bank,
            min_occ=self.occ_min, max_occ=self.occ_max,
            occ_count_min=self.occ_count_min, occ_count_max=self.occ_count_max,
            use_geom=self.use_geom,
            edge_bias=self.edge_bias,
            edge_band_px=self.edge_band_px,
            edge_only=self.edge_only,
        )

        # positive prompt point from visible region (fallback to full mask)
        pt = _rand_point_in_mask(vis_mask)
        if pt is None:
            pt = _rand_point_in_mask(m_sq)

        return {
            "image": img_occ,                  # HxWx3 uint8
            "target": (m_sq > 0).astype(np.uint8),
            "point": pt                        # (x, y)
        }



# ----------------------------
# 3) SAM forward helper (SAM2 or SAM1)
# ----------------------------

def _get_prompt_encoder(model, is_sam2: bool):
    """Get prompt encoder - SAM2 uses 'sam_prompt_encoder', SAM v1 uses 'prompt_encoder'."""
    if is_sam2:
        return model.sam_prompt_encoder
    return model.prompt_encoder


def _get_mask_decoder(model, is_sam2: bool):
    """Get mask decoder - SAM2 uses 'sam_mask_decoder', SAM v1 uses 'mask_decoder'."""
    if is_sam2:
        return model.sam_mask_decoder
    return model.mask_decoder


def _normalize_point(pt_xy):
    """Normalize point input to a 2-tuple (x, y) or None."""
    if pt_xy is None:
        return None

    def _scalar_to_int(v):
        if torch.is_tensor(v):
            if v.numel() == 0:
                return None
            return int(v.view(-1)[0].item())
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return None
            return int(v.reshape(-1)[0])
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return int(v)
        return None

    # Torch tensor (e.g., shape (1,2) from DataLoader collate)
    if torch.is_tensor(pt_xy):
        if pt_xy.numel() < 2:
            return None
        flat = pt_xy.view(-1)
        return (int(flat[0].item()), int(flat[1].item()))

    # Numpy array
    if isinstance(pt_xy, np.ndarray):
        if pt_xy.size < 2:
            return None
        flat = pt_xy.reshape(-1)
        return (int(flat[0]), int(flat[1]))

    # List/tuple (may be wrapped, e.g., [ (x,y) ])
    if isinstance(pt_xy, (list, tuple)):
        if len(pt_xy) == 0:
            return None
        if len(pt_xy) == 1:
            return _normalize_point(pt_xy[0])
        # Common case: list/tuple of two scalars or 0-d tensors/arrays
        x = _scalar_to_int(pt_xy[0])
        y = _scalar_to_int(pt_xy[1])
        if x is not None and y is not None:
            return (x, y)
        # Fallback: take first element (e.g., list of points)
        return _normalize_point(pt_xy[0])

    return None


def sam_forward_logits(model, is_sam2: bool, img_rgb_uint8: np.ndarray, pt_xy: tuple[int,int] | None):
    """
    Returns mask logits upsampled to image size: (1,1,H,W) torch.float32
    """
    device = next(model.parameters()).device
    H, W = img_rgb_uint8.shape[:2]

    prompt_encoder = _get_prompt_encoder(model, is_sam2)
    mask_decoder = _get_mask_decoder(model, is_sam2)

    if is_sam2:
        # SAM2 expects normalized input + special feature prep
        from sam2.utils.transforms import SAM2Transforms

        # cache transforms by resolution
        global _SAM2_TRANSFORMS_CACHE
        try:
            _SAM2_TRANSFORMS_CACHE
        except NameError:
            _SAM2_TRANSFORMS_CACHE = {}

        res = int(getattr(model, "image_size", 1024))
        if res not in _SAM2_TRANSFORMS_CACHE:
            _SAM2_TRANSFORMS_CACHE[res] = SAM2Transforms(
                resolution=res, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0
            )
        transforms = _SAM2_TRANSFORMS_CACHE[res]

        img_t = transforms(img_rgb_uint8).unsqueeze(0).to(device)  # (1,3,res,res)

        with torch.no_grad():
            backbone_out = model.forward_image(img_t)
            _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
            if getattr(model, "directly_add_no_mem_embed", False):
                vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(feat.size(1), feat.size(2), *feat_size)
                for feat, feat_size in zip(vision_feats, feat_sizes)
            ]
            image_embed = feats[-1]
            high_res_feats = feats[:-1]

        # Build prompt embeddings (coords in original image space)
        pt_xy = _normalize_point(pt_xy)
        if pt_xy is not None:
            point_coords = torch.tensor([[pt_xy[0], pt_xy[1]]], device=device).float()
            point_coords = transforms.transform_coords(point_coords, normalize=True, orig_hw=(H, W))
            point_coords = point_coords.unsqueeze(0)  # (1,1,2)
            point_labels = torch.tensor([[1]], device=device, dtype=torch.int)
            sparse, dense = prompt_encoder(points=(point_coords, point_labels), boxes=None, masks=None)
        else:
            sparse, dense = prompt_encoder(points=None, boxes=None, masks=None)

        low_res_out = mask_decoder(
            image_embeddings=image_embed,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )
        low_res_logits = low_res_out[0]
        logit = transforms.postprocess_masks(low_res_logits, (H, W))
        return logit

    # SAM v1 path
    img_t = to_tensor(img_rgb_uint8).unsqueeze(0).to(device)  # (1,3,H,W), [0..1]
    with torch.no_grad():
        image_embed = model.image_encoder(img_t)  # (1,C,h,w)

    pt_xy = _normalize_point(pt_xy)
    if pt_xy is not None:
        pts = torch.tensor([[pt_xy[0], pt_xy[1]]], device=device).float().unsqueeze(0)  # (1,1,2)
        lbl = torch.tensor([[1]], device=device).float()                                # (1,1)
        sparse, dense = prompt_encoder(points=(pts, lbl), boxes=None, masks=None)
    else:
        sparse, dense = prompt_encoder(points=None, boxes=None, masks=None)

    low_res_logits, _ = mask_decoder(
        image_embeddings=image_embed,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False
    )
    logit = F.interpolate(low_res_logits, size=(H, W), mode="bilinear", align_corners=False)
    return logit  # (1,1,H,W)


# ----------------------------
# 4) Train
# ----------------------------
def dice_loss_from_logits(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2*(p*target).sum(dim=(1,2,3))
    den = (p+target).sum(dim=(1,2,3)) + eps
    return 1 - (num/den).mean()


def train(
    data_dir: str,
    checkpoint: str,
    config_yaml: str|None,
    out_path: str,
    resume: str|None = None,
    image_size: int = 512,
    batch_size: int = 4,
    steps: int = 20000,
    lr: float = 1e-4,
    device: str = "cuda",
    occ_min: float = 0.15,
    occ_max: float = 0.50,
    occ_count_min: int = 1,
    occ_count_max: int = 3,
    edge_bias: float = 0.0,
    edge_band_px: int = 8,
    edge_only: bool = False,
    images_dir: str | None = None,
    masks_dir: str | None = None,
):
    model, is_sam2 = build_model(checkpoint, config_yaml, device=device)
    # Note: build_sam2 already moves model to device, but SAM v1 doesn't
    if not is_sam2:
        model.to(device)

    # Optional resume: load previous finetuned weights
    if resume:
        try:
            ckpt = torch.load(resume, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    sd = ckpt["state_dict"]
                elif "model" in ckpt:
                    sd = ckpt["model"]
                else:
                    sd = ckpt
            else:
                sd = ckpt
            model.load_state_dict(sd, strict=False)
            print(f"Resumed weights from: {resume}")
        except Exception as e:
            print(f"WARN: Failed to resume from {resume}: {e}")

    # Get the correct encoder/decoder based on model type
    prompt_encoder = _get_prompt_encoder(model, is_sam2)
    mask_decoder = _get_mask_decoder(model, is_sam2)

    # Freeze image encoder; train prompt+mask decoder
    for p in model.image_encoder.parameters(): p.requires_grad = False
    for p in prompt_encoder.parameters(): p.requires_grad = True
    for p in mask_decoder.parameters():   p.requires_grad = True

    opt = torch.optim.AdamW(
        list(prompt_encoder.parameters()) + list(mask_decoder.parameters()),
        lr=lr, weight_decay=1e-4
    )

    ds_train = LeafSegmentsDataset(
        data_dir, resize_to=image_size, val=False, split=0.1,
        occ_min=occ_min, occ_max=occ_max,
        occ_count_min=occ_count_min, occ_count_max=occ_count_max,
        edge_bias=edge_bias, edge_band_px=edge_band_px, edge_only=edge_only,
        images_dir=images_dir, masks_dir=masks_dir,
    )
    ds_val = LeafSegmentsDataset(
        data_dir, resize_to=image_size, val=True, split=0.1,
        occ_min=occ_min, occ_max=occ_max,
        occ_count_min=occ_count_min, occ_count_max=occ_count_max,
        edge_bias=edge_bias, edge_band_px=edge_band_px, edge_only=edge_only,
        images_dir=images_dir, masks_dir=masks_dir,
    )

    # We’ll iterate samples one-by-one inside the step loop (SAM prompts are per-sample anyway)
    loader = DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    vloader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=2)

    bce = torch.nn.BCEWithLogitsLoss()

    step = 0
    losses = []
    best_val = 1e9

    while step < steps:
        for batch in loader:
            model.train()
            img = batch["image"][0].numpy()         # HWC uint8
            tgt = (batch["target"][0].numpy()>0).astype(np.float32)
            pt  = batch["point"]

            logits = sam_forward_logits(model, is_sam2, img, pt)
            target = torch.from_numpy(tgt).to(logits.device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            loss = bce(logits, target) + 0.5 * dice_loss_from_logits(logits, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(prompt_encoder.parameters()) + list(mask_decoder.parameters()), 1.0)
            opt.step()

            step += 1
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                print(f"[{step}/{steps}] loss={np.mean(losses[-100:]):.4f}")

            if step % 1000 == 0:
                # quick val
                model.eval()
                with torch.no_grad():
                    vl = []
                    for vb in vloader:
                        vimg = vb["image"][0].numpy()
                        vtgt = (vb["target"][0].numpy()>0).astype(np.float32)
                        vpt  = vb["point"]
                        vlog = sam_forward_logits(model, is_sam2, vimg, vpt)
                        vtar = torch.from_numpy(vtgt).to(vlog.device).unsqueeze(0).unsqueeze(0)
                        vloss = bce(vlog, vtar) + 0.5 * dice_loss_from_logits(vlog, vtar)
                        vl.append(float(vloss.cpu()))
                    vmean = float(np.mean(vl))
                    print(f"  val_loss={vmean:.4f}")
                    # save best
                    if vmean < best_val:
                        best_val = vmean
                        save_dict = {
                            "state_dict": model.state_dict(),
                            "meta": {
                                "sam2": is_sam2,
                                "checkpoint": checkpoint,
                                "config": config_yaml,
                                "image_size": image_size,
                            }
                        }
                        torch.save(save_dict, out_path)
                        print(f"  saved: {out_path}")

            if step >= steps:
                break

    # final save
    save_dict = {
        "state_dict": model.state_dict(),
        "meta": {
            "sam2": is_sam2,
            "checkpoint": checkpoint,
            "config": config_yaml,
            "image_size": image_size,
        }
    }
    torch.save(save_dict, out_path)
    print(f"Done. Saved to {out_path}")


# ----------------------------
# 5) CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Fine-tune SAM2/SAM for leaf completion with occlusion augmentation")
    ap.add_argument("--data", required=True, help="Folder with individual leaf segments")
    ap.add_argument("--images", default=None, help="Optional: folder with leaf images (paired with --masks)")
    ap.add_argument("--masks", default=None, help="Optional: folder with masks (same stem as images)")
    ap.add_argument("--ckpt", required=True, help="Path to base SAM2 (or SAM) checkpoint .pt")
    ap.add_argument("--config", default=None, help="SAM2 YAML (ignored for SAM v1)")
    ap.add_argument("--out", required=True, help="Where to save finetuned .pth")
    ap.add_argument("--resume", default=None, help="Optional existing .pth to resume from")
    ap.add_argument("--size", type=int, default=512, help="square training size")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--occ-min", type=float, default=0.15)
    ap.add_argument("--occ-max", type=float, default=0.50)
    ap.add_argument("--occ-count-min", type=int, default=1)
    ap.add_argument("--occ-count-max", type=int, default=1)
    ap.add_argument("--edge-bias", type=float, default=0.0, help="Probability of placing occluders near leaf edges")
    ap.add_argument("--edge-band", type=int, default=8, help="Edge band thickness (px) for edge-biased occlusions")
    ap.add_argument("--edge-only", action="store_true", help="Place occluders only near leaf edges")
    ap.add_argument("--bs", type=int, default=1, help="internal batch is 1 (prompts per-sample); keep 1")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if str(args.device).startswith("mps"):
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("MPS fallback enabled (PYTORCH_ENABLE_MPS_FALLBACK=1) for unsupported ops.")

    train(
        data_dir=args.data,
        checkpoint=args.ckpt,
        config_yaml=args.config,
        out_path=args.out,
        resume=args.resume,
        image_size=args.size,
        batch_size=args.bs,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        occ_min=args.occ_min,
        occ_max=args.occ_max,
        occ_count_min=args.occ_count_min,
        occ_count_max=args.occ_count_max,
        edge_bias=args.edge_bias,
        edge_band_px=args.edge_band,
        edge_only=args.edge_only,
        images_dir=args.images,
        masks_dir=args.masks,
    )



# Run
# example
#python train_leaf_completion.py \
#  --data /path/to/leaves_folder \  # Folder of single-leaf images. Each file is either: PNG with alpha (mask from alpha), or img.jpg + img_mask.png (same basename).
#  --ckpt /path/to/sam2.1_hiera_l.pt \ # Base SAM2 (or SAM v1) checkpoint to fine-tune.
#  --config /path/to/sam2.1_hiera_l.yaml \ # SAM2 model config (ignored if you’re using SAM v1).
#  --out /path/to/finetuned_sam2_occ.pth \ # Where the fine-tuned weights are saved.
#  --occ-min 0.20 --occ-max 0.55 \ 
#  --occ-count-min 2 --occ-count-max 4
#  --size 512 \ # Training resolution. Each image is resized & padded to a 512×512 square. Bigger → sharper but more VRAM;
#  --steps 20000 \ # Number of training iterations (batch size is effectively 1). With 100 images, each image is seen ~200 times on average, each time with a fresh, random occlusion.
#  --device cuda # Compute device. Use cuda / cuda:0 (NVIDIA), mps (Apple Silicon), or cpu.
