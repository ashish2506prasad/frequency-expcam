"""Shared CLI helpers used by every section script."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

import torch

from config import DEFAULT_BACKBONES, MASK_MODEL, SEED, STEPS, add_where_args
from data import load_eval_set

_CUDA_OK = None
_CUDA_NOTE = ""


def cuda_is_runnable() -> bool:
    """True only if a CUDA kernel can actually run, not merely if a GPU is present.

    Kaggle P100 is sm_60; current PyTorch wheels are sm_70+ and will crash at conv2d
    with cudaErrorNoKernelImageForDevice even though torch.cuda.is_available() is True.
    """
    global _CUDA_OK, _CUDA_NOTE
    if _CUDA_OK is not None:
        return _CUDA_OK
    if not torch.cuda.is_available():
        _CUDA_OK, _CUDA_NOTE = False, "CUDA not available"
        return False
    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        getter = getattr(torch.cuda, "get_arch_list", None)
        archs = [a for a in getter() if str(a).startswith("sm_")] if callable(getter) else []
        if archs and sm not in archs:
            _CUDA_OK = False
            _CUDA_NOTE = (
                f"{name} is {sm}; this PyTorch build only has {', '.join(archs)}. "
                "On Kaggle switch the accelerator from P100 to T4 (or any Volta+ GPU), "
                "or pass --device cpu."
            )
            return False
        x = torch.zeros(1, 3, 8, 8, device="cuda")
        w = torch.zeros(1, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        _CUDA_OK, _CUDA_NOTE = True, f"{name} ({sm})"
        return True
    except Exception as e:
        _CUDA_OK = False
        _CUDA_NOTE = f"CUDA probe failed ({type(e).__name__}: {e}). Pass --device cpu or use a T4 GPU."
        return False


def device_from_flag(s: str) -> torch.device:
    s = (s or "auto").strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    want_cuda = s in ("auto", "cuda") or s.startswith("cuda")
    if want_cuda and cuda_is_runnable():
        return torch.device(s if s.startswith("cuda:") else "cuda")
    if want_cuda and s != "cpu":
        print(f"[device] {_CUDA_NOTE}", flush=True)
        print("[device] using CPU", flush=True)
    return torch.device("cpu")


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_where_args(p)
    p.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA if the GPU can actually run kernels, else CPU), cuda, or cpu. "
        "Kaggle P100 is often unusable with current PyTorch; pick a T4 or pass cpu.",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--n-images", type=int, default=None, help="Subset of the cached eval set (still stratified order).")
    p.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--quick", action="store_true", help="Tiny debug run: 1 image/class, 40 steps, resnet18 only.")
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print loss every N U-Net steps (0 = silent). Notebooks used 10.",
    )
    p.add_argument(
        "--no-ce",
        action="store_true",
        help="Omit cross-entropy (CE) loss from U-Net training (all backbones).",
    )
    p.add_argument(
        "--mask-model",
        choices=("unet", "tiny_unet", "none"),
        default=None,
        help=(
            "FFT mask: unet (polar bins), tiny_unet (one-skip polar), "
            f"or none (direct rFFT spectrum). Default: {MASK_MODEL} from config.py."
        ),
    )
    return p


def add_loss_lambda_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """CLI overrides for each loss term (λ_act, λ_ce, …)."""
    p.add_argument("--lam-act", type=float, default=None, help="Override YAML λ_act.")
    p.add_argument("--lam-ce", type=float, default=None, help="Override YAML λ_ce.")
    p.add_argument("--lam-kl", type=float, default=None, help="Override YAML λ_kl.")
    p.add_argument(
        "--lam-area",
        type=float,
        default=None,
        help="Override YAML λ_area (pixel only; ignored for FFT Section 0).",
    )
    p.add_argument("--lam-energy", type=float, default=None, help="Override YAML λ_energy (FFT p_energy).")
    p.add_argument("--lam-bin", type=float, default=None, help="Override YAML λ_bin.")
    p.add_argument("--lam-tv", type=float, default=None, help="Override YAML λ_tv.")
    p.add_argument("--lam-rob", type=float, default=None, help="Override YAML λ_rob.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML with Section 0 loss weights (default: configs/fft.yaml).",
    )
    return p


def fft_settings_from_args(args) -> dict:
    """YAML defaults, then CLI λ / bin overrides."""
    import config as C

    yaml_path = getattr(args, "config", None) or C.DEFAULT_FFT_YAML
    parsed = C.parse_fft_yaml(yaml_path)
    lambdas = lambdas_from_args(args, base=parsed["lambdas"])
    lambdas["area"] = 0.0
    if getattr(args, "lam_area", None) is not None:
        print("[fft] ignoring --lam-area (area loss is not used in polar FFT)", flush=True)
    out = {
        **parsed,
        "lambdas": lambdas,
        "yaml_path": parsed["path"],
    }
    cli_model = getattr(args, "mask_model", None)
    if cli_model:
        out["mask_model"] = str(cli_model).strip().lower()
    elif not out.get("mask_model"):
        out["mask_model"] = C.MASK_MODEL
    return out


def lambdas_from_args(args, base: dict | None = None) -> dict:
    import config as C

    out = dict(base if base is not None else C.DEFAULT_LAMBDAS)
    for key in C.LOSS_TERMS:
        val = getattr(args, f"lam_{key}", None)
        if val is not None:
            out[key] = float(val)
    return out


def resolve_run(args: argparse.Namespace):
    import config as C

    C.apply_where(
        getattr(args, "where", "auto"),
        data_root=getattr(args, "data_root", None),
        runs_dir=getattr(args, "out", None),
        zip_out=getattr(args, "zip_out", None),
    )
    if getattr(args, "out", None) is None:
        args.out = C.RUNS_DIR
    if args.quick:
        args.steps = min(args.steps, 40)
        args.backbones = ["resnet18"]
        args.n_images = args.n_images or 7
    eval_path = getattr(args, "eval_set", None)
    if eval_path is not None:
        from io_utils import load_json

        eval_set = load_json(Path(eval_path))
        eval_set["root"] = str(C.TRAINX_DIR)
        print(f"Reusing eval set ({len(eval_set.get('images', []))} images) at {eval_path}", flush=True)
    else:
        eval_set = load_eval_set()
    images = _stratified_subset(eval_set["images"], args.n_images)
    device = device_from_flag(args.device)
    return eval_set, images, device


def _stratified_subset(images, n):
    """Keep class balance when taking a prefix of the cached set."""
    if n is None or n >= len(images):
        return list(images)
    by_cls = {}
    for rec in images:
        by_cls.setdefault(rec["wnid"], []).append(rec)
    classes = list(by_cls.keys())
    out = []
    i = 0
    while len(out) < n:
        wnid = classes[i % len(classes)]
        slot = i // len(classes)
        if slot < len(by_cls[wnid]):
            out.append(by_cls[wnid][slot])
        i += 1
        if i > n * len(classes):
            break
    return out
