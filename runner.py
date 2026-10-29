"""Train/eval one (image, backbone, config) with JSON + mask caching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from classifiers import ClassifierBundle
from data import load_tensor
from io_utils import load_json, save_json
from metrics import evaluate_explanation
from train import TrainConfig, TrainResult, image_seed, log, save_mask_tensors, train_mask
from train_fft import FFTTrainResult, save_fft_mask_tensors, train_mask_fft


def run_dir(root: Path, section: str, backbone: str, config_name: str, image_id: str) -> Path:
    return root / section / backbone / _safe(config_name) / image_id


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def maybe_train_and_eval(
    bundle: ClassifierBundle,
    rec: dict,
    eval_root: Path,
    out_dir: Path,
    cfg: TrainConfig,
    skip_existing: bool,
    device: torch.device,
    extra_metrics: Optional[Dict] = None,
) -> Dict:
    metrics_path = out_dir / "metrics.json"
    tag = ""
    if extra_metrics:
        tag = extra_metrics.get("config") or extra_metrics.get("protocol") or ""
    who = f"{bundle.name}  img={rec['id']} {rec['class_name']}  {tag}".strip()
    if skip_existing and metrics_path.exists():
        log(f"[skip] {who}")
        return load_json(metrics_path)

    log(f"[train] {who}  steps={cfg.steps}")
    x = load_tensor(rec, bundle.preprocess, device, root=eval_root)
    result: TrainResult = train_mask(bundle, x, cfg)
    metrics = evaluate_explanation(
        bundle,
        x,
        result.mask_hard,
        mask_prob=result.mask_prob,
    )
    payload = {
        "image_id": rec["id"],
        "relpath": rec["relpath"],
        "wnid": rec["wnid"],
        "class_name": rec["class_name"],
        "backbone": bundle.name,
        "seed": result.seed,
        "steps": result.steps,
        "lr": cfg.lr,
        "optimizer": "Adam",
        "seconds": result.seconds,
        "lambdas": result.lambdas,
        "constraint": cfg.constraint,
        "p_target": cfg.p_target,
        "mu_final": result.mu_final,
        "auto_balance": cfg.auto_balance,
        "p_actual": result.p_actual,
        "best_ef1": result.best_ef1,
        "best_ef1_step": result.best_ef1_step,
        **metrics,
        **(extra_metrics or {}),
    }
    save_json(metrics_path, payload)
    log(
        f"[done]  img={rec['id']}  {result.seconds:.1f}s  "
        f"p={metrics['p_actual_pct']:.1f}%  top1={int(metrics['top1_agreement'])}  "
        f"conf={metrics['conf_e']:.3f}  kl={metrics['kl_divergence']:.3f}"
    )
    if cfg.record_curve and result.loss_curve:
        save_json(out_dir / "loss_curve.json", result.loss_curve)
    if result.weight_curve:
        save_json(out_dir / "weight_curve.json", result.weight_curve)
    if result.ef1_curve:
        save_json(out_dir / "ef1_curve.json", result.ef1_curve)
    save_mask_tensors(out_dir, result)
    save_json(
        out_dir / "train_meta.json",
        {
            "seconds": result.seconds,
            "steps": result.steps,
            "seed": result.seed,
            "optimizer": "Adam",
            "lr": cfg.lr,
            "per_image": True,
            "architecture": "UNetCSAE",
        },
    )
    return payload


def maybe_train_and_eval_fft(
    bundle: ClassifierBundle,
    rec: dict,
    eval_root: Path,
    out_dir: Path,
    cfg: TrainConfig,
    skip_existing: bool,
    device: torch.device,
    extra_metrics: Optional[Dict] = None,
) -> Dict:
    metrics_path = out_dir / "metrics.json"
    tag = (extra_metrics or {}).get("config") or (extra_metrics or {}).get("protocol") or ""
    who = f"{bundle.name}  img={rec['id']} {rec['class_name']}  FFT  {tag}".strip()
    if skip_existing and metrics_path.exists():
        log(f"[skip] {who}")
        return load_json(metrics_path)

    log(f"[train] {who}  steps={cfg.steps}")
    x = load_tensor(rec, bundle.preprocess, device, root=eval_root)
    result: FFTTrainResult = train_mask_fft(bundle, x, cfg)
    metrics = evaluate_explanation(
        bundle,
        x,
        result.mask_hard,
        mask_prob=result.mask_prob,
        e=result.e,
        p_energy=result.p_energy,
    )
    payload = {
        "image_id": rec["id"],
        "relpath": rec["relpath"],
        "wnid": rec["wnid"],
        "class_name": rec["class_name"],
        "backbone": bundle.name,
        "domain": "fft",
        "seed": result.seed,
        "steps": result.steps,
        "lr": cfg.lr,
        "optimizer": "Adam",
        "seconds": result.seconds,
        "lambdas": result.lambdas,
        "p_energy": result.p_energy,
        "p_energy_pct": 100.0 * result.p_energy,
        "p_bins": result.p_bins,
        "radial_kept": result.radial_kept,
        "best_ef1": result.best_ef1,
        "best_ef1_conf": result.best_ef1_conf,
        "best_ef1_step": result.best_ef1_step,
        "conf_e": result.conf_e,
        "conf_x": result.conf_x,
        "n_lpf_radii": result.n_lpf_radii,
        "radial_bins": result.radial_bins,
        "angular_bins": result.angular_bins,
        **metrics,
        **(extra_metrics or {}),
    }
    save_json(metrics_path, payload)
    log(
        f"[done]  img={rec['id']}  {result.seconds:.1f}s  "
        f"retained p_E={100.0 * result.p_energy:.1f}% of ||FFT(x)||²  "
        f"p_bins={100.0 * result.p_bins:.1f}%  conf(x)={result.conf_x:.3f}  "
        f"conf(e)={result.conf_e:.3f}  EF1={result.best_ef1:.3f} @ step {result.best_ef1_step}"
    )
    if result.ef1_curve:
        save_json(out_dir / "ef1_curve.json", result.ef1_curve)
    save_fft_mask_tensors(out_dir, result)
    save_json(
        out_dir / "train_meta.json",
        {
            "seconds": result.seconds,
            "steps": result.steps,
            "seed": result.seed,
            "optimizer": "Adam",
            "lr": cfg.lr,
            "per_image": True,
            "architecture": {
                "none": "SpectrumMask",
                "tiny_unet": "TinyFourierMaskNet_polar",
                "unet": "FourierMaskNet_polar",
            }.get(getattr(result, "mask_model", "unet"), "FourierMaskNet_polar"),
            "mask_model": getattr(result, "mask_model", "unet"),
            "domain": "fft",
            "checkpoint": "ef1_hm_conf_energy",
        },
    )
    return payload


def load_masks(out_dir: Path, device: torch.device):
    blob = torch.load(out_dir / "masks.pt", map_location=device)
    return blob["mask_hard"].to(device), blob["mask_prob"].to(device)
