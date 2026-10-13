"""Spatio-frequency overlay: spatial EXP-CAM mask × FFT explanation.

Loads pixel masks from expcam_spatial/spatial_masks/{backbone}/{id}_{class}/
and polar FFT masks from a Section 0 FFT run. Builds

    e_sf = m_s ⊙ e_f + (1 − m_s) ⊙ grey

then reports EF1, Δconf, and saves the combined explanation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from cli import add_common_args, resolve_run
from classifiers import load_classifier
from config import GREY_BG, IMAGENET_MEAN, IMAGENET_STD, SECTION0_IMAGE_SLOTS, SECTION0_PER_CLASS
from data import load_tensor, section0_image_records
from fft import FourierPolarMasker
from io_utils import ensure_dir, load_json, rows_to_excel, save_json
from metrics import (
    denormalize,
    ef1_new_version,
    evaluate_explanation,
    explanation_f1,
    mask_size,
)
from train import log
from viz import panel_mask, panel_original


def _run_dir(root: Path, section: str, backbone: str, config_name: str, image_id: str) -> Path:
    safe = str(config_name).replace("/", "_").replace(" ", "_")
    return Path(root) / section / backbone / safe / image_id


def _apply_subset_args(args) -> None:
    if getattr(args, "quick", False):
        args.section0_per_class = 1
        args.section0_slots = [0]
    elif getattr(args, "subset", False):
        args.section0_per_class = 2
        args.section0_slots = [0, 1]


def _spatial_dir(spatial_root: Path, backbone: str, image_id: str) -> Path | None:
    packed = spatial_root / "spatial_masks" / backbone
    if packed.is_dir():
        hits = sorted(packed.glob(f"{image_id}_*"))
        if hits and (hits[0] / "masks.pt").exists():
            return hits[0]
    for section in ("section0", "section0_fft"):
        d = _run_dir(spatial_root, section, backbone, "ef1_full", image_id)
        if (d / "masks.pt").exists():
            return d
    bases = [spatial_root / backbone, spatial_root]
    for base in bases:
        if not base.is_dir():
            continue
        hits = sorted(base.glob(f"{image_id}_*"))
        if hits and (hits[0] / "masks.pt").exists():
            return hits[0]
        direct = base / image_id
        if (direct / "masks.pt").exists():
            return direct
    return None


def _fft_dir(fft_root: Path, backbone: str, image_id: str) -> Path | None:
    for section in ("section0_fft", "section0"):
        d = _run_dir(fft_root, section, backbone, "ef1_full", image_id)
        if (d / "masks.pt").exists():
            return d
    return None


def _load_pt_masks(path: Path, device: torch.device):
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    hard = blob["mask_hard"].to(device).float()
    prob = blob.get("mask_prob", hard)
    if isinstance(prob, torch.Tensor):
        prob = prob.to(device).float()
    else:
        prob = hard
    return hard, prob, blob


def _resize_mask(mask: torch.Tensor, hw) -> torch.Tensor:
    h, w = int(hw[-2]), int(hw[-1])
    m = mask.float()
    if m.dim() == 2:
        m = m.view(1, 1, *m.shape)
    elif m.dim() == 3:
        m = m.unsqueeze(0)
    if m.shape[-2] != h or m.shape[-1] != w:
        m = torch.nn.functional.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
    return m


def _grey_fill(e: torch.Tensor, mask: torch.Tensor, grey: float = GREY_BG) -> torch.Tensor:
    """Apply spatial mask in pixel space, grey outside, return ImageNet-normalized tensor."""
    vis = denormalize(e)
    m = mask.float()
    if m.dim() == 3:
        m = m.unsqueeze(0)
    if m.shape[1] == 1:
        m = m.expand_as(vis)
    vis_sf = m * vis + (1.0 - m) * grey
    mean = e.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = e.new_tensor(IMAGENET_STD)[None, :, None, None]
    return ((vis_sf - mean) / std).clamp(-10, 10)


def _save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def _plot_overlay(path: Path, panels: list, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.1))
    if n == 1:
        axes = [axes]
    for ax, (img, cap) in zip(axes, panels):
        ax.axis("off")
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(cap, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_overlay(args) -> None:
    eval_set, _images, device = resolve_run(args)
    _apply_subset_args(args)
    subset = section0_image_records(
        eval_set,
        per_class=getattr(args, "section0_per_class", SECTION0_PER_CLASS),
        slots=tuple(getattr(args, "section0_slots", list(SECTION0_IMAGE_SLOTS))),
    )
    spatial_root = Path(args.spatial_root)
    fft_root = Path(args.fft_root)
    out_root = Path(args.out)
    eval_root = Path(eval_set["root"])
    log(
        f"Spatio-freq  device={device}  n={len(subset)}  "
        f"spatial={spatial_root}  fft={fft_root}  out={out_root}"
    )

    rows = []
    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        fig_dir = ensure_dir(out_root / backbone)
        for rec in subset:
            dest = ensure_dir(fig_dir / f"{rec['id']}_{rec['class_name'].replace(' ', '_')}")
            metrics_path = dest / "metrics.json"
            if args.skip_existing and metrics_path.exists():
                rows.append(load_json(metrics_path))
                continue

            sdir = _spatial_dir(spatial_root, backbone, rec["id"])
            fdir = _fft_dir(fft_root, backbone, rec["id"])
            if sdir is None or not (sdir / "masks.pt").exists():
                log(f"[skip] no spatial mask for {backbone} {rec['id']}")
                continue
            if fdir is None:
                log(f"[skip] no FFT mask for {backbone} {rec['id']}")
                continue

            x = load_tensor(rec, bundle.preprocess, device, root=eval_root)
            m_s, m_s_prob, _ = _load_pt_masks(sdir / "masks.pt", device)
            m_s = _resize_mask(m_s, x.shape)
            m_s_prob = _resize_mask(m_s_prob, x.shape)
            m_s_hard = (m_s > 0.5).float()

            m_f, m_f_prob, fblob = _load_pt_masks(fdir / "masks.pt", device)
            radial = int(fblob.get("radial_bins", 16) or 16)
            angular = int(fblob.get("angular_bins", 16) or 16)
            m_f = m_f.reshape(1, 1, radial, angular)
            m_f_prob = m_f_prob.reshape(1, 1, radial, angular)
            fourier = FourierPolarMasker(radial, angular).to(device)
            e_f = fourier.construct_explanation(x, m_f)
            e_s = _grey_fill(x, m_s_hard)
            e_sf = _grey_fill(e_f, m_s_hard)

            p_spatial = mask_size(m_s_hard)
            coeff = fourier.fft2(x)
            half_f = fourier.expand_polar_mask(m_f, x.shape[-2], x.shape[-1])
            p_energy = float(fourier.kept_energy_fraction(coeff, half_f).mean().item())

            met_sf = evaluate_explanation(
                bundle, x, m_s_hard, mask_prob=m_s_prob, e=e_sf, p_energy=p_energy
            )
            met_s = evaluate_explanation(bundle, x, m_s_hard, mask_prob=m_s_prob, e=e_s)
            met_f = evaluate_explanation(bundle, x, m_f, mask_prob=m_f_prob, e=e_f, p_energy=p_energy)

            dconf = float(met_sf["conf_x"] - met_sf["conf_e"])
            ef1_area = explanation_f1(met_sf["confidence_delta"], p_spatial)
            ef1_e = explanation_f1(met_sf["confidence_delta"], p_energy)
            ef1_conf = ef1_new_version(met_sf["conf_e"], p_energy)

            row = {
                "image_id": rec["id"],
                "relpath": rec["relpath"],
                "wnid": rec["wnid"],
                "class_name": rec["class_name"],
                "backbone": backbone,
                "conf_x": met_sf["conf_x"],
                "conf_spatial": met_s["conf_e"],
                "conf_fft": met_f["conf_e"],
                "conf_sf": met_sf["conf_e"],
                "delta_conf": dconf,
                "confidence_delta": met_sf["confidence_delta"],
                "p_spatial": p_spatial,
                "p_spatial_pct": 100.0 * p_spatial,
                "p_energy": p_energy,
                "p_energy_pct": 100.0 * p_energy,
                "ef1": ef1_area,
                "ef1_energy": ef1_e,
                "ef1_hm_conf": ef1_conf,
                "top1_sf": met_sf["top1_agreement"],
                "spatial_dir": str(sdir),
                "fft_dir": str(fdir),
            }
            save_json(metrics_path, row)
            rows.append(row)

            x_np = panel_original(x)
            _save_png(dest / "x.png", x_np)
            _save_png(dest / "mask_spatial.png", panel_mask(m_s_hard))
            _save_png(dest / "e_spatial.png", panel_original(e_s))
            _save_png(dest / "e_fft.png", panel_original(e_f))
            _save_png(dest / "e_sf.png", panel_original(e_sf))
            torch.save(
                {"e_sf": e_sf.cpu(), "e_fft": e_f.cpu(), "mask_spatial": m_s_hard.cpu()},
                dest / "e_sf.pt",
            )
            _plot_overlay(
                dest / "overlay.png",
                [
                    (x_np, f"x\nconf={row['conf_x']:.3f}"),
                    (panel_mask(m_s_hard), f"m_s  p={100 * p_spatial:.1f}%"),
                    (panel_original(e_f), f"e_fft\nconf={row['conf_fft']:.3f}"),
                    (
                        panel_original(e_sf),
                        f"e_sf = m_s⊙e_fft\nconf={row['conf_sf']:.3f}  "
                        f"Δconf={dconf:.3f}  EF1={ef1_area:.3f}",
                    ),
                ],
                title=f"{backbone}  {rec['id']} {rec['class_name']}",
            )
            log(
                f"[sf] {backbone} {rec['id']} {rec['class_name']}  "
                f"conf(x)={row['conf_x']:.3f}  conf(sf)={row['conf_sf']:.3f}  "
                f"Δconf={dconf:.3f}  EF1={ef1_area:.3f}  p_E={100 * p_energy:.1f}%"
            )

        bb_rows = [r for r in rows if r.get("backbone") == backbone]
        if bb_rows:
            def _ms(key):
                vals = [float(r[key]) for r in bb_rows]
                return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), "n": len(vals)}

            save_json(
                fig_dir / "summary.json",
                {
                    "ef1": _ms("ef1"),
                    "delta_conf": _ms("delta_conf"),
                    "conf_sf": _ms("conf_sf"),
                    "p_spatial": _ms("p_spatial"),
                    "p_energy": _ms("p_energy"),
                },
            )

    save_json(out_root / "spatio_freq_runs.json", rows)
    rows_to_excel(out_root / "spatio_freq_results.xlsx", {"overlay": rows})
    log(f"Spatio-freq complete under {out_root}")


def main(argv=None) -> None:
    p = add_common_args(
        argparse.ArgumentParser(description="Spatial mask × FFT explanation overlay")
    )
    repo = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--spatial-root",
        type=Path,
        default=repo / "expcam_spatial",
        help="Pixel EXP-CAM run dir (contains spatial_masks/{backbone}/{id}_class/).",
    )
    p.add_argument(
        "--fft-root",
        type=Path,
        default=repo / "freq_expcam_fft_energy_lambda_80",
        help="FFT Section 0 run dir (contains section0_fft/{backbone}/ef1_full/{id}/).",
    )
    p.add_argument(
        "--subset",
        action="store_true",
        help="2 images per class, slots 0 and 1 (same as FFT --subset).",
    )
    p.add_argument("--section0-per-class", type=int, default=SECTION0_PER_CLASS)
    p.add_argument("--section0-slots", type=int, nargs="+", default=list(SECTION0_IMAGE_SLOTS))
    args = p.parse_args(argv)
    if args.out is None:
        args.out = repo / "spatio_freq"
    if getattr(args, "eval_set", None) is None:
        for cand in (Path(args.fft_root) / "eval_set.json", Path(args.spatial_root) / "eval_set.json"):
            if cand.is_file():
                args.eval_set = cand
                break
    run_overlay(args)


if __name__ == "__main__":
    main()
