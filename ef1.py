"""Section 0 — polar FFT EXP-CAM (generative + attribution-only IG).

Trains FourierMaskNet for SECTION0_STEPS with CE off by default, logs
HM(1−p_E, conf_e) and HM(1−p_E, conf_rob), checkpoints the CLI/YAML metric,
and writes WaveX-style FFT figures plus LPF vs greedy Δconf curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from cli import (
    add_common_args,
    add_loss_lambda_args,
    dwt_settings_from_args,
    fft_settings_from_args,
    lambdas_from_args,
    resolve_run,
)
from classifiers import load_classifier
from config import (
    SECTION0_IMAGE_SLOTS,
    SECTION0_PER_CLASS,
    SECTION0_STEPS,
    ef1_snapshot_steps,
    figure_dir,
)
from data import load_pil, load_tensor, section0_image_records
from io_utils import load_json, rows_to_excel, save_json
from runner import maybe_train_and_eval, maybe_train_and_eval_dwt, maybe_train_and_eval_fft, run_dir
from train import TrainConfig, image_seed, log
from train_dwt import save_dwt_best_band_expl_from_run
from viz import (
    plot_dwt_freq_bias_bars,
    plot_ef1_section0_figures,
    plot_lpf_vs_greedy,
    plot_section0_step_progression_grids,
)

DEFAULT_STEP_GRID_EVERY = (40,)


def _section_name(args) -> str:
    if getattr(args, "dwt", False):
        return "section0_dwt"
    if getattr(args, "pixel", False):
        return "section0_new_score" if getattr(args, "ef1_new_version", False) else "section0"
    return "section0_fft"


def _curve_rows_from_runs(out_root: Path, section: str, backbone: str, subset) -> list:
    curves = []
    for rec in subset:
        out_dir = run_dir(out_root, section, backbone, "ef1_full", rec["id"])
        curve_path = out_dir / "ef1_curve.json"
        if not curve_path.exists():
            continue
        curve = load_json(curve_path)
        if not curve:
            continue
        curves.append(
            {
                "class_name": rec["class_name"],
                "image_id": rec["id"],
                "wnid": rec["wnid"],
                "steps": [p["step"] for p in curve],
                "ef1": [p["ef1"] for p in curve],
                "ef1_energy": [p["ef1_energy"] for p in curve] if curve and "ef1_energy" in curve[0] else None,
                "p_energy": [p["p_energy"] for p in curve] if curve and "p_energy" in curve[0] else None,
                "p_actual": [p["p_actual"] for p in curve] if curve and "p_actual" in curve[0] else None,
            }
        )
    return curves


def _plot_step_grids_for_backbone(out_root, section, backbone, subset, grid_every, eval_root) -> None:
    fig_dir = figure_dir(out_root, section, backbone)
    n = 0
    for rec in subset:
        out_dir = run_dir(out_root, section, backbone, "ef1_full", rec["id"])
        snap_dir = out_dir / "snapshots"
        curve_path = out_dir / "ef1_curve.json"
        if not snap_dir.exists() or not curve_path.exists():
            continue
        curve = load_json(curve_path)
        if not curve:
            continue
        pil = load_pil(rec, root=eval_root)
        import numpy as np

        original = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
        plot_section0_step_progression_grids(
            fig_dir,
            snap_dir,
            image_id=rec["id"],
            class_name=rec["class_name"],
            backbone=backbone,
            curve=curve,
            grid_every=grid_every,
            original_img=original,
        )
        n += 1
    if n:
        log(
            f"[figures] {backbone}: step grids every {list(grid_every)} "
            f"-> {fig_dir}  ({n} images)"
        )


def _plot_dwt_band_grids_for_backbone(out_root, section, backbone, subset, eval_root, device) -> None:
    from shutil import copy2

    from classifiers import load_classifier
    from dwt import dwt_level1, reconstruct_explanation, unet_masks_to_dict
    from runner import load_masks

    fig_dir = figure_dir(out_root, section, backbone)
    bundle = load_classifier(backbone, device)
    n = 0
    for rec in subset:
        out_dir = run_dir(out_root, section, backbone, "ef1_full", rec["id"])
        best_dir = out_dir / "best_ef1"
        grid_src = best_dir / "band_expl_grid.png"
        cname = rec["class_name"].replace(" ", "_")
        grid_dst = fig_dir / f"Fig0_dwt_bands_{rec['id']}_{cname}.png"
        if grid_src.exists():
            copy2(grid_src, grid_dst)
            n += 1
            continue
        if not (out_dir / "masks.pt").exists():
            continue
        x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_root))
        mask_hard, _ = load_masks(out_dir, device)
        meta_path = best_dir / "best_ef1_meta.json"
        meta = load_json(meta_path) if meta_path.exists() else load_json(out_dir / "metrics.json")
        hard_dict = unet_masks_to_dict(mask_hard)
        bands = dwt_level1(x)
        e_full = reconstruct_explanation(bands, unet_masks_to_dict(mask_hard.float()), hard_dict)
        save_dwt_best_band_expl_from_run(out_dir, x, mask_hard, e_full, meta)
        if (best_dir / "band_expl_grid.png").exists():
            copy2(best_dir / "band_expl_grid.png", grid_dst)
            n += 1
    if n:
        log(f"[figures] {backbone}: DWT best-EF1 band grids -> {fig_dir}  ({n} images)")


def _plot_dwt_freq_bias(out_root, section, backbones, subset, eval_root, device) -> None:
    """Per-image grouped bars: kept energy fraction per subband, ResNet vs ViT."""
    from classifiers import load_classifier
    from dwt import SUBBANDS, band_energy_kept_fractions, dwt_level1, unet_masks_to_dict
    from runner import load_masks

    pair = [b for b in ("resnet18", "vit_b_16") if b in backbones]
    if len(pair) < 2:
        pair = list(backbones)
    if not pair:
        return
    dest = Path(out_root) / section / "figures_freq_bias"
    bundle = load_classifier(pair[0], device)
    n = 0
    for rec in subset:
        kept = {}
        x = None
        for bb in pair:
            out_dir = run_dir(out_root, section, bb, "ef1_full", rec["id"])
            if not (out_dir / "masks.pt").exists():
                continue
            if x is None:
                x = load_tensor(rec, bundle.preprocess, device, root=Path(eval_root))
            mask_hard, _ = load_masks(out_dir, device)
            bands = dwt_level1(x)
            hard_dict = unet_masks_to_dict(mask_hard)
            kept[bb] = band_energy_kept_fractions(bands, hard_dict)
        if len(kept) < 1:
            continue
        cname = rec["class_name"].replace(" ", "_")
        plot_dwt_freq_bias_bars(
            dest / f"Fig0_freq_bias_{rec['id']}_{cname}.png",
            kept,
            title=f"Frequency bias — {rec['class_name']} (img {rec['id']})",
            bands=SUBBANDS,
        )
        save_json(dest / f"Fig0_freq_bias_{rec['id']}_{cname}.json", {"image_id": rec["id"], "kept_energy_frac": kept})
        n += 1
    if n:
        log(f"[figures] freq-bias bars -> {dest}  ({n} images, models={pair})")


def _plot_fft_grids_for_backbone(out_root, section, backbone, subset) -> None:
    from shutil import copy2

    fig_dir = figure_dir(out_root, section, backbone)
    n = 0
    for rec in subset:
        out_dir = run_dir(out_root, section, backbone, "ef1_full", rec["id"])
        cname = rec["class_name"].replace(" ", "_")
        src = out_dir / "best_ef1" / "fft_expl_grid.png"
        if src.exists():
            copy2(src, fig_dir / f"Fig0_fft_wavex_{rec['id']}_{cname}.png")
            n += 1
        ig = out_dir / "attribution_ig" / "Fig0_fft_ig.png"
        if ig.exists():
            copy2(ig, fig_dir / f"Fig0_fft_ig_{rec['id']}_{cname}.png")
        lpf = out_dir / "best_ef1" / "Fig0_lpf_vs_greedy.png"
        if lpf.exists():
            copy2(lpf, fig_dir / f"Fig0_lpf_vs_greedy_{rec['id']}_{cname}.png")
    if n:
        log(f"[figures] {backbone}: FFT WaveX grids + IG + LPF -> {fig_dir}")


def _plot_fft_freq_bias(out_root, section, backbones, subset) -> None:
    pair = [b for b in ("resnet18", "vit_b_16") if b in backbones] or list(backbones)
    dest = Path(out_root) / section / "figures_freq_bias"
    n = 0
    for rec in subset:
        kept = {}
        for bb in pair:
            met = run_dir(out_root, section, bb, "ef1_full", rec["id"]) / "metrics.json"
            if not met.exists():
                continue
            data = load_json(met)
            radial = data.get("radial_kept") or {}
            if radial:
                kept[bb] = radial
        if not kept:
            continue
        cname = rec["class_name"].replace(" ", "_")
        plot_dwt_freq_bias_bars(
            dest / f"Fig0_freq_bias_{rec['id']}_{cname}.png",
            kept,
            title=f"Radial frequency bias — {rec['class_name']} (img {rec['id']})",
            bands=("low", "mid", "high"),
        )
        save_json(dest / f"Fig0_freq_bias_{rec['id']}_{cname}.json", {"image_id": rec["id"], "kept_energy_frac": kept})
        n += 1
    if n:
        log(f"[figures] FFT freq-bias bars -> {dest}  ({n} images)")


def _plot_fft_lpf_mean(out_root, section, backbone, subset) -> None:
    curves_l, curves_g, radii = [], [], None
    for rec in subset:
        js = run_dir(out_root, section, backbone, "ef1_full", rec["id"]) / "best_ef1" / "lpf_vs_greedy.json"
        if not js.exists():
            continue
        data = load_json(js)
        rows = data.get("rows") or []
        if not rows:
            continue
        if radii is None:
            radii = [r["radius"] for r in rows]
        curves_l.append([r["delta_conf_lpf"] for r in rows])
        curves_g.append([r["delta_conf_greedy"] for r in rows])
    if not curves_l or radii is None:
        return
    import numpy as np

    arr_l = np.asarray(curves_l, dtype=float)
    arr_g = np.asarray(curves_g, dtype=float)
    fig_dir = figure_dir(out_root, section, backbone)
    plot_lpf_vs_greedy(
        fig_dir / "Fig0_lpf_vs_greedy_mean.png",
        radii,
        arr_l.mean(axis=0).tolist(),
        arr_g.mean(axis=0).tolist(),
        title=f"Mean Δconf vs r — {backbone} (n={len(curves_l)})",
        delta_lpf_std=arr_l.std(axis=0).tolist(),
        delta_greedy_std=arr_g.std(axis=0).tolist(),
    )
    log(f"[figures] {backbone}: mean LPF vs greedy -> {fig_dir / 'Fig0_lpf_vs_greedy_mean.png'}")


def _plot_ef1_for_backbone(out_root, section, backbone, subset, hline) -> dict:
    curves = _curve_rows_from_runs(out_root, section, backbone, subset)
    if not curves:
        return {}
    fig_dir = figure_dir(out_root, section, backbone)
    plot_ef1_section0_figures(fig_dir, curves, backbone, hline=hline)
    log(f"[figures] {backbone} -> {fig_dir}  ({len(curves)} per-image + 1 all-classes)")
    series = {
        f"{c['class_name']} ({c['image_id']})": (c["steps"], c["ef1"]) for c in curves
    }
    if hline is not None:
        save_json(fig_dir / "ef1_threshold_stats.json", _threshold_stats(series, hline))
    return series


def run_section0(args) -> None:
    eval_set, _images, device = resolve_run(args)
    out_root = args.out
    section = _section_name(args)
    steps = args.steps
    if getattr(args, "quick", False):
        args.section0_per_class = 1
        args.section0_slots = [0]
    subset = section0_image_records(
        eval_set,
        per_class=args.section0_per_class,
        slots=tuple(args.section0_slots),
    )
    snap_steps = tuple(args.snapshot_steps) if args.snapshot_steps else ef1_snapshot_steps(max_step=steps)
    rows = []
    use_fft = not args.dwt and not getattr(args, "pixel", False)
    fft_cfg = fft_settings_from_args(args) if use_fft else {}
    if args.dwt:
        dwt_cfg = dwt_settings_from_args(args)
        lambdas = dwt_cfg["lambdas"]
        band_lambdas = dwt_cfg["band_lambdas"]
        disabled_subbands = dwt_cfg["disabled_subbands"]
        wavelet = dwt_cfg["wavelet"]
        ef1_version = "legacy"
        train_fn = maybe_train_and_eval_dwt
        domain = "dwt"
    elif use_fft:
        lambdas = fft_cfg["lambdas"]
        band_lambdas = {}
        disabled_subbands = ()
        wavelet = None
        ef1_version = "new"
        train_fn = maybe_train_and_eval_fft
        domain = "fft"
    else:
        lambdas = lambdas_from_args(args)
        band_lambdas = {}
        disabled_subbands = ()
        wavelet = None
        ef1_version = "new" if args.ef1_new_version else "legacy"
        train_fn = maybe_train_and_eval
        domain = "pixel"
    eval_root = Path(eval_set["root"])

    log(
        f"Section 0  device={device}  subset={len(subset)} images "
        f"({args.section0_per_class}/class)  steps={steps}  section={section}  "
        f"domain={domain}  fft_metric={fft_cfg.get('fft_metric', '—')}  "
        f"lambdas={lambdas}  snapshots={snap_steps[0]}…{snap_steps[-1] if snap_steps else '—'}"
    )

    for backbone in args.backbones:
        log(f"Loading classifier {backbone} ...")
        bundle = load_classifier(backbone, device)
        for rec in subset:
            out_dir = run_dir(out_root, section, backbone, "ef1_full", rec["id"])
            expl_dir = out_dir / "snapshots"
            cfg = TrainConfig(
                lambdas=lambdas,
                band_lambdas=band_lambdas,
                disabled_subbands=disabled_subbands,
                wavelet=wavelet or "db1",
                steps=steps,
                seed=image_seed(rec["id"], backbone, args.seed),
                log_every=args.log_every,
                no_ce=args.no_ce,
                record_ef1=True,
                keep_best_ef1=True,
                ef1_version=ef1_version,
                save_expl_steps=snap_steps,
                save_expl_dir=str(expl_dir),
                mask_thresh=float(fft_cfg.get("mask_thresh", 0.5)) if use_fft else 0.5,
                fft_radial_bins=int(fft_cfg.get("fft_radial_bins", 16)),
                fft_angular_bins=int(fft_cfg.get("fft_angular_bins", 16)),
                fft_mask_temperature=float(fft_cfg.get("fft_mask_temperature", 0.25)),
                fft_noise_scale=float(fft_cfg.get("fft_noise_scale", 0.1)),
                fft_metric=str(fft_cfg.get("fft_metric", "energy_conf")),
                n_lpf_radii=int(fft_cfg.get("n_lpf_radii", 50)),
                ig_n_cutoffs=int(fft_cfg.get("ig_n_cutoffs", 8)),
                ig_steps=int(fft_cfg.get("ig_steps", 32)),
            )
            extra = {
                "protocol": "ef1_fft" if use_fft else ("ef1_dwt" if args.dwt else "ef1"),
                "config": "Full_no_ce_fft" if use_fft else ("Full_no_ce_dwt" if args.dwt else "Full_no_ce"),
                "ef1_version": ef1_version,
            }
            if use_fft:
                row = train_fn(
                    bundle,
                    rec,
                    eval_root,
                    out_dir,
                    cfg,
                    skip_existing=args.skip_existing,
                    device=device,
                    extra_metrics=extra,
                    run_ig=not getattr(args, "no_ig", False),
                )
            else:
                row = train_fn(
                    bundle,
                    rec,
                    eval_root,
                    out_dir,
                    cfg,
                    skip_existing=args.skip_existing,
                    device=device,
                    extra_metrics=extra,
                )
            curve = load_json(out_dir / "ef1_curve.json") if (out_dir / "ef1_curve.json").exists() else []
            if curve:
                row["ef1_final"] = curve[-1]["ef1"]
                row["ef1_max"] = max(p["ef1"] for p in curve)
                row["ef1_mean"] = sum(p["ef1"] for p in curve) / len(curve)
                if "ef1_energy" in curve[0]:
                    row["ef1_energy_final"] = curve[-1]["ef1_energy"]
                    row["ef1_energy_max"] = max(p["ef1_energy"] for p in curve)
                    row["p_energy_at_best"] = max(curve, key=lambda p: p["ef1_energy"]).get("p_energy")
            rows.append(row)

        _plot_ef1_for_backbone(out_root, section, backbone, subset, args.ef1_threshold)
        _plot_step_grids_for_backbone(
            out_root, section, backbone, subset, tuple(args.step_grid_every), eval_root
        )
        if args.dwt:
            _plot_dwt_band_grids_for_backbone(out_root, section, backbone, subset, eval_root, device)
        if use_fft:
            _plot_fft_grids_for_backbone(out_root, section, backbone, subset)
            _plot_fft_lpf_mean(out_root, section, backbone, subset)

    if args.dwt:
        _plot_dwt_freq_bias(out_root, section, args.backbones, subset, eval_root, device)
    if use_fft:
        _plot_fft_freq_bias(out_root, section, args.backbones, subset)

    save_json(out_root / section / "ef1_runs.json", rows)
    rows_to_excel(out_root / section / "section0_results.xlsx", {"ef1": rows})
    log(f"Section 0 complete under {out_root / section}")


def _threshold_stats(series, tau: float) -> dict:
    out = {}
    for label, (steps, vals) in series.items():
        hits = [s for s, v in zip(steps, vals) if v >= tau]
        out[label] = {
            "first_step_at_or_above": hits[0] if hits else None,
            "max_ef1": max(vals) if vals else None,
            "final_ef1": vals[-1] if vals else None,
        }
    return {"threshold": tau, "per_image": out}


def run_section0_figures_only(args) -> None:
    eval_set, _, device = resolve_run(args)
    out_root = args.out
    section = _section_name(args)
    subset = section0_image_records(
        eval_set,
        per_class=args.section0_per_class,
        slots=tuple(args.section0_slots),
    )
    grid_every = tuple(args.step_grid_every)
    eval_root = Path(eval_set["root"])
    if not getattr(args, "step_grids_only", False):
        js = out_root / section / "ef1_runs.json"
        if not js.exists():
            log(f"No {section}/ef1_runs.json — run Section 0 first.")
            return
        load_json(js)
    for backbone in args.backbones:
        if not getattr(args, "step_grids_only", False):
            _plot_ef1_for_backbone(out_root, section, backbone, subset, args.ef1_threshold)
        _plot_step_grids_for_backbone(
            out_root, section, backbone, subset, grid_every, eval_root
        )
        if args.dwt:
            _plot_dwt_band_grids_for_backbone(out_root, section, backbone, subset, eval_root, device)
        if not args.dwt and not getattr(args, "pixel", False):
            _plot_fft_grids_for_backbone(out_root, section, backbone, subset)
            _plot_fft_lpf_mean(out_root, section, backbone, subset)
    if args.dwt:
        _plot_dwt_freq_bias(out_root, section, args.backbones, subset, eval_root, device)
    if not args.dwt and not getattr(args, "pixel", False):
        _plot_fft_freq_bias(out_root, section, args.backbones, subset)


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 0: EF1 diagnostic training"))
    add_loss_lambda_args(p)
    p.set_defaults(steps=SECTION0_STEPS, no_ce=True)
    p.add_argument(
        "--fft-metric",
        choices=["energy_conf", "energy_rob"],
        default=None,
        help="Checkpoint + LPF/greedy comparison: HM(1−p_E, conf_e) or HM(1−p_E, conf_rob). Default from YAML.",
    )
    p.add_argument(
        "--no-ig",
        action="store_true",
        help="Skip attribution-only Fourier IG after generative training.",
    )
    p.add_argument(
        "--pixel",
        action="store_true",
        help="Pixel EXP-CAM Section 0 (legacy). Default is polar FFT.",
    )
    p.add_argument(
        "--dwt",
        action="store_true",
        help="Legacy DWT domain (kept for old runs). Default is polar FFT.",
    )
    p.add_argument(
        "--ef1-new-version",
        action="store_true",
        help="Pixel mode only: use ef1_new_version=HM(conf, 1−p). Ignored with --dwt.",
    )
    p.add_argument(
        "--ef1-threshold",
        type=float,
        default=None,
        help="Optional horizontal line on EF1 plot to inspect a quality threshold.",
    )
    p.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="*",
        default=None,
        help="Override snapshot steps (default: every 40 from step 40).",
    )
    p.add_argument(
        "--section0-per-class",
        type=int,
        default=SECTION0_PER_CLASS,
        help="Images per class for Section 0 training (default: 10).",
    )
    p.add_argument(
        "--section0-slots",
        type=int,
        nargs="+",
        default=list(SECTION0_IMAGE_SLOTS),
        help="Eval-set slot indices per class (default: 0…9).",
    )
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate EF1 plots and/or step grids from saved runs.",
    )
    p.add_argument(
        "--step-grids-only",
        action="store_true",
        help="With --figures-only: only write step progression grids (skip EF1 curves).",
    )
    p.add_argument(
        "--step-grid-every",
        type=int,
        nargs="+",
        default=list(DEFAULT_STEP_GRID_EVERY),
        help="Sampling interval for step progression grids (default: 40).",
    )
    args = p.parse_args(argv)
    if args.step_grids_only:
        args.figures_only = True
    if args.figures_only:
        run_section0_figures_only(args)
    else:
        run_section0(args)


if __name__ == "__main__":
    main()
