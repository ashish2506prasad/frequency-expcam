"""Section 0 — polar FFT EXP-CAM (generative).

Trains FourierMaskNet for STEPS with CE off by default, logs
EF1 = HM(1−p_E, conf(e)), checkpoints the step with highest EF1,
and writes WaveX-style FFT figures plus LPF vs greedy curves.
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
    fft_settings_from_args,
    lambdas_from_args,
    resolve_run,
)
from classifiers import load_classifier
from config import (
    SECTION0_IMAGE_SLOTS,
    SECTION0_PER_CLASS,
    SNAPSHOT_EVERY,
    STEPS,
    ef1_snapshot_steps,
    figure_dir,
)
from data import load_pil, section0_image_records
from io_utils import load_json, rows_to_excel, save_json
from runner import maybe_train_and_eval, maybe_train_and_eval_fft, run_dir
from train import TrainConfig, image_seed, log
from viz import (
    plot_delta_conf_histograms,
    plot_ef1_section0_figures,
    plot_freq_bias_bars,
    plot_lpf_vs_greedy,
    plot_section0_step_progression_grids,
)

def _section_name(args) -> str:
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
                "ef1_energy": None,
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
        lpf = out_dir / "best_ef1" / "Fig0_lpf_vs_greedy.png"
        if lpf.exists():
            copy2(lpf, fig_dir / f"Fig0_lpf_vs_greedy_{rec['id']}_{cname}.png")
    if n:
        log(f"[figures] {backbone}: FFT WaveX grids + LPF -> {fig_dir}")


def _is_top1(met: dict) -> bool:
    if met.get("top1_agreement") is not None:
        return float(met["top1_agreement"]) >= 0.5
    if "pred_x" in met and "pred_e" in met:
        return int(met["pred_x"]) == int(met["pred_e"])
    return False


def _delta_conf_drop(met: dict) -> float:
    """conf(x) − conf(e); how much original-class confidence the explanation loses."""
    if met.get("conf_x") is not None and met.get("conf_e") is not None:
        return float(met["conf_x"]) - float(met["conf_e"])
    return -float(met.get("confidence_delta", 0.0) or 0.0)


def _plot_fft_freq_bias(out_root, section, backbones, subset) -> None:
    """Radial-band energy bias on top-1 survivors + Δconf histograms per backbone.

    Frequency-bias bars use only images where argmax f(e) = argmax f(x).
    Histograms show Δconf = conf(x) − conf(e) for survivors vs failures.
    Bands: Nyquist-normalized r; kept = ||m ⊙ 1_band ⊙ FFT(x)||² / ||1_band ⊙ FFT(x)||².
    """
    pair = [b for b in ("resnet18", "vit_b_16") if b in backbones] or list(backbones)
    dest = Path(out_root) / section / "figures_freq_bias"
    bands = ("low", "mid", "high")
    per_bb = {bb: {b: [] for b in bands} for bb in pair}
    hist = {bb: {"top1": [], "fail": []} for bb in pair}
    n_total = {bb: 0 for bb in pair}
    n_top1 = {bb: 0 for bb in pair}
    rows = []
    for rec in subset:
        for bb in pair:
            met_path = run_dir(out_root, section, bb, "ef1_full", rec["id"]) / "metrics.json"
            if not met_path.exists():
                continue
            met = load_json(met_path)
            n_total[bb] += 1
            dconf = _delta_conf_drop(met)
            top1 = _is_top1(met)
            if top1:
                n_top1[bb] += 1
                hist[bb]["top1"].append(dconf)
                radial = met.get("radial_kept") or {}
                if radial:
                    for b in bands:
                        per_bb[bb][b].append(float(radial.get(b, 0.0)))
            else:
                hist[bb]["fail"].append(dconf)
            rows.append(
                {
                    "image_id": rec["id"],
                    "class_name": rec["class_name"],
                    "backbone": bb,
                    "top1": top1,
                    "delta_conf": dconf,
                    "conf_x": met.get("conf_x"),
                    "conf_e": met.get("conf_e"),
                    "radial_kept": met.get("radial_kept") or {},
                }
            )

    mean_kept, std_kept = {}, {}
    for bb in pair:
        if n_top1[bb] == 0 or not any(per_bb[bb][b] for b in bands):
            continue
        mean_kept[bb] = {}
        std_kept[bb] = {}
        for b in bands:
            vals = per_bb[bb][b]
            if not vals:
                mean_kept[bb][b] = 0.0
                std_kept[bb][b] = 0.0
                continue
            mu = float(sum(vals) / len(vals))
            mean_kept[bb][b] = mu
            if len(vals) < 2:
                std_kept[bb][b] = 0.0
            else:
                std_kept[bb][b] = float((sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5)

    n_note = ", ".join(f"{bb} {n_top1[bb]}/{n_total[bb]}" for bb in pair)
    if mean_kept:
        plot_freq_bias_bars(
            dest / "Fig0_freq_bias_mean.png",
            mean_kept,
            title=f"Mean retained energy (top-1 preserved)  n={n_note}",
            bands=bands,
            std_by_backbone=std_kept,
            xlabel="radial band  (low r<1/3, mid 1/3–2/3, high r>2/3)",
            ylabel="retained energy / band energy (%)",
        )
    if any(hist[bb]["top1"] or hist[bb]["fail"] for bb in pair):
        plot_delta_conf_histograms(
            dest / "Fig0_delta_conf_hist.png",
            hist,
            title=r"$\Delta$conf = conf(x) − conf(e)  (top-1 kept vs lost)",
        )

    def _ms(vals):
        vals = [float(v) for v in vals]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        mu = float(sum(vals) / len(vals))
        std = float((sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5) if len(vals) > 1 else 0.0
        return {"mean": mu, "std": std, "n": len(vals)}

    save_json(
        dest / "Fig0_freq_bias_mean.json",
        {
            "filter": "freq-bias bars: top-1 agreement only (argmax f(e) = argmax f(x))",
            "n_total": n_total,
            "n_top1": n_top1,
            "survival_rate": {
                bb: (float(n_top1[bb] / n_total[bb]) if n_total[bb] else 0.0) for bb in pair
            },
            "delta_conf_top1": {bb: _ms(hist[bb]["top1"]) for bb in pair},
            "delta_conf_fail": {bb: _ms(hist[bb]["fail"]) for bb in pair},
            "bands": {
                "low": "[0, 1/3)",
                "mid": "[1/3, 2/3)",
                "high": "[2/3, 1]",
                "note": "Nyquist-normalized radius; retained = ||m ⊙ 1_band ⊙ X||² / ||1_band ⊙ X||²",
            },
            "mean_kept_frac": mean_kept,
            "std_kept_frac": std_kept,
            "rows": rows,
        },
    )
    log(
        f"[figures] FFT freq-bias (top-1 only) + Δconf hist -> {dest}  "
        f"survive={n_note}"
    )


def _plot_fft_lpf_mean(out_root, section, backbone, subset) -> None:
    curves_l, curves_g, curves_e, radii = [], [], [], None
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
        curves_e.append([r.get("p_energy", 0.0) for r in rows])
    if not curves_l or radii is None:
        return
    import numpy as np

    arr_l = np.asarray(curves_l, dtype=float)
    arr_g = np.asarray(curves_g, dtype=float)
    arr_e = np.asarray(curves_e, dtype=float) if curves_e else None
    fig_dir = figure_dir(out_root, section, backbone)
    plot_lpf_vs_greedy(
        fig_dir / "Fig0_lpf_vs_greedy_mean.png",
        radii,
        arr_l.mean(axis=0).tolist(),
        arr_g.mean(axis=0).tolist(),
        title=f"Mean Δconf vs r — {backbone} (n={len(curves_l)})",
        delta_lpf_std=arr_l.std(axis=0).tolist(),
        delta_greedy_std=arr_g.std(axis=0).tolist(),
        p_energy=arr_e.mean(axis=0).tolist() if arr_e is not None and arr_e.size else None,
        p_energy_std=arr_e.std(axis=0).tolist() if arr_e is not None and arr_e.size else None,
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


def _apply_subset_args(args) -> None:
    """--subset: 2 images/class. --quick still wins (1 image, 40 steps)."""
    if getattr(args, "quick", False):
        args.section0_per_class = 1
        args.section0_slots = [0]
    elif getattr(args, "subset", False):
        args.section0_per_class = 2
        args.section0_slots = [0, 1]


def run_section0(args) -> None:
    eval_set, _images, device = resolve_run(args)
    out_root = args.out
    section = _section_name(args)
    steps = args.steps
    _apply_subset_args(args)
    subset = section0_image_records(
        eval_set,
        per_class=args.section0_per_class,
        slots=tuple(args.section0_slots),
    )
    snap_steps = tuple(args.snapshot_steps) if args.snapshot_steps else ef1_snapshot_steps(max_step=steps)
    rows = []
    use_fft = not getattr(args, "pixel", False)
    fft_cfg = fft_settings_from_args(args) if use_fft else {}
    if use_fft:
        lambdas = fft_cfg["lambdas"]
        ef1_version = "new"
        train_fn = maybe_train_and_eval_fft
        domain = "fft"
    else:
        lambdas = lambdas_from_args(args)
        ef1_version = "new" if args.ef1_new_version else "legacy"
        train_fn = maybe_train_and_eval
        domain = "pixel"
    eval_root = Path(eval_set["root"])

    log(
        f"Section 0  device={device}  subset={len(subset)} images "
        f"({args.section0_per_class}/class)  steps={steps}  section={section}  "
        f"domain={domain}  mask_model={fft_cfg.get('mask_model', '—')}  "
        f"checkpoint=EF1=HM(1−p_E, conf(e))  "
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
                n_lpf_radii=int(fft_cfg.get("n_lpf_radii", 50)),
                mask_model=fft_cfg.get("mask_model"),
            )
            extra = {
                "protocol": "ef1_fft" if use_fft else "ef1",
                "config": "Full_no_ce_fft" if use_fft else "Full_no_ce",
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
                best_pt = max(curve, key=lambda p: p.get("ef1", 0.0))
                row["p_energy_at_best"] = best_pt.get("p_energy")
                row["best_ef1_curve_step"] = best_pt.get("step")
            rows.append(row)

        _plot_ef1_for_backbone(out_root, section, backbone, subset, args.ef1_threshold)
        _plot_step_grids_for_backbone(
            out_root, section, backbone, subset, tuple(args.step_grid_every), eval_root
        )
        if use_fft:
            _plot_fft_grids_for_backbone(out_root, section, backbone, subset)
            _plot_fft_lpf_mean(out_root, section, backbone, subset)

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
    eval_set, _, _device = resolve_run(args)
    out_root = args.out
    section = _section_name(args)
    _apply_subset_args(args)
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
        if not getattr(args, "pixel", False):
            _plot_fft_grids_for_backbone(out_root, section, backbone, subset)
            _plot_fft_lpf_mean(out_root, section, backbone, subset)
    if not getattr(args, "pixel", False):
        _plot_fft_freq_bias(out_root, section, args.backbones, subset)


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description="Section 0: EF1 diagnostic training"))
    add_loss_lambda_args(p)
    p.set_defaults(steps=STEPS, no_ce=True)
    p.add_argument(
        "--pixel",
        action="store_true",
        help="Pixel EXP-CAM Section 0 (legacy). Default is polar FFT.",
    )
    p.add_argument(
        "--ef1-new-version",
        action="store_true",
        help="Pixel mode only: use ef1_new_version=HM(conf, 1−p).",
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
        "--subset",
        action="store_true",
        help="Small run: 2 images per class (eval-set slots 0 and 1). Default is 10 per class.",
    )
    p.add_argument(
        "--section0-per-class",
        type=int,
        default=SECTION0_PER_CLASS,
        help="Images per class for Section 0 training (default: 10). Overridden by --subset / --quick.",
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
        default=[SNAPSHOT_EVERY],
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
