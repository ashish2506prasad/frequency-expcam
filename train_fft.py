"""FFT-domain polar-mask training (WaveX generative) for Section 0."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()

from classifiers import ClassifierBundle, run_with_acts
from config import MASK_MODEL
from fft import FourierPolarMasker
from losses import (
    binarization_loss,
    ce_loss,
    hard_st,
    kl_loss,
    layer_act_loss,
)
from metrics import class_confidence, confidence_delta, ef1_new_version
from models import FourierMaskNet, SpectrumMask
from train import TrainConfig, TrainResult, log, seed_all


@dataclass
class FFTTrainResult(TrainResult):
    p_energy: float = 0.0
    p_bins: float = 0.0
    best_ef1_conf: float = 0.0
    conf_e: float = 0.0
    conf_x: float = 0.0
    radial_kept: Dict[str, float] = field(default_factory=dict)
    n_lpf_radii: int = 50
    noise_scale: float = 0.1
    radial_bins: int = 16
    angular_bins: int = 16
    mask_model: str = "unet"


def train_mask_fft(bundle: ClassifierBundle, x: torch.Tensor, cfg: TrainConfig) -> FFTTrainResult:
    device = x.device
    seed_all(cfg.seed)
    radial = int(getattr(cfg, "fft_radial_bins", 16) or 16)
    angular = int(getattr(cfg, "fft_angular_bins", 16) or 16)
    temp = float(getattr(cfg, "fft_mask_temperature", 0.25) or 0.25)
    noise_scale = float(getattr(cfg, "fft_noise_scale", 0.1) or 0.1)
    n_lpf = int(getattr(cfg, "n_lpf_radii", 50) or 50)

    fourier = FourierPolarMasker(radial, angular).to(device)
    model_name = str(getattr(cfg, "mask_model", None) or MASK_MODEL).strip().lower()
    use_spectrum = model_name in ("none", "pixel", "direct", "spectrum")
    if use_spectrum:
        mask_net = SpectrumMask(int(x.shape[-2]), int(x.shape[-1])).to(device)
        arch = "SpectrumMask"
    elif model_name in ("unet", "polar"):
        mask_net = FourierMaskNet(
            in_channels=3, radial_bins=radial, angular_bins=angular, temperature=temp
        ).to(device)
        arch = "FourierMaskNet_polar"
    else:
        raise ValueError(f"Unknown mask_model={model_name!r}; expected 'unet' or 'none'")
    mask_net.train()
    opt = torch.optim.Adam(mask_net.parameters(), lr=cfg.lr)

    acts_x, logits_x = run_with_acts(bundle, x, require_grad=False)
    y = logits_x.argmax(dim=1)
    conf_x = class_confidence(logits_x, y)
    shallow = set(list(bundle.tap_names)[:2])
    ef1_curve: List[Dict[str, float]] = []
    save_expl_set = set(cfg.save_expl_steps or ())
    track = cfg.record_ef1 or cfg.keep_best_ef1
    p_e = 0.0
    conf_e = 0.0
    ef1_conf = 0.0
    best_score = -1.0
    best_step = 0
    best_soft: Optional[torch.Tensor] = None
    best_hard: Optional[torch.Tensor] = None
    best_e: Optional[torch.Tensor] = None
    best_row: Dict[str, float] = {}

    t0 = time.perf_counter()
    if cfg.log_every:
        log(
            f"  training FFT {arch}  steps={cfg.steps}  "
            + (
                f"spectrum {x.shape[-2]}x{x.shape[-1] // 2 + 1}  "
                if use_spectrum
                else f"bins={radial}x{angular}  "
            )
            + f"T={temp:g}  checkpoint=EF1=HM(1−p_E, conf(e))  "
            + f"λ_energy={cfg.lambdas.get('energy', 0):g}  "
            + f"λ_rob={cfg.lambdas.get('rob', 0):g}  λ_kl={cfg.lambdas.get('kl', 0):g}  "
            + f"lr={cfg.lr}  seed={cfg.seed}"
        )

    for t in range(1, cfg.steps + 1):
        opt.zero_grad()
        m_soft = mask_net(x)
        m_st = hard_st(m_soft, cfg.mask_thresh)
        e = fourier.construct_explanation(x, m_st)
        acts_e, logits_e = run_with_acts(bundle, e, require_grad=True)

        loss_ce = ce_loss(logits_x, logits_e) if not cfg.no_ce else x.new_zeros(())
        loss_kl = kl_loss(logits_x, logits_e, temperature=cfg.temperature)
        loss_act = x.new_zeros(())
        for name, w in bundle.layer_weights.items():
            if name not in acts_x or name not in acts_e:
                continue
            loss_act = loss_act + w * layer_act_loss(name, acts_x[name], acts_e[name], shallow)

        coeff = fourier.fft2(x)
        half_soft = fourier.as_half_mask(m_soft, x.shape[-2], x.shape[-1])
        p_energy_t = fourier.kept_energy_fraction(coeff, half_soft).mean()
        loss_energy = p_energy_t
        loss_bin = binarization_loss(m_soft)
        loss_tv = x.new_zeros(())

        loss_rob = x.new_zeros(())
        if float(cfg.lambdas.get("rob", 0.0) or 0.0) > 0:
            e_rob_t = fourier.perturb_explanation(x, m_st, noise_scale=noise_scale)
            logits_rob_t = bundle.model(e_rob_t)
            loss_rob = kl_loss(logits_x, logits_rob_t, temperature=cfg.temperature)

        components = {
            "act": loss_act,
            "ce": loss_ce,
            "kl": loss_kl,
            "energy": loss_energy,
            "bin": loss_bin,
            "tv": loss_tv,
            "rob": loss_rob,
        }
        components_for_sum = dict(components)
        if cfg.no_ce:
            components_for_sum.pop("ce", None)
        loss_total = x.new_zeros(())
        for k, L in components_for_sum.items():
            loss_total = loss_total + float(cfg.lambdas.get(k, 0.0) or 0.0) * L

        loss_total.backward()
        opt.step()

        if track or t in save_expl_set or (cfg.log_every and (t == 1 or t % cfg.log_every == 0 or t == cfg.steps)):
            with torch.no_grad():
                m_hard = (m_soft > cfg.mask_thresh).float()
                e_hard = fourier.construct_explanation(x, m_hard)
                logits_eh = bundle.model(e_hard)
                half_h = fourier.as_half_mask(m_hard, x.shape[-2], x.shape[-1])
                p_e = float(fourier.kept_energy_fraction(coeff.detach(), half_h).mean().item())
                p_bins = float(m_hard.mean().item())
                conf_e = class_confidence(logits_eh, y)
                ef1_conf = ef1_new_version(conf_e, p_e)
                d_conf = confidence_delta(logits_x, logits_eh)
                row = {
                    "step": t,
                    "ef1": ef1_conf,
                    "ef1_conf": ef1_conf,
                    "p_energy": p_e,
                    "p_actual": p_bins,
                    "conf_e": conf_e,
                    "conf_x": conf_x,
                    "delta_conf": d_conf,
                }
                if track:
                    ef1_curve.append(row)
                if cfg.keep_best_ef1 and ef1_conf >= best_score:
                    best_score = ef1_conf
                    best_step = t
                    best_soft = m_soft.detach().clone()
                    best_hard = m_hard.detach().clone()
                    best_e = e_hard.detach().clone()
                    best_row = dict(row)
                if t in save_expl_set and cfg.save_expl_dir:
                    _save_fft_snapshot(
                        x, m_hard, e_hard, fourier, cfg.save_expl_dir, t, row
                    )

        if cfg.log_every and (t == 1 or t % cfg.log_every == 0 or t == cfg.steps):
            with torch.no_grad():
                top1 = int(logits_e.argmax(dim=1).item() == int(y.item()))
            log(
                f"  step {t:>4d}/{cfg.steps}  total={loss_total.item():.3f}  "
                f"act={loss_act.item():.3f}  kl={loss_kl.item():.3f}  "
                f"energy={loss_energy.item():.3f}  "
                f"rob={loss_rob.item():.3f}  p_E={100.0 * p_e:.1f}%  "
                f"EF1={ef1_conf:.3f}  conf(e)={conf_e:.3f}  top1={'Y' if top1 else 'N'}"
            )

    seconds = time.perf_counter() - t0
    if cfg.keep_best_ef1 and best_soft is not None:
        m_soft = best_soft
        m_hard = best_hard
        e = best_e
        p_e = float(best_row.get("p_energy", 0.0))
        p_bins = float(best_row.get("p_actual", 0.0))
        conf_e = float(best_row.get("conf_e", 0.0))
        if cfg.log_every:
            log(
                f"  best EF1={best_score:.4f} at step {best_step}  "
                f"p_E={100.0 * p_e:.1f}% of ||FFT(x)||²  conf(x)={conf_x:.3f}  "
                f"conf(e)={conf_e:.3f}"
            )
    else:
        mask_net.eval()
        with torch.no_grad():
            m_soft = mask_net(x)
            m_hard = (m_soft > cfg.mask_thresh).float()
            e = fourier.construct_explanation(x, m_hard)
            coeff = fourier.fft2(x)
            half_h = fourier.as_half_mask(m_hard, x.shape[-2], x.shape[-1])
            p_e = float(fourier.kept_energy_fraction(coeff, half_h).mean().item())
            p_bins = float(m_hard.mean().item())
            logits_eh = bundle.model(e)
            conf_e = class_confidence(logits_eh, y)
            best_score = ef1_new_version(conf_e, p_e)
            best_step = cfg.steps
            best_row = {
                "ef1": best_score,
                "ef1_conf": best_score,
                "p_energy": p_e,
                "conf_e": conf_e,
                "conf_x": conf_x,
            }

    with torch.no_grad():
        coeff = fourier.fft2(x)
        half_h = fourier.as_half_mask(m_hard, x.shape[-2], x.shape[-1])
        radial_kept = fourier.radial_band_kept_fractions(coeff, half_h)

    if cfg.log_every:
        log(f"  finished in {seconds:.1f}s  p_E={100.0 * p_e:.1f}%  p_bins={100.0 * p_bins:.1f}%")

    save_root = Path(cfg.save_expl_dir).parent if cfg.save_expl_dir else None
    if save_root is not None and m_hard is not None:
        save_fft_best_expl(
            save_root,
            x,
            m_soft,
            m_hard,
            e,
            fourier,
            best_row,
            best_step,
            radial_kept,
        )
        lpf_vs_greedy_comparison(
            save_root,
            bundle,
            x,
            m_soft,
            fourier,
            n_radii=n_lpf,
            noise_scale=noise_scale,
        )

    return FFTTrainResult(
        mask_prob=m_soft.detach(),
        mask_hard=m_hard.detach(),
        e=e.detach(),
        seconds=seconds,
        steps=cfg.steps,
        lambdas=dict(cfg.lambdas),
        seed=cfg.seed,
        loss_curve=[],
        weight_curve=[],
        mu_final=0.0,
        p_actual=p_bins,
        ef1_curve=ef1_curve,
        best_ef1=float(best_score),
        best_ef1_step=int(best_step),
        p_energy=float(p_e),
        p_bins=float(p_bins),
        best_ef1_conf=float(best_row.get("ef1_conf", 0.0)),
        conf_e=float(conf_e),
        conf_x=float(conf_x),
        radial_kept=radial_kept,
        n_lpf_radii=n_lpf,
        noise_scale=noise_scale,
        radial_bins=radial,
        angular_bins=angular,
        mask_model="none" if use_spectrum else "unet",
    )


def _save_fft_snapshot(x, m_hard, e, fourier, out_dir, step, row) -> None:
    from pathlib import Path

    from PIL import Image

    from io_utils import ensure_dir
    from viz import panel_mask, panel_original, triple_panel

    dest = ensure_dir(Path(out_dir) / f"step_{step:04d}")
    x_np = panel_original(x)
    e_np = panel_original(e)
    m_vis = m_hard[0, 0].detach().cpu().numpy()
    m_label = "m spectrum (hard)" if fourier.is_spectrum_mask(m_hard, x.shape[-2], x.shape[-1]) else "m polar (hard)"
    triple_panel(
        dest / "x_m_e.png",
        x_np,
        m_vis,
        e_np,
        labels=("x (original)", m_label, "e (IFFT)"),
        suptitle=(
            f"step={step}  conf(x)={row['conf_x']:.3f}  conf(e)={row['conf_e']:.3f}  "
            f"retained p_E={100 * row['p_energy']:.1f}% of ||FFT(x)||²  Δconf={row['delta_conf']:.3f}"
        ),
    )
    Image.fromarray((np.clip(e_np, 0, 1) * 255).astype(np.uint8)).save(dest / "e.png")


def save_fft_best_expl(
    out_dir,
    x,
    m_soft,
    m_hard,
    e,
    fourier: FourierPolarMasker,
    best_row: dict,
    best_step: int,
    radial_kept: dict,
) -> None:
    from pathlib import Path

    from PIL import Image

    from io_utils import ensure_dir, save_json
    from viz import panel_original, plot_fft_wavex_grid

    dest = ensure_dir(Path(out_dir) / "best_ef1")
    with torch.no_grad():
        coeff = fourier.fft2(x)
        half_s = fourier.as_half_mask(m_soft, x.shape[-2], x.shape[-1])
        half_h = fourier.as_half_mask(m_hard, x.shape[-2], x.shape[-1])
        e_soft = fourier.ifft2_real(coeff * half_s, signal_hw=x.shape[-2:])
        mag_x = fourier.fft_log_magnitude(coeff, x.shape[-1])
        mag_s = fourier.fft_log_magnitude(coeff * half_s, x.shape[-1])
        mag_h = fourier.fft_log_magnitude(coeff * half_h, x.shape[-1])
        full_s = torch.fft.fftshift(
            fourier.one_sided_to_full(half_s, width=x.shape[-1]), dim=(-2, -1)
        )
        full_h = torch.fft.fftshift(
            fourier.one_sided_to_full(half_h, width=x.shape[-1]), dim=(-2, -1)
        )
    p_e = float(best_row.get("p_energy", 0.0) or 0.0)
    conf_x = float(best_row.get("conf_x", 0.0) or 0.0)
    conf_e = float(best_row.get("conf_e", 0.0) or 0.0)
    plot_fft_wavex_grid(
        dest / "fft_expl_grid.png",
        panel_original(x),
        panel_original(e_soft),
        panel_original(e),
        m_soft[0, 0].detach().cpu().numpy(),
        m_hard[0, 0].detach().cpu().numpy(),
        mag_x[0, 0].detach().cpu().numpy(),
        mag_s[0, 0].detach().cpu().numpy(),
        mag_h[0, 0].detach().cpu().numpy(),
        full_s[0, 0].detach().cpu().numpy(),
        full_h[0, 0].detach().cpu().numpy(),
        title=(
            f"Best EF1=HM(1−p_E, conf(e)) @ step {best_step}  "
            f"conf(x)={conf_x:.3f}  conf(e)={conf_e:.3f}  "
            f"retained p_E={100.0 * p_e:.1f}% of ||FFT(x)||²"
        ),
        conf_x=conf_x,
        conf_e=conf_e,
        p_energy=p_e,
        mask_kind=(
            "spectrum"
            if fourier.is_spectrum_mask(m_soft, x.shape[-2], x.shape[-1])
            else "polar"
        ),
    )
    Image.fromarray((np.clip(panel_original(e), 0, 1) * 255).astype(np.uint8)).save(dest / "e.png")
    save_json(
        dest / "best_ef1_meta.json",
        {
            "checkpoint": "ef1_hm_conf_energy",
            "best_ef1_step": best_step,
            **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in best_row.items()},
            "radial_kept": radial_kept,
        },
    )


def lpf_vs_greedy_comparison(
    out_dir,
    bundle: ClassifierBundle,
    x: torch.Tensor,
    m_soft: torch.Tensor,
    fourier: FourierPolarMasker,
    n_radii: int = 50,
    noise_scale: float = 0.1,
) -> dict:
    """Δconf(x, filtered) vs cutoff radius for circular LPF vs score-greedy polar bins."""
    from pathlib import Path

    from io_utils import ensure_dir, save_json
    from viz import plot_lpf_vs_greedy

    dest = ensure_dir(Path(out_dir) / "best_ef1")
    radii = np.linspace(0.0, 1.0, int(n_radii))
    with torch.no_grad():
        logits_x = bundle.model(x)
        y = logits_x.argmax(dim=1)
        conf_x = class_confidence(logits_x, y)
        coeff = fourier.fft2(x)
        total_e = float((coeff.abs() ** 2).sum().item())
        spectrum = fourier.is_spectrum_mask(m_soft, x.shape[-2], x.shape[-1])
        bin_e = None if spectrum else fourier.polar_bin_energies(coeff, x.shape[-2], x.shape[-1])[0]
        scores = m_soft.detach()
        rows = []
        for r in radii:
            e_lpf, p_e, _ = fourier.lowpass_image(x, float(r))
            target = float(p_e.item()) * total_e
            if spectrum:
                m_g = fourier.greedy_spectrum_mask_for_energy(scores, coeff, target)
            else:
                m_g = fourier.greedy_polar_mask_for_energy(scores, bin_e, target)
            e_g = fourier.construct_explanation(x, m_g)
            conf_lpf = class_confidence(bundle.model(e_lpf), y)
            conf_g = class_confidence(bundle.model(e_g), y)
            rows.append(
                {
                    "radius": float(r),
                    "p_energy": float(p_e.item()),
                    "conf_lpf": conf_lpf,
                    "conf_greedy": conf_g,
                    "delta_conf_lpf": float(conf_x - conf_lpf),
                    "delta_conf_greedy": float(conf_x - conf_g),
                    "conf_x": conf_x,
                }
            )
    save_json(dest / "lpf_vs_greedy.json", {"n_radii": n_radii, "noise_scale": noise_scale, "rows": rows})
    plot_lpf_vs_greedy(
        dest / "Fig0_lpf_vs_greedy.png",
        [r["radius"] for r in rows],
        [r["delta_conf_lpf"] for r in rows],
        [r["delta_conf_greedy"] for r in rows],
        title="Δconf vs circular cutoff radius (same energy)",
        p_energy=[r["p_energy"] for r in rows],
        greedy_label="greedy spectrum bins" if spectrum else "greedy polar bins",
    )
    return {"rows": rows, "conf_x": conf_x}


def save_fft_mask_tensors(out_dir, result: FFTTrainResult) -> None:
    from pathlib import Path

    from io_utils import ensure_dir

    out_dir = ensure_dir(Path(out_dir))
    torch.save(
        {
            "mask_hard": result.mask_hard.cpu(),
            "mask_prob": result.mask_prob.cpu(),
            "p_energy": result.p_energy,
            "p_bins": result.p_bins,
            "radial_kept": result.radial_kept,
            "domain": "fft",
            "radial_bins": result.radial_bins,
            "angular_bins": result.angular_bins,
            "mask_model": result.mask_model,
            "mask_kind": "spectrum" if result.mask_model == "none" else "polar",
        },
        out_dir / "masks.pt",
    )
