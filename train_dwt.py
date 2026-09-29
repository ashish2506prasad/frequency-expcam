"""DWT-domain mask training: RGB U-Net → subband masks → IDWT → pixel explanation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._path import ensure_pkg_path
except ImportError:
    from _path import ensure_pkg_path

ensure_pkg_path()


from classifiers import ClassifierBundle, run_with_acts
from dwt import (
    SUBBANDS,
    band_energy_kept_fractions,
    band_energy_shares,
    dwt_level1,
    downsample_masks,
    energy_weighted_p,
    force_disabled_masks,
    idwt_level1,
    mean_subband_p,
    reconstruct_explanation,
    reconstruct_single_band_explanation,
    subband_p_actual,
    unet_masks_to_dict,
)
from losses import (
    abductive_loss,
    area_loss,
    binarization_loss,
    ce_loss,
    hard_st,
    kl_loss,
    lagrangian_area_penalty,
    layer_act_loss,
    project_topk_mask,
    sample_gaussian_backgrounds,
    tv8_loss,
)
from metrics import confidence_delta, explanation_f1
from models import UNetDWT
from train import TrainConfig, TrainResult, log, seed_all


class DWTMaskNet(nn.Module):
    """RGB-image U-Net → full-res masks → pool to one-level DWT grid.

    Disabled subbands (``--no-ll``, …) are not predicted: those channels are
    forced to zero so the corresponding coefficients are dropped in IDWT.
    """

    def __init__(self, disabled_subbands: Sequence[str] = ()):
        super().__init__()
        disabled = {s.upper() for s in disabled_subbands}
        self.active = [b for b in SUBBANDS if b not in disabled]
        self.disabled = tuple(b for b in SUBBANDS if b in disabled)
        n_out = max(len(self.active), 1)
        self.unet = UNetDWT(in_channels=3, out_channels=n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.unet(x)
        continuous = torch.sigmoid(logits)
        pooled = downsample_masks(continuous)
        bsz, _, sh, sw = pooled.shape
        zeros = pooled.new_zeros(bsz, 1, sh, sw)
        channels = []
        idx = 0
        for name in SUBBANDS:
            if name in self.active:
                channels.append(pooled[:, idx : idx + 1])
                idx += 1
            else:
                channels.append(zeros)
        return torch.cat(channels, dim=1)


def _mask_tensors_from_prob(
    mask_prob: torch.Tensor,
    disabled: Sequence[str],
    thresh: float,
) -> Tuple[torch.Tensor, Dict, Dict]:
    mask_prob = force_disabled_masks(mask_prob, disabled)
    prob_dict = unet_masks_to_dict(mask_prob)
    hard_dict = {b: hard_st(prob_dict[b], thresh) for b in SUBBANDS}
    for b in disabled:
        hard_dict[b] = torch.zeros_like(hard_dict[b])
        prob_dict[b] = torch.zeros_like(prob_dict[b])
    hard_stacked = torch.cat([hard_dict[b] for b in SUBBANDS], dim=1)
    return mask_prob, prob_dict, hard_dict, hard_stacked


def _per_band_area_tv_loss(
    mask_prob: torch.Tensor,
    band_lambdas: Dict[str, float],
    disabled: Sequence[str],
    energy_share: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Area: Σ λ_b · (E_b / E_tot) · mean(m_b). TV uses λ_tv_b only (unscaled)."""
    disabled = {s.upper() for s in disabled}
    share = energy_share or {b: 1.0 for b in SUBBANDS}
    ref = mask_prob[:, 0:1]
    loss_area = ref.new_zeros(())
    loss_tv = ref.new_zeros(())
    for i, b in enumerate(SUBBANDS):
        if b in disabled:
            continue
        ch = mask_prob[:, i : i + 1]
        la = float(band_lambdas.get(f"area_{b}", 0.0))
        lt = float(band_lambdas.get(f"tv_{b}", 0.0))
        eb = float(share.get(b, 0.0))
        if la != 0.0:
            loss_area = loss_area + la * eb * area_loss(ch)
        if lt != 0.0:
            loss_tv = loss_tv + lt * tv8_loss(ch)
    return loss_area, loss_tv


def _per_band_rob_loss(
    model,
    x: torch.Tensor,
    bands: Dict[str, torch.Tensor],
    mask_prob: torch.Tensor,
    backgrounds: Sequence[torch.Tensor],
    band_lambdas: Dict[str, float],
    disabled: Sequence[str],
    wave: str,
    temperature: float,
) -> torch.Tensor:
    """Abductive KL per subband in coeff space: mix only band b with DWT(background).

    e_r = IDWT( …, X_b ← m_b ⊙ X_b + (1−m_b) ⊙ R_b, other bands = X … )
    L = Σ_b λ_rob_b · KL(f(x) || f(e_r^{(b)}))
    """
    disabled = {s.upper() for s in disabled}
    weights = {b: float(band_lambdas.get(f"rob_{b}", 0.0)) for b in SUBBANDS}
    if not any(weights[b] > 0 and b not in disabled for b in SUBBANDS):
        return x.new_zeros(())
    with torch.no_grad():
        px = F.softmax(model(x) / temperature, dim=1)
    bg_bands = [dwt_level1(r.to(x.device), wave=wave) for r in backgrounds]
    n_bg = max(1, len(bg_bands))
    total = x.new_zeros(())
    for i, b in enumerate(SUBBANDS):
        w = weights[b]
        if w <= 0 or b in disabled:
            continue
        m = hard_st(mask_prob[:, i : i + 1])
        band_loss = x.new_zeros(())
        for rb in bg_bands:
            mixed = {k: bands[k] for k in SUBBANDS}
            mixed[b] = m * bands[b] + (1.0 - m) * rb[b]
            e_r = idwt_level1(mixed, wave=wave)
            log_pe = F.log_softmax(model(e_r) / temperature, dim=1)
            band_loss = band_loss + F.kl_div(log_pe, px, reduction="batchmean") * (temperature ** 2)
        total = total + w * (band_loss / n_bg)
    return total


def _subband_bin_loss(mask_prob: torch.Tensor, disabled: Sequence[str]) -> torch.Tensor:
    active = [i for i, b in enumerate(SUBBANDS) if b not in {s.upper() for s in disabled}]
    if not active:
        return mask_prob.new_zeros(())
    return sum(binarization_loss(mask_prob[:, i : i + 1]) for i in active) / float(len(active))


def _project_subband_masks(
    mask_prob: torch.Tensor, p_target: float, disabled: Sequence[str]
) -> torch.Tensor:
    disabled = {s.upper() for s in disabled}
    out = []
    for i, b in enumerate(SUBBANDS):
        ch = mask_prob[:, i : i + 1]
        if b in disabled:
            out.append(torch.zeros_like(ch))
        else:
            out.append(project_topk_mask(ch, p_target))
    return torch.cat(out, dim=1)


@dataclass
class DWTTrainResult(TrainResult):
    p_subbands: Dict[str, float] = field(default_factory=dict)
    mask_prob_subbands: Optional[torch.Tensor] = None
    disabled_subbands: Tuple[str, ...] = ()
    p_energy: float = 0.0
    p_energy_bands: Dict[str, float] = field(default_factory=dict)
    best_ef1_area: float = 0.0


def train_mask_dwt(
    bundle: ClassifierBundle,
    x: torch.Tensor,
    cfg: TrainConfig,
) -> DWTTrainResult:
    device = x.device
    seed_all(cfg.seed)
    wave = getattr(cfg, "wavelet", "db1") or "db1"
    disabled = tuple(s.upper() for s in (cfg.disabled_subbands or ()))
    mask_net = DWTMaskNet(disabled_subbands=disabled).to(device)
    mask_net.train()
    opt = torch.optim.Adam(mask_net.parameters(), lr=cfg.lr)

    acts_x, logits_x = run_with_acts(bundle, x, require_grad=False)
    shallow = set(list(bundle.tap_names)[:2])
    mu = float(cfg.mu_init)
    ef1_curve: List[Dict[str, float]] = []
    save_expl_set = set(cfg.save_expl_steps or ())
    track_ef1 = cfg.record_ef1 or cfg.keep_best_ef1
    best_ef1 = -1.0
    best_ef1_step = 0
    best_mask_prob: Optional[torch.Tensor] = None
    best_mask_hard: Optional[torch.Tensor] = None
    best_e: Optional[torch.Tensor] = None
    best_p_sub: Dict[str, float] = {}
    best_p_energy: float = 0.0
    best_ef1_area_at_best: float = 0.0
    best_bands: Optional[Dict[str, torch.Tensor]] = None
    best_hard_dict: Optional[Dict[str, torch.Tensor]] = None
    best_p_energy_bands: Dict[str, float] = {}
    band_lambdas = dict(cfg.band_lambdas or {})
    with torch.no_grad():
        bands0 = dwt_level1(x, wave=wave)
        energy_share = band_energy_shares(bands0)

    t0 = time.perf_counter()
    if cfg.log_every:
        log(
            f"  training DWT U-Net (image→masks)  steps={cfg.steps}  "
            f"wave={wave}  disabled={disabled or 'none'}  "
            f"λ_rob={cfg.lambdas.get('rob', 0):g}  "
            f"E_share="
            + ",".join(f"{b}={energy_share[b]:.3f}" for b in SUBBANDS)
            + "  band_area="
            + ",".join(f"{b}={band_lambdas.get(f'area_{b}', 0):g}" for b in SUBBANDS)
            + f"  lr={cfg.lr}  seed={cfg.seed}"
        )

    for t in range(1, cfg.steps + 1):
        opt.zero_grad()
        bands = dwt_level1(x, wave=wave)
        mask_prob = mask_net(x)
        if cfg.constraint == "project" and cfg.p_target is not None:
            mask_prob = _project_subband_masks(mask_prob, cfg.p_target, disabled)
        mask_prob, prob_dict, hard_dict, hard_stacked = _mask_tensors_from_prob(
            mask_prob, disabled, cfg.mask_thresh
        )

        e = reconstruct_explanation(bands, prob_dict, hard_dict, wave=wave)
        acts_e, logits_e = run_with_acts(bundle, e, require_grad=True)

        loss_ce = ce_loss(logits_x, logits_e) if not cfg.no_ce else x.new_zeros(())
        loss_kl = kl_loss(logits_x, logits_e, temperature=cfg.temperature)
        loss_act = x.new_zeros(())
        for name, w in bundle.layer_weights.items():
            if name not in acts_x or name not in acts_e:
                continue
            loss_act = loss_act + w * layer_act_loss(name, acts_x[name], acts_e[name], shallow)

        # Per-band area: λ_b * (E_b / E_tot) * mean(m_b); added unscaled.
        loss_area_band, loss_tv_band = _per_band_area_tv_loss(
            mask_prob, band_lambdas, disabled, energy_share=energy_share
        )
        loss_area_uniform = x.new_zeros(())
        loss_tv_uniform = x.new_zeros(())
        active_idx = [i for i, b in enumerate(SUBBANDS) if b not in set(disabled)]
        if cfg.lambdas.get("area", 0.0) > 0 and active_idx:
            loss_area_uniform = sum(area_loss(mask_prob[:, i : i + 1]) for i in active_idx) / len(active_idx)
        if cfg.lambdas.get("tv", 0.0) > 0 and active_idx:
            loss_tv_uniform = sum(tv8_loss(mask_prob[:, i : i + 1]) for i in active_idx) / len(active_idx)
        loss_bin = _subband_bin_loss(mask_prob, disabled)
        want_pix_rob = cfg.lambdas.get("rob", 0.0) > 0
        want_band_rob = any(
            float(band_lambdas.get(f"rob_{b}", 0.0)) > 0 and b not in set(disabled) for b in SUBBANDS
        )
        bgs = None
        if want_pix_rob or want_band_rob:
            bgs = list(cfg.extra_backgrounds) if cfg.extra_backgrounds else sample_gaussian_backgrounds(x, cfg.rob_k)
        if want_pix_rob:
            m_pix = F.interpolate(
                mask_prob.mean(dim=1, keepdim=True),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss_rob = abductive_loss(bundle.model, x, m_pix, bgs, temperature=cfg.temperature)
        else:
            loss_rob = x.new_zeros(())
        loss_rob_band = (
            _per_band_rob_loss(
                bundle.model,
                x,
                bands,
                mask_prob,
                bgs,
                band_lambdas,
                disabled,
                wave,
                cfg.temperature,
            )
            if want_band_rob
            else x.new_zeros(())
        )

        components = {
            "act": loss_act,
            "ce": loss_ce,
            "kl": loss_kl,
            "area": loss_area_uniform,
            "bin": loss_bin,
            "tv": loss_tv_uniform,
            "rob": loss_rob,
        }
        if cfg.constraint in ("project", "lagrangian"):
            components_for_sum = {k: v for k, v in components.items() if k != "area"}
        else:
            components_for_sum = components
        if cfg.no_ce:
            components_for_sum = {k: v for k, v in components_for_sum.items() if k != "ce"}

        loss_total = x.new_zeros(())
        for k, L in components_for_sum.items():
            loss_total = loss_total + cfg.lambdas.get(k, 0.0) * L
        # WaveX: l_area = Σ w_b mean(m_b) is *not* multiplied by a global λ_area.
        if cfg.constraint not in ("project", "lagrangian"):
            loss_total = loss_total + loss_area_band
        loss_total = loss_total + loss_tv_band
        loss_total = loss_total + loss_rob_band

        if cfg.constraint == "lagrangian" and cfg.p_target is not None:
            p_now = mask_prob.mean()
            loss_total = loss_total + lagrangian_area_penalty(p_now, cfg.p_target, mu)

        loss_total.backward()
        opt.step()

        if cfg.constraint == "lagrangian" and cfg.p_target is not None:
            with torch.no_grad():
                viol = float(torch.relu(mask_prob.mean() - cfg.p_target).item())
            mu = max(0.0, mu + cfg.mu_eta * viol)

        if track_ef1 or save_expl_set:
            with torch.no_grad():
                p_sub = subband_p_actual(hard_dict)
                p_act = mean_subband_p(p_sub)
                p_e = energy_weighted_p(bands, hard_dict)
                d_conf = confidence_delta(logits_x, logits_e)
                ef1 = explanation_f1(d_conf, p_act)
                ef1_e = explanation_f1(d_conf, p_e)
                if track_ef1:
                    row = {
                        "step": t,
                        "ef1": ef1,
                        "ef1_energy": ef1_e,
                        "delta_conf": d_conf,
                        "p_actual": p_act,
                        "p_energy": p_e,
                    }
                    row.update({f"p_{b.lower()}": p_sub[b] for b in SUBBANDS})
                    ef1_curve.append(row)
                if cfg.keep_best_ef1 and ef1_e >= best_ef1:
                    best_ef1 = ef1_e
                    best_ef1_step = t
                    best_ef1_area_at_best = ef1
                    best_p_energy = p_e
                    best_mask_prob = mask_prob.detach().clone()
                    best_mask_hard = hard_stacked.detach().clone()
                    best_e = e.detach().clone()
                    best_p_sub = dict(p_sub)
                    best_p_energy_bands = band_energy_kept_fractions(bands, hard_dict)
                    best_bands = {b: bands[b].detach().clone() for b in SUBBANDS}
                    best_hard_dict = {b: hard_dict[b].detach().clone() for b in SUBBANDS}
                if t in save_expl_set and cfg.save_expl_dir:
                    _save_dwt_snapshot(
                        x, bands, hard_dict, e, cfg.save_expl_dir, t, ef1, ef1_e, p_sub, p_e, d_conf
                    )

        if cfg.log_every and (t == 1 or t % cfg.log_every == 0 or t == cfg.steps):
            with torch.no_grad():
                p_sub = subband_p_actual(hard_dict)
                top1 = int(logits_e.argmax(dim=1).item() == logits_x.argmax(dim=1).item())
            log(
                f"  step {t:>4d}/{cfg.steps}  total={loss_total.item():.3f}  "
                f"act={loss_act.item():.3f}  "
                + ("" if cfg.no_ce else f"ce={loss_ce.item():.3f}  ")
                + f"kl={loss_kl.item():.3f}  bin={loss_bin.item():.3f}  "
                f"area={float(loss_area_band + cfg.lambdas.get('area', 0.0) * loss_area_uniform):.3f}  "
                + (f"rob={loss_rob.item():.3f}  " if cfg.lambdas.get("rob", 0.0) > 0 else "")
                + (f"rob_b={loss_rob_band.item():.3f}  " if want_band_rob else "")
                + f"p_ll={100*p_sub['LL']:.1f}% lh={100*p_sub['LH']:.1f}% "
                f"hl={100*p_sub['HL']:.1f}% hh={100*p_sub['HH']:.1f}%  "
                f"p_E={100*energy_weighted_p(bands, hard_dict):.1f}%  "
                f"top1={'Y' if top1 else 'N'}"
            )

    seconds = time.perf_counter() - t0
    p_energy = best_p_energy
    if cfg.keep_best_ef1 and best_mask_prob is not None:
        mask_prob = best_mask_prob
        mask_hard = best_mask_hard
        e = best_e if best_e is not None else x
        p_sub = best_p_sub
        p_energy = best_p_energy
        p_energy_bands = best_p_energy_bands
    else:
        mask_net.eval()
        with torch.no_grad():
            bands = dwt_level1(x, wave=wave)
            mask_prob = mask_net(x)
            if cfg.constraint == "project" and cfg.p_target is not None:
                mask_prob = _project_subband_masks(mask_prob, cfg.p_target, disabled)
            mask_prob, prob_dict, hard_dict, mask_hard = _mask_tensors_from_prob(
                mask_prob, disabled, cfg.mask_thresh
            )
            e = reconstruct_explanation(bands, prob_dict, hard_dict, wave=wave)
            p_sub = subband_p_actual(hard_dict)
            p_energy = energy_weighted_p(bands, hard_dict)
            best_p_energy_bands = band_energy_kept_fractions(bands, hard_dict)
        best_ef1 = max((row.get("ef1_energy", row["ef1"]) for row in ef1_curve), default=0.0)
        best_ef1_step = next(
            (row["step"] for row in ef1_curve if row.get("ef1_energy", row["ef1"]) == best_ef1),
            0,
        )
        best_ef1_area_at_best = next(
            (row["ef1"] for row in ef1_curve if row.get("ef1_energy", row["ef1"]) == best_ef1),
            0.0,
        )

    p_energy_bands = dict(best_p_energy_bands)
    p_actual = mean_subband_p(p_sub)
    if cfg.log_every:
        log(
            f"  finished in {seconds:.1f}s  p_mean={100.0 * p_actual:.1f}%  "
            f"p_E={100.0 * p_energy:.1f}%  "
            f"p_ll={100*p_sub['LL']:.1f}% … hh={100*p_sub['HH']:.1f}%  "
            f"best_EF1_E={best_ef1:.3f} @ step {best_ef1_step}  "
            f"EF1_area@best={best_ef1_area_at_best:.3f}"
        )

    if best_bands is not None and best_hard_dict is not None:
        save_root = Path(cfg.save_expl_dir).parent if cfg.save_expl_dir else None
        if save_root is not None:
            save_dwt_best_band_expl(
                save_root,
                x,
                best_bands,
                best_hard_dict,
                best_e if best_e is not None else e,
                best_p_sub,
                best_ef1,
                best_ef1_step,
                wave=wave,
                p_energy=best_p_energy,
                best_ef1_area=best_ef1_area_at_best,
            )

    return DWTTrainResult(
        mask_prob=mask_prob.detach(),
        mask_hard=mask_hard.detach(),
        e=e.detach(),
        seconds=seconds,
        steps=cfg.steps,
        lambdas=dict(cfg.lambdas),
        seed=cfg.seed,
        loss_curve=[],
        weight_curve=[],
        mu_final=mu,
        p_actual=p_actual,
        ef1_curve=ef1_curve,
        best_ef1=float(best_ef1),
        best_ef1_step=int(best_ef1_step),
        p_subbands=p_sub,
        disabled_subbands=disabled,
        p_energy=float(p_energy),
        p_energy_bands=p_energy_bands,
        best_ef1_area=float(best_ef1_area_at_best),
    )


def _save_dwt_snapshot(
    x,
    bands,
    hard_dict,
    e,
    out_dir: str,
    step: int,
    ef1: float,
    ef1_e: float,
    p_sub: dict,
    p_e: float,
    d_conf: float,
) -> None:
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from io_utils import ensure_dir
    from viz import panel_mask, panel_original, triple_panel

    dest = ensure_dir(Path(out_dir) / f"step_{step:04d}")
    x_np = panel_original(x)
    e_np = panel_original(e)
    m_stack = torch.cat([hard_dict[b] for b in SUBBANDS], dim=1)
    m_up = F.interpolate(m_stack.mean(dim=1, keepdim=True), size=x.shape[-2:], mode="nearest")
    triple_panel(
        dest / "x_m_e.png",
        x_np,
        panel_mask(m_up),
        e_np,
        suptitle=(
            f"step={step}  EF1={ef1:.3f}  EF1_E={ef1_e:.3f}  Δconf={d_conf:.3f}\n"
            f"p={100*sum(p_sub.values())/4:.1f}%  p_E={100*p_e:.1f}%  "
            f"ll={100*p_sub['LL']:.1f}% lh={100*p_sub['LH']:.1f}% "
            f"hl={100*p_sub['HL']:.1f}% hh={100*p_sub['HH']:.1f}%"
        ),
    )
    Image.fromarray((np.clip(e_np, 0, 1) * 255).astype(np.uint8)).save(dest / "e.png")


def save_dwt_best_band_expl(
    out_dir,
    x: torch.Tensor,
    bands: Dict[str, torch.Tensor],
    hard_dict: Dict[str, torch.Tensor],
    e_full: torch.Tensor,
    p_sub: Dict[str, float],
    best_ef1: float,
    best_step: int,
    wave: str = "db1",
    p_energy: float = 0.0,
    best_ef1_area: float = 0.0,
) -> None:
    """Save per-subband IDWT explanations and a summary grid at best energy-weighted EF1."""
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from io_utils import ensure_dir, save_json
    from viz import panel_mask, panel_original, plot_dwt_best_band_grid

    dest = ensure_dir(Path(out_dir) / "best_ef1")
    x_np = panel_original(x)
    e_full_np = panel_original(e_full)
    band_e_np = {}
    band_m_np = {}
    for b in SUBBANDS:
        e_b = reconstruct_single_band_explanation(bands, hard_dict, b, wave=wave)
        band_e_np[b] = panel_original(e_b)
        m_up = F.interpolate(hard_dict[b], size=x.shape[-2:], mode="nearest")
        band_m_np[b] = panel_mask(m_up)
        Image.fromarray((np.clip(band_e_np[b], 0, 1) * 255).astype(np.uint8)).save(dest / f"e_{b.lower()}.png")
        Image.fromarray((np.clip(band_m_np[b], 0, 1) * 255).astype(np.uint8)).save(dest / f"m_{b.lower()}.png")

    plot_dwt_best_band_grid(
        dest / "band_expl_grid.png",
        x_np,
        e_full_np,
        band_e_np,
        band_m_np,
        p_sub,
        best_ef1,
        best_step,
        p_energy=p_energy,
    )
    save_json(
        dest / "best_ef1_meta.json",
        {
            "best_ef1": best_ef1,
            "best_ef1_energy": best_ef1,
            "best_ef1_area": best_ef1_area,
            "best_ef1_step": best_step,
            "p_subbands": p_sub,
            "p_energy": p_energy,
            "p_energy_bands": band_energy_kept_fractions(bands, hard_dict) if bands and hard_dict else {},
        },
    )


def save_dwt_best_band_expl_from_run(
    out_dir,
    x: torch.Tensor,
    mask_hard: torch.Tensor,
    e_full: torch.Tensor,
    meta: dict,
    wave: str = "db1",
) -> None:
    """Regenerate best-EF1 band grid from saved masks (figures-only)."""
    bands = dwt_level1(x, wave=wave)
    hard_dict = unet_masks_to_dict(mask_hard)
    save_dwt_best_band_expl(
        out_dir,
        x,
        bands,
        hard_dict,
        e_full,
        meta.get("p_subbands") or {},
        float(meta.get("best_ef1_energy", meta.get("best_ef1", 0))),
        int(meta.get("best_ef1_step", 0)),
        wave=wave,
        p_energy=float(meta.get("p_energy", 0.0)),
        best_ef1_area=float(meta.get("best_ef1_area", 0.0)),
    )


def save_dwt_mask_tensors(out_dir, result: DWTTrainResult) -> None:
    from pathlib import Path

    from io_utils import ensure_dir

    out_dir = ensure_dir(Path(out_dir))
    torch.save(
        {
            "mask_hard": result.mask_hard.cpu(),
            "mask_prob": result.mask_prob.cpu(),
            "p_subbands": result.p_subbands,
            "p_energy": result.p_energy,
            "p_energy_bands": result.p_energy_bands,
            "disabled_subbands": list(result.disabled_subbands),
            "domain": "dwt",
        },
        out_dir / "masks.pt",
    )
