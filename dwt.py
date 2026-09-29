"""Single-level DWT / IDWT for EXP-CAM (pytorch_wavelets, db1/Haar)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SUBBANDS = ("LL", "LH", "HL", "HH")

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pytorch_wavelets is required for DWT training. "
        "Install with: pip install pytorch-wavelets"
    ) from e

# Cached (wave, mode, device) → (forward, inverse) modules.
_DWT_CACHE: Dict[Tuple[str, str, str], Tuple[nn.Module, nn.Module]] = {}


def _dwt_pair(device: torch.device, wave: str = "db1", mode: str = "periodization"):
    key = (wave, mode, str(device))
    if key not in _DWT_CACHE:
        fwd = DWTForward(J=1, wave=wave, mode=mode).to(device).eval()
        inv = DWTInverse(wave=wave, mode=mode).to(device).eval()
        for m in (fwd, inv):
            for p in m.parameters():
                p.requires_grad_(False)
        _DWT_CACHE[key] = (fwd, inv)
    return _DWT_CACHE[key]


def dwt_level1(x: torch.Tensor, wave: str = "db1") -> Dict[str, torch.Tensor]:
    """One-level 2D DWT. x: (B,C,H,W) even H,W → bands (B,C,H/2,W/2)."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(f"DWT expects even H,W; got {tuple(x.shape)}")
    fwd, _ = _dwt_pair(x.device, wave=wave)
    yl, yh = fwd(x)
    # yh[0]: (B, C, 3, H/2, W/2) with dim 2 = [LH, HL, HH]
    return {
        "LL": yl,
        "LH": yh[0][:, :, 0],
        "HL": yh[0][:, :, 1],
        "HH": yh[0][:, :, 2],
    }


def idwt_level1(bands: Dict[str, torch.Tensor], wave: str = "db1") -> torch.Tensor:
    """Inverse one-level DWT from LL,LH,HL,HH back to (B,C,H,W)."""
    _, inv = _dwt_pair(bands["LL"].device, wave=wave)
    yh = [torch.stack([bands["LH"], bands["HL"], bands["HH"]], dim=2)]
    return inv((bands["LL"], yh))


# Back-compat aliases used by figures code.
dwt_haar1 = dwt_level1
idwt_haar1 = idwt_level1


def unet_masks_to_dict(mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split 4-channel mask into per-subband dict (B,1,H,W)."""
    if mask.shape[1] != 4:
        raise ValueError(f"Expected 4 mask channels, got {mask.shape[1]}")
    return {b: mask[:, i : i + 1] for i, b in enumerate(SUBBANDS)}


def apply_subband_masks(
    bands: Dict[str, torch.Tensor],
    mask_prob: Dict[str, torch.Tensor],
    mask_hard: Dict[str, torch.Tensor],
    *,
    use_hard: bool = True,
) -> Dict[str, torch.Tensor]:
    """Zero-fill outside mask in coefficient domain: e_band = m_band ⊙ band."""
    out = {}
    for b in SUBBANDS:
        m = mask_hard[b] if use_hard else mask_prob[b]
        out[b] = m * bands[b]
    return out


def subband_p_actual(mask_hard: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {b: float(mask_hard[b].mean().item()) for b in SUBBANDS}


def mean_subband_p(p_dict: Dict[str, float]) -> float:
    return sum(p_dict.values()) / len(SUBBANDS)


def band_energy_totals(bands: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Σ coeff² per subband (scalar tensors, no grad)."""
    return {b: bands[b].float().pow(2).sum().detach() for b in SUBBANDS}


def band_energy_shares(bands: Dict[str, torch.Tensor], eps: float = 1e-12) -> Dict[str, float]:
    """Fraction of total DWT energy in each subband: E_b / Σ_k E_k."""
    totals = band_energy_totals(bands)
    z = sum(totals.values()).clamp_min(eps)
    return {b: float((totals[b] / z).item()) for b in SUBBANDS}


def band_energy_kept_fractions(
    bands: Dict[str, torch.Tensor],
    mask_hard: Dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Kept energy within each band: ||m_b ⊙ X_b||² / ||X_b||²."""
    out: Dict[str, float] = {}
    for b in SUBBANDS:
        coeff = bands[b].float()
        total = coeff.pow(2).sum().clamp_min(eps)
        kept = (mask_hard[b].float() * coeff).pow(2).sum()
        out[b] = float((kept / total).item())
    return out


def energy_weighted_p(
    bands: Dict[str, torch.Tensor],
    mask_hard: Dict[str, torch.Tensor],
) -> float:
    """Kept DWT energy / full image energy: Σ ||m_b ⊙ X_b||² / Σ ||X_b||².

    Denominator is all subbands (including disabled/zeroed), so dropping LL is not hidden.
    """
    total = bands["LL"].new_zeros(())
    kept = bands["LL"].new_zeros(())
    for b in SUBBANDS:
        coeff = bands[b].float()
        energy = coeff.pow(2).sum()
        total = total + energy
        kept = kept + (mask_hard[b].float() * coeff).pow(2).sum()
    if float(total.item()) <= 0.0:
        return 0.0
    return float((kept / total).item())


def force_disabled_masks(
    mask_prob: torch.Tensor,
    disabled: Iterable[str],
) -> torch.Tensor:
    """Zero entire channels for disabled subbands (e.g. --no-ll)."""
    disabled = {s.upper() for s in disabled}
    if not disabled:
        return mask_prob
    out = mask_prob.clone()
    for i, b in enumerate(SUBBANDS):
        if b in disabled:
            out[:, i].zero_()
    return out


def reconstruct_explanation(
    bands: Dict[str, torch.Tensor],
    mask_prob: Dict[str, torch.Tensor],
    mask_hard: Dict[str, torch.Tensor],
    wave: str = "db1",
) -> torch.Tensor:
    """Masked coeffs (zero outside m) → IDWT → pixel explanation for classifier."""
    e_bands = apply_subband_masks(bands, mask_prob, mask_hard, use_hard=True)
    return idwt_level1(e_bands, wave=wave)


def reconstruct_single_band_explanation(
    bands: Dict[str, torch.Tensor],
    mask_hard: Dict[str, torch.Tensor],
    active: str,
    wave: str = "db1",
) -> torch.Tensor:
    """IDWT with only one subband masked; all other bands zeroed in coeff domain."""
    e_bands = {}
    for b in SUBBANDS:
        if b == active:
            e_bands[b] = mask_hard[b] * bands[b]
        else:
            e_bands[b] = bands[b].new_zeros(bands[b].shape)
    return idwt_level1(e_bands, wave=wave)


def downsample_masks(mask_full: torch.Tensor) -> torch.Tensor:
    """Full-res (B,4,H,W) sigmoid masks → subband grid (B,4,H/2,W/2)."""
    return F.avg_pool2d(mask_full, kernel_size=2, stride=2)


def parse_disabled_subbands(names: Optional[Iterable[str]]) -> Tuple[str, ...]:
    out = []
    seen = set()
    for n in names or ():
        b = str(n).upper()
        if b not in SUBBANDS:
            raise ValueError(f"Unknown subband {n!r}; expected one of {SUBBANDS}")
        if b not in seen:
            seen.add(b)
            out.append(b)
    return tuple(out)
