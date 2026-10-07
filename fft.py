"""Polar RFFT utilities (WaveX / minimal_frequency_explanation.ipynb)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class FourierPolarMasker(nn.Module):
    """One-sided rfft2/irfft2 with a shared polar (radius, angle) bin table."""

    def __init__(self, radial_bins: int = 16, angular_bins: int = 16):
        super().__init__()
        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)
        self._index_cache: Dict[tuple, torch.Tensor] = {}
        self._radius_cache: Dict[tuple, torch.Tensor] = {}

    @property
    def n_bins(self) -> int:
        return self.radial_bins * self.angular_bins

    def fft2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

    def ifft2_real(self, coeff: torch.Tensor, signal_hw) -> torch.Tensor:
        h, w = int(signal_hw[-2]), int(signal_hw[-1])
        return torch.fft.irfft2(coeff, s=(h, w), dim=(-2, -1), norm="ortho")

    def radius_map(self, height: int, width: int, device) -> torch.Tensor:
        height, width = int(height), int(width)
        key = (height, width, str(device), "r")
        if key in self._radius_cache:
            return self._radius_cache[key]
        coeff_w = width // 2 + 1
        fy = torch.fft.fftfreq(height, d=1.0, device=device).view(height, 1)
        fx = torch.fft.rfftfreq(width, d=1.0, device=device).view(1, coeff_w)
        radius = torch.sqrt(fx.square() + fy.square())
        radius = radius / radius.max().clamp(min=1e-8)
        self._radius_cache[key] = radius
        return radius

    def _polar_indices(self, height, width, device) -> torch.Tensor:
        height, width = int(height), int(width)
        key = (height, width, str(device), self.radial_bins, self.angular_bins)
        if key in self._index_cache:
            return self._index_cache[key]
        coeff_w = width // 2 + 1
        fy = torch.fft.fftfreq(height, d=1.0, device=device).view(height, 1)
        fx = torch.fft.rfftfreq(width, d=1.0, device=device).view(1, coeff_w)
        radius = torch.sqrt(fx.square() + fy.square())
        radius = radius / radius.max().clamp(min=1e-8)
        radial_idx = torch.clamp((radius * self.radial_bins).long(), max=self.radial_bins - 1)
        angle = torch.atan2(fy, fx.clamp(min=1e-12))
        angle_norm = ((angle + (torch.pi / 2.0)) / torch.pi).clamp(0.0, 1.0 - 1e-7)
        angular_idx = torch.clamp((angle_norm * self.angular_bins).long(), max=self.angular_bins - 1)
        flat_idx = (radial_idx * self.angular_bins + angular_idx).view(1, 1, -1)
        self._index_cache[key] = flat_idx
        return flat_idx

    def expand_polar_mask(self, m_polar: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b = m_polar.shape[0]
        height, width = int(height), int(width)
        coeff_w = width // 2 + 1
        flat_mask = m_polar.reshape(b, 1, self.n_bins)
        flat_idx = self._polar_indices(height, width, m_polar.device).expand(b, 1, -1)
        return flat_mask.gather(dim=2, index=flat_idx).view(b, 1, height, coeff_w)

    def polar_bin_energies(self, coeff: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Per-polar-bin coefficient energy, summed over RGB. Shape (B, n_bins)."""
        energy = (coeff.abs() ** 2).sum(dim=1)
        bsz = energy.shape[0]
        flat = energy.reshape(bsz, -1)
        idx = self._polar_indices(height, width, coeff.device).view(1, -1).expand(bsz, -1)
        out = flat.new_zeros(bsz, self.n_bins)
        out.scatter_add_(1, idx, flat)
        return out

    def kept_energy_fraction(self, coeff: torch.Tensor, half_mask: torch.Tensor) -> torch.Tensor:
        """p_E = ||m ⊙ X||² / ||X||², averaged over the batch."""
        kept = ((half_mask * coeff).abs() ** 2).sum(dim=(1, 2, 3))
        tot = (coeff.abs() ** 2).sum(dim=(1, 2, 3)).clamp(min=1e-8)
        return kept / tot

    def construct_explanation(self, x: torch.Tensor, m_polar: torch.Tensor) -> torch.Tensor:
        coeff = self.fft2(x)
        half = self.expand_polar_mask(m_polar, x.shape[-2], x.shape[-1])
        return self.ifft2_real(coeff * half, signal_hw=x.shape[-2:])

    def perturb_explanation(
        self,
        x: torch.Tensor,
        m_polar: torch.Tensor,
        noise_scale: float = 0.1,
    ) -> torch.Tensor:
        """White-noise the dropped Fourier bins, then IFFT (WaveX perturb_explanation)."""
        coeff = self.fft2(x)
        half = self.expand_polar_mask(m_polar, x.shape[-2], x.shape[-1])
        coeff_std = coeff.abs().std().detach().clamp(min=1e-6)
        noise_real = torch.randn_like(coeff.real) * (noise_scale * coeff_std)
        noise_imag = torch.randn_like(coeff.imag) * (noise_scale * coeff_std)
        noise = torch.complex(noise_real, noise_imag)
        mixed = half * coeff + (1.0 - half) * noise
        return self.ifft2_real(mixed, signal_hw=x.shape[-2:])

    def circular_lowpass_mask(self, height: int, width: int, radius: float, device) -> torch.Tensor:
        rmap = self.radius_map(height, width, device)
        return (rmap <= float(radius)).to(dtype=torch.float32)

    def apply_coeff_mask(self, x: torch.Tensor, half_mask: torch.Tensor) -> torch.Tensor:
        coeff = self.fft2(x)
        if half_mask.dim() == 2:
            half_mask = half_mask.view(1, 1, *half_mask.shape)
        elif half_mask.dim() == 3:
            half_mask = half_mask.unsqueeze(1)
        return self.ifft2_real(coeff * half_mask, signal_hw=x.shape[-2:])

    def lowpass_image(self, x: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (filtered image, p_energy, half_mask)."""
        coeff = self.fft2(x)
        half = self.circular_lowpass_mask(x.shape[-2], x.shape[-1], radius, x.device)
        half = half.view(1, 1, *half.shape)
        p_e = self.kept_energy_fraction(coeff, half)
        e = self.ifft2_real(coeff * half, signal_hw=x.shape[-2:])
        return e, p_e, half

    def greedy_polar_mask_for_energy(
        self,
        scores_polar: torch.Tensor,
        bin_energy: torch.Tensor,
        target_energy: float,
    ) -> torch.Tensor:
        """Keep highest-score polar bins until cumulative energy ≥ target_energy.

        scores_polar: (1,1,R,A) or (R,A). Higher score = more important.
        bin_energy: (n_bins,) or (1, n_bins).
        """
        scores = scores_polar.reshape(-1)
        energy = bin_energy.reshape(-1)
        if float(target_energy) <= 0:
            return torch.zeros(1, 1, self.radial_bins, self.angular_bins, device=scores_polar.device, dtype=scores_polar.dtype)
        order = torch.argsort(scores, descending=True)
        keep = torch.zeros_like(scores)
        accum = scores.new_zeros(())
        target = float(target_energy)
        for idx in order:
            keep[idx] = 1.0
            accum = accum + energy[idx]
            if float(accum.item()) >= target:
                break
        return keep.view(1, 1, self.radial_bins, self.angular_bins)

    def one_sided_to_full(self, tensor_one_sided: torch.Tensor, width: Optional[int] = None) -> torch.Tensor:
        w_half = int(tensor_one_sided.shape[-1])
        if width is None:
            width = 2 * (w_half - 1)
        width = int(width)
        if width % 2 == 0:
            core = tensor_one_sided[..., 1 : w_half - 1]
        else:
            core = tensor_one_sided[..., 1:w_half]
        core = torch.flip(core, dims=[-1])
        if torch.is_complex(tensor_one_sided):
            core = torch.conj(core)
        return torch.cat([tensor_one_sided, core], dim=-1)

    def fft_log_magnitude(self, coeff: torch.Tensor, width: int) -> torch.Tensor:
        full = self.one_sided_to_full(coeff, width=width)
        shifted = torch.fft.fftshift(full, dim=(-2, -1))
        mag = torch.log1p(shifted.abs()).mean(dim=1, keepdim=True)
        return mag

    def fft_magnitude(self, coeff: torch.Tensor, width: int) -> torch.Tensor:
        full = self.one_sided_to_full(coeff, width=width)
        shifted = torch.fft.fftshift(full, dim=(-2, -1))
        return shifted.abs().mean(dim=1, keepdim=True)

    def radial_band_kept_fractions(
        self,
        coeff: torch.Tensor,
        half_mask: torch.Tensor,
        n_bands: int = 3,
    ) -> Dict[str, float]:
        """Kept energy / band energy for low / mid / high radius thirds.

        Radius is Nyquist-normalized to [0, 1] (DC → image corner).
        low = r ∈ [0, 1/3), mid = [1/3, 2/3), high = [2/3, 1].
        Value is ||m ⊙ 1_band ⊙ X||² / ||1_band ⊙ X||² (retained fraction of that band).
        """
        h, w = coeff.shape[-2], 2 * (coeff.shape[-1] - 1)
        rmap = self.radius_map(coeff.shape[-2], w, coeff.device)
        names = ("low", "mid", "high") if n_bands == 3 else tuple(f"r{i}" for i in range(n_bands))
        out: Dict[str, float] = {}
        for i, name in enumerate(names):
            lo, hi = i / n_bands, (i + 1) / n_bands
            if i == n_bands - 1:
                band = (rmap >= lo) & (rmap <= hi + 1e-6)
            else:
                band = (rmap >= lo) & (rmap < hi)
            band_f = band.to(dtype=coeff.real.dtype).view(1, 1, *band.shape)
            tot = ((band_f * coeff).abs() ** 2).sum()
            kept = ((band_f * half_mask * coeff).abs() ** 2).sum()
            out[name] = float((kept / tot.clamp(min=1e-12)).item())
        return out
