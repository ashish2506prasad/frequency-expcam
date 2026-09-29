"""Attribution-only Fourier IG (WaveX frequency_ig_explanation_fourier.ipynb).

Baselines are circular low-pass images at cutoff radii r in [0, 1].
IG(r) attributes from low-pass(r) → x; ΔAttr_i = IG(r_i) − IG(r_{i+1}).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from fft import FourierPolarMasker
from io_utils import ensure_dir, save_json
from viz import panel_original, to_numpy_image


def circular_lowpass(x: torch.Tensor, radius: float, fourier: FourierPolarMasker) -> torch.Tensor:
    e, _, _ = fourier.lowpass_image(x, radius)
    return e


def integrated_gradients(
    model: torch.nn.Module,
    img: torch.Tensor,
    baseline: torch.Tensor,
    target_class: int,
    steps: int = 32,
    batch_size: int = 8,
) -> torch.Tensor:
    """IG from baseline to img for the target class logit. Returns |IG| averaged over RGB."""
    alphas = torch.linspace(0.0, 1.0, steps, device=img.device, dtype=img.dtype)
    delta = img - baseline
    acc = torch.zeros_like(img)
    n = 0
    for start in range(0, steps, batch_size):
        chunk = alphas[start : start + batch_size]
        z = baseline + chunk.view(-1, 1, 1, 1) * delta
        z = z.detach().requires_grad_(True)
        logits = model(z)
        score = logits[:, int(target_class)].sum()
        grads = torch.autograd.grad(score, z)[0]
        acc = acc + grads.sum(dim=0, keepdim=True)
        n += int(chunk.numel())
    ig = delta * (acc / max(n, 1))
    return ig.abs().mean(dim=1, keepdim=True)


def run_fourier_ig(
    model: torch.nn.Module,
    x: torch.Tensor,
    fourier: FourierPolarMasker,
    radii: Sequence[float],
    steps: int = 32,
) -> dict:
    with torch.no_grad():
        logits = model(x)
        y = int(logits.argmax(dim=1).item())
        conf = float(F.softmax(logits, dim=1)[0, y].item())
    attrs: List[torch.Tensor] = []
    baselines: List[torch.Tensor] = []
    for r in radii:
        with torch.no_grad():
            base = circular_lowpass(x, float(r), fourier)
        baselines.append(base.detach())
        attrs.append(integrated_gradients(model, x, base, y, steps=steps).detach())
    deltas = []
    for i in range(len(attrs) - 1):
        deltas.append((attrs[i] - attrs[i + 1]).clamp(min=0))
    return {
        "y": y,
        "conf": conf,
        "radii": [float(r) for r in radii],
        "attrs": attrs,
        "deltas": deltas,
        "baselines": baselines,
    }


def plot_fourier_ig(path: Path, x: torch.Tensor, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radii = result["radii"]
    attrs = result["attrs"]
    deltas = result["deltas"]
    n = len(radii)
    fig, axes = plt.subplots(3, n, figsize=(1.9 * n, 6.2))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    x_np = panel_original(x)
    for i, r in enumerate(radii):
        axes[0, i].imshow(np.clip(x_np, 0, 1))
        axes[0, i].set_title(f"x  r={r:.2f}", fontsize=7)
        axes[0, i].axis("off")
        a = to_numpy_image(attrs[i])
        if a.ndim == 3:
            a = a.mean(axis=-1)
        axes[1, i].imshow(a, cmap="inferno")
        axes[1, i].set_title("IG", fontsize=7)
        axes[1, i].axis("off")
        if i < len(deltas):
            d = to_numpy_image(deltas[i])
            if d.ndim == 3:
                d = d.mean(axis=-1)
            axes[2, i].imshow(d, cmap="magma")
            axes[2, i].set_title(f"Δ r{i}→r{i+1}", fontsize=7)
        else:
            axes[2, i].axis("off")
            continue
        axes[2, i].axis("off")
    fig.suptitle(
        f"Fourier IG (circular low-pass baselines)  class={result['y']}  conf={result['conf']:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_fourier_ig_run(
    out_dir: Path,
    x: torch.Tensor,
    model: torch.nn.Module,
    fourier: FourierPolarMasker,
    n_cutoffs: int = 8,
    ig_steps: int = 32,
) -> dict:
    radii = np.linspace(0.0, 1.0, int(n_cutoffs)).tolist()
    result = run_fourier_ig(model, x, fourier, radii, steps=ig_steps)
    dest = ensure_dir(Path(out_dir) / "attribution_ig")
    plot_fourier_ig(dest / "Fig0_fft_ig.png", x, result)
    save_json(
        dest / "ig_meta.json",
        {"radii": result["radii"], "y": result["y"], "conf": result["conf"]},
    )
    return {"radii": result["radii"], "y": result["y"], "conf": result["conf"]}
