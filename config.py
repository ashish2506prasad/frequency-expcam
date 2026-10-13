"""Canonical hyperparameters, paths, and experiment constants.

Values below are the paper's later full-model defaults, taken from the
AM4/AM5/AM6 training cells (not the earlier AM1/AM2 hand-tuning grid).
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
TRAINX_DIR = PROJECT_ROOT / "trainX1_subset"
RUNS_DIR = PROJECT_ROOT / "expcam_runs"
EVAL_SET_PATH = RUNS_DIR / "eval_set.json"
WHERE = "local"
ZIP_OUT = False
_ZIP_ATEXIT = False

# Kaggle dataset layout (class folders such as n01484850 live under this root).
# Override with --data-root if the attached dataset slug differs.
KAGGLE_TRAINX = Path("/kaggle/input/trainx1-subset/trainX1_subset")
KAGGLE_RUNS = Path("/kaggle/working/expcam_runs")
KAGGLE_WORKING = Path("/kaggle/working")

SEED = 0
# None = every TRAINX_CLASSES folder with at least N_PER_CLASS images (skips e.g. n01749939=7).
N_CLASSES = None
N_PER_CLASS = 10
N_IMAGES = None  # filled when the eval set is built
QUALITATIVE_PER_CLASS = 2  # images per class in Fig A11, B5, C4, E4, etc.
SECTION0_PER_CLASS = 10  # Section 0 trains 10 images per class
SECTION0_IMAGE_SLOTS = tuple(range(10))
SECTION0_STEPS = 800
SECTION0_EF1_SNAPSHOT_START = 0
SECTION0_EF1_SNAPSHOT_EVERY = 40
SECTION3_EXP_STEPS = 800  # EXP-CAM training in Section 3; best-EF1 checkpoint saved
IMAGE_SIZE = 224

DEFAULT_BACKBONES = ("resnet18", "efficientnet_b0", "vit_b_16")

# Per-backbone figure output folders under section1/ … section5/.
FIGURE_DIRS = {
    "resnet18": "figures_resnet",
    "efficientnet_b0": "figures_efficient",
    "vit_b_16": "figures_vit",
}


def figure_dir(out_root: Path, section: str, backbone: str) -> Path:
    """Return e.g. section1/figures_resnet/ for a given backbone."""
    sub = FIGURE_DIRS.get(backbone, f"figures_{backbone}")
    return Path(out_root) / section / sub

# Per-image U-Net training (AM6 cell 15 / AM4–AM5 full-model cells)
STEPS = 700
SECTION1_STEPS = 300
LR = 5e-5
OPTIMIZER_NAME = "adam"
TEMPERATURE = 2.0
MASK_THRESH = 0.5
ROB_K_BACKGROUNDS = 3

DEFAULT_LAMBDAS = {
    "act": 10.0,
    "ce": 15.0,
    "kl": 5.0,
    "area": 100.0,
    "energy": 0.0,  # FFT Section 0; unused in pixel EXP-CAM
    "bin": 1.0,
    "tv": 500.0,
    "rob": 100.0,  # abductive / L_rob in the notebooks (lam_abd)
}

DEFAULT_LAMBDAS_DWT = {
    "act": 10.0,
    "ce": 0.0,
    "kl": 5.0,
    "area": 0.0,  # uniform extra; WaveX per-band area lives in DEFAULT_DWT_BAND_LAMBDAS
    "energy": 0.0,
    "bin": 1.0,
    "tv": 0.0,
    "rob": 0.0,
}

DEFAULT_LAMBDAS_FFT = {
    "act": 10.0,
    "ce": 0.0,
    "kl": 5.0,
    "area": 0.0,  # unused in polar FFT (kept so the lambda dict is complete)
    "energy": 10.0,
    "bin": 1.0,
    "tv": 0.0,
    "rob": 6.0,
}

# Per Haar subband area / TV weights (keys area_LL, tv_LH, …). CLI: --lam-ll-area, etc.
# WaveX adds Σ w_b * mean(m_b) directly (not multiplied by λ_area). HH > LH=HL > LL.
DWT_SUBBANDS = ("LL", "LH", "HL", "HH")
DWT_SUBBAND_CLI = ("ll", "lh", "hl", "hh")
DEFAULT_DWT_BAND_LAMBDAS = {
    "area_LL": 5.0,
    "area_LH": 10.0,
    "area_HL": 10.0,
    "area_HH": 20.0,
    "tv_LL": 0.0,
    "tv_LH": 0.0,
    "tv_HL": 0.0,
    "tv_HH": 0.0,
    "rob_LL": 0.0,
    "rob_LH": 0.0,
    "rob_HL": 0.0,
    "rob_HH": 0.0,
}
DEFAULT_DWT_YAML = PACKAGE_DIR / "configs" / "dwt.yaml"
DEFAULT_FFT_YAML = PACKAGE_DIR / "configs" / "fft.yaml"
DEFAULT_DWT_WAVELET = "db1"

LOSS_TERMS = ("act", "ce", "kl", "area", "energy", "bin", "tv", "rob")

BUDGETS = (0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
# Section 2 knapsack sweep on Section 0 best mask_prob (fraction of pixels kept)
KNAPSACK_BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
FIDELITY_TAUS = (0.95, 0.90, 1.0)
P_TARGETS = (0.05, 0.10, 0.20)

LAYER_WEIGHTS_RESNET = {
    "relu": 1.0,
    "layer1": 2.0,
    "layer2": 2.0,
    "layer3": 4.0,
    "layer4": 8.0,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GREY_BG = 0.5

# ImageNet-1K synsets present in trainX1_subset/
TRAINX_CLASSES = {
    "n01484850": {"name": "great_white_shark", "imagenet_idx": 2},
    "n01494475": {"name": "hammerhead", "imagenet_idx": 4},
    "n01531178": {"name": "goldfinch", "imagenet_idx": 11},
    "n01632777": {"name": "axolotl", "imagenet_idx": 29},
    "n01665541": {"name": "leatherback_turtle", "imagenet_idx": 34},
    "n01687978": {"name": "agama", "imagenet_idx": 42},
    "n01695060": {"name": "komodo_dragon", "imagenet_idx": 48},
    "n01749939": {"name": "green_mamba", "imagenet_idx": 64},
    "n01818515": {"name": "macaw", "imagenet_idx": 88},
    "n01833805": {"name": "hummingbird", "imagenet_idx": 94},
    "n02012849": {"name": "crane", "imagenet_idx": 134},
    "n02018795": {"name": "bustard", "imagenet_idx": 138},
    "n02037110": {"name": "oystercatcher", "imagenet_idx": 143},
    "n01914609": {"name": "jellyfish", "imagenet_idx": 107},
    "n01924916": {"name": "flatworm", "imagenet_idx": 110},
    "n01930112": {"name": "nematode", "imagenet_idx": 111},
    "n01950731": {"name": "sea_slug", "imagenet_idx": 115},
    "n01978455": {"name": "rock_crab", "imagenet_idx": 119},
    "n01984695": {"name": "spiny_lobster", "imagenet_idx": 123},
}


def ef1_snapshot_steps(
    start: int = SECTION0_EF1_SNAPSHOT_START,
    every: int = SECTION0_EF1_SNAPSHOT_EVERY,
    max_step: int = SECTION0_STEPS,
) -> tuple[int, ...]:
    """Steps at which to save explanation snapshots (40, 80, …, max_step)."""
    return tuple(range(start + every, max_step + 1, every))


def ef1_grid_steps(
    every: int,
    max_step: int = SECTION0_STEPS,
    first: int | None = None,
) -> tuple[int, ...]:
    """Steps for Section 0 progression grids (40, 80, …, max_step)."""
    if first is None:
        first = every
    return tuple(range(first, max_step + 1, every))

N_BG_DRAWS = 5
N_INSERT_STEPS = 50
N_RISE_MASKS = 800
N_INSERT_SEEDS = 5
AUG_ROT_DEG = 5.0
AUG_TRANSLATE_FRAC = 0.05
AUG_BRIGHTNESS = 0.10

# Retrospective count of distinct λ tuples in AM0–AM6 (Section 5 "tuning cost")
MANUAL_TRIAL_CONFIGS = 18

# Section 6 — local causal LM (no Hugging Face hub on air-gapped clusters)
DISTILGPT2_LOCAL = PROJECT_ROOT / "distilgpt2-local"


def detect_where() -> str:
    if Path("/kaggle/input").exists() or Path("/kaggle/working").exists():
        return "kaggle"
    return "local"


def _kaggle_trainx_dir() -> Path:
    """Resolve trainX1_subset under /kaggle/input, or fall back to KAGGLE_TRAINX."""
    markers = ("n01484850", "n01531178", "n01494475")
    candidates = [
        KAGGLE_TRAINX,
        Path("/kaggle/input/trainx1-subset"),
        Path("/kaggle/input/trainX1_subset"),
        Path("/kaggle/input/trainX1_subset/trainX1_subset"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for child in sorted(input_root.iterdir()):
            if child.is_dir():
                candidates.append(child)
                candidates.append(child / "trainX1_subset")
    seen = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.is_dir() and (c.name == "trainX1_subset" or any((c / m).is_dir() for m in markers)):
            return c
    return KAGGLE_TRAINX


def zip_runs_dir(runs_dir=None) -> Path | None:
    """Write <runs_dir>.zip next to the output folder (Kaggle-downloadable)."""
    import shutil

    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        print(f"[zip] skip, not a directory: {runs_dir}", flush=True)
        return None
    archive = Path(shutil.make_archive(str(runs_dir), "zip", root_dir=str(runs_dir)))
    print(f"[zip] {runs_dir} -> {archive}", flush=True)
    return archive


def maybe_zip_runs(runs_dir=None) -> Path | None:
    if not ZIP_OUT:
        return None
    return zip_runs_dir(runs_dir)


def apply_where(where: str = "auto", data_root=None, runs_dir=None, zip_out=None) -> str:
    """Switch trainX1_subset/ and output roots for local vs Kaggle.

    local:  <repo>/trainX1_subset  and  <repo>/expcam_runs
    kaggle: /kaggle/input/.../trainX1_subset  and  /kaggle/working/expcam_runs
    --data-root / --out override those defaults.
    """
    global PROJECT_ROOT, TRAINX_DIR, RUNS_DIR, EVAL_SET_PATH, WHERE, ZIP_OUT, _ZIP_ATEXIT
    if where in (None, "auto"):
        where = detect_where()
    WHERE = where
    if where == "kaggle":
        PROJECT_ROOT = KAGGLE_WORKING
        TRAINX_DIR = _kaggle_trainx_dir()
        RUNS_DIR = KAGGLE_RUNS
    elif where == "local":
        PROJECT_ROOT = PACKAGE_DIR.parent
        TRAINX_DIR = PROJECT_ROOT / "trainX1_subset"
        RUNS_DIR = PROJECT_ROOT / "expcam_runs"
    else:
        raise ValueError(f"Unknown --where {where!r}; expected local, kaggle, or auto")
    if data_root is not None:
        TRAINX_DIR = Path(data_root)
    if runs_dir is not None:
        RUNS_DIR = Path(runs_dir)
    EVAL_SET_PATH = RUNS_DIR / "eval_set.json"
    ZIP_OUT = bool(zip_out) if zip_out is not None else (WHERE == "kaggle")
    print(f"[paths] where={WHERE} data-root={TRAINX_DIR} out={RUNS_DIR} zip-out={ZIP_OUT}", flush=True)
    if ZIP_OUT and not _ZIP_ATEXIT:
        import atexit

        atexit.register(maybe_zip_runs)
        _ZIP_ATEXIT = True
    return WHERE


def add_where_args(p):
    p.add_argument(
        "--where",
        choices=("local", "kaggle", "auto"),
        default="auto",
        help=(
            "Path layout. local: <repo>/trainX1_subset and <repo>/expcam_runs. "
            "kaggle: /kaggle/input/.../trainX1_subset and /kaggle/working/expcam_runs. "
            "auto: kaggle if /kaggle exists."
        ),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Image folder with synset dirs (n01484850, …). Overrides the --where default.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run directory. Default: expcam_runs (local) or /kaggle/working/expcam_runs (kaggle).",
    )
    p.add_argument(
        "--eval-set",
        type=Path,
        default=None,
        help="Reuse this eval_set.json (do not rebuild). Use the FFT run's file so image ids match.",
    )
    p.set_defaults(zip_out=None)
    p.add_argument(
        "--zip-out",
        action="store_true",
        dest="zip_out",
        help="Zip --out after the run. On by default when --where kaggle.",
    )
    p.add_argument(
        "--no-zip-out",
        action="store_false",
        dest="zip_out",
        help="Do not zip --out (overrides the Kaggle default).",
    )
    return p


_YAML_LAMBDA_KEYS = {
    "lambda_act": "act",
    "lambda_CE": "ce",
    "lambda_ce": "ce",
    "lambda_KL": "kl",
    "lambda_kl": "kl",
    "lambda_area": "area",
    "lambda_energy": "energy",
    "lambda_bins": "area",
    "lambda_bin": "bin",
    "lambda_tv": "tv",
    "lambda_rob": "rob",
}


def load_yaml_file(path) -> dict:
    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML {path} must be a mapping, got {type(data)}")
    return data


def parse_dwt_yaml(path=None) -> dict:
    """Load DWT training weights. Missing file → package defaults.

    Accepts nested ``lambdas`` / ``band_lambdas`` or WaveX-style ``lambda_act``,
    ``lambda_area_LL``, ``disabled_subbands``, ``wavelet``.
    """
    path = Path(path) if path is not None else DEFAULT_DWT_YAML
    lambdas = dict(DEFAULT_LAMBDAS_DWT)
    band = dict(DEFAULT_DWT_BAND_LAMBDAS)
    disabled: list[str] = []
    wavelet = DEFAULT_DWT_WAVELET
    if path.is_file():
        data = load_yaml_file(path)
        nested = data.get("lambdas") or {}
        for k, v in nested.items():
            if k in lambdas and v is not None:
                lambdas[k] = float(v)
        for src, dst in _YAML_LAMBDA_KEYS.items():
            if data.get(src) is not None:
                lambdas[dst] = float(data[src])
        nested_band = data.get("band_lambdas") or data.get("band") or {}
        for k, v in nested_band.items():
            if v is not None:
                band[k] = float(v)
        for b in DWT_SUBBANDS:
            for kind in ("area", "tv", "rob"):
                for key in (f"lambda_{kind}_{b}", f"lam_{kind}_{b}", f"{kind}_{b}"):
                    if data.get(key) is not None:
                        band[f"{kind}_{b}"] = float(data[key])
        raw_dis = data.get("disabled_subbands") or []
        disabled = [str(s).upper() for s in raw_dis]
        wavelet = str(data.get("wavelet") or data.get("wavelet_name") or wavelet)
        if wavelet.lower() in ("haar", "db1"):
            wavelet = "db1"
    return {
        "lambdas": lambdas,
        "band_lambdas": band,
        "disabled_subbands": disabled,
        "wavelet": wavelet,
        "path": str(path),
    }


def parse_fft_yaml(path=None) -> dict:
    """Load polar-FFT Section 0 weights and bin settings from configs/fft.yaml."""
    path = Path(path) if path is not None else DEFAULT_FFT_YAML
    lambdas = dict(DEFAULT_LAMBDAS_FFT)
    radial = 16
    angular = 16
    temp = 0.25
    thresh = 0.5
    noise = 0.1
    n_lpf = 50
    if path.is_file():
        data = load_yaml_file(path)
        nested = data.get("lambdas") or {}
        for k, v in nested.items():
            if k in lambdas and v is not None and k != "area":
                lambdas[k] = float(v)
        for src, dst in _YAML_LAMBDA_KEYS.items():
            if dst == "area":
                continue
            if data.get(src) is not None:
                lambdas[dst] = float(data[src])
        radial = int(data.get("fft_radial_bins", radial))
        angular = int(data.get("fft_angular_bins", angular))
        temp = float(data.get("mask_temperature", temp))
        thresh = float(data.get("mask_threshold", thresh))
        noise = float(data.get("noise_scale", noise))
        n_lpf = int(data.get("n_lpf_radii", n_lpf))
    lambdas["area"] = 0.0
    return {
        "lambdas": lambdas,
        "fft_radial_bins": radial,
        "fft_angular_bins": angular,
        "fft_mask_temperature": temp,
        "mask_thresh": thresh,
        "fft_noise_scale": noise,
        "n_lpf_radii": n_lpf,
        "path": str(path),
    }


def lambdas_with_overrides(**overrides) -> dict:
    out = dict(DEFAULT_LAMBDAS)
    out.update(overrides)
    return out


def sequential_configs() -> dict:
    """Cumulative loss addition ending with the full model (+ L_rob)."""
    zero = {k: 0.0 for k in DEFAULT_LAMBDAS}
    cfgs = {}

    lam = dict(zero)
    lam["act"] = DEFAULT_LAMBDAS["act"]
    cfgs["seq_act"] = lam

    lam = dict(cfgs["seq_act"])
    lam["kl"] = DEFAULT_LAMBDAS["kl"]
    cfgs["seq_act_kl"] = lam

    lam = dict(cfgs["seq_act_kl"])
    lam["ce"] = DEFAULT_LAMBDAS["ce"]
    cfgs["seq_act_kl_ce"] = lam

    lam = dict(cfgs["seq_act_kl_ce"])
    lam["area"] = DEFAULT_LAMBDAS["area"]
    lam["bin"] = DEFAULT_LAMBDAS["bin"]
    lam["tv"] = DEFAULT_LAMBDAS["tv"]
    cfgs["seq_act_kl_ce_min"] = lam

    cfgs["Full"] = dict(DEFAULT_LAMBDAS)
    return cfgs


SEQUENTIAL_PLOT_ORDER = ["seq_act", "seq_act_kl", "seq_act_kl_ce", "seq_act_kl_ce_min", "Full"]
