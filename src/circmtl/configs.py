from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]   # config.py -> circmtl -> src -> project root


def load_paths(cfg_file: str = "configs/paths.yaml") -> dict[str, Path]:
    """Resolve all configured paths against the project root and validate the inputs."""
    with open(ROOT / cfg_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = cfg["datasets"]
    paths = {
        "raw": ROOT / cfg["inputs"]["raw_dir"],
        "processed": ROOT / cfg["outputs"]["processed_dir"],
        "circr2disease": ROOT / ds["circr2disease"]["dir"],
        "encori_mirna": ROOT / ds["encori_mirna"]["dir"] / ds["encori_mirna"]["file"],
        "encori_rbp": ROOT / ds["encori_rbp"]["dir"],
    }

    missing = [k for k in ("raw", "circr2disease", "encori_mirna", "encori_rbp")
               if not paths[k].exists()]
    if missing:
        raise FileNotFoundError(
            "Missing inputs: " + ", ".join(f"{k} -> {paths[k]}" for k in missing))

    paths["processed"].mkdir(parents=True, exist_ok=True)
    return paths


PATHS = load_paths()
