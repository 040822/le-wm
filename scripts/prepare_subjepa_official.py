#!/usr/bin/env python3
"""Download, convert and numerically verify the four published Sub-JEPA models.

Requires a separate legacy site directory containing transformers==4.57.1
and huggingface-hub==0.36.0. Existing runtime dependencies are not changed.
Use --offline after downloading once to repeat conversion/verification locally.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch(url, destination, expected, *, blob=False, offline=False):
    destination = Path(destination)
    digest = git_blob if blob else sha256
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"Existing asset hash mismatch: {destination}")
        return
    if offline:
        raise FileNotFoundError(f"Offline asset is missing: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if digest(temporary) != expected:
            raise ValueError(f"Downloaded asset hash mismatch: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_task(task, spec, registry, output_root, legacy_site, source_dir, offline):
    import torch
    import transformers
    import yaml

    from source.common.remap import remap_state_dict
    from source.model.subjepa.official import (
        FORMAT, OfficialSubJEPA, conversion_checks, infer_model_config,
    )

    task_dir = output_root / task
    raw_dir = task_dir / "raw"
    revision = registry["revision"]
    base_url = f"https://huggingface.co/{registry['repository']}/resolve/{revision}/subjepa/{task}"
    original = raw_dir / f"{task}_subjepa_object.ckpt"
    config_path = raw_dir / "config.yml"
    fetch(f"{base_url}/{original.name}", original, spec["weights_sha256"], offline=offline)
    fetch(f"{base_url}/config.yml", config_path, spec["config_git_blob"], blob=True, offline=offline)
    training_config = yaml.safe_load(config_path.read_text())
    if int(training_config["wm"]["sigreg_subspaces"]) != int(spec["num_subspaces"]):
        raise ValueError(f"Unexpected published subspace count for {task}")
    if int(training_config["wm"]["sigreg_init_mode"]) != 2:
        raise ValueError("Expected frozen orthogonal projections in the published configuration")
    provenance = {
        "method": "subjepa_official", "task": task,
        "repository": registry["repository"], "revision": revision,
        "source_url": f"{base_url}/{original.name}",
        "original_sha256": sha256(original), "config_sha256": sha256(config_path),
        "reference_lewm_revision": registry["lewm_revision"],
        "training_seed": int(training_config["seed"]),
        "training_epochs_configured": int(training_config["trainer"]["max_epochs"]),
        "num_subspaces": int(spec["num_subspaces"]),
        "subspace_dim": int(training_config["wm"]["embed_dim"]) // int(spec["num_subspaces"]),
        "projection_mode": "orthogonal_frozen",
        "frameskip": int(training_config["data"]["dataset"]["frameskip"]),
        "benchmark_training_dataset": training_config["data"]["dataset"]["name"],
        "runtime_transformers": transformers.__version__,
        "reference_transformers": registry["legacy_transformers"],
    }
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    # The subprocess is the only place the official object pickle is loaded.
    env["PYTHONPATH"] = str(legacy_site)
    with tempfile.TemporaryDirectory(prefix="subjepa-export-") as temporary:
        exported = Path(temporary) / "reference.pt"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/subjepa_legacy_export.py"),
            "--source-dir", str(source_dir), "--checkpoint", str(original),
            "--output", str(exported),
        ], env=env, check=True)
        reference = torch.load(exported, map_location="cpu", weights_only=True)
    model_config = infer_model_config(
        reference["state_dict"], training_config, reference["encoder_config"],
    )
    payload = {
        "format": FORMAT, "model_config": model_config,
        "state_dict": remap_state_dict(reference["state_dict"]), "provenance": provenance,
    }
    model = OfficialSubJEPA.from_payload(payload)
    verification = conversion_checks(model, reference["fixture"], reference["expected"])
    checkpoint = task_dir / "subjepa.pt"
    if checkpoint.exists():
        existing = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if existing["provenance"] != provenance or existing["model_config"] != model_config:
            raise ValueError(f"Refusing to overwrite a different converted asset: {checkpoint}")
        conversion_checks(OfficialSubJEPA.from_payload(existing), reference["fixture"], reference["expected"])
        if existing["state_dict"].keys() != payload["state_dict"].keys() or any(
            not torch.equal(existing["state_dict"][key], value)
            for key, value in payload["state_dict"].items()
        ):
            raise ValueError(f"Converted weights differ from the pinned original: {checkpoint}")
    else:
        temporary = checkpoint.with_suffix(".pt.partial")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
    manifest = {
        **provenance, "checkpoint": str(checkpoint.relative_to(output_root)),
        "converted_sha256": sha256(checkpoint), "verification": verification,
        "model_config": model_config, "original_training_config": training_config,
    }
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (task_dir / "model_config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    print(json.dumps({"task": task, "checkpoint": str(checkpoint), "verification": verification}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "config/baselines/subjepa_official.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--legacy-site-packages", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=("tworoom", "pusht", "reacher", "cube"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    output_root = (args.output_root or ROOT / registry["output_root"]).resolve()
    legacy_site = args.legacy_site_packages.resolve()
    if not legacy_site.is_dir():
        parser.error("--legacy-site-packages must be an existing isolated dependency directory")
    source_dir = output_root / "raw_source"
    source_base = f"https://raw.githubusercontent.com/lucas-maes/le-wm/{registry['lewm_revision']}"
    for name, expected in registry["source_files"].items():
        fetch(f"{source_base}/{name}", source_dir / name, expected, offline=args.offline)
    tasks = args.tasks or list(registry["tasks"])
    for task in tasks:
        prepare_task(task, registry["tasks"][task], registry, output_root, legacy_site, source_dir, args.offline)


if __name__ == "__main__":
    main()
