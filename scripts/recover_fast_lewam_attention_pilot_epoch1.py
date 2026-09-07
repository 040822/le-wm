"""Safely adopt completed epoch-1 attention-pilot artifacts after a compatible fix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.experiments.fast_lewam_attention_pilots import (
    StaleArtifactError,
    atomic_write_json,
    cohort_for_level,
    file_sha256,
    load_pilot_manifest,
    phase_identity,
    read_json,
    validate_artifact_identity,
)
from scripts.run_fast_lewam_attention_pilot_worker import (
    _baseline_eval_leaf,
    _evaluation_cache_identity,
    _evaluation_identity_path,
    _eval_leaf,
    _phase_marker_path,
    _result_payload,
    _stage_root,
    _validate_result_cohort,
)


def _require_compatible_identity(old, new, previous_sha, *, source):
    if not isinstance(old, dict):
        raise StaleArtifactError(f"missing identity: {source}")
    if old.get("manifest_sha256") != previous_sha:
        raise StaleArtifactError(f"unexpected previous manifest: {source}")
    adopted = dict(old)
    adopted["manifest_sha256"] = new["manifest_sha256"]
    if adopted != new:
        raise StaleArtifactError(f"identity differs beyond manifest SHA: {source}")


def _archive(path: Path, archive_root: Path) -> None:
    if not path.exists():
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / path.name
    if target.exists():
        raise FileExistsError(f"recovery archive target already exists: {target}")
    path.replace(target)


def _adopt_pilot(manifest, pilot, previous_sha: str, *, dry_run: bool) -> dict:
    cohort = cohort_for_level(manifest, pilot, 1)
    marker_path = _phase_marker_path(pilot, 1)
    marker = read_json(marker_path)
    expected_train = phase_identity(
        manifest,
        pilot,
        epoch=1,
        phase="train",
        parent_checkpoint=None,
        cohort=cohort,
    )
    _require_compatible_identity(
        marker.get("identity"), expected_train, previous_sha, source=marker_path
    )
    if marker.get("status") != "complete":
        raise StaleArtifactError(f"epoch-1 training is not complete: {marker_path}")
    if not validate_artifact_identity(marker.get("artifacts", {})):
        raise StaleArtifactError(f"epoch-1 artifacts changed: {marker_path}")

    stage_identity_path = _stage_root(pilot, 1) / "stage_identity.json"
    stage_identity = read_json(stage_identity_path)
    _require_compatible_identity(
        stage_identity.get("identity"),
        expected_train,
        previous_sha,
        source=stage_identity_path,
    )
    if stage_identity.get("status") != "complete":
        raise StaleArtifactError(f"epoch-1 stage identity is not complete: {stage_identity_path}")

    eval_specs = (
        (
            _baseline_eval_leaf(pilot, 1),
            pilot.baseline_checkpoints[1],
            pilot.baseline_config,
        ),
        (
            _eval_leaf(pilot, 1),
            _stage_root(pilot, 1) / "checkpoints" / "fast_lewam_weights_epoch_1.pt",
            _stage_root(pilot, 1) / "config.yaml",
        ),
    )
    adopted_evals = []
    for output_leaf, checkpoint, run_config in eval_specs:
        result_path = output_leaf / "result.json"
        sidecar_path = _evaluation_identity_path(output_leaf)
        sidecar = read_json(sidecar_path)
        expected_eval = _evaluation_cache_identity(
            manifest,
            pilot,
            epoch=1,
            checkpoint=checkpoint,
            run_config=run_config,
            cohort=cohort,
        )
        _require_compatible_identity(
            sidecar.get("identity"), expected_eval, previous_sha, source=sidecar_path
        )
        if sidecar.get("result_sha256") != file_sha256(result_path):
            raise StaleArtifactError(f"evaluation result changed: {result_path}")
        _validate_result_cohort(_result_payload(result_path), cohort)
        adopted_evals.append((sidecar_path, sidecar, expected_eval))

    summary_path = pilot.output_dir / "summaries" / "epoch_1.json"
    summary = read_json(summary_path)
    if (
        summary.get("manifest_sha256") != previous_sha
        or summary.get("status") != "ok"
        or summary.get("decision") != "continue"
        or summary.get("pilot") != pilot.name
        or summary.get("epoch") != 1
        or summary.get("cohort_sha256") != expected_train["cohort_sha256"]
    ):
        raise StaleArtifactError(f"epoch-1 summary is not compatible: {summary_path}")

    archive_root = (
        pilot.output_dir
        / "recovery_archive"
        / f"{previous_sha[:12]}_to_{manifest.sha256[:12]}"
    )
    if dry_run:
        return {"pilot": pilot.name, "archive": str(archive_root), "status": "validated"}

    # All checks precede writes. The migration only changes the manifest SHA.
    marker["identity"] = expected_train
    atomic_write_json(marker_path, marker)
    stage_identity["identity"] = expected_train
    atomic_write_json(stage_identity_path, stage_identity)
    for sidecar_path, sidecar, expected_eval in adopted_evals:
        sidecar["identity"] = expected_eval
        atomic_write_json(sidecar_path, sidecar)
    summary["manifest_sha256"] = manifest.sha256
    atomic_write_json(summary_path, summary)

    _archive(_stage_root(pilot, 3), archive_root)
    _archive(pilot.output_dir / "process.json", archive_root)
    _archive(pilot.output_dir / "worker_failure.json", archive_root)
    return {"pilot": pilot.name, "archive": str(archive_root), "status": "adopted"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--previous-manifest-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_pilot_manifest(args.manifest)
    if args.previous_manifest_sha256 == manifest.sha256:
        raise ValueError("previous and current manifest identities are equal")
    rows = [
        _adopt_pilot(
            manifest,
            pilot,
            args.previous_manifest_sha256,
            dry_run=args.dry_run,
        )
        for pilot in manifest.pilots.values()
    ]
    print({"manifest_sha256": manifest.sha256, "pilots": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
