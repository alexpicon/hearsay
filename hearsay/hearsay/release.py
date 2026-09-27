# Author: Alex Picon <alexnpc@me.com>
"""Build a verifiable release record without changing model or submission scores."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from hearsay.model import MODEL_PATH, Detector
from hearsay.ssl import model_revision
from hearsay.submission import (
    audio_paths,
    read_scores,
    require_coverage,
    validate_submission,
)


def fingerprint(path: Path) -> dict[str, object]:
    """Return a file's size and content digest without loading it into memory."""
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def compare_scores(
    reference: Path, recomputed: Path, tolerance: float = 1e-5
) -> dict[str, object]:
    """Check fresh inference against every released score by exact filename.

    Args:
        reference: Released submission TSV.
        recomputed: Freshly computed TSV from the same audio/model.
        tolerance: Largest permitted absolute score difference.

    Returns:
        Reproduction statistics; raises when coverage or tolerance fails.

    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    names, scores = read_scores(reference)
    fresh_names, fresh_scores = read_scores(recomputed)
    require_coverage(fresh_names, names, source="released submission")
    by_name = dict(zip(fresh_names, fresh_scores, strict=True))
    difference = np.abs(scores - np.array([by_name[n] for n in names]))
    maximum = float(difference.max())
    if maximum > tolerance:
        raise ValueError(f"score reproduction differs by {maximum:g} > {tolerance:g}")
    return {
        "rows": len(names),
        "max_absolute_difference": maximum,
        "mean_absolute_difference": float(difference.mean()),
        "different_rounded_scores": int(np.count_nonzero(difference)),
        "tolerance": tolerance,
        "recomputed_tsv": fingerprint(recomputed),
        "note": "Score reproduction is not an accuracy evaluation.",
    }


def input_manifest(folder: Path) -> list[dict[str, object]]:
    """Fingerprint the exact audio bytes and expose the SSL prefix coverage."""
    rows = []
    for path in audio_paths(folder):
        info = sf.info(path)
        rows.append(
            {
                "filename": path.name,
                **fingerprint(path),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_s": info.frames / info.samplerate,
                "ssl_truncated": info.frames / info.samplerate > 12.0,
            }
        )
    return sorted(rows, key=lambda row: row["filename"])


def source_manifest(root: Path) -> dict[str, dict[str, object]]:
    """Hash inference/research source and the dependency/build contract."""
    paths = [
        *root.joinpath("hearsay").rglob("*.py"),
        *root.joinpath("forensics").rglob("*.py"),
        root / "forensics" / "calibration.json",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "Dockerfile",
        root / ".dockerignore",
    ]
    return {
        path.relative_to(root).as_posix(): fingerprint(path)
        for path in sorted(paths)
        if path.is_file()
    }


def make_release(args: argparse.Namespace) -> dict[str, object]:
    """Validate and write a release manifest plus a hashed audio inventory."""
    validation = validate_submission(
        args.scores, template=args.template, folder=args.audio_dir
    )
    comparisons = [
        compare_scores(args.scores, path, args.tolerance) for path in args.recomputed
    ]
    inputs = input_manifest(args.audio_dir)
    input_bytes = (json.dumps(inputs, indent=2) + "\n").encode()
    root = Path(__file__).resolve().parents[1]
    detector = Detector.load(args.model)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "code_revision_at_validation": revision.stdout.strip() or None,
        "model": fingerprint(args.model),
        "submission": fingerprint(args.scores),
        "template": fingerprint(args.template),
        "score_direction": "0=bonafide, 1=synthetic",
        "coverage": {k: v for k, v in validation.items() if k != "coverage"},
        "backbone": {
            "model_id": detector.ssl_model,
            "revision": model_revision(detector.ssl_model),
            "transformer_blocks": detector.n_layers,
            "pooled_layers": list(detector.linear.layers),
            "ssl_max_seconds": 12.0,
        },
        "input_manifest": {
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
            "files": len(inputs),
            "ssl_truncated_files": sum(row["ssl_truncated"] for row in inputs),
        },
        "reproduction": comparisons,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "transformers",
                    "numpy",
                    "scikit-learn",
                    "lightgbm",
                ]
            },
        },
        "source_files": source_manifest(root),
        "accuracy": "Unknown: official test labels are not available.",
        "evaluation_contract": {
            "archive_spoof_prior": 0.5,
            "alternative_spoof_prior": 0.3,
            "alternative_status": (
                "Alternative prior sensitivity analysis; not an organizer-confirmed setting."
            ),
            "cost_readings": ["analyst", "as_coded"],
            "note": "Both readings remain explicit; no official ranking is claimed.",
        },
    }
    if args.backbone_file:
        report["backbone"]["weights"] = fingerprint(args.backbone_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    inputs_out = args.out.with_name(args.out.stem + ".inputs.json")
    inputs_out.write_bytes(input_bytes)
    report["input_manifest"]["file"] = inputs_out.name
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Create a release record after successful fresh inference runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--recomputed", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--backbone-file", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = make_release(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"manifest": str(args.out), "coverage": report["coverage"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
