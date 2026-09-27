# Author: Alex Picon <alexnpc@me.com>
"""Strict filename, score and coverage checks for HEARSAY release TSVs."""

import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm"}


def validate_filenames(
    values: Iterable[str], *, source: str = "filenames"
) -> list[str]:
    """Require nonempty, unique names without altering their spelling."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{source} must be a sequence of filenames")
    names = list(values)
    if not names:
        raise ValueError(f"{source} contains no filenames")
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or any(char in name for char in "\t\r\n\0")
        ):
            raise ValueError(f"{source} contains an empty or invalid filename")
    duplicate = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate:
        raise ValueError(f"{source} contains duplicate filenames: {duplicate[:5]}")
    return names


def validate_probabilities(values: Iterable[float]) -> np.ndarray:
    """Require a nonempty finite one-dimensional score vector in [0, 1]."""
    try:
        raw = np.asarray(values if isinstance(values, np.ndarray) else list(values))
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be a sequence of real numbers in [0, 1]") from exc
    if np.iscomplexobj(raw):
        raise ValueError("scores must be finite real numbers in [0, 1]")
    try:
        scores = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be finite real numbers in [0, 1]") from exc
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite one-dimensional vector")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be in [0, 1]; values are not clipped")
    return scores


def _read_tsv(path: Path, required: tuple[str, ...]) -> dict[str, list[str]]:
    """Read literal TSV fields and reject malformed rows or duplicate headers."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        try:
            header = next(reader)
            if len(set(header)) != len(header) or not set(required) <= set(header):
                raise ValueError(f"{path}: require unique columns {required}")
            columns: dict[str, list[str]] = {name: [] for name in header}
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"{path}: malformed TSV row {reader.line_num}")
                for name, value in zip(header, row, strict=True):
                    columns[name].append(value)
        except StopIteration as exc:
            raise ValueError(f"{path}: empty TSV") from exc
        except csv.Error as exc:
            raise ValueError(f"{path}: malformed TSV: {exc}") from exc
    columns["filename"] = validate_filenames(columns["filename"], source=str(path))
    return columns


def read_scores(path: Path) -> tuple[list[str], np.ndarray]:
    """Read unique literal filenames and validated synthetic probabilities."""
    table = _read_tsv(path, ("filename", "cm-score"))
    return table["filename"], validate_probabilities(table["cm-score"])


def read_key(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a key accepting only the exact labels ``bonafide`` and ``spoof``."""
    table = _read_tsv(path, ("filename", "cm-label"))
    labels = table["cm-label"]
    invalid = sorted(set(labels) - {"bonafide", "spoof"})
    if invalid:
        raise ValueError(f"{path}: invalid cm-label values: {invalid[:5]}")
    return table["filename"], np.array(
        [label == "spoof" for label in labels], dtype=int
    )


def template_filenames(path: Path) -> list[str]:
    """Read the template's literal IDs without interpreting placeholder scores."""
    return _read_tsv(path, ("filename",))["filename"]


def audio_paths(folder: Path) -> list[Path]:
    """List visible audio paths recursively, rejecting duplicate basenames."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"audio folder does not exist or is not a directory: {folder}")
    paths = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(folder).parts)
        and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    validate_filenames([path.name for path in paths], source=f"audio folder {folder}")
    return paths


def audio_filenames(folder: Path) -> list[str]:
    """Return the exact release IDs from the shared validated audio scan."""
    return [path.name for path in audio_paths(folder)]


def require_coverage(
    actual: Iterable[str], expected: Iterable[str], *, source: str
) -> None:
    """Require every expected ID exactly once and reject unexpected IDs."""
    actual_ids = set(validate_filenames(actual, source="score filenames"))
    expected_ids = set(validate_filenames(expected, source=source))
    missing, extra = (
        sorted(expected_ids - actual_ids),
        sorted(actual_ids - expected_ids),
    )
    if missing or extra:
        raise ValueError(
            f"filename coverage differs from {source}: "
            f"missing {len(missing)} {missing[:5]}; extra {len(extra)} {extra[:5]}"
        )


def validate_submission(
    scores_path: Path, *, template: Path | None = None, folder: Path | None = None
) -> dict[str, object]:
    """Validate release scores against a template, an audio folder, or both."""
    if template is None and folder is None:
        raise ValueError("provide a template and/or audio folder for exact coverage")
    names, scores = read_scores(scores_path)
    coverage = {}
    if template is not None:
        require_coverage(
            names, template_filenames(template), source=f"template {template}"
        )
        coverage["template"] = str(template)
    if folder is not None:
        require_coverage(
            names, audio_filenames(folder), source=f"audio folder {folder}"
        )
        coverage["folder"] = str(folder)
    return {
        "valid": True,
        "rows": len(names),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_direction": "0=bonafide, 1=synthetic",
        "coverage": coverage,
    }
