# Author: Alex Picon <alexnpc@me.com>
"""Train, score, evaluate and validate HEARSAY submissions."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from hearsay.corpus import CHANNELS, SETS, build_manifest, feature_path, model_tag
from hearsay.ssl import DEFAULT_MODEL

logger = logging.getLogger("hearsay")


def _extract(a: argparse.Namespace) -> None:
    """Extract (and cache) features for a named set."""
    from hearsay.extract import extract

    spec = SETS[a.set]
    m = build_manifest(spec, seed=a.seed)
    m = m.iloc[a.shard :: a.n_shards] if a.n_shards > 1 else m
    out = feature_path(a.set, model_tag(a.model), a.shard, a.n_shards)
    logger.info("%s: %d clips -> %s", a.set, len(m), out)
    extract(
        m,
        out,
        model=a.model,
        n_layers=a.layers,
        augment=spec.augment,
        seed=a.seed,
        max_seconds=a.max_seconds if spec.augment else 12.0,
        batch=a.batch if spec.augment else 1,
        threads=a.threads,
        channel=CHANNELS[spec.channel],
    )


def _feature_files(names: str, tag: str) -> list[Path]:
    """All cache files (sharded or not) of comma-separated set names."""
    paths: list[Path] = []
    for name in names.split(","):
        whole = feature_path(name, tag)
        shards = sorted(whole.parent.glob(f"{name}.*of*.npz"))
        paths += [whole] if whole.exists() else shards
        if not whole.exists() and not shards:
            msg = f"no cached features for set {name!r} ({whole})"
            raise FileNotFoundError(msg)
    return paths


def _train(a: argparse.Namespace) -> None:
    """Fit the detector, evaluate on the dev sets and save the bundle."""
    from hearsay.model import Features
    from hearsay.train import composition, default_score, evaluate, fit_detector

    tag = model_tag(a.model)
    train = Features.load(_feature_files(a.train, tag))
    layers = tuple(int(x) for x in a.use_layers.split(","))
    logger.info(
        "training on %d clips (%d synthetic)", len(train), int(train.label.sum())
    )
    det = fit_detector(
        train, ssl_model=a.model, n_layers=a.layers, layers=layers, c=a.c
    )
    report: dict = {
        "train_sets": a.train.split(","),
        "n_train": len(train),
        "layers": list(layers),
        "c": a.c,
        "ssl_model": a.model,
    }
    if a.dev:
        dev = Features.load(_feature_files(a.dev, tag))
        report["dev_sets"] = a.dev.split(",")
        report["dev"] = evaluate(det, dev)
        det.thresholds = report["dev"]["operating_points"]
        det.default_score = default_score(det.thresholds)
    report["operating_points"] = det.thresholds
    report["default_score"] = det.default_score
    report["oof_attack_disjoint"] = det.meta.get("oof_attack_disjoint")
    report["train_composition"] = composition(train)
    det.meta = report
    det.save(Path(a.out))
    if a.metrics:
        Path(a.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(a.metrics).write_text(json.dumps(report, indent=2, default=float))


def _evaluate(a: argparse.Namespace) -> None:
    """Evaluate scores only when both TSVs cover exactly the same unique IDs."""
    from hearsay.metrics import evaluate_scores
    from hearsay.submission import read_key, read_scores, require_coverage

    names, scores = read_scores(Path(a.scores))
    key_names, labels = read_key(Path(a.key))
    require_coverage(names, key_names, source="evaluation key")
    by_name = dict(zip(names, scores, strict=True))
    aligned = np.array([by_name[name] for name in key_names])
    report = evaluate_scores(aligned, labels, p_spoof=getattr(a, "spoof_prior", 0.5))
    print(json.dumps(report, indent=2))


def _predict(a: argparse.Namespace) -> None:
    """Score a folder and write the submission TSV."""
    from hearsay.inference import score_directory

    res = score_directory(Path(a.folder), Path(a.out), a.model_path, a.threads)
    p = np.array([r["probability"] for r in res])
    logger.info(
        "%d files, mean score %.3f, %d undecodable",
        len(res),
        p.mean(),
        sum(r["error"] is not None for r in res),
    )


def _profile(a: argparse.Namespace) -> None:
    """Write the channel profile of a folder (JSON summary, optional CSV)."""
    from hearsay.profiling import profile_folder

    df, summary = profile_folder(Path(a.folder), a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=float))
    if a.csv:
        df.to_csv(a.csv, index=False)
    print(json.dumps(summary["percentiles"], indent=1, default=float))


def _submit(a: argparse.Namespace) -> None:
    """Write the submission TSV from cached (unaugmented) test features."""
    from hearsay.inference import write_tsv
    from hearsay.model import Detector, Features
    from hearsay.submission import (
        require_coverage,
        template_filenames,
        validate_filenames,
        validate_probabilities,
    )

    det = Detector.load(Path(a.model_path)) if a.model_path else Detector.load()
    f = Features.load(_feature_files(a.set, model_tag(det.ssl_model)))
    ids = validate_filenames(f.uid, source="cached features")
    names = template_filenames(Path(a.template)) if a.template else ids
    require_coverage(ids, names, source="submission template")
    p = validate_probabilities(det.predict(f.ssl, f.spec))
    rows = dict(zip(ids, p, strict=True))
    write_tsv(
        [{"file": n, "probability": float(rows[n])} for n in sorted(names)],
        Path(a.out),
    )


def _validate_submission(a: argparse.Namespace) -> None:
    """Check release scores and exact coverage without loading a detector."""
    from hearsay.submission import validate_submission

    report = validate_submission(
        Path(a.scores),
        template=Path(a.template) if a.template else None,
        folder=Path(a.folder) if a.folder else None,
    )
    print(json.dumps(report, indent=2))


def _score(a: argparse.Namespace) -> None:
    """Print the full evidence dictionary of one file."""
    from hearsay.inference import score_file

    print(json.dumps(score_file(a.file, a.model_path), indent=2))


def main() -> None:
    """Entry point of the ``hearsay`` command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="hearsay", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="cache features of a named set")
    p.add_argument("--set", required=True, choices=sorted(SETS))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=4.5)
    p.add_argument("--batch", type=int, default=8)
    p.set_defaults(func=_extract)

    p = sub.add_parser("train", help="fit, evaluate on dev and save the detector")
    p.add_argument("--train", required=True, help="comma-separated set names")
    p.add_argument("--dev", default="", help="comma-separated held-out set names")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--use-layers", default="3,4,5,6,7")
    p.add_argument("--c", type=float, default=0.01)
    p.add_argument("--out", default="models/detector.joblib")
    p.add_argument("--metrics", default="")
    p.set_defaults(func=_train)

    p = sub.add_parser("evaluate", help="both minDCF readings of a TSV against a key")
    p.add_argument("--scores", required=True)
    p.add_argument("--key", required=True, help="TSV with filename and cm-label")
    p.add_argument("--spoof-prior", type=float, default=0.5, help="default: 0.5")
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("predict", help="score a folder into the submission TSV")
    p.add_argument("folder")
    p.add_argument("--out", required=True)
    p.add_argument("--model-path", default=None)
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=_predict)

    p = sub.add_parser("profile", help="channel profile of a folder of clips")
    p.add_argument("folder")
    p.add_argument("--out", required=True, help="JSON summary path")
    p.add_argument("--csv", default="", help="optional per-clip CSV path")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_profile)

    p = sub.add_parser("submit", help="submission TSV from cached test features")
    p.add_argument("--set", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--template", default="", help="answer-key template to cover")
    p.add_argument("--model-path", default=None)
    p.set_defaults(func=_submit)

    p = sub.add_parser("validate-submission", help="validate scores and exact coverage")
    p.add_argument("--scores", required=True)
    p.add_argument("--template", default="", help="TSV containing expected filenames")
    p.add_argument("--folder", default="", help="folder containing expected audio")
    p.set_defaults(func=_validate_submission)

    p = sub.add_parser("score", help="explain the score of one file")
    p.add_argument("file")
    p.add_argument("--model-path", default=None)
    p.set_defaults(func=_score)
    args = ap.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
