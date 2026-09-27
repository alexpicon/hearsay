# Author: Alex Picon <alexnpc@me.com>
"""Plain-language wording for the detector's evidence."""

import re

FIXED = {
    "flatness_mean": "spectral flatness (how noise-like the spectrum is)",
    "flatness_std": "variation of spectral flatness over time",
    "hi_flatness_mean": "flatness of the 4-7 kHz band (vocoder noise shaping)",
    "centroid_mean": "spectral centroid (brightness)",
    "centroid_std": "variation of brightness over time",
    "rolloff95_mean": "frequency below which 95% of the energy lies",
    "energy_range_db": "loudness dynamic range between speech and pauses",
    "pause_ratio": "fraction of frames that are pauses or near-silence",
    "frame_energy_std": "frame-to-frame loudness variation",
    "dc_offset": "DC offset of the waveform",
    "crest_factor_db": "peak-to-RMS ratio",
    "clip_ratio": "fraction of clipped samples",
    "zcr_mean": "zero-crossing rate (fricative and noise content)",
    "zcr_std": "variation of the zero-crossing rate",
    "spectral_flux_mean": "average frame-to-frame spectral change",
    "spectral_flux_std": "burstiness of frame-to-frame spectral change",
    "duration_s": "clip duration",
    "lead_silence_s": "leading silence",
    "trail_silence_s": "trailing silence",
}


def describe_feature(name: str) -> str:
    """Human-readable meaning of a hand-crafted feature name.

    Args:
        name: A name from ``hearsay.spectral.feature_names()``.

    Returns:
        A short description.

    """
    if name in FIXED:
        return FIXED[name]
    if m := re.fullmatch(r"(d?)lfcc(\d+)_(mean|std)", name):
        coef = f"cepstral coefficient {m.group(2)} (spectral envelope detail)"
        if m.group(1):
            return f"frame-to-frame jitter of {coef}"
        return (
            f"{'average' if m.group(3) == 'mean' else 'variation over time'} of {coef}"
        )
    if m := re.fullmatch(r"band_(\d+)_(\d+)_(mean|std)", name):
        stat = "share of energy" if m.group(3) == "mean" else "variability of energy"
        return f"{stat} in the {m.group(1)}-{m.group(2)} Hz band"
    return name


def summarize(
    p: float,
    ssl_p: float,
    spec_p: float,
    layers: dict[str, float],
    fusion_weights: dict[str, float] | None = None,
) -> str:
    """Describe scores and head influence without assigning another verdict.

    Args:
        p: Fused probability of synthetic speech.
        ssl_p: Probability from the self-supervised head.
        spec_p: Probability from the hand-crafted head.
        layers: Per-layer logit shares of the self-supervised head.
        fusion_weights: Exact, unrounded logit weights of the two heads.

    Returns:
        Neutral score evidence; the caller reports its threshold-based verdict.

    """
    weights = fusion_weights or {}
    lines = [f"Fused synthetic score: {p:.2f}."]
    for label, key, score in (
        ("SSL", "ssl_head", ssl_p),
        ("Spectral", "spectral_head", spec_p),
    ):
        weight = weights.get(key)
        influence = ""
        if weight == 0:
            influence = (
                "; fusion weight 0, diagnostic only with zero influence "
                "on the fused score"
            )
        elif weight is not None:
            influence = f"; logit fusion weight {weight:g}"
        lines.append(f"{label} head synthetic score: {score:.2f}{influence}.")
    if layers:
        strongest = max(layers, key=lambda key: abs(layers[key]))
        lines.append(
            "Largest absolute SSL head layer contribution: "
            f"{strongest.replace('_', ' ')}."
        )
    return " ".join(lines)
