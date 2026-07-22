# coding=utf-8
"""Audio duration checks for Qwen3-TTS fine-tuning (recommended 5–30 seconds)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import soundfile as sf


def get_audio_duration(path: str) -> float:
    """Return duration in seconds without fully decoding the file."""
    info = sf.info(path)
    return float(info.duration)


def check_samples(
    samples: List[Dict[str, Any]],
    min_duration: float = 5.0,
    max_duration: float = 30.0,
    audio_key: str = "audio",
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Probe each sample's audio length and keep only those in [min_duration, max_duration].

    Returns:
        kept: copies of valid samples with ``duration_sec`` set
        df: DataFrame with one row per input sample
    """
    rows: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        audio = sample.get(audio_key)
        text = sample.get("text", "")
        text_preview = text if not isinstance(text, str) else (text[:80] + ("…" if len(text) > 80 else ""))
        row: Dict[str, Any] = {
            "index": idx,
            "audio": audio,
            "ref_audio": sample.get("ref_audio"),
            "text": text_preview,
            "duration_sec": None,
            "status": "kept",
            "split": "n/a",
            "reason": "",
        }

        if not audio or not isinstance(audio, str):
            row["status"] = "missing"
            row["reason"] = f"missing or invalid '{audio_key}' path"
            rows.append(row)
            continue

        if not os.path.isfile(audio):
            row["status"] = "missing"
            row["reason"] = "file not found"
            rows.append(row)
            continue

        try:
            duration = get_audio_duration(audio)
        except Exception as exc:  # noqa: BLE001 — report any probe failure
            row["status"] = "error"
            row["reason"] = str(exc)
            rows.append(row)
            continue

        row["duration_sec"] = round(duration, 4)

        if duration < min_duration:
            row["status"] = "too_short"
            row["reason"] = f"{duration:.2f}s < {min_duration}"
            rows.append(row)
            continue

        if duration > max_duration:
            row["status"] = "too_long"
            row["reason"] = f"{duration:.2f}s > {max_duration}"
            rows.append(row)
            continue

        row["reason"] = f"{duration:.2f}s in [{min_duration}, {max_duration}]"
        rows.append(row)

        kept_sample = dict(sample)
        kept_sample["duration_sec"] = row["duration_sec"]
        kept.append(kept_sample)

    df = pd.DataFrame(rows, columns=[
        "index", "audio", "ref_audio", "text", "duration_sec", "status", "split", "reason",
    ])
    return kept, df


def assign_split(
    df: pd.DataFrame,
    train_samples: List[Dict[str, Any]],
    eval_samples: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Mark split column for kept samples using audio path identity."""
    train_paths = {s.get("audio") for s in train_samples}
    eval_paths = {s.get("audio") for s in eval_samples}

    def _split_for(row) -> str:
        if row["status"] != "kept":
            return "n/a"
        path = row["audio"]
        if path in train_paths:
            return "train"
        if path in eval_paths:
            return "eval"
        return "kept"

    out = df.copy()
    out["split"] = out.apply(_split_for, axis=1)
    return out


def print_duration_summary(df: pd.DataFrame, min_duration: float, max_duration: float) -> None:
    counts = df["status"].value_counts().to_dict()
    kept = df[df["status"] == "kept"]
    print("\n=== Audio duration filter (Qwen3-TTS recommends 5–30s) ===")
    print(f"Range kept: [{min_duration}, {max_duration}] seconds")
    print(f"Total probed: {len(df)}")
    for key in ("kept", "too_short", "too_long", "missing", "error"):
        if key in counts:
            print(f"  {key}: {counts[key]}")
    if len(kept) > 0:
        print(
            f"Kept duration_sec — min={kept['duration_sec'].min():.2f} "
            f"mean={kept['duration_sec'].mean():.2f} max={kept['duration_sec'].max():.2f}"
        )
    rejected = df[df["status"] != "kept"]
    if len(rejected) > 0:
        print("Examples rejected:")
        for _, r in rejected.head(5).iterrows():
            print(f"  [{r['status']}] {r['audio']} — {r['reason']}")
    print("=" * 56)


def save_duration_report(df: pd.DataFrame, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"✅ Audio length report saved: {path} ({len(df)} rows)")
