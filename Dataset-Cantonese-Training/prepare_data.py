# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Filters audio to Qwen3-TTS recommended length (default 5–30s), records
# duration_sec on each kept sample, and writes a per-audio length CSV.

import argparse
import json
import os

from qwen_tts import Qwen3TTSTokenizer

from audio_duration import check_samples, print_duration_summary, save_duration_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--min_duration", type=float, default=5.0,
                        help="Minimum training audio length in seconds (default: 5)")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Maximum training audio length in seconds (default: 30)")
    parser.add_argument("--duration_report", type=str, default=None,
                        help="CSV path for per-audio length report (default: next to output)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with error if any sample is rejected by the duration filter")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Tokenizer encode batch size (lower = less VRAM; default: 4, try 1–2 on 12GB GPUs)")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        total_lines = [json.loads(line.strip()) for line in f.readlines() if line.strip()]

    print(f"Total samples: {len(total_lines)}")

    kept_lines, duration_df = check_samples(
        total_lines,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    print_duration_summary(duration_df, args.min_duration, args.max_duration)

    rejected_n = int((duration_df["status"] != "kept").sum())
    if args.strict and rejected_n > 0:
        raise SystemExit(f"--strict: rejected {rejected_n} samples outside [{args.min_duration}, {args.max_duration}]s")

    if len(kept_lines) == 0:
        raise SystemExit("No samples left after duration filter. Relax --min_duration/--max_duration or fix audio paths.")

    report_path = args.duration_report
    if report_path is None:
        base, _ = os.path.splitext(args.output_jsonl)
        report_path = f"{base}_audio_lengths.csv"
    duration_df = duration_df.copy()
    duration_df.loc[duration_df["status"] == "kept", "split"] = "train"
    save_duration_report(duration_df, report_path)

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    print(f"Encoding audio_codes (batch_size={args.batch_size})...")
    final_lines = []
    batch_lines = []
    batch_audios = []
    for line in kept_lines:
        batch_lines.append(line)
        batch_audios.append(line["audio"])

        if len(batch_lines) >= args.batch_size:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, sample in zip(enc_res.audio_codes, batch_lines):
                sample["audio_codes"] = code.cpu().tolist()
                final_lines.append(sample)
            batch_lines.clear()
            batch_audios.clear()

    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, sample in zip(enc_res.audio_codes, batch_lines):
            sample["audio_codes"] = code.cpu().tolist()
            final_lines.append(sample)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for line in final_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"✅ Saved: {args.output_jsonl} ({len(final_lines)} samples)")
    print(f"   → Duration report: {report_path}")


if __name__ == "__main__":
    main()
