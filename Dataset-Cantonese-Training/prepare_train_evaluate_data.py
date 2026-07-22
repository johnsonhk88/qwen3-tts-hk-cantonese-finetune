# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Modified: prepares BOTH training (with audio_codes) + evaluation datasets,
# filters audio to Qwen3-TTS recommended length (default 5–30s), and writes
# a per-sample duration DataFrame/CSV report.
#
# Usage example:
# python prepare_train_evaluate_data.py \
#   --input_jsonl raw_hk_cantonese.jsonl \
#   --output_train_jsonl train_prepared.jsonl \
#   --output_eval_jsonl eval_prepared.jsonl \
#   --eval_ratio 0.1 \
#   --speaker_name hk_cantonese_speaker \
#   --min_duration 5.0 \
#   --max_duration 30.0

import argparse
import json
import os
import random

from qwen_tts import Qwen3TTSTokenizer

from audio_duration import (
    assign_split,
    check_samples,
    print_duration_summary,
    save_duration_report,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare BOTH training and evaluation datasets for Qwen3-TTS")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Raw input jsonl (must contain 'text' and 'audio' keys)")
    parser.add_argument("--output_train_jsonl", type=str, required=True,
                        help="Output for training (will add 'audio_codes')")
    parser.add_argument("--output_eval_jsonl", type=str, default=None,
                        help="Optional: Output for evaluation (clean, no audio_codes needed)")
    parser.add_argument("--eval_ratio", type=float, default=0.1,
                        help="Ratio of data to use for evaluation (0.0–1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split")
    parser.add_argument("--speaker_name", type=str, default=None,
                        help="Optional: Inject speaker_name into every sample")
    parser.add_argument("--min_duration", type=float, default=5.0,
                        help="Minimum training audio length in seconds (default: 5)")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Maximum training audio length in seconds (default: 30)")
    parser.add_argument("--duration_report", type=str, default=None,
                        help="CSV path for per-audio length report (default: next to train output)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with error if any sample is rejected by the duration filter")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Tokenizer encode batch size (lower = less VRAM; default: 4, try 1–2 on 12GB GPUs)")

    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")

    # ==================== Load raw data ====================
    print(f"Loading raw data from: {args.input_jsonl}")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        total_lines = [json.loads(line.strip()) for line in f.readlines() if line.strip()]

    print(f"Total samples: {len(total_lines)}")

    # ==================== Duration filter (5–30s) ====================
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

    # ==================== Train/Eval Split ====================
    random.seed(args.seed)
    random.shuffle(kept_lines)

    if args.output_eval_jsonl and args.eval_ratio > 0:
        split_idx = int(len(kept_lines) * (1 - args.eval_ratio))
        # ensure at least one train sample when possible
        split_idx = max(1, min(split_idx, len(kept_lines) - 1)) if len(kept_lines) > 1 else len(kept_lines)
        train_data = kept_lines[:split_idx]
        eval_data = kept_lines[split_idx:]
    else:
        train_data = kept_lines
        eval_data = []

    duration_df = assign_split(duration_df, train_data, eval_data)

    report_path = args.duration_report
    if report_path is None:
        base, _ = os.path.splitext(args.output_train_jsonl)
        report_path = f"{base}_audio_lengths.csv"
    save_duration_report(duration_df, report_path)

    print(f"Train samples: {len(train_data)} | Eval samples: {len(eval_data)}")

    # ==================== Prepare TRAINING dataset (with audio_codes) ====================
    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    print(f"Encoding audio_codes for TRAINING dataset (batch_size={args.batch_size})...")
    final_train_lines = []
    batch_lines = []
    batch_audios = []

    for line in train_data:
        batch_lines.append(line.copy())
        batch_audios.append(line["audio"])

        if len(batch_lines) >= args.batch_size:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, sample in zip(enc_res.audio_codes, batch_lines):
                sample["audio_codes"] = code.cpu().tolist()
                if args.speaker_name:
                    sample["speaker_name"] = args.speaker_name
                final_train_lines.append(sample)
            batch_lines.clear()
            batch_audios.clear()

    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, sample in zip(enc_res.audio_codes, batch_lines):
            sample["audio_codes"] = code.cpu().tolist()
            if args.speaker_name:
                sample["speaker_name"] = args.speaker_name
            final_train_lines.append(sample)

    with open(args.output_train_jsonl, "w", encoding="utf-8") as f:
        for line in final_train_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"✅ Training dataset saved: {args.output_train_jsonl} ({len(final_train_lines)} samples)")

    # ==================== Prepare EVALUATION dataset (clean, no codes) ====================
    if args.output_eval_jsonl:
        final_eval_lines = []
        for line in eval_data:
            sample = line.copy()
            if args.speaker_name:
                sample["speaker_name"] = args.speaker_name
            final_eval_lines.append(sample)

        with open(args.output_eval_jsonl, "w", encoding="utf-8") as f:
            for line in final_eval_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"✅ Evaluation dataset saved: {args.output_eval_jsonl} ({len(final_eval_lines)} samples)")

    print("\n🎉 All datasets prepared successfully!")
    print(f"   → Use --train_jsonl {args.output_train_jsonl}")
    print(f"   → Use --val_jsonl or --test_jsonl {args.output_eval_jsonl or 'N/A'}")
    print(f"   → Duration report: {report_path}")


if __name__ == "__main__":
    main()
