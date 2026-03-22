# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Grok: Now prepares BOTH training (with audio_codes) + evaluation (clean) datasets
# Usage example:
# python prepare_data.py \
#   --input_jsonl raw_hk_cantonese.jsonl \
#   --output_train_jsonl train_prepared.jsonl \
#   --output_eval_jsonl eval_prepared.jsonl \
#   --eval_ratio 0.1 \
#   --speaker_name hk_cantonese_speaker

import argparse
import json
import random
from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 16  # Reduce if you run out of GPU memory

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
    
    args = parser.parse_args()

    # ==================== Load raw data ====================
    print(f"Loading raw data from: {args.input_jsonl}")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        total_lines = [json.loads(line.strip()) for line in f.readlines()]

    print(f"Total samples: {len(total_lines)}")

    # ==================== Train/Eval Split ====================
    random.seed(args.seed)
    random.shuffle(total_lines)

    split_idx = int(len(total_lines) * (1 - args.eval_ratio))
    train_data = total_lines[:split_idx]
    eval_data = total_lines[split_idx:]

    print(f"Train samples: {len(train_data)} | Eval samples: {len(eval_data)}")

    # ==================== Prepare TRAINING dataset (with audio_codes) ====================
    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    print("Encoding audio_codes for TRAINING dataset (this may take a while)...")
    final_train_lines = []
    batch_lines = []
    batch_audios = []

    for line in train_data:
        batch_lines.append(line.copy())           # copy to avoid modifying original
        batch_audios.append(line['audio'])

        if len(batch_lines) >= BATCH_INFER_NUM:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, sample in zip(enc_res.audio_codes, batch_lines):
                sample['audio_codes'] = code.cpu().tolist()
                if args.speaker_name:
                    sample['speaker_name'] = args.speaker_name
                final_train_lines.append(sample)
            batch_lines.clear()
            batch_audios.clear()

    # Last batch
    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, sample in zip(enc_res.audio_codes, batch_lines):
            sample['audio_codes'] = code.cpu().tolist()
            if args.speaker_name:
                sample['speaker_name'] = args.speaker_name
            final_train_lines.append(sample)

    # Save training jsonl
    with open(args.output_train_jsonl, 'w', encoding='utf-8') as f:
        for line in final_train_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f"✅ Training dataset saved: {args.output_train_jsonl} ({len(final_train_lines)} samples)")

    # ==================== Prepare EVALUATION dataset (clean, no codes) ====================
    if args.output_eval_jsonl:
        final_eval_lines = []
        for line in eval_data:
            sample = line.copy()
            if args.speaker_name:
                sample['speaker_name'] = args.speaker_name
            # No audio_codes needed for evaluation
            final_eval_lines.append(sample)

        with open(args.output_eval_jsonl, 'w', encoding='utf-8') as f:
            for line in final_eval_lines:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

        print(f"✅ Evaluation dataset saved: {args.output_eval_jsonl} ({len(final_eval_lines)} samples)")

    print("\n🎉 All datasets prepared successfully!")
    print(f"   → Use --train_jsonl {args.output_train_jsonl}")
    print(f"   → Use --val_jsonl or --test_jsonl {args.output_eval_jsonl or 'N/A'}")

if __name__ == "__main__":
    main()