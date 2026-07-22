# evaluate_qwen3_tts.py
# FINAL STABLE VERSION — Manual WER/CER for Cantonese + English code-mixing
# Adapted for new eval_prepared.jsonl format: uses "ref_audio" for speaker similarity

import argparse
import json
import os
import torch
import pandas as pd
import soundfile as sf
import torchaudio
import warnings
import re
from transformers import pipeline
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from tqdm.auto import tqdm
import jiwer

warnings.filterwarnings("ignore")


# ====================== Lazy-loaded tools ======================
asr_pipeline = None
speaker_encoder = None
utmos_model = None


def clean_reference_text(text_field: str) -> str:
    """Remove training prefix if present (for compatibility with ASR-style data)"""
    if "<asr_text>" in text_field:
        return text_field.split("<asr_text>", 1)[1].strip()
    return text_field.strip()


def tokenize_mixed_text(text: str):
    """Custom tokenizer for Cantonese + English code-mixing"""
    text = re.sub(r'([a-zA-Z0-9\'\-]+)', r' \1 ', text)   # space around English words
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()


def get_asr_pipeline():
    global asr_pipeline
    if asr_pipeline is None:
        print("🔄 Loading Whisper-large-v3-turbo (excellent for HK Cantonese)...")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    return asr_pipeline


def get_speaker_encoder():
    global speaker_encoder
    if speaker_encoder is None:
        try:
            # === MONKEY-PATCH for new torchaudio ===
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]
            
            print("🔄 Loading ECAPA-TDNN speaker encoder...")
            from speechbrain.pretrained import EncoderClassifier
            speaker_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
            )
            print("✅ Speaker encoder loaded!")
        except Exception as e:
            print(f"⚠️ Speaker similarity skipped: {e}")
            return None
    return speaker_encoder


def get_utmos_model():
    global utmos_model
    if utmos_model is None:
        print("🔄 Loading UTMOS (first time only, may take 20-60s)...")
        try:
            utmos_model = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
            print("✅ UTMOS loaded!")
        except Exception as e:
            print(f"⚠️ UTMOS failed: {e}")
            return None
    return utmos_model


def evaluate_checkpoint(checkpoint_dir, test_jsonl, speaker_name, output_dir, language="chinese", 
                       max_samples=None, skip_speaker_sim=False, skip_utmos=False):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading fine-tuned model from: {checkpoint_dir}")
    model = Qwen3TTSModel.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else None
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(test_jsonl, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f.readlines()]
    if max_samples:
        test_data = test_data[:max_samples]
    print(f"✅ Loaded {len(test_data)} test samples (new format with ref_audio)")

    asr = get_asr_pipeline()
    spk_enc = get_speaker_encoder() if not skip_speaker_sim else None
    utmos = get_utmos_model() if not skip_utmos else None

    print("✅ All models ready!")
    print("🚀 Starting evaluation with MANUAL WER/CER (custom tokenizer for code-mixing)...\n")

    results = []
    total_cer = total_wer = total_sim = total_utmos = 0.0
    sim_available = not skip_speaker_sim
    utmos_available = utmos is not None

    for i, sample in tqdm(enumerate(test_data), total=len(test_data),
                          desc="Generating & Evaluating", unit="sample"):
        text = sample["text"]
        # === NEW: use "ref_audio" from the dataset (not "audio") ===
        ref_audio_path = sample["ref_audio"]

        # === Generate with custom voice ===
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker_name,
            max_new_tokens=2048
        )
        gen_wav = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]

        # Save generated audio
        gen_path = os.path.join(output_dir, f"sample_{i:03d}.wav")
        sf.write(gen_path, gen_wav, sr)

        # === ASR Transcription (in-memory — torchcodec-free) ===
        # Whisper expects 16 kHz; TTS output is typically 24 kHz. Without
        # resampling, mel length is overstated and >~20s audio hits the
        # 3000-frame / 30s long-form limit.
        gen_wav_16k = torchaudio.functional.resample(
            torch.from_numpy(gen_wav).float().unsqueeze(0),
            orig_freq=sr,
            new_freq=16000,
        ).squeeze(0).numpy()
        transcription = asr(
            {"array": gen_wav_16k, "sampling_rate": 16000},
            return_timestamps=True,
            generate_kwargs={"language": "yue", "task": "transcribe"},
        )["text"]
        pred_text = transcription.strip()

        # === Clean reference & compute MANUAL WER / CER ===
        ref_text = clean_reference_text(text)

        # WORD-LEVEL (WER) — custom tokenizer for Cantonese + English code-mixing
        ref_tokens = tokenize_mixed_text(ref_text)
        pred_tokens = tokenize_mixed_text(pred_text)

        ref_str = " ".join(ref_tokens)
        pred_str = " ".join(pred_tokens)

        wer_output = jiwer.process_words(ref_str, pred_str)
        wer_sub = wer_output.substitutions
        wer_del = wer_output.deletions
        wer_ins = wer_output.insertions
        wer_errors = wer_sub + wer_del + wer_ins
        ref_words = len(ref_tokens) or 1
        wer = wer_errors / ref_words if ref_words > 0 else 0.0

        ref_words_array = ", ".join(ref_tokens)
        pred_words_array = ", ".join(pred_tokens)

        # CHARACTER-LEVEL (CER) — manual
        cer_output = jiwer.process_characters(ref_text, pred_text)
        cer_sub = cer_output.substitutions
        cer_del = cer_output.deletions
        cer_ins = cer_output.insertions
        cer_errors = cer_sub + cer_del + cer_ins
        ref_chars = len(cer_output.references[0]) if cer_output.references else 1
        cer = cer_errors / ref_chars if ref_chars > 0 else 0.0

        ref_chars_array = ", ".join(cer_output.references[0]) if cer_output.references else ""
        pred_chars_array = ", ".join(cer_output.hypotheses[0]) if cer_output.hypotheses else ""

        # === Speaker Similarity ===
        if spk_enc is not None:
            try:
                audio_data, orig_sr = sf.read(ref_audio_path)
                if len(audio_data.shape) == 1:
                    audio_data = audio_data[None, :]
                ref_sig = torch.from_numpy(audio_data).float().to(spk_enc.device)
                ref_emb = spk_enc.encode_batch(ref_sig)
                
                gen_sig = torch.from_numpy(gen_wav).unsqueeze(0).float().to(spk_enc.device)
                gen_emb = spk_enc.encode_batch(gen_sig)
                
                sim = torch.nn.functional.cosine_similarity(ref_emb.mean(1), gen_emb.mean(1)).item()
            except Exception as e:
                print(f"⚠️ Speaker sim failed for sample {i}: {e}")
                sim = 0.0
        else:
            sim = 0.0
            sim_available = False

        # === UTMOS ===
        if utmos is not None:
            waveform_16k = torchaudio.functional.resample(
                torch.from_numpy(gen_wav).unsqueeze(0), orig_freq=sr, new_freq=16000
            )
            utmos_score = utmos(waveform_16k, 16000).item()
        else:
            utmos_score = 0.0

        results.append({
            "index": i,
            "text": text,
            "ref_text": ref_text,
            "pred_text": pred_text,
            "ref_audio": ref_audio_path,
            "gen_audio": gen_path,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "speaker_sim": round(sim, 4) if spk_enc else "N/A",
            "utmos": round(utmos_score, 3) if utmos_available else "N/A",
            # === Debug columns (same as ASR evaluator) ===
            "ref_words_array": ref_words_array,
            "pred_words_array": pred_words_array,
            "ref_words": ref_words,
            "wer_substitutions": wer_sub,
            "wer_deletions": wer_del,
            "wer_insertions": wer_ins,
            "wer_errors_(S+D+I)": wer_errors,
            "ref_chars_array": ref_chars_array,
            "pred_chars_array": pred_chars_array,
            "ref_chars": ref_chars,
            "cer_substitutions": cer_sub,
            "cer_deletions": cer_del,
            "cer_insertions": cer_ins,
            "cer_errors_(S+D+I)": cer_errors,
        })

        total_cer += cer
        total_wer += wer
        total_sim += sim
        total_utmos += utmos_score

    # === Summary ===
    n = len(results)
    avg_cer = total_cer / n if n > 0 else 0
    avg_wer = total_wer / n if n > 0 else 0
    avg_sim = total_sim / n if sim_available else "N/A"
    avg_utmos = total_utmos / n if utmos_available else "N/A"

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False, encoding="utf-8")

    summary = {
        "checkpoint": checkpoint_dir,
        "test_samples": n,
        "avg_cer": round(avg_cer, 4),
        "avg_wer": round(avg_wer, 4),
        "avg_speaker_similarity": avg_sim,
        "avg_utmos": avg_utmos,
        "language": language,
        "speaker_name": speaker_name,
        "asr_model": "whisper-large-v3-turbo",
        "calculation_method": "manual_jiwer_code_mixing"
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE — MANUAL WER/CER with custom tokenizer")
    print("="*80)
    print(f"Checkpoint       : {checkpoint_dir}")
    print(f"Test samples     : {n}")
    print(f"Avg CER          : {avg_cer:.4f}  (lower = better)")
    print(f"Avg WER          : {avg_wer:.4f}  (lower = better) ← accurate for Cantonese+English code-mixing")
    print(f"Avg Speaker SIM  : {avg_sim}  (higher = better)")
    print(f"Avg UTMOS        : {avg_utmos}  (higher = better)")
    print(f"Results saved to : {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Evaluation (manual WER/CER for code-mixing + new dataset format)")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--speaker_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--language", type=str, default="chinese")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_speaker_sim", action="store_true")
    parser.add_argument("--skip_utmos", action="store_true")

    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint_dir,
        args.test_jsonl,
        args.speaker_name,
        args.output_dir,
        args.language,
        args.max_samples,
        args.skip_speaker_sim,
        args.skip_utmos
    )