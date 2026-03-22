# evaluate_qwen3_tts.py
# Standalone post-training evaluation for Qwen3-TTS Hong Kong Cantonese custom voice
# Run after training: python evaluate_qwen3_tts.py --checkpoint_dir ...

import argparse
import json
import os
import torch
import pandas as pd
import soundfile as sf
import torchaudio
from transformers import pipeline
from speechbrain.pretrained import EncoderClassifier
import evaluate
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# ====================== Lazy load evaluation tools ======================
cer_metric = evaluate.load("cer")
asr_pipeline = None
speaker_encoder = None
utmos_model = None


def get_asr_pipeline():
    global asr_pipeline
    if asr_pipeline is None:
        print("Loading SenseVoiceSmall (best for HK Cantonese)...")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="FunAudioLLM/SenseVoiceSmall",
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return asr_pipeline


def get_speaker_encoder():
    global speaker_encoder
    if speaker_encoder is None:
        print("Loading ECAPA-TDNN speaker encoder...")
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
    return speaker_encoder


def get_utmos_model():
    global utmos_model
    if utmos_model is None:
        print("Loading UTMOS (naturalness scorer)...")
        utmos_model = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
    return utmos_model


def evaluate_checkpoint(checkpoint_dir, test_jsonl, speaker_name, output_dir, language="yue", max_samples=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading fine-tuned model from: {checkpoint_dir}")
    model = Qwen3TTSModel.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Load test data
    with open(test_jsonl, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f.readlines()]
    if max_samples:
        test_data = test_data[:max_samples]
    print(f"Evaluating {len(test_data)} test samples...")

    results = []
    total_cer = total_sim = total_utmos = 0.0

    asr = get_asr_pipeline()
    spk_enc = get_speaker_encoder()
    utmos = get_utmos_model()

    for i, sample in enumerate(test_data):
        text = sample["text"]          # ← change key if your jsonl uses different name
        ref_audio_path = sample["audio"]  # ← change key if needed

        print(f"[{i+1}/{len(test_data)}] Generating: {text[:60]}...")

        # === Generate with custom Hong Kong voice ===
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

        # === CER ===
        transcription = asr(gen_path, return_timestamps=False, generate_kwargs={"language": "yue"})["text"]
        cer = cer_metric.compute(predictions=[transcription], references=[text])

        # === Speaker Similarity ===
        ref_sig, _ = torchaudio.load(ref_audio_path)
        ref_emb = spk_enc.encode_batch(ref_sig.unsqueeze(0).to(spk_enc.device))
        gen_sig = torch.from_numpy(gen_wav).unsqueeze(0).to(spk_enc.device)
        gen_emb = spk_enc.encode_batch(gen_sig)
        sim = torch.nn.functional.cosine_similarity(ref_emb.mean(1), gen_emb.mean(1)).item()

        # === UTMOS (resample to 16kHz as required by model) ===
        waveform_16k = torchaudio.functional.resample(
            torch.from_numpy(gen_wav).unsqueeze(0), orig_freq=sr, new_freq=16000
        )
        utmos_score = utmos(waveform_16k, 16000).item()

        results.append({
            "index": i,
            "text": text,
            "ref_audio": ref_audio_path,
            "gen_audio": gen_path,
            "cer": round(cer, 4),
            "speaker_sim": round(sim, 4),
            "utmos": round(utmos_score, 3)
        })

        total_cer += cer
        total_sim += sim
        total_utmos += utmos_score

    # === Summary ===
    avg_cer = total_cer / len(results)
    avg_sim = total_sim / len(results)
    avg_utmos = total_utmos / len(results)

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    summary = {
        "checkpoint": checkpoint_dir,
        "test_samples": len(results),
        "avg_cer": round(avg_cer, 4),
        "avg_speaker_similarity": round(avg_sim, 4),
        "avg_utmos": round(avg_utmos, 3),
        "language": language,
        "speaker_name": speaker_name
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    print(f"Checkpoint       : {checkpoint_dir}")
    print(f"Test samples     : {len(results)}")
    print(f"Avg CER          : {avg_cer:.4f}  (lower = better)")
    print(f"Avg Speaker SIM  : {avg_sim:.4f}  (higher = better)")
    print(f"Avg UTMOS        : {avg_utmos:.3f}  (higher = better)")
    print(f"Results saved to : {output_dir}/")
    print("   • results.csv      (full table)")
    print("   • summary.json     (summary)")
    print("   • sample_000.wav   (all generated audios)")
    print("="*60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training evaluation for Qwen3-TTS custom voice")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint (e.g. output/checkpoint-epoch-9)")
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="Path to test jsonl (can be larger than validation)")
    parser.add_argument("--speaker_name", type=str, required=True,
                        help="Your speaker name (same as training)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Folder to save audios + report")
    parser.add_argument("--language", type=str, default="yue",
                        help="Language code (yue for HK Cantonese)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (for quick test)")

    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint_dir,
        args.test_jsonl,
        args.speaker_name,
        args.output_dir,
        args.language,
        args.max_samples
    )