# evaluate_qwen3_tts.py
# FINAL STABLE VERSION — Fixed torchcodec error + tqdm + detailed logging

import argparse
import json
import os
import torch
import pandas as pd
import soundfile as sf
import torchaudio
import warnings
from transformers import pipeline
import evaluate
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ====================== Lazy-loaded tools ======================
cer_metric = evaluate.load("cer")
asr_pipeline = None
speaker_encoder = None
utmos_model = None


def get_asr_pipeline():
    global asr_pipeline
    if asr_pipeline is None:
        print("🔄 Loading Whisper-large-v3-turbo (excellent for HK Cantonese)...")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    return asr_pipeline


def get_speaker_encoder():
    global speaker_encoder
    if speaker_encoder is None:
        try:
            print("🔄 Loading ECAPA-TDNN speaker encoder...")
            from speechbrain.pretrained import EncoderClassifier
            speaker_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
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
        device_map="auto" if torch.cuda.is_available() else None
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(test_jsonl, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f.readlines()]
    if max_samples:
        test_data = test_data[:max_samples]
    print(f"✅ Loaded {len(test_data)} test samples")

    asr = get_asr_pipeline()
    spk_enc = get_speaker_encoder() if not skip_speaker_sim else None
    utmos = get_utmos_model() if not skip_utmos else None

    print("✅ All models ready!")
    print("🚀 Starting evaluation...\n")

    results = []
    total_cer = total_sim = total_utmos = 0.0
    sim_available = not skip_speaker_sim
    utmos_available = utmos is not None

    for i, sample in tqdm(enumerate(test_data), total=len(test_data),
                          desc="Generating & Evaluating", unit="sample"):
        text = sample["text"]
        ref_audio_path = sample["audio"]

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

        # === CER ===
        transcription = asr(gen_path, generate_kwargs={"language": "yue"})["text"]
        cer = cer_metric.compute(predictions=[transcription], references=[text])

        # === Speaker Similarity (fixed loading with soundfile) ===
        if spk_enc is not None:
            try:
                # Use soundfile instead of torchaudio.load to avoid torchcodec error
                audio_data, orig_sr = sf.read(ref_audio_path)
                if len(audio_data.shape) == 1:
                    audio_data = audio_data[None, :]  # add channel dim
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
            "ref_audio": ref_audio_path,
            "gen_audio": gen_path,
            "cer": round(cer, 4),
            "speaker_sim": round(sim, 4) if spk_enc else "N/A",
            "utmos": round(utmos_score, 3) if utmos_available else "N/A"
        })

        total_cer += cer
        total_sim += sim
        total_utmos += utmos_score

    # === Summary ===
    avg_cer = total_cer / len(results)
    avg_sim = total_sim / len(results) if sim_available else "N/A"
    avg_utmos = total_utmos / len(results) if utmos_available else "N/A"

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False, encoding="utf-8")

    summary = {
        "checkpoint": checkpoint_dir,
        "test_samples": len(results),
        "avg_cer": round(avg_cer, 4),
        "avg_speaker_similarity": avg_sim,
        "avg_utmos": avg_utmos,
        "language": language,
        "speaker_name": speaker_name,
        "asr_model": "whisper-large-v3-turbo"
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"Checkpoint       : {checkpoint_dir}")
    print(f"Test samples     : {len(results)}")
    print(f"Avg CER          : {avg_cer:.4f}  (lower = better)")
    print(f"Avg Speaker SIM  : {avg_sim}  (higher = better)")
    print(f"Avg UTMOS        : {avg_utmos}  (higher = better)")
    print(f"Results saved to : {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Evaluation (stable version)")
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