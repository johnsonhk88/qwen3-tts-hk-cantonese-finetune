# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import shutil
import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, pipeline
import mlflow
import evaluate
import soundfile as sf
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# ====================== NEW: Global lazy-loaded evaluation tools ======================
cer_metric = evaluate.load("cer")
asr_pipeline = None
speaker_encoder = None
target_speaker_embedding = None


def compute_cer_and_sim(eval_model, val_data, num_samples, speaker_name, output_dir, epoch):
    """Compute CER + Speaker Similarity + save audio samples for MLflow"""
    global asr_pipeline, speaker_encoder

    if asr_pipeline is None:
        print("Loading SenseVoiceSmall ASR (best for Cantonese yue + Traditional Chinese)...")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="FunAudioLLM/SenseVoiceSmall",
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    if speaker_encoder is None:
        print("Loading ECAPA-TDNN speaker encoder for similarity...")
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

    cers = []
    sims = []
    sample_audio_paths = []

    for i in range(min(num_samples, len(val_data))):
        sample = val_data[i]
        text = sample["text"]                    # ← Change key if your jsonl uses different name
        ref_audio_path = sample["audio"]         # ← Change key if your jsonl uses different name (e.g. "ref_audio")

        # === Generate with your fine-tuned custom Hong Kong voice ===
        wavs, sr = eval_model.generate_custom_voice(
            text=text,
            language="yue",                      # "yue" for Hong Kong Cantonese (best)
            speaker=speaker_name,
            max_new_tokens=2048
        )
        gen_wav = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]

        # Save sample for listening (will be uploaded to MLflow)
        sample_path = os.path.join(output_dir, f"epoch_{epoch}_sample_{i:03d}.wav")
        sf.write(sample_path, gen_wav, sr)
        sample_audio_paths.append(sample_path)

        # === CER (Character Error Rate) ===
        # Resample to 16 kHz for Whisper (TTS is typically 24 kHz)
        gen_wav_16k = torchaudio.functional.resample(
            torch.from_numpy(gen_wav).float().unsqueeze(0),
            orig_freq=sr,
            new_freq=16000,
        ).squeeze(0).numpy()
        transcription = asr_pipeline(
            {"array": gen_wav_16k, "sampling_rate": 16000},
            return_timestamps=True,
            generate_kwargs={"language": "yue", "task": "transcribe"},
        )["text"]
        cer = cer_metric.compute(predictions=[transcription], references=[text])
        cers.append(cer)

        # === Speaker Similarity (cosine) ===
        ref_signal, _ = torchaudio.load(ref_audio_path)
        ref_emb = speaker_encoder.encode_batch(ref_signal.unsqueeze(0).to(speaker_encoder.device))

        gen_signal = torch.from_numpy(gen_wav).unsqueeze(0).to(speaker_encoder.device)
        gen_emb = speaker_encoder.encode_batch(gen_signal)

        sim = torch.nn.functional.cosine_similarity(
            ref_emb.mean(dim=1), gen_emb.mean(dim=1)
        ).item()
        sims.append(sim)

    avg_cer = sum(cers) / len(cers)
    avg_sim = sum(sims) / len(sims)
    return avg_cer, avg_sim, sample_audio_paths[0] if sample_audio_paths else None  # return one sample for logging


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-TTS with MLflow + CER + Speaker Similarity")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, default=None,
                        help="Path to validation jsonl (held-out samples) for CER + SIM")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=2,
                        help="Run full CER+SIM evaluation every N epochs")
    parser.add_argument("--num_eval_samples", type=int, default=8,
                        help="Number of validation samples to evaluate")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention backend. Use flash_attention_2 if installed")
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI (default: http://localhost:5000)"
    )
    
    args = parser.parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    print(f"Using MLflow tracking URI: {args.mlflow_tracking_uri}")

    mlflow.set_experiment("Qwen3-TTS-Finetune")
    mlflow.config.enable_async_logging()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    with mlflow.start_run(run_name=f"finetune-{args.speaker_name}-bs{args.batch_size}-lr{args.lr}"):
        if accelerator.is_main_process:
            mlflow.log_params({
                "init_model_path": args.init_model_path,
                "output_model_path": args.output_model_path,
                "train_jsonl": args.train_jsonl,
                "val_jsonl": args.val_jsonl,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "num_epochs": args.num_epochs,
                "speaker_name": args.speaker_name,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "eval_every": args.eval_every,
                "num_eval_samples": args.num_eval_samples,
                "attn_implementation": args.attn_implementation,
                "mixed_precision": "bf16",
            })

        MODEL_PATH = args.init_model_path
        qwen3tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        )
        config = AutoConfig.from_pretrained(MODEL_PATH)

        train_data = [json.loads(line) for line in open(args.train_jsonl).readlines()]
        dataset = TTSDataset(train_data, qwen3tts.processor, config)
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        # ====================== NEW: Load validation data ======================
        val_data = None
        if args.val_jsonl and accelerator.is_main_process:
            val_data = [json.loads(line) for line in open(args.val_jsonl).readlines()]
            print(f"Loaded {len(val_data)} validation samples for CER + Speaker Similarity evaluation")

        optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

        model, optimizer, train_dataloader = accelerator.prepare(
            qwen3tts.model, optimizer, train_dataloader
        )

        model.train()

        global_step = 0

        for epoch in range(args.num_epochs):
            if accelerator.is_main_process:
                print(f"\n=== Starting Epoch {epoch} ===")

            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    input_ids = batch['input_ids']
                    codec_ids = batch['codec_ids']
                    ref_mels = batch['ref_mels']
                    text_embedding_mask = batch['text_embedding_mask']
                    codec_embedding_mask = batch['codec_embedding_mask']
                    attention_mask = batch['attention_mask']
                    codec_0_labels = batch['codec_0_labels']
                    codec_mask = batch['codec_mask']

                    speaker_embedding = model.speaker_encoder(
                        ref_mels.to(model.device).to(model.dtype)
                    ).detach()

                    if target_speaker_embedding is None:
                        target_speaker_embedding = speaker_embedding

                    input_text_ids = input_ids[:, :, 0]
                    input_codec_ids = input_ids[:, :, 1]

                    text_emb = model.talker.model.text_embedding(input_text_ids)
                    input_text_embedding = model.talker.text_projection(text_emb) * text_embedding_mask

                    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                    input_codec_embedding[:, 6, :] = speaker_embedding

                    input_embeddings = input_text_embedding + input_codec_embedding

                    for i in range(1, 16):
                        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                            codec_ids[:, :, i]
                        )
                        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                        input_embeddings = input_embeddings + codec_i_embedding

                    outputs = model.talker(
                        inputs_embeds=input_embeddings[:, :-1, :],
                        attention_mask=attention_mask[:, :-1],
                        labels=codec_0_labels[:, 1:],
                        output_hidden_states=True
                    )

                    hidden_states = outputs.hidden_states[0][-1]
                    # talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                    talker_hidden_states = hidden_states[codec_mask[:, :-1]] # fix fine tune bug
                    talker_codec_ids = codec_ids[codec_mask]

                    sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )

                    loss = outputs.loss + 0.3 * sub_talker_loss

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                current_loss = loss.item()
                epoch_loss += current_loss
                num_batches += 1
                global_step += 1

                if step % 10 == 0:
                    accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {current_loss:.4f}")
                    if accelerator.is_main_process:
                        mlflow.log_metric("train_loss", current_loss, step=global_step)

            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                if accelerator.is_main_process:
                    mlflow.log_metric("epoch_avg_loss", avg_epoch_loss, step=epoch)
                accelerator.print(f"Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}")

            # ====================== Save checkpoint (unchanged) ======================
            if accelerator.is_main_process:
                output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
                print(f"Saving checkpoint to: {output_dir}")
                shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

                input_config_file = os.path.join(MODEL_PATH, "config.json")
                output_config_file = os.path.join(output_dir, "config.json")

                with open(input_config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)

                config_dict["tts_model_type"] = "custom_voice"
                talker_config = config_dict.get("talker_config", {})
                talker_config["spk_id"] = {args.speaker_name: 3000}
                talker_config["spk_is_dialect"] = {args.speaker_name: False}
                config_dict["talker_config"] = talker_config

                with open(output_config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

                drop_prefix = "speaker_encoder"
                keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
                for k in keys_to_drop:
                    del state_dict[k]

                weight = state_dict['talker.model.codec_embedding.weight']
                state_dict['talker.model.codec_embedding.weight'][3000] = \
                    target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)

                save_path = os.path.join(output_dir, "model.safetensors")
                save_file(state_dict, save_path)

                print(f"Checkpoint epoch-{epoch} saved locally successfully")

            # ====================== NEW: Evaluation every N epochs (CER + Speaker Similarity) ======================
            if (accelerator.is_main_process and
                args.val_jsonl and
                val_data and
                (epoch % args.eval_every == 0 or epoch == args.num_epochs - 1)):

                print(f"\n=== Running full evaluation (CER + Speaker SIM) on epoch {epoch} ===")
                checkpoint_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")

                eval_model = Qwen3TTSModel.from_pretrained(
                    checkpoint_dir,
                    torch_dtype=torch.bfloat16,
                )

                avg_cer, avg_sim, sample_audio_path = compute_cer_and_sim(
                    eval_model, val_data, args.num_eval_samples,
                    args.speaker_name, checkpoint_dir, epoch
                )

                mlflow.log_metric("val_cer", avg_cer, step=epoch)
                mlflow.log_metric("speaker_similarity", avg_sim, step=epoch)
                # if sample_audio_path:
                #     mlflow.log_artifact(sample_audio_path, artifact_path=f"audio_samples/epoch_{epoch}")

                print(f"Epoch {epoch} Eval → CER: {avg_cer:.4f} ↓ | Speaker SIM: {avg_sim:.4f} ↑")
                print(f"Sample audio saved & logged to MLflow → check Artifacts")

    if accelerator.is_main_process:
        print("Training finished successfully. Checkpoints + metrics saved locally + MLflow.")

if __name__ == "__main__":
    train()