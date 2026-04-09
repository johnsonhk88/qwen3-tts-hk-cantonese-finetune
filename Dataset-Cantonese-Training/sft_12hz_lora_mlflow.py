# coding=utf-8
# Copyright 2026 The Alibaba Qwen team + LoRA modification by Grok
# SPDX-License-Identifier: Apache-2.0

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
from transformers import AutoConfig, get_linear_schedule_with_warmup
import mlflow

from peft import LoraConfig, get_peft_model

target_speaker_embedding = None


def clean_merged_state_dict(peft_model):
    """After merge_adapter(), extract COMPLETELY clean state_dict
    - removes ALL lora_* keys
    - removes base_layer.weight → .weight
    - removes base_model.model. prefix (fixes checkpoint warnings)
    """
    raw_state = peft_model.state_dict()
    clean_dict = {}
    for k, v in raw_state.items():
        if "lora_" in k:
            continue

        clean_k = k
        # Strip PEFT wrapper prefix
        if clean_k.startswith("base_model.model."):
            clean_k = clean_k[len("base_model.model."):]
        if clean_k.startswith("model."):
            clean_k = clean_k[6:]

        # Convert base_layer.weight → weight
        clean_k = clean_k.replace(".base_layer.weight", ".weight")

        clean_dict[clean_k] = v.detach().to("cpu")
    return clean_dict


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-TTS with MLflow + Warmup")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--speaker_name", type=str, default="hk_cantonese_speaker")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps to use for linear LR warmup (default: 0.1 = 10%)")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://localhost:5000")

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("Qwen3-TTS-LoRA-Finetune")
    mlflow.config.enable_async_logging()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    with mlflow.start_run(run_name=f"lora-r{args.lora_rank}-{args.speaker_name}-bs{args.batch_size}-warmup{args.warmup_ratio}"):
        if accelerator.is_main_process:
            mlflow.log_params(vars(args))

        # === Model loading ===
        qwen3tts = Qwen3TTSModel.from_pretrained(
            args.init_model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        config = AutoConfig.from_pretrained(args.init_model_path)

        # === LoRA setup ===
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(qwen3tts.model, lora_config)
        if accelerator.is_main_process:
            peft_model.print_trainable_parameters()

        # Dataset
        train_data = [json.loads(line) for line in open(args.train_jsonl).readlines()]
        dataset = TTSDataset(train_data, qwen3tts.processor, config)
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        optimizer = AdamW(peft_model.parameters(), lr=args.lr, weight_decay=0.01)

        # === Prepare with Accelerator ===
        model, optimizer, train_dataloader = accelerator.prepare(
            peft_model, optimizer, train_dataloader
        )

        # === Linear Warmup Scheduler (prevents early divergence) ===
        total_steps = len(train_dataloader) * args.num_epochs
        warmup_steps = int(args.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        scheduler = accelerator.prepare(scheduler)

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
                    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                    talker_codec_ids = codec_ids[codec_mask]

                    sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )

                    loss = outputs.loss + 0.3 * sub_talker_loss

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()          # ← Warmup happens here
                        optimizer.zero_grad()

                current_loss = loss.item()
                epoch_loss += current_loss
                num_batches += 1
                global_step += 1

                if step % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {current_loss:.4f} | LR: {current_lr:.2e}")
                    if accelerator.is_main_process:
                        mlflow.log_metric("train_loss", current_loss, step=global_step)
                        mlflow.log_metric("learning_rate", current_lr, step=global_step)

            # === Calculate and log average loss ===
            if num_batches > 0:
                epoch_avg_loss = epoch_loss / num_batches
                if accelerator.is_main_process:
                    print(f"✅ Epoch {epoch} finished | Avg Loss: {epoch_avg_loss:.4f}")
                    mlflow.log_metric("epoch_avg_loss", epoch_avg_loss, step=epoch)

            # ==================== SAVE FULL MERGED CHECKPOINT EVERY EPOCH ====================
            if accelerator.is_main_process and num_batches > 0:
                output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
                print(f"Saving full merged checkpoint to: {output_dir}")
                shutil.copytree(args.init_model_path, output_dir, dirs_exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)
                unwrapped.merge_adapter()

                state_dict = clean_merged_state_dict(unwrapped)

                # Update config
                with open(os.path.join(args.init_model_path, "config.json"), 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                config_dict["tts_model_type"] = "custom_voice"
                talker_config = config_dict.get("talker_config", {})
                talker_config["spk_id"] = {args.speaker_name: 3000}
                talker_config["spk_is_dialect"] = {args.speaker_name: False}
                config_dict["talker_config"] = talker_config
                with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                # Drop speaker_encoder
                keys_to_drop = [k for k in state_dict if k.startswith("speaker_encoder")]
                for k in keys_to_drop:
                    del state_dict[k]

                # Inject speaker embedding
                codec_key = None
                for k in state_dict:
                    if k.endswith("codec_embedding.weight"):
                        codec_key = k
                        break
                if codec_key:
                    weight = state_dict[codec_key]
                    state_dict[codec_key][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
                    print(f"✓ Speaker embedding injected at index 3000")

                save_path = os.path.join(output_dir, "model.safetensors")
                save_file(state_dict, save_path)
                print(f"✅ Full merged model saved at {save_path}")

                # Restore adapter for next epoch
                unwrapped.unmerge_adapter()

        # ==================== FINAL MERGED FOLDER ====================
        if accelerator.is_main_process:
            print("\n=== Creating final_merged folder ===")
            final_dir = os.path.join(args.output_model_path, "final_merged")
            shutil.copytree(args.init_model_path, final_dir, dirs_exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)
            clean_model = unwrapped.merge_and_unload()

            state_dict = {k: v.detach().to("cpu") for k, v in clean_model.state_dict().items()}

            with open(os.path.join(args.init_model_path, "config.json"), 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config
            with open(os.path.join(final_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # Drop speaker_encoder
            keys_to_drop = [k for k in state_dict if k.startswith("speaker_encoder")]
            for k in keys_to_drop:
                del state_dict[k]

            # Inject speaker embedding
            codec_key = None
            for k in state_dict:
                if k.endswith("codec_embedding.weight"):
                    codec_key = k
                    break
            if codec_key:
                weight = state_dict[codec_key]
                state_dict[codec_key][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
                print(f"✓ Speaker embedding injected at index 3000")

            save_file(state_dict, os.path.join(final_dir, "model.safetensors"))
            print(f"✅ Final merged model saved at {final_dir}")

            unwrapped.save_pretrained(os.path.join(final_dir, "lora_adapter"))

if __name__ == "__main__":
    train()