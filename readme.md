# Qwen3-TTS Fine-Tuning for Hong Kong Cantonese 🇭🇰

This repository provides a **complete workflow** for fine-tuning **Qwen3-TTS** (0.6B-Base or 1.7B-Base) to generate high-quality **Hong Kong Cantonese** speech with authentic accent, natural intonation, prosody, and local slang.

---

## ✨ Features

- Single-speaker voice cloning / fine-tuning
- **LoRA fine-tuning** (recommended for 8–12 GB GPUs)
- **Voice Clone Server** – Web UI / API for instant inference
- **Model Evaluation** – Compare original pre-trained vs. fine-tuned model (WER/CER, speaker similarity, UTMOS)
- Full Cantonese dataset preparation pipeline
- Training with **MLflow** experiment tracking
- Gradient accumulation + Flash Attention-2 support

## 📋 Requirements

### Hardware (Updated for LoRA)

| Method                  | GPU VRAM Required     | Recommended Model     | Speed     |
|-------------------------|-----------------------|-----------------------|-----------|
| **LoRA (Recommended)** | **8–12 GB**          | 0.6B-Base            | Fast     |
| Full Fine-Tuning       | 16–24+ GB            | 0.6B or 1.7B         | Slower   |

- NVIDIA GPU with CUDA 12.x
- ~25 GB+ free disk space

### Software

- Python 3.11 or 3.12
- FFmpeg
- `pip install -r requirements.txt`

**Install Flash Attention-2 (highly recommended):**

```bash
./flash-attent-install.sh
```

### 🚀 Quick Start
1. Clone Repository
```Bash
git clone https://github.com/johnsonhk88/qwen3-tts-hk-cantonese-finetune.git
cd qwen3-tts-hk-cantonese-finetune
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Base Models
```bash
# Recommended: Start with 0.6B (faster training)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
```



### 4. Prepare Dataset (Inside Dataset-Cantonese-Training/)
```bash
cd Dataset-Cantonese-Training

# Put your clean .wav files in audio/
# Create train_raw.jsonl (see example inside folder)
# Run data preparation
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path ../../models/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

### Important for HK Cantonese:

- Use clean 5–15 second mono WAVs
- Use one fixed reference audio (ref.wav) for all samples
- Include natural HK slang and expressions

### 5. Fine-Tuning 
#### Option A: LoRA (Recommended for 8–12 GB GPUs)

```bash
python sft_12hz_lora_mlflow.py \
  --init_model_path ./Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path ./output_hk_cantonese_lora \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 1 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name hk_cantonese_speaker \
  --gradient_accumulation_steps 8 \
  --lora_rank 8

```

#### Option B: Full Fine-Tuning (16+ GB VRAM)

```bash
python sft_12hz_mlflow.py \
  --init_model_path ./Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path ./output_hk_cantonese \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-6 \
  --num_epochs 8 \
  --speaker_name hk_cantonese_speaker \
  --gradient_accumulation_steps 4
```

#### Start MLflow dashboard:
```bash
mlflow ui
```

### 6. Inference (works with both LoRA & full checkpoints)
```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "../output_hk_cantonese_lora/checkpoint-epoch-9",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

wavs, sr = model.generate_custom_voice(
    text="喂，你食咗飯未呀？今晚想唔想去打邊爐？",
    speaker="hk_cantonese_speaker"
)
sf.write("output_hk.wav", wavs[0], sr)
```

## 7. Voice Clone Server (Web UI / API)
```bash
# Run from project root
python voice_clone_server.py \
  --model_path output_hk_cantonese_lora/checkpoint-epoch-9 \
  --port 7860

```
### Open http://localhost:7860 

## 8. Model Evaluation (New)
#### Compare the original pre-trained model vs. your fine-tuned model using evaluate_qwen3_tts.py.

```bash
cd Dataset-Cantonese-Training

# Evaluate original pre-trained model
python evaluate_qwen3_tts.py \
  --checkpoint_dir ../../models/Qwen3-TTS-12Hz-0.6B-Base \
  --test_jsonl eval_prepared.jsonl \
  --speaker_name hk_cantonese_speaker \
  --output_dir ../evaluation_results_original

# Evaluate your fine-tuned model (LoRA or full)
python evaluate_qwen3_tts.py \
  --checkpoint_dir ../output_hk_cantonese_lora/checkpoint-epoch-9 \
  --test_jsonl eval_prepared.jsonl \
  --speaker_name hk_cantonese_speaker \
  --output_dir ../evaluation_results_finetuned

```


## What it computes:
- Word Error Rate (WER) / Character Error Rate (CER) with Cantonese-English code-mixing support
- Speaker similarity (ECAPA-TDNN)
- Audio quality (UTMOS)
- Saves per-sample results + summary JSON + generated audio


## 📁 Project Structure (April 2026)
```text
qwen3-tts-hk-cantonese-finetune/
├── Dataset-Cantonese-Training/
│   ├── evaluate_qwen3_tts.py          ← Model evaluation (pre-trained + fine-tuned)
│   ├── eval_prepared.jsonl
│   ├── sft_12hz_lora_mlflow.py
│   ├── sft_12hz_mlflow.py
│   ├── voice_clone_server.py
│   ├── prepare_data.py
│   ├── audio/
│   └── ...
├── models/
├── output_hk_cantonese_lora/
├── output_hk_cantonese/
├── flash-attent-install.sh
├── requirements.txt
└── README.md
```


## Recent Updates (April 2026)

- Apr 07: Added LoRA support, Voice Clone Server, and Model Evaluation section
- Mar 19: Fixed MLflow logging + dataset improvements

## Best Practices for HK Cantonese

- Use clean 5–15 second mono WAVs from native speakers
- Keep one fixed reference audio for all samples
- Start with LoRA + 0.6B-Base

Made for the Hong Kong Cantonese community ❤️
Pull requests and pre-trained models welcome!
## References

- Official Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- License: MIT


#### ⚙️ Hyperparameter Recommendations

- batch_size: 1–4 (depending on VRAM)
- lr: 2e-5 – 5e-4 (start low)
- num_epochs: 3–10 (more data → fewer epochs)
- Use smaller 0.6B-Base if VRAM-limited

#### Limitations

- Single-speaker only (multi-speaker coming soon)
- Base model has strong Chinese support; fine-tuning adapts dialect/accent


#### License
MIT License (or your choice)

#### Contributing
Pull requests welcome! Add your HK Cantonese datasets, scripts, or pre-trained checkpoints (upload to Hugging Face).



