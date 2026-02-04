# Qwen3-TTS Fine-Tuning for Hong Kong Cantonese

This repository provides a complete, ready-to-use workflow for **fine-tuning Qwen3-TTS** (12Hz-1.7B-Base or 0.6B-Base) to create high-quality, speaker-specific **Hong Kong Cantonese** voices.

Qwen3-TTS natively supports Chinese (Mandarin) and multiple dialects. Fine-tuning with Hong Kong Cantonese audio improves accent fidelity, intonation, prosody, and colloquial phrasing (e.g., "食咗飯未？", "今晚打邊爐").<grok-card data-id="0d40e3" data-type="citation_card" data-plain-type="render_inline_citation" ></grok-card><grok-card data-id="520e3c" data-type="citation_card" data-plain-type="render_inline_citation" ></grok-card>

## ✨ Features
- Single-speaker voice fine-tuning
- Dataset preparation scripts & templates
- Training & checkpoint management
- Inference examples with custom HK Cantonese speaker
- Ready-to-use JSONL templates for Cantonese
- ComfyUI node compatibility notes (optional)

## 📋 Requirements

### Hardware
- NVIDIA GPU with ≥16 GB VRAM (24+ GB recommended for 1.7B model)
- CUDA 12.x compatible
- 20+ GB disk space for dataset + checkpoints

### Software
- Python 3.11–3.12
- FFmpeg (for audio conversion)
- `pip install -U qwen-tts torch soundfile`
- Optional: FlashAttention-2 (`pip install -U flash-attn --no-build-isolation`)

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/qwen3-tts-hk-cantonese-finetune.git
cd qwen3-tts-hk-cantonese-finetune
pip install -r requirements.txt
```

### 2. Downalod base model & Tokenizer
```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir Qwen3-TTS-12Hz-1.7B-Base
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir Qwen3-TTS-Tokenizer-12Hz
```

### 3. Dataset Preparation (Hong Kong Cantonese)
##### Audio & Reference Requirements

- audio: 5–20 second WAV clips (clean, mono, 16-bit, 44.1/48 kHz)
- ref_audio: One fixed 3–5 second neutral clip from the same speaker (e.g., "你好，我係香港人。") — used for every sample
- Total duration: 1–10+ hours for good results (5+ hours recommended for strong HK accent)

#### Steps

1. Place all .wav files in data/audio/ and transcripts in matching .txt files (or manual JSONL).
2. Choose/create data/ref.wav (same speaker).
3.  Create train_raw.jsonl (one line per sample):
```json
{"audio": "data/audio/utt0001.wav", "text": "香港天氣好熱呀，出去飲杯凍奶茶啦。", "ref_audio": "data/ref.wav"}
{"audio": "data/audio/utt0002.wav", "text": "今晚打邊爐定係去食火鍋？", "ref_audio": "data/ref.wav"}
```

**Tips for HK Cantonese:**

- Use Traditional Chinese characters.
- Source from HK YouTube, TVB, radio, or record native speakers.
- Include natural slang, question tones, and fast-paced delivery.
- Avoid mixing Mandarin or other accents.


4. Process dataset:
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

### 4. Fine-Tuning

```bash
python sft_12hz.py \
  --init_model_path Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output_hk_cantonese \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 5 \
  --speaker_name hk_cantonese_speaker
```

Checkpoints saved in output_hk_cantonese/checkpoint-epoch-X.

### 5. Inference

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "output_hk_cantonese/checkpoint-epoch-4",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="香港股市今日升咗好多，你有冇買？",
    speaker="hk_cantonese_speaker",
    # language="Chinese"  # optional
)
sf.write("hk_output.wav", wavs[0], sr)
```

### 📁 Project Structure
```text
data/
├── audio/          # your wav files
├── ref.wav
├── train_raw.jsonl
└── train_with_codes.jsonl
output_hk_cantonese/   # checkpoints
scripts/               # helper scripts
prepare_data.py
sft_12hz.py
```


#### ⚙️ Hyperparameter Recommendations

- batch_size: 1–4 (depending on VRAM)
- lr: 2e-5 – 5e-4 (start low)
- num_epochs: 3–10 (more data → fewer epochs)
- Use smaller 0.6B-Base if VRAM-limited

#### 📝 Tips for Best HK Cantonese Results

- High-quality, noise-free recordings
- Consistent ref_audio (critical for voice stability)
- Diverse sentence lengths & emotions
- At least 5 hours for noticeable accent improvement
- Evaluate by listening to generated samples

#### Limitations

- Single-speaker only (multi-speaker coming soon)
- Base model has strong Chinese support; fine-tuning adapts dialect/accent

#### References

- Official Qwen3-TTS Repo: https://github.com/QwenLM/Qwen3-TTS
- Fine-tuning Folder: https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning

#### License
MIT License (or your choice)

#### Contributing
Pull requests welcome! Add your HK Cantonese datasets, scripts, or pre-trained checkpoints (upload to Hugging Face).
