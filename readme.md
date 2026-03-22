# Qwen3-TTS Fine-Tuning for Hong Kong Cantonese 🇭🇰

This repository provides a **complete and up-to-date workflow** for fine-tuning **Qwen3-TTS** (0.6B-Base or 1.7B-Base) to generate high-quality **Hong Kong Cantonese** speech with authentic accent, natural intonation, prosody, and local slang.

---

## ✨ Features

- Single-speaker voice cloning / fine-tuning
- Full Cantonese dataset preparation pipeline
- Training with **MLflow** experiment tracking (recommended)
- Support for `gradient_accumulation_steps`
- Flash Attention-2 ready
- Custom voice inference examples
- Optimized for Traditional Chinese + Hong Kong colloquial Cantonese

## 📋 Requirements

### Hardware
- NVIDIA GPU with ≥16GB VRAM (24GB+ strongly recommended for 1.7B)
- CUDA 12.x
- ~25GB+ free disk space

### Software
- Python 3.11 or 3.12
- FFmpeg
- `pip install -r requirements.txt`

**Install Flash Attention-2 (recommended):**
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

### 5. Fine-Tuning (MLflow version recommended)

```bash
python sft_12hz_mlflow.py \
  --init_model_path ../../models/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path ../output_hk_cantonese \
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

### 6. Inference Example
```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "../output_hk_cantonese/checkpoint-final",
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

## 📁 Current Project Structure (March 2026)
```text
qwen3-tts-hk-cantonese-finetune/
├── Dataset-Cantonese-Training/          ← Main training folder
│   ├── audio/
│   ├── prepare_data.py
│   ├── sft_12hz_mlflow.py               ← Recommended
│   ├── sft_12hz.py
│   ├── train_raw.jsonl
│   ├── train_with_codes.jsonl
│   └── ...
├── models/                              ← Base models
├── output_hk_cantonese/                 ← Your fine-tuned checkpoints
├── src/
├── Qwen3-TTS/ (submodule)
├── flash-attent-install.sh
├── requirements.txt
└── readme.md
```


## Recent Updates (March 2026)

- Mar 19: Fixed MLflow logging issues + dataset refresh
- Mar 18: Added sft_12hz_mlflow.py + MLflow tracking
- Mar 19 (gradient): Added --gradient_accumulation_steps support
- Mar 17–15: Flash Attention-2 integration + training improvements

## Best Practices for HK Cantonese

1. Record with native Hong Kong speakers (no background noise)
2. Maintain consistent reference audio across all samples
3. Use Traditional Chinese in transcripts
4. Include varied sentence lengths, emotions, and slang
5. Start with 0.6B model + more epochs, then scale to 1.7B


### Made for the Hong Kong Cantonese community ❤️
Pull requests, better datasets, and pre-trained models on Hugging Face are welcome!

```text
**How to apply this fully modified `readme.md`**  
1. Copy **everything** above (from `# Qwen3-TTS...` to the end).  
2. Open your repo → replace the entire content of `readme.md`.  
3. Commit & push:  
   ```bash
   git add readme.md
   git commit -m "docs: fully update readme.md with latest March 2026 structure, MLflow, gradient accumulation, and accurate paths"
   git push
   ```
    
```


#### ⚙️ Hyperparameter Recommendations

- batch_size: 1–4 (depending on VRAM)
- lr: 2e-5 – 5e-4 (start low)
- num_epochs: 3–10 (more data → fewer epochs)
- Use smaller 0.6B-Base if VRAM-limited

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



