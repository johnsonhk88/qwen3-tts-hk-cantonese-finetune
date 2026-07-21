## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base (HK Cantonese)

Single-speaker fine-tuning for Hong Kong Cantonese. Run all commands from this directory
(`Dataset-Cantonese-Training/`). See the root `../readme.md` for the full project workflow.

```bash
pip install -r ../requirements.txt
```

### 1) Input JSONL format

Prepare your training file as a JSONL (one JSON object per line). Each line must contain:

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

`ref_audio` recommendation:
- Strongly recommended: use the same `ref_audio` for all samples.
- Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.


### 2) Prepare data (extract `audio_codes`)

Convert `train_raw.jsonl` into a training JSONL that includes `audio_codes`:

```bash
# Option A: training file only
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# Option B: train + eval split (recommended)
python prepare_train_evaluate_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_train_jsonl train_prepared.jsonl \
  --output_eval_jsonl eval_prepared.jsonl \
  --eval_ratio 0.1 \
  --speaker_name hk_cantonese_speaker
```


### 3) Fine-tune

**LoRA (recommended, 8–12 GB VRAM):**

```bash
python sft_12hz_lora_mlflow.py \
  --init_model_path ./Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path ./output_hk_cantonese_lora \
  --train_jsonl train_prepared.jsonl \
  --batch_size 1 \
  --lr 2e-4 \
  --num_epochs 10 \
  --speaker_name hk_cantonese_speaker \
  --gradient_accumulation_steps 8 \
  --lora_rank 8
```

**Full fine-tuning (16+ GB VRAM):**

```bash
python sft_12hz_mlflow.py \
  --init_model_path ./Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path ./output_hk_cantonese \
  --train_jsonl train_prepared.jsonl \
  --batch_size 2 \
  --lr 2e-6 \
  --num_epochs 8 \
  --speaker_name hk_cantonese_speaker \
  --gradient_accumulation_steps 4
```

Optional: `--attn_implementation flash_attention_2` if Flash Attention-2 is installed.

Checkpoints are written to:
- `output_*/checkpoint-epoch-0`
- `output_*/checkpoint-epoch-1`
- ...
- `output_hk_cantonese_lora/final_merged` (LoRA only)


### 4) Quick inference test

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output_hk_cantonese_lora/checkpoint-epoch-9",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

wavs, sr = tts.generate_custom_voice(
    text="喂，你食咗飯未呀？今晚想唔想去打邊爐？",
    speaker="hk_cantonese_speaker",
)
sf.write("output.wav", wavs[0], sr)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="./Qwen3-TTS-12Hz-0.6B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_prepared.jsonl"
EVAL_JSONL="eval_prepared.jsonl"
OUTPUT_DIR="output_hk_cantonese_lora"

BATCH_SIZE=1
LR=2e-4
EPOCHS=10
SPEAKER_NAME="hk_cantonese_speaker"

python prepare_train_evaluate_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_train_jsonl ${TRAIN_JSONL} \
  --output_eval_jsonl ${EVAL_JSONL} \
  --eval_ratio 0.1 \
  --speaker_name ${SPEAKER_NAME}

python sft_12hz_lora_mlflow.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME} \
  --gradient_accumulation_steps 8 \
  --lora_rank 8
```