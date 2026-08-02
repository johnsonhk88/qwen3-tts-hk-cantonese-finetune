## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base (HK Cantonese)

Single-speaker and multi-speaker fine-tuning for Hong Kong Cantonese. Run all commands from this directory
(`Dataset-Cantonese-Training/`). See the root `../readme.md` for the full project workflow.

```bash
pip install -r ../requirements.txt
```

### Choose a path

| Path | Script | JSONL requirement | Inference `speaker=` |
|------|--------|-------------------|----------------------|
| **Single-speaker** | `sft_12hz_lora_mlflow.py` | optional `speaker_name` via prep | one name, e.g. `hk_cantonese_speaker` |
| **Multi-speaker** | `sft_12hz_lora_mlflow_multi_speaker.py` | **required** `speaker_id` on every line | each id, e.g. `spk_021` |

### 1) Input JSONL format

Prepare your training file as a JSONL (one JSON object per line).

**Required for all paths:**

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)

**Multi-speaker only:**

- `speaker_id`: stable speaker label (e.g. `spk_021`). Required by `sft_12hz_lora_mlflow_multi_speaker.py`.

**Single-speaker example:**
```jsonl
{"audio":"./audio/utt0001.wav","text":"喂，你食咗飯未呀？","ref_audio":"./audio/ref.wav"}
{"audio":"./audio/utt0002.wav","text":"今晚想唔想去打邊爐？","ref_audio":"./audio/ref.wav"}
```

**Multi-speaker example** (after `cluster_speakers.py` + prep):
```jsonl
{"audio":"./audio/40.wav","text":"喂，你食咗飯未呀？","ref_audio":"./audio/40.wav","speaker_id":"spk_115"}
{"audio":"./audio/6858.wav","text":"今晚想唔想去打邊爐？","ref_audio":"./audio/5612.wav","speaker_id":"spk_083"}
```

`ref_audio` recommendation:

- **Single-speaker:** use the same `ref_audio` for all samples.
- **Multi-speaker:** use one fixed `ref_audio` **per** `speaker_id` (same voice within a speaker). `cluster_speakers.py` does this automatically.

### 1b) Multi-speaker clustering (optional but recommended)

If your corpus has many voices and no `speaker_id` yet:

```bash
python cluster_speakers.py
```

This writes `train_raw.jsonl` with `speaker_id` and speaker-matched `ref_audio`. Then run prepare (step 2). Prep preserves existing fields including `speaker_id`.

### 2) Prepare data (extract `audio_codes`)

Convert `train_raw.jsonl` into a training JSONL that includes `audio_codes`.

Audio length is checked first: only clips in **[5, 30] seconds** are kept (Qwen3-TTS fine-tune requirement). Each kept sample gets `duration_sec`. A CSV report is written with columns `index`, `audio`, `ref_audio`, `text`, `duration_sec`, `status`, `split`, `reason`.

```bash
# Option A: training file only
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl \
  --min_duration 5.0 \
  --max_duration 30.0 \
  --duration_report train_with_codes_audio_lengths.csv

# Option B: train + eval split (recommended)
# Single-speaker: pass --speaker_name
# Multi-speaker: omit --speaker_name if lines already have speaker_id (prep keeps it)
python prepare_train_evaluate_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_train_jsonl train_prepared.jsonl \
  --output_eval_jsonl eval_prepared.jsonl \
  --eval_ratio 0.1 \
  --speaker_name hk_cantonese_speaker \
  --min_duration 5.0 \
  --max_duration 30.0 \
  --duration_report train_prepared_audio_lengths.csv \
  --batch_size 2
```

`--batch_size` (default **4**) is the tokenizer encode batch size. Lower it (`1`–`2`) to reduce peak VRAM; the old hardcoded value was 16 and could use ~20–24 GB.

### 3) Fine-tune

#### A) Single-speaker LoRA (recommended, 8–12 GB VRAM)

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

For the exact 1-epoch safe-run setup used here, see `run-full-lora-1epoch-bs6.sh`.

#### B) Single-speaker full fine-tuning (16+ GB VRAM)

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

#### C) Multi-speaker LoRA

Use when every line in `train_prepared.jsonl` has `speaker_id` (e.g. from `cluster_speakers.py`). Do **not** pass `--speaker_name`.

Export rules:

- One checkpoint, many speaker slots
- Slot IDs start at `3000`, assigned in sorted `speaker_id` order
- Each speaker gets the first `ref_audio` seen for that id
- Config writes `talker_config.spk_id` so inference can list speakers via `get_supported_speakers()`

```bash
python sft_12hz_lora_mlflow_multi_speaker.py \
  --init_model_path ./Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path ./output_hk_cantonese_multi_lora \
  --train_jsonl train_prepared.jsonl \
  --batch_size 1 \
  --lr 2e-4 \
  --num_epochs 10 \
  --gradient_accumulation_steps 8 \
  --lora_rank 8
```

Optional: `--attn_implementation flash_attention_2` if Flash Attention-2 is installed.

Checkpoints are written to:

- `output_*/checkpoint-epoch-0`, `checkpoint-epoch-1`, ...
- `output_hk_cantonese_lora/final_merged` (single-speaker LoRA)
- `output_hk_cantonese_multi_lora/final_merged` (multi-speaker LoRA)

Helper unit test for multi-speaker slot mapping:

```bash
python test_multi_speaker_export.py
```

### 4) Quick inference test

**Single-speaker:**

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output_hk_cantonese_lora/final_merged",
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

**Multi-speaker** (use any id from the exported map):

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "output_hk_cantonese_multi_lora/final_merged",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

print(sorted(tts.get_supported_speakers()))  # e.g. ['spk_021', 'spk_083', ...]

wavs, sr = tts.generate_custom_voice(
    text="喂，你食咗飯未呀？今晚想唔想去打邊爐？",
    speaker="spk_021",
)
sf.write("output_multi.wav", wavs[0], sr)
```

### One-click shell script example (single-speaker)

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

### One-click shell script example (multi-speaker)

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="./Qwen3-TTS-12Hz-0.6B-Base"

# train_raw.jsonl must already include speaker_id (run cluster_speakers.py first)
RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_prepared.jsonl"
EVAL_JSONL="eval_prepared.jsonl"
OUTPUT_DIR="output_hk_cantonese_multi_lora"

python prepare_train_evaluate_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_train_jsonl ${TRAIN_JSONL} \
  --output_eval_jsonl ${EVAL_JSONL} \
  --eval_ratio 0.1 \
  --min_duration 5.0 \
  --max_duration 30.0 \
  --batch_size 2

python sft_12hz_lora_mlflow_multi_speaker.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size 1 \
  --lr 2e-4 \
  --num_epochs 10 \
  --gradient_accumulation_steps 8 \
  --lora_rank 8
```
