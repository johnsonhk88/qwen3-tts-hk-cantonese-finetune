# Multi-Speaker LoRA Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new multi-speaker LoRA fine-tuning/export entrypoint for HK Cantonese that exports one merged checkpoint with multiple speaker slots while leaving the existing single-speaker script unchanged.

**Architecture:** The new script will reuse the current Qwen3-TTS LoRA training loop, but will derive an ordered `speaker_id -> slot_id` mapping from `train_jsonl` and inject one exported speaker embedding per speaker during checkpoint merge. The existing single-speaker path remains intact; only the new entrypoint and user-facing docs change.

**Tech Stack:** Python, PyTorch, Accelerate, PEFT/LoRA, safetensors, MLflow, Qwen3-TTS

## Global Constraints

- Keep `Dataset-Cantonese-Training/sft_12hz_lora_mlflow.py` unchanged.
- Add `Dataset-Cantonese-Training/sft_12hz_lora_mlflow_multi_speaker.py` as a separate entrypoint.
- Use `speaker_id` from `train_jsonl` as the only speaker identifier for the new script.
- Assign speaker slots in sorted `speaker_id` order, starting at `3000`.
- Use the first `ref_audio` seen for each speaker when exporting that speaker's embedding.
- Do not average or cluster speaker embeddings during export.
- Keep the current training loop, LoRA setup, and merge/save flow.
- Exported checkpoints must load through `Qwen3TTSModel.from_pretrained()` and expose all speaker names through `get_supported_speakers()`.

---

### Task 1: Add the multi-speaker exporter script

**Files:**
- Create: `Dataset-Cantonese-Training/sft_12hz_lora_mlflow_multi_speaker.py`

**Interfaces:**
- Consumes: `train_jsonl` records with `audio`, `text`, `ref_audio`, `audio_codes`, and `speaker_id`.
- Produces: a separate fine-tuning command that trains with the same loop as the current LoRA script and exports a checkpoint with multiple speaker slots.

- [ ] **Step 1: Write the failing script-presence check**

```python
# Save as /tmp/opencode/check_multi_speaker_script.py
from pathlib import Path

script = Path("Dataset-Cantonese-Training/sft_12hz_lora_mlflow_multi_speaker.py")
assert script.exists(), script
```

- [ ] **Step 2: Run the check and confirm it fails before implementation**

Run: `python /tmp/opencode/check_multi_speaker_script.py`
Expected: `AssertionError` because the file does not exist yet.

- [ ] **Step 3: Implement the script by copying the current LoRA flow and changing only the multi-speaker pieces**

Create the new script with these behaviors:

```python
def build_speaker_slot_map(train_data: list[dict[str, object]], start_slot: int = 3000) -> dict[str, int]:
    speakers = sorted({str(sample["speaker_id"]) for sample in train_data})
    return {speaker_id: start_slot + i for i, speaker_id in enumerate(speakers)}


def build_first_ref_audio_map(train_data: list[dict[str, object]]) -> dict[str, str]:
    first_ref_audio = {}
    for sample in train_data:
        speaker_id = str(sample["speaker_id"])
        ref_audio = str(sample["ref_audio"])
        if speaker_id in first_ref_audio and first_ref_audio[speaker_id] != ref_audio:
            raise ValueError(f"speaker_id {speaker_id} has conflicting ref_audio values")
        first_ref_audio.setdefault(speaker_id, ref_audio)
    return first_ref_audio
```

The training loop should still use `TTSDataset`, `DataLoader`, `Accelerator`, LoRA, and the current loss computation from `sft_12hz_lora_mlflow.py`.

At export time, compute the speaker embeddings from the unwrapped model before merge, then inject them after merge:

```python
speaker_slot_map = build_speaker_slot_map(train_data)
first_ref_audio_map = build_first_ref_audio_map(train_data)

speaker_model = accelerator.unwrap_model(model)
export_dataset = TTSDataset([], qwen3tts.processor, config)


@torch.inference_mode()
def build_speaker_embedding(speaker_model, ref_audio_path: str) -> torch.Tensor:
    wav, sr = export_dataset._load_audio_to_np(ref_audio_path)
    ref_mel = export_dataset.extract_mels(audio=wav, sr=sr)
    return speaker_model.speaker_encoder(ref_mel.to(speaker_model.device).to(speaker_model.dtype)).detach()[0]


speaker_embeddings = {
    speaker_id: build_speaker_embedding(speaker_model, first_ref_audio_map[speaker_id])
    for speaker_id in speaker_slot_map
}


def inject_speaker_embeddings(state_dict, speaker_slot_map, speaker_embeddings):
    codec_key = next(k for k in state_dict if k.endswith("codec_embedding.weight"))
    weight = state_dict[codec_key]

    if max(speaker_slot_map.values()) >= weight.shape[0]:
        raise ValueError("speaker slot exceeds codec embedding table size")

    for speaker_id, slot_id in speaker_slot_map.items():
        state_dict[codec_key][slot_id] = speaker_embeddings[speaker_id].to(weight.device).to(weight.dtype)
```

Then populate `talker_config["spk_id"]` and `talker_config["spk_is_dialect"]` from that mapping, call `speaker_model.merge_and_unload()`, build the final `state_dict`, and call `inject_speaker_embeddings(...)` before saving.

```python
talker_config["spk_id"] = speaker_slot_map
talker_config["spk_is_dialect"] = {speaker_id: False for speaker_id in speaker_slot_map}
```

Keep the export path identical to the current script except that the speaker embedding injection loops over every speaker in `speaker_slot_map`.

- [ ] **Step 4: Run a syntax check**

Run: `python -m py_compile Dataset-Cantonese-Training/sft_12hz_lora_mlflow_multi_speaker.py`
Expected: no output.

- [ ] **Step 5: Commit the script**

```bash
git add Dataset-Cantonese-Training/sft_12hz_lora_mlflow_multi_speaker.py
git commit -m "feat: add multi-speaker lora exporter"
```

### Task 2: Update the Cantonese training docs

**Files:**
- Modify: `Dataset-Cantonese-Training/README.md`
- Modify: `readme.md`

**Interfaces:**
- Consumes: the new multi-speaker script name and its `speaker_id`-driven input contract.
- Produces: updated usage docs that tell users when to choose the single-speaker path versus the multi-speaker path.

- [ ] **Step 1: Update the fine-tune section in `Dataset-Cantonese-Training/README.md`**

Add a short multi-speaker subsection with a command like:

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

State explicitly that `train_prepared.jsonl` must already include `speaker_id`.

- [ ] **Step 2: Update the root `readme.md` project structure and fine-tuning section**

Mention the new script in the file tree and add a sentence that the new entrypoint should be used for multi-speaker datasets produced by `cluster_speakers.py`.

- [ ] **Step 3: Verify the docs render as plain Markdown**

Run: `python - <<'PY'
from pathlib import Path
for path in [Path('Dataset-Cantonese-Training/README.md'), Path('readme.md')]:
    text = path.read_text(encoding='utf-8')
    assert 'sft_12hz_lora_mlflow_multi_speaker.py' in text
PY`
Expected: no output.

- [ ] **Step 4: Commit the docs**

```bash
git add Dataset-Cantonese-Training/README.md readme.md
git commit -m "docs: add multi-speaker lora usage"
```

### Task 3: Validate export compatibility with Qwen3-TTS inference

**Files:**
- Test: `Qwen3-TTS/examples/test_model_12hz_custom_voice.py` or a temporary smoke script under `/tmp/opencode`

**Interfaces:**
- Consumes: the exported checkpoint from the new multi-speaker script.
- Produces: confirmation that `Qwen3TTSModel.from_pretrained()` loads the checkpoint and exposes the expected speaker names.

- [ ] **Step 1: Run a tiny smoke training/export pass with a small subset of `train_prepared.jsonl`**

Create a tiny balanced subset first, then train on it:

```python
# Save as /tmp/opencode/multi_speaker_smoke_subset.py
import json
from pathlib import Path

src = Path("Dataset-Cantonese-Training/train_prepared.jsonl")
dst = Path("/tmp/opencode/multi_speaker_smoke.jsonl")

samples = []
seen = set()
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    sample = json.loads(line)
    speaker_id = sample["speaker_id"]
    if speaker_id in seen:
        continue
    seen.add(speaker_id)
    samples.append(sample)
    if len(samples) == 3:
        break

with dst.open("w", encoding="utf-8") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
```

Run:

```bash
python /tmp/opencode/multi_speaker_smoke_subset.py
python Dataset-Cantonese-Training/sft_12hz_lora_mlflow_multi_speaker.py \
  --init_model_path Dataset-Cantonese-Training/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path Dataset-Cantonese-Training/output/lora-multi-speaker-smoke \
  --train_jsonl /tmp/opencode/multi_speaker_smoke.jsonl \
  --batch_size 1 \
  --lr 2e-4 \
  --num_epochs 1 \
  --gradient_accumulation_steps 1 \
  --lora_rank 8
```

- [ ] **Step 2: Load the exported checkpoint and inspect supported speakers**

Run a smoke script shaped like:

```python
import torch
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "Dataset-Cantonese-Training/output/lora-multi-speaker-smoke/final_merged",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
print(sorted(tts.get_supported_speakers()))
```

Expected: the printed speaker list matches the `speaker_id` set from the smoke dataset.

- [ ] **Step 3: Exercise one custom-voice generation call**

Use one speaker name from the exported set and confirm generation returns at least one waveform:

```python
wavs, sr = tts.generate_custom_voice(
    text="喂，你食咗飯未呀？今晚想唔想去打邊爐？",
    speaker=sorted(tts.get_supported_speakers())[0],
)
assert len(wavs) == 1
assert sr > 0
```

- [ ] **Step 4: Commit any smoke helper you decide to keep**

If a reusable smoke script is added to the repo, commit it. Otherwise keep the validation script disposable in `/tmp/opencode`.
