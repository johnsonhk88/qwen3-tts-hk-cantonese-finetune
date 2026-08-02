# Multi-Speaker LoRA Export Design

## Goal
Add a new multi-speaker fine-tuning/export entrypoint for HK Cantonese that keeps the existing single-speaker script unchanged, while producing one merged checkpoint compatible with Qwen3-TTS custom-voice inference.

## Scope
- Add `Dataset-Cantonese-Training/sft_12hz_lora_mlflow_multi_speaker.py`.
- Keep `Dataset-Cantonese-Training/sft_12hz_lora_mlflow.py` unchanged.
- Use `speaker_id` from `train_jsonl` to build the exported speaker map.
- Export one slot per speaker, starting at `3000`.
- Keep the current training loop, LoRA setup, and checkpoint merging behavior.

## Non-Goals
- No dataset schema rewrite.
- No changes to `dataset.py` unless implementation later proves it is required.
- No attempt to average or cluster speaker embeddings during export.

## Inputs
- Training JSONL entries already contain:
  - `audio`
  - `text`
  - `ref_audio`
  - `audio_codes`
  - `speaker_id`
- `speaker_id` is the only speaker identifier the new script should trust.

## Speaker Slot Rules
- Speaker slots are assigned in sorted `speaker_id` order for deterministic exports.
- Slot IDs begin at `3000`.
- Each `speaker_id` maps to one slot ID.
- `talker_config["spk_id"]` and `talker_config["spk_is_dialect"]` are populated for every speaker in the export config.
- `spk_is_dialect` is `False` for every speaker.

## Speaker Embedding Rules
- The script scans `train_jsonl` in file order to capture each speaker's first `ref_audio`.
- Slot assignment uses the sorted speaker list, but embedding selection uses the first-seen sample.
- For each speaker, use the first `ref_audio` seen for that `speaker_id`.
- Compute the speaker embedding from that `ref_audio` during export.
- Inject exactly one embedding per speaker into `codec_embedding.weight` at the assigned slot.
- The training path continues to use the per-sample `ref_audio` already present in the dataset.

## Runtime Behavior
- The new script loads the same base model and tokenizer path as the existing LoRA script.
- The training loop remains unchanged from the single-speaker LoRA flow.
- After training, the script merges LoRA weights, copies the base model folder, edits `config.json`, and writes `model.safetensors`.
- The export config must remain loadable through `Qwen3TTSModel.from_pretrained()` and expose all speaker names via `get_supported_speakers()`.

## Failure Conditions
The script should fail fast if:
- `speaker_id` is missing from any training sample.
- A speaker appears with conflicting `ref_audio` values.
- The computed slot ID would exceed the codec embedding table size.
- The target codec embedding parameter cannot be found.

## Verification
Minimum checks:
- `python -m py_compile Dataset-Cantonese-Training/sft_12hz_lora_mlflow_multi_speaker.py`
- Run a small smoke training/export pass on a tiny subset.
- Load the exported checkpoint with `Qwen3TTSModel.from_pretrained()` and confirm the speaker list matches the exported `speaker_id` set.

## Notes
- This design intentionally preserves the single-speaker workflow as a separate path.
- Deterministic sorted speaker ordering makes exports stable even if the JSONL line order changes.
