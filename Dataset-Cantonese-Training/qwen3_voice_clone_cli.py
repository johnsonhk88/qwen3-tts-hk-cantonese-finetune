import torch
import soundfile as sf
from pathlib import Path
import argparse
import os

from qwen_tts import Qwen3TTSModel
from qwen_asr import Qwen3ASRModel


# Simple model cache
_model_cache: dict = {}


def _get_asr_model(asr_model_id: str, device: str = None):
    key = f"asr_{asr_model_id}_{device}"
    if key not in _model_cache:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen3-ASR model: {asr_model_id} on {device}...")
        model = Qwen3ASRModel.from_pretrained(
            asr_model_id,
            dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device,
        )
        _model_cache[key] = model
    return _model_cache[key]


def _get_tts_model(tts_model_id: str, device: str = None, no_flash_attn: bool = False):
    key = f"tts_{tts_model_id}_{device}_{no_flash_attn}"
    if key not in _model_cache:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Auto-detect if Flash Attention 2 is supported (Ampere or newer GPU)
        use_flash = False
        if "cuda" in device and not no_flash_attn:
            try:
                major, minor = torch.cuda.get_device_capability(0)
                if major >= 8:  # Ampere (8.x) or newer
                    use_flash = True
            except Exception:
                pass

        attn_impl = "flash_attention_2" if use_flash else None

        print(f"Loading Qwen3-TTS model: {tts_model_id} on {device}...")
        print(f"   → Flash Attention 2: {'ENABLED' if use_flash else 'DISABLED (older GPU or --no-flash-attn)'}")

        model = Qwen3TTSModel.from_pretrained(
            tts_model_id,
            dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device,
            attn_implementation=attn_impl,
        )
        _model_cache[key] = model
    return _model_cache[key]


def qwen3_voice_clone(
    reference_audio: str,
    generate_text: list,
    qwen3_asr_model: str = "Qwen/Qwen3-ASR-1.7B",
    qwen3_tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    output_path: str = "cloned_voice.wav",
    language: str = None,
    device: str = None,
    no_flash_attn: bool = False,
    **generate_kwargs,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    asr_model = _get_asr_model(qwen3_asr_model, device)
    tts_model = _get_tts_model(qwen3_tts_model, device, no_flash_attn)

    # Step 1: Auto-transcribe
    print(f"🔊 Transcribing reference audio with {qwen3_asr_model}...")
    results = asr_model.transcribe(reference_audio, language=None)
    ref_text = results[0].text.strip()
    detected_lang = results[0].language or "English"
    
    if language is None:
        language = detected_lang
    
    print(f"✅ Auto-generated ref_text: {ref_text[:150]}{'...' if len(ref_text) > 150 else ''}")
    print(f"🎤 Language: {language}")

    # Step 2: Generate cloned voice
    print(f"🎙️ Cloning voice with {qwen3_tts_model}... (generation params: {generate_kwargs})")
    wavs, sr = tts_model.generate_voice_clone(
        text=generate_text,
        language=language,
        ref_audio=reference_audio,
        ref_text=ref_text,
        **generate_kwargs,
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if len(wavs) == 1:
        sf.write(output_path, wavs[0], sr)
        print(f"💾 Saved → {output_path}")
    else:
        for i, wav in enumerate(wavs):
            sf.write(output_path.with_stem(f"{output_path.stem}_{i}"), wav, sr)
        print(f"💾 Saved {len(wavs)} files")

    return wavs, sr


# ============================ COMMAND LINE ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3 Voice Clone CLI (fixed for older GPUs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qwen3_voice_clone_cli.py --reference_audio alienkevin.wav --generate_text "Hello! This is my cloned voice."
  python qwen3_voice_clone_cli.py --reference_audio alienkevin.wav --generate_text "今天天氣真好！" --language Chinese --temperature 0.8 --top_p 0.95 --no-flash-attn
        """
    )

    # Required
    parser.add_argument("--reference_audio", type=str, required=True,
                        help="Reference voice (local path OR public URL)")
    parser.add_argument("--generate_text", type=str, nargs="+", required=True,
                        help="Text to speak (can be multiple sentences)")

    # Model selection
    parser.add_argument("--qwen3_asr_model", type=str, default="./Qwen3-ASR-0.6B",
                        help="Qwen3-ASR model path/ID")
    parser.add_argument("--qwen3_tts_model", type=str, default="./Qwen3-TTS-12Hz-0.6B-Base",
                        help="Qwen3-TTS model path/ID")

    # Output & language
    parser.add_argument("--output_path", type=str, default="cloned_voice.wav",
                        help="Output WAV file")
    parser.add_argument("--language", type=str, default=None,
                        help="Force language (English, Chinese, Cantonese, etc.)")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p nucleus sampling")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                        help="Repetition penalty")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum new tokens")

    # Flash Attention fix
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable Flash Attention 2 (use if you get 'FlashAttention only supports Ampere GPUs' error)")

    args = parser.parse_args()

    # Validation: check reference audio
    if not args.reference_audio.startswith(("http://", "https://")):
        if not os.path.exists(args.reference_audio):
            print(f"\n❌ ERROR: Reference audio file not found!\n"
                  f"   Path: {args.reference_audio}\n"
                  f"   Please use a real local file or a public URL.\n")
            exit(1)

    # Collect generation kwargs
    generate_kwargs = {}
    for k in ["temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"]:
        v = getattr(args, k)
        if v is not None:
            generate_kwargs[k] = v

    # Run
    wavs, sr = qwen3_voice_clone(
        reference_audio=args.reference_audio,
        generate_text=args.generate_text,
        qwen3_asr_model=args.qwen3_asr_model,
        qwen3_tts_model=args.qwen3_tts_model,
        output_path=args.output_path,
        language=args.language,
        no_flash_attn=args.no_flash_attn,
        **generate_kwargs,
    )

    print("🎉 Voice cloning completed successfully!")