import torch
import soundfile as sf
from pathlib import Path
import gradio as gr
import os
import tempfile

from qwen_tts import Qwen3TTSModel
from qwen_asr import Qwen3ASRModel

# ====================== MODEL CACHE ======================
_model_cache = {}


def _get_asr_model(asr_model_id: str, device: str = None):
    key = f"asr_{asr_model_id}_{device}"
    if key not in _model_cache:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Qwen3-ASR: {asr_model_id} on {device}...")
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

        use_flash = False
        if "cuda" in device and not no_flash_attn:
            try:
                major, _ = torch.cuda.get_device_capability(0)
                if major >= 8:
                    use_flash = True
            except Exception:
                pass

        attn_impl = "flash_attention_2" if use_flash else None
        print(f"🚀 Loading Qwen3-TTS: {tts_model_id} on {device} → Flash Attn: {'ON' if use_flash else 'OFF'}")

        model = Qwen3TTSModel.from_pretrained(
            tts_model_id,
            dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device,
            attn_implementation=attn_impl,
        )
        _model_cache[key] = model
    return _model_cache[key]


def _normalize_language(lang: str | None) -> str | None:
    """Qwen3-TTS only accepts lowercase 'chinese' (covers Cantonese too)."""
    if not lang:
        return None
    lang = str(lang).lower().strip()
    if lang in ["cantonese", "yue", "cantonese (china)"]:
        return "chinese"
    return lang


# ====================== CORE VOICE CLONE FUNCTION ======================
def voice_clone(
    reference_audio_path: str,
    generate_text: str,
    qwen3_asr_model: str,
    qwen3_tts_model: str,
    language: str | None,
    temperature: float,
    top_p: float,
    no_flash_attn: bool,
):
    if not reference_audio_path or not generate_text.strip():
        return None, "❌ Please upload a reference audio and enter text."

    device = "cuda" if torch.cuda.is_available() else "cpu"

    asr_model = _get_asr_model(qwen3_asr_model, device)
    tts_model = _get_tts_model(qwen3_tts_model, device, no_flash_attn)

    # Step 1: Auto-transcribe reference
    print("🔊 Transcribing reference audio with Qwen3-ASR...")
    results = asr_model.transcribe(reference_audio_path, language=None)
    ref_text = results[0].text.strip()
    detected_lang = results[0].language or "chinese"

    # Normalize detected language (Cantonese → chinese)
    if not language:  # user chose Auto
        language = _normalize_language(detected_lang)
    else:
        language = _normalize_language(language)

    print(f"✅ Ref text: {ref_text[:120]}... | Final language for TTS: {language}")

    # Step 2: Generate cloned voice
    print("🎙️ Generating cloned voice with Qwen3-TTS...")
    generate_kwargs = {}
    if temperature is not None and 0.1 <= temperature <= 1.2:
        generate_kwargs["temperature"] = temperature
    if top_p is not None and 0.1 <= top_p <= 1.0:
        generate_kwargs["top_p"] = top_p

    wavs, sr = tts_model.generate_voice_clone(
        text=[generate_text],
        language=language,
        ref_audio=reference_audio_path,
        ref_text=ref_text,
        **generate_kwargs,
    )

    # Save to temporary file for Gradio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wavs[0], sr)
        output_path = tmp.name

    return output_path, f"✅ Success!\nAuto-detected ref text: {ref_text[:150]}..."


# ====================== GRADIO FRONTEND (Gradio 6.0+ compatible) ======================
def create_demo():
    with gr.Blocks(title="🎤 Qwen3 Voice Clone Server") as demo:
        gr.Markdown("# 🎤 Qwen3 Voice Clone Web Server\n"
                    "**Upload your voice → Type any text → Get perfect clone** (HK Cantonese supported!)")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="📤 Upload Reference Voice (your voice to clone)",
                    type="filepath",
                )
                text_input = gr.Textbox(
                    label="✍️ Text to Speak in Cloned Voice",
                    placeholder="例如：昨晚我嘗試咗去做一個 vegetarian pizza...",
                    lines=3,
                )

            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="🔊 Generated Cloned Voice",
                    type="filepath",
                    interactive=False,
                )
                status = gr.Textbox(label="Status", interactive=False)

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                asr_dropdown = gr.Dropdown(
                    choices=["./Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"],
                    value="./Qwen3-ASR-0.6B",
                    label="Qwen3-ASR Model",
                )
                tts_dropdown = gr.Dropdown(
                    choices=["./Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"],
                    value="./Qwen3-TTS-12Hz-0.6B-Base",
                    label="Qwen3-TTS Model",
                )

            language_dropdown = gr.Dropdown(
                choices=["Auto", "Chinese", "Cantonese", "English"],
                value="Auto",
                label="Language (Auto recommended)",
            )

            with gr.Row():
                temperature_slider = gr.Slider(0.1, 1.2, value=0.8, step=0.05, label="Temperature (creativity)")
                top_p_slider = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")

            no_flash = gr.Checkbox(label="Disable Flash Attention (use if GPU is old)", value=False)

        generate_btn = gr.Button("🚀 Generate Cloned Voice", variant="primary", size="large")

        # Click handler with language normalization
        def generate_wrapper(ref_audio, text, asr, tts, lang, temp, tp, no_flash):
            # Map user selection to what TTS accepts
            if lang == "Auto":
                lang = None
            elif lang == "Cantonese":
                lang = "chinese"

            return voice_clone(
                reference_audio_path=ref_audio,
                generate_text=text,
                qwen3_asr_model=asr,
                qwen3_tts_model=tts,
                language=lang,
                temperature=temp,
                top_p=tp,
                no_flash_attn=no_flash,
            )

        generate_btn.click(
            fn=generate_wrapper,
            inputs=[audio_input, text_input, asr_dropdown, tts_dropdown, language_dropdown,
                    temperature_slider, top_p_slider, no_flash],
            outputs=[output_audio, status],
        )

        gr.Markdown("### 💡 Tips\n"
                    "- First generation loads models (10–30 seconds)\n"
                    "- Cantonese voice works perfectly (auto-mapped to 'chinese')\n"
                    "- Best with 5–30 second clear reference audio")

    return demo


if __name__ == "__main__":
    print("🌐 Starting Qwen3 Voice Clone Web Server...")
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )