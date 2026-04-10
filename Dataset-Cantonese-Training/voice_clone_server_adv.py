import torch
import soundfile as sf
import numpy as np
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
    if not lang:
        return None
    lang = str(lang).lower().strip()
    if lang in ["cantonese", "yue", "cantonese (china)"]:
        return "chinese"
    return lang


# ====================== CORE VOICE CLONE FUNCTION ======================
def voice_clone(
    reference_audio_path: str,
    instructions: str,
    texts: str,
    qwen3_asr_model: str,
    qwen3_tts_model: str,
    language: str | None,
    temperature: float,
    top_p: float,
    no_flash_attn: bool,
):
    if not reference_audio_path:
        return None, "❌ 請上傳參考音頻。"

    if not instructions.strip() or not texts.strip():
        return None, "❌ 請輸入 Style Instruction 同 Dialog Text。"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    asr_model = _get_asr_model(qwen3_asr_model, device)
    tts_model = _get_tts_model(qwen3_tts_model, device, no_flash_attn)

    # Step 1: Auto-transcribe reference
    print("🔊 Transcribing reference audio with Qwen3-ASR...")
    results = asr_model.transcribe(reference_audio_path, language=None)
    ref_text = results[0].text.strip()
    detected_lang = results[0].language or "chinese"

    if not language:
        language = _normalize_language(detected_lang)
    else:
        language = _normalize_language(language)

    print(f"✅ Ref text: {ref_text[:120]}... | Language: {language}")

    # Step 2: Split instructions and texts
    inst_lines = [line.strip() for line in instructions.split("\n") if line.strip()]
    text_lines = [line.strip() for line in texts.split("\n") if line.strip()]

    if len(inst_lines) != len(text_lines):
        return None, f"❌ Instruction 行數 ({len(inst_lines)}) 同 Text 行數 ({len(text_lines)}) 唔一樣！"

    print(f"🎙️ 準備生成 {len(inst_lines)} 段香港粵語對話（使用獨立 instruct 參數）...")

    all_wavs = []
    sr_final = None

    generate_kwargs = {}
    if 0.1 <= temperature <= 1.2:
        generate_kwargs["temperature"] = temperature
    if 0.1 <= top_p <= 1.0:
        generate_kwargs["top_p"] = top_p

    for i, (inst, txt) in enumerate(zip(inst_lines, text_lines)):
        print(f"   ├─ 第 {i+1}/{len(inst_lines)} 段 → Instruction: {inst[:40]}... | Text: {txt[:45]}...")

        # ✅ 使用 Qwen3-TTS 官方獨立 instruct 參數
        wavs, sr = tts_model.generate_voice_clone(
            text=[txt],           # 只放真正要講的文字
            language=language,
            ref_audio=reference_audio_path,
            ref_text=ref_text,
            instruct=inst,        # 獨立語調指令（不會被讀出來）
            **generate_kwargs,
        )
        all_wavs.append(wavs[0])
        if sr_final is None:
            sr_final = sr

    # Step 3: 合併成單一WAV
    if len(all_wavs) > 1:
        print("🔗 正在合併所有語音段落...")
        concatenated = np.concatenate(all_wavs, axis=0)
    else:
        concatenated = all_wavs[0]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, concatenated, sr_final)
        output_path = tmp.name

    total_seconds = len(concatenated) / sr_final
    return output_path, f"✅ 成功！\n生成 {len(inst_lines)} 段對話\n總長度 ≈ {total_seconds:.1f} 秒\n已自動合併成單一WAV\n（已使用獨立 instruct）"


# ====================== GRADIO FRONTEND ======================
def create_demo():
    with gr.Blocks(title="🎤 Qwen3 Voice Clone Server - HK Cantonese（獨立 instruct）") as demo:
        gr.Markdown("# 🎤 Qwen3 Voice Clone Web Server\n"
                    "**香港廣東話專用版**｜**已使用獨立 instruct 參數**")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="📤 上傳參考聲音（要Clone嘅聲音）",
                    type="filepath",
                )

            with gr.Column(scale=2):
                instruction_input = gr.Textbox(
                    label="🎯 Style Instruction（每行一句語調指令）",
                    placeholder="每行輸入一句語調指令，例如：\n用熱情專業香港客服語調，語速正常，帶微笑\n用急切擔心嘅語氣，語速稍快",
                    lines=18,
                    max_lines=20,
                )
                text_input = gr.Textbox(
                    label="✍️ Dialog Text（每行對應講嘅文字）",
                    placeholder="每行輸入對應的對話文字（行數必須同左邊一樣）\n例如：\n您好，歡迎致電The Club客戶服務中心！請問有咩可以幫到您？",
                    lines=18,
                    max_lines=20,
                )

            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="🔊 生成後嘅完整克隆語音（已合併）",
                    type="filepath",
                    interactive=False,
                )
                status = gr.Textbox(label="Status", interactive=False)

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                asr_dropdown = gr.Dropdown(
                    choices=["./Qwen3-ASR-0.6B", "./Qwen3-ASR-1.7B"],
                    value="./Qwen3-ASR-0.6B",
                    label="Qwen3-ASR Model",
                )
                tts_dropdown = gr.Dropdown(
                    choices=["./Qwen3-TTS-12Hz-0.6B-Base", "./Qwen3-TTS-12Hz-1.7B-Base"],
                    value="./Qwen3-TTS-12Hz-0.6B-Base",
                    label="Qwen3-TTS Model",
                )

            language_dropdown = gr.Dropdown(
                choices=["Auto", "Chinese", "Cantonese", "English"],
                value="Auto",
                label="Language（粵語請選Auto或Cantonese）",
            )

            with gr.Row():
                temperature_slider = gr.Slider(0.1, 1.2, value=0.75, step=0.05, label="Temperature（0.7左右最自然）")
                top_p_slider = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")

            no_flash = gr.Checkbox(label="Disable Flash Attention", value=False)

        generate_btn = gr.Button("🚀 生成完整香港粵語對話（自動合併）", variant="primary", size="large")

        def generate_wrapper(ref_audio, inst_text, txt_text, asr, tts, lang, temp, tp, no_flash):
            if lang == "Auto":
                lang = None
            elif lang == "Cantonese":
                lang = "chinese"

            return voice_clone(
                reference_audio_path=ref_audio,
                instructions=inst_text,
                texts=txt_text,
                qwen3_asr_model=asr,
                qwen3_tts_model=tts,
                language=lang,
                temperature=temp,
                top_p=tp,
                no_flash_attn=no_flash,
            )

        generate_btn.click(
            fn=generate_wrapper,
            inputs=[audio_input, instruction_input, text_input, asr_dropdown, tts_dropdown,
                    language_dropdown, temperature_slider, top_p_slider, no_flash],
            outputs=[output_audio, status],
        )

        gr.Markdown("### 💡 使用提示\n"
                    "• 兩個文字框已清空（無預設值）\n"
                    "• 直接貼上你之前嘅17行 Instruction 同 Text\n"
                    "• Instruction 會作為獨立 instruct 參數（不會被讀出來）\n"
                    "• 上傳 reference audio → 按生成 → 一次出完整對話WAV")

    return demo


if __name__ == "__main__":
    print("🌐 Starting Qwen3 Voice Clone Server - HK Cantonese（獨立 instruct + 無預設值）...")
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )