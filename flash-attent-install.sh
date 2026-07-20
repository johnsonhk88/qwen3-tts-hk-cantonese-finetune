# The "hanging" is normal — it's not broken, it's just compiling CUDA kernels and taking a long time (or forever) because ninja is missing.
# Without ninja, flash-attn falls back to single-threaded compilation and can look completely stuck for 30–120+ minutes (many Ubuntu 22.04 # # users report exactly this). With ninja it usually finishes in 3–10 minutes.
# Fix it step-by-step (Ubuntu 22.04)

# 1. Install system build tools
sudo apt update
sudo apt install -y build-essential python3-dev ninja-build

# 2. Activate your venv and upgrade basics
source  ./venv-qwen3tts/bin/activate   # ← change to your venv path if different

pip install --upgrade pip wheel setuptools packaging psutil
pip install --upgrade pip setuptools wheel

# Force reinstall ninja (very important)
pip uninstall -y ninja
pip install ninja


#Recommended fix: Use a pre-built wheel for torch 2.10 + cu12 (no compilation!)
# Official flash-attn doesn't ship pre-built wheels for torch 2.10 yet (it's a newer version), but community members have built and shared# # compatible ones in the flash-attention GitHub iss
# The best match for you (Python 3.12, torch 2.10, CUDA 12.x family including 12.8):

# Direct install into your envirnment
pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"


# check install 
python -c "import flash_attn; print(flash_attn.__version__)"



pip install xformers==0.0.35 ---no-deps
pip install flash-attn==2.8.3 --no-deps