python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
set HF_HUB_DISABLE_XET=1
hf download bartowski/Qwen2.5-0.5B-Instruct-GGUF --include "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" --local-dir ./models
