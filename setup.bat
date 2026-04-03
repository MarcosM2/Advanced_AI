python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install llama-cpp-python huggingface_hub
hf download bartowski/Qwen2.5-0.5B-Instruct-GGUF --include "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" --local-dir ./models
