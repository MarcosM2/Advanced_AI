REM python -m venv venv
REM call venv\Scripts\activate
REM python -m pip install --upgrade pip
REM pip install -r requirements.txt
REM pip install llama-cpp-python huggingface_hub
REM hf download bartowski/Qwen2.5-0.5B-Instruct-GGUF --include "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" --local-dir ./models
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --prefer-binary llama-cpp-python
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/Qwen2.5-0.5B-Instruct-GGUF', filename='Qwen2.5-0.5B-Instruct-Q4_K_M.gguf', local_dir='./models')"
