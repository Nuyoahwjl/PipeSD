# install packages
pip install transformers ipdb accelerate==1.7.0 numpy==1.26.3 shortuuid fschat fastchat fastapi uvicorn pydantic msgpack pynvml --index-url https://pypi.tuna.tsinghua.edu.cn/simple

sudo apt install -y aria2
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh TheBloke/deepseek-coder-1.3b-instruct-GGUF --include deepseek-coder-1.3b-instruct.Q4_K_M.gguf --tool aria2c -x 10  --local-dir pre_models
./hfd.sh TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --include tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --tool aria2c -x 10  --local-dir pre_models