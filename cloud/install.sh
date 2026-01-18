# install packages
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers tqdm ipdb accelerate==1.7.0 numpy==1.26.3 shortuuid fschat fastchat fastapi uvicorn pydantic msgpack pynvml --index-url https://pypi.tuna.tsinghua.edu.cn/simple

sudo apt install -y aria2
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh TheBloke/deepseek-coder-6.7B-instruct-GGUF --include deepseek-coder-6.7b-instruct.Q4_K_M.gguf --tool aria2c -x 10 --local-dir pre_models
./hfd.sh TheBloke/Llama-2-7b-Chat-GGUF --include llama-2-7b-chat.Q4_K_M.gguf --tool aria2c -x 10 --local-dir pre_models
