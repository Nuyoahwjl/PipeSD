# install packages
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers==4.55.4 tqdm ipdb accelerate numpy=1.26.3 shortuuid fschat fastchat scikit-optimize

# get fastchat for mt-bench
# git clone https://github.com/lm-sys/FastChat.git
# mv FastChat/fastchat/ fastchat/
# rm -rf FastChat

# get ouroboros
# git clone https://github.com/thunlp/Ouroboros.git