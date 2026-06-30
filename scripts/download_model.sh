# bash scripts/download_model.sh

export http_proxy=http://10.126.126.3:7889
export https_proxy=http://10.126.126.3:7889

# export http_proxy=http://10.126.126.5:7890
# export https_proxy=http://10.126.126.5:7890

export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/data}"
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_ENDPOINT="https://huggingface.co"
# export HF_DEBUG=1

mkdir -p "$STABLEWM_HOME/checkpoints" # -p 参数会在parent directory不存在时也创建parent directory

echo "$HF_ENDPOINT"

# 下载全部数据集.
for repo in lewm-pusht lewm-cube lewm-reacher lewm-tworooms; do
MODEL_PATH="$STABLEWM_HOME/checkpoints/quentinll/$repo"
  hf download "quentinll/$repo" \
    --local-dir "$MODEL_PATH"
done
