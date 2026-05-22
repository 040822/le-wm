# bash scripts/download_dataset.sh
# TODO:目前无法下载，后续需要解决下载问题

export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/data}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DEBUG=1
export http_proxy=http://10.126.126.1:7897
export https_proxy=http://10.126.126.1:7897

mkdir -p "$STABLEWM_HOME" # -p 参数会在parent directory不存在时也创建parent directory

echo "$HF_ENDPOINT"

# 下载全部数据集，太大了下载不下来，先下载 lewm-cube，后续需要再下载其他数据集
# for repo in lewm-pusht lewm-cube lewm-reacher lewm-tworooms; do
#   hf download "quentinll/$repo" \
#     --repo-type dataset \
#     --include "*.h5.zst" \
#     --local-dir "$STABLEWM_HOME"
# done

hf download "quentinll/lewm-tworooms" \
    --repo-type dataset \
    --local-dir "$STABLEWM_HOME"

cd 

tar --zstd -xvf *.tar.zst