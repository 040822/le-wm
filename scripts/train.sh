# Examples:
# tmux new -s train

# bash scripts/train.sh tworoom 7


data_name=${1}
gpu_id=${2}

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python -u train.py data=${data_name}

# if [[ -z "${info}" ]]; then
#     python -u train.py data=${data_name} 2>&1 | tee "Temp/${data_name}.out"
# else
#     python -u train.py data=${data_name} 2>&1 | tee "Temp/${data_name}_${info}.out"
# fi

