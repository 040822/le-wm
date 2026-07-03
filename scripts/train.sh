# Examples:
# tmux new -s train

# bash scripts/train.sh lewm tworoom 1
# bash scripts/train.sh lewm pusht 0

# bash scripts/train.sh value_jepa tworoom 0,1
# bash scripts/train.sh value_jepa pusht 0,1,2,3
# bash scripts/train.sh value_jepa cube 0,1,2,3
# bash scripts/train.sh value_jepa reacher 4,5,6,7



config_name=${1}
data_name=${2}
gpu_id=${3}
run_time=$(date +"%Y%m%d_%H%M%S_%N")
output_dir="./outputs/${config_name}/${data_name}/${run_time}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/data}"

# python -u train.py data=${data_name}

if [[ -z "${info}" ]]; then
    python -u train.py --config-name=${config_name} data=${data_name} hydra.run.dir=${output_dir} 2>&1 | tee "Temp/${config_name}_${data_name}.out"
else
    python -u train.py --config-name=${config_name} data=${data_name} hydra.run.dir=${output_dir} 2>&1 | tee "Temp/${config_name}_${data_name}_${info}.out"
fi
