# Examples:
# tmux new -s train

# bash scripts/train.sh lewm tworoom 0
# bash scripts/train.sh lewm pusht 1
# bash scripts/train.sh lewm reacher 2
# bash scripts/train.sh lewm cube 3


# bash scripts/train.sh value_jepa tworoom 0,1
# bash scripts/train.sh value_jepa pusht 0,1,2,3
# bash scripts/train.sh value_jepa cube 0,1,2,3
# bash scripts/train.sh value_jepa reacher 4,5,6,7


# bash scripts/train.sh fast_lewam tworoom 0
# bash scripts/train.sh fast_lewam pusht 1
# bash scripts/train.sh fast_lewam reacher 2
# bash scripts/train.sh fast_lewam cube 3



config_name=${1}
data_name=${2}
gpu_id=${3}
info=${4:-""}
run_time=$(date +"%m%d")
id_name="${run_time}_${info}"
output_dir="./outputs/${config_name}/${data_name}/${id_name}"

if [[ $# -ge 4 ]]; then
    shift 4
    extra_args=("$@")
fi


export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/data}"

# python -u train.py data=${data_name}

python -u train.py --config-name=${config_name} data=${data_name} hydra.run.dir=${output_dir} info=${info} "${extra_args[@]}" 2>&1 | tee "Temp/${config_name}_${data_name}_${info}.out"
