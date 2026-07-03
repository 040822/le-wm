# bash scripts/train_resume.sh

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/data}"

python -u train.py --config-name=value_jepa data=pusht \
  subdir=20260703_011737_977269 \
  hydra.run.dir=outputs/20260703_011737_977269 \
  loader.batch_size=128 \
  '+resume_ckpt="/home/wenxin/office/pre-exp/le-wm/outputs/20260703_011737_977269/LeWorldModel/20260703_011737_977269/checkpoints/epoch=14-step=52245.ckpt"'