# bash scripts/eval.sh

# python eval.py --config-name=pusht.yaml policy=quentinll/lewm-pusht/weights.pt
python eval.py --config-name=tworoom.yaml policy=quentinll/lewm-tworooms/weights.pt

python eval.py --config-name=tworoom.yaml policy=/home/wenxin/office/pre-exp/le-wm/outputs/20260630_214928_485796/value_jepa_object.ckpt

