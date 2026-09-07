# bash scripts/eval.sh



python eval.py --config-name=cube.yaml policy=quentinll/lewm-cube/weights.pt
python eval.py --config-name=pusht.yaml policy=quentinll/lewm-pusht/weights.pt
python eval.py --config-name=reacher.yaml policy=quentinll/lewm-reacher/weights.pt
python eval.py --config-name=tworoom.yaml policy=quentinll/lewm-tworooms/weights.pt

python eval.py --config-name=cube.yaml policy=quentinll/lewm-cube/weights.pt
python eval.py --config-name=pusht.yaml policy=quentinll/lewm-pusht/weights.pt
python eval.py --config-name=reacher.yaml policy=quentinll/lewm-reacher/weights.pt
python eval.py --config-name=tworoom.yaml policy=outputs/lewm/tworoom/0721_/checkpoints/lewm_weights_epoch_10.pt

