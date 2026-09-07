# Repository Agent Instructions

## GPU Usage

- Only GPU0, GPU1, GPU2, and GPU3 may be used in this repository.
- GPU4, GPU5, GPU6, and GPU7 are strictly prohibited.
- Every GPU command must explicitly restrict visibility to the permitted devices, for example with `CUDA_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=1`, `CUDA_VISIBLE_DEVICES=2`, `CUDA_VISIBLE_DEVICES=3`, or a subset/list containing only devices 0–3.
- Never run a command, experiment, test, training job, evaluation, or monitoring task on GPU4–7.
- Before launching GPU work, verify that the selected device IDs are all within 0–3.
