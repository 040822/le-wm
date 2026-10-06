#!/usr/bin/env python3
"""Reacher baseline prefix ablation; keep five-block proposals for both methods.

LeWM keeps its native CEM; LeFlow keeps its native 64-path/16-step sampler.
Only the rollout scoring prefix and execution prefix change. This is not
short-horizon LeFlow path generation (which does not support one block).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def gpu_snapshot(gpu):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    query = subprocess.run(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,uuid,name,driver_version', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()
    fields = next(csv.reader([query], skipinitialspace=True))
    if len(fields) != 8 or int(fields[0]) != gpu:
        raise RuntimeError(f'unexpected nvidia-smi preflight response: {query!r}')
    snapshot = {
        'physical_id': int(fields[0]), 'memory_total_mib': int(fields[1]),
        'memory_used_mib': int(fields[2]), 'memory_free_mib': int(fields[3]),
        'utilization_percent': int(fields[4]), 'uuid': fields[5],
        'name': fields[6], 'driver_version': fields[7],
    }
    if snapshot['memory_free_mib'] < 12 * 1024:
        raise RuntimeError(
            f"GPU {gpu} has {snapshot['memory_free_mib'] / 1024:.2f} GiB free; "
            'the evaluation safety margin is 12 GiB'
        )
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['lewm', 'leflow'])
    parser.add_argument('--steps', type=int, choices=[25, 10, 5], required=True)
    parser.add_argument('--budget', type=int, choices=[50, 60, 75, 90, 110, 125], default=50)
    parser.add_argument('--output-root', default='outputs/round5/phase5_baseline_horizons')
    args = parser.parse_args()
    from source.common.gpu_environment import configure_mujoco_egl_device
    gpu = configure_mujoco_egl_device()
    if gpu not in range(4):
        raise ValueError('This runner is restricted to GPU0–3')
    gpu_before = gpu_snapshot(gpu)
    import numpy as np
    import torch
    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import DatasetEvaluationSession, EvaluationIdentity, compose_eval_config
    from source.common.phase3_compat import patch_phase3_environments
    from source.common.round3_phase1 import CohortManifest
    from source.common.round4_eval import _TimedSolver

    patch_phase3_environments()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU fallback')
    free, _ = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise RuntimeError('Less than 12 GiB free VRAM')
    checkpoint = ROOT / ('data/checkpoints/leflow/reacher/latent_planner.pt'
                         if args.method == 'leflow' else
                         'outputs/lewm/reacher/0721_/checkpoints/lewm_weights_epoch_10.pt')
    manifest = CohortManifest.load(ROOT / 'outputs/round3/phase1/cohorts/reacher/legacy_50.json')
    expected = json.loads((ROOT / 'config/round5/phase5_pre_report2.json').read_text())['tasks']['reacher']['cohort_sha256']
    assert manifest.computed_sha256 == expected and len(manifest.entries) == 50
    output_root = ROOT / args.output_root
    if args.budget != 50:
        output_root = output_root / f'budget_{args.budget}'
    output = output_root / args.method / f'{args.steps}_{args.steps}'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    cfg = compose_eval_config('reacher', overrides=[
        'output.save_video=false', 'plan_config.horizon=5',
        f'plan_config.receding_horizon={args.steps // 5}',
        f'eval.eval_budget={args.budget}',
    ])
    model, resolved = load_policy_or_model(str(checkpoint))
    model = getattr(model, 'model', model)
    blocks = args.steps // 5
    score_calls = []
    if args.method == 'lewm':
        original = model.get_cost
        def prefix_cost(self, info, actions):
            # Native CEM supplies candidate blocks directly; current observation
            # history length is one in this legacy protocol.
            if info['pixels'].shape[2] != 1:
                raise ValueError('Unexpected observation history; prefix alignment requires audit')
            assert actions.shape[2] == 5
            if not score_calls:
                score_calls.append({'candidate_blocks': 5, 'scored_blocks': blocks})
            return original(info, actions[:, :, :blocks])
        model.get_cost = types.MethodType(prefix_cost, model)
    else:
        original = model.rollout_final_latent
        def prefix_rollout(self, start, actions, history_size=3):
            assert actions.shape[2] == 5
            if not score_calls:
                score_calls.append({'candidate_blocks': 5, 'scored_blocks': blocks})
            return original(start, actions[:, :, :blocks], history_size)
        model.rollout_final_latent = types.MethodType(prefix_rollout, model)
    session = DatasetEvaluationSession(cfg, task='reacher', cohort=manifest.to_evaluation_cohort())
    identity = EvaluationIdentity(entrypoint='round5_phase5_baseline_horizons',
        policy_kind=args.method, checkpoint=str(resolved or checkpoint), epoch=10, stage=None)
    policy = session._build_policy(model, identity, 'cuda:0')
    timer = _TimedSolver(policy.solver)
    policy.solver = timer
    session.evaluate(policy, identity=identity, output_dir=output, device='cuda:0')
    gpu_after = gpu_snapshot(gpu)
    path = output / 'result.json'
    result = json.loads(path.read_text())
    assert len(result['episodes']) == 50 and score_calls
    result['horizon_ablation'] = {
        'method': args.method, 'execute_steps': args.steps, 'score_steps': args.steps,
        'eval_budget': args.budget,
        'proposal_steps': 25, 'action_block': 5, 'gpu': gpu,
        'gpu_before': gpu_before, 'gpu_after': gpu_after,
        'checkpoint_sha256': sha(checkpoint), 'runner_sha256': sha(__file__),
        'cohort_sha256': expected, 'score_audit': score_calls,
        'protocol': 'legacy native baseline; no added candidate clipping',
        'planning_events': timer.events,
    }
    path.write_text(json.dumps(result, indent=2) + '\n')
    successes = sum(e['success'] for e in result['episodes'])
    print(json.dumps({'method': args.method, 'steps': args.steps, 'successes': successes,
                      'n': 50, 'result': str(path)}), flush=True)


if __name__ == '__main__':
    main()
