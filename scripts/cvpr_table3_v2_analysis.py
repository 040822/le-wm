#!/usr/bin/env python3
"""Independent raw-record Table 3 v2 analysis; never writes human reports."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'outputs/cvpr/table3/v1'
V2 = ROOT / 'outputs/cvpr/table3/v2'
ANALYSIS_LOCK = V2 / 'analysis/analysis_lock.json'
METHODS = ('cowm_po', 'lewm_po', 'cowm_gf')
B = 10_000
RNG_SEED = 20261005


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        result = [json.loads(line) for line in f if line.strip()]
    if len(result) != 50 or [r['slot'] for r in result] != list(range(50)):
        raise ValueError(f'invalid 50-slot coverage: {path}')
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(path, expected):
    require(sha(path) == expected, f'SHA256 mismatch: {path}')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def cost(task, current, goal):
    a, b = np.asarray(current, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    require(a.ndim == b.ndim == 1 and a.shape == b.shape and a.size > 0, f'invalid {task} state shape')
    require(np.isfinite(a).all() and np.isfinite(b).all(), f'nonfinite {task} state')
    d = a - b
    if task == 'pusht':
        require(d.size >= 5, 'PushT state lacks angle')
        return float(max(np.linalg.norm(d[:4]) / 20., abs((d[4] + np.pi) % (2*np.pi) - np.pi) / (np.pi/9)))
    return float(np.linalg.norm(d))


def ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='stable')
    out = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2.
        i = j
    return out


def ranking(predicted, truth):
    p, t = np.asarray(predicted, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    require(len(p) == len(t) == 64 and np.isfinite(p).all() and np.isfinite(t).all(), 'ranking requires finite complete 64-pool')
    ordered = np.lexsort((np.arange(64), p))
    oracle = np.flatnonzero(t == t.min())
    pr, tr = ranks(p), ranks(t)
    rho = None if np.ptp(pr) == 0 or np.ptp(tr) == 0 else float(np.corrcoef(pr, tr)[0, 1])
    m = len(oracle)
    return dict(spearman=rho, hit_at_5=float(bool(set(ordered[:5]) & set(oracle))),
                hit_at_1=float(ordered[0] in oracle), selection_regret=float(t[ordered[0]]-t.min()),
                random_hit_at_5=1-(math.comb(64-m, 5)/math.comb(64, 5) if 64-m >= 5 else 0.),
                uniform_expected_regret=float(t.mean()-t.min()), oracle_count=m, selected_index=int(ordered[0]),
                true_constant=bool(np.ptp(t) == 0), predicted_constant=bool(np.ptp(p) == 0))


def at25(task, row):
    if row.get('valid_at_25') is not True:
        return None
    state = row.get('milestone_state_at_25')
    require(isinstance(state, dict) and row.get('effective_valid_length', 0) >= 25, 'valid25 lacks actual milestone')
    require(state['local_primitive_step'] == 25, 'wrong actual step25 milestone')
    value = cost(task, state['current'], state['goal'])
    require(math.isclose(value, row['physical_cost_at_25'], rel_tol=0, abs_tol=1e-6), 'step25 source cost mismatch')
    return value


def identity(a, b):
    for key in ('slot', 'state_id', 'episode_id', 'row_index', 'start_step', 'candidate_index'):
        require(a[key] == b[key], f'identity mismatch: {key}')


def accepted(path):
    a = read(path / 'acceptance.json')
    require(a.get('status') == 'pass' and not a.get('errors'), f'unaccepted cell: {path}')
    check(path / 'summary.json', a['summary_sha256'])
    return a, read(path / 'summary.json')


def source_cell(task, seed, ref):
    aroot, broot = V1 / '3a' / task / f'seed_{seed}', V1 / '3b' / task / f'seed_{seed}'
    for root, rr in ((aroot, ref['3a']), (broot, ref['3b'])):
        for name in ('acceptance', 'summary', 'frozen_config'):
            check(root / f'{name}.json', rr[f'{name}_sha256'])
        acceptance, summary = accepted(root)
        if root == aroot:
            require(acceptance['fixed_pool_sha256'] == summary['fixed_pool_sha256'] == rr['fixed_pool_sha256'], '3a pool acceptance hash chain mismatch')
        else:
            require(acceptance['action_archive_sha256'] == summary['action_archive_sha256'] == rr['actions_sha256'], '3b action acceptance hash chain mismatch')
            require(acceptance['frozen_config_sha256'] == rr['frozen_config_sha256'], '3b frozen acceptance hash chain mismatch')
            require(acceptance.get('checks', {}).get('candidate0_branch_matches_3a') is True, 'accepted 3b source lacks its candidate0-to-3a match check')
            baseline_check = summary.get('restore_check', {})
            require(baseline_check.get('status') == 'pass' and baseline_check.get('checked_states') == 50, '3b baseline replay did not cover all 50 3a source states')
        require(summary['task'] == task and summary['evaluation_seed'] == seed, 'source task/seed mismatch')
        require(summary['branch_file_sha256'] == rr['branch_file_sha256'], 'source branch manifest mismatch')
    check(aroot / 'fixed_pool.npz', ref['3a']['fixed_pool_sha256'])
    check(broot / 'actions.npz', ref['3b']['actions_sha256'])
    af, bf = read(aroot / 'frozen_config.json'), read(broot / 'frozen_config.json')
    dataset_ref = ref['3a']['dataset']
    stat = (ROOT / dataset_ref['path']).stat()
    require(stat.st_size == dataset_ref['file_size_bytes'] and stat.st_mtime_ns == dataset_ref['mtime_ns'], 'dataset changed since full-file hash audit')
    require(af['dataset']['sha256'] == dataset_ref['sha256'], 'source dataset hash identity mismatch')
    require(bf['source_fixed_pool_sha256'] == ref['3a']['fixed_pool_sha256'], '3b wrong source pool')
    require(bf['protocol']['improvement_tolerance'] == 1e-6, 'wrong 3b improvement tolerance')
    pool = dict(np.load(aroot / 'fixed_pool.npz', allow_pickle=False))
    action = dict(np.load(broot / 'actions.npz', allow_pickle=False))
    require(np.array_equal(action['candidate0_original_normalized'], pool['candidates'][:, 0]), '3b baseline not exact index0')
    require(np.array_equal(action['candidate0_original_physical'], pool['physical_actions'][:, 0]), '3b physical baseline mismatch')
    require(np.array_equal(action['candidate0_noise'], pool['candidate_noise'][:, 0]), '3b noise mismatch')
    old = []
    for i in range(64):
        name = f'candidate_{i:04d}.jsonl'
        check(aroot / 'branches' / name, ref['3a']['branch_file_sha256'][name])
        rr = rows(aroot / 'branches' / name)
        for slot, row in enumerate(rr):
            require(row['candidate_index'] == i, 'candidate index mismatch')
            require(np.array_equal(row['action'], pool['candidates'][slot, i]), 'frozen candidate action mismatch')
            require(np.array_equal(row['physical_action'], pool['physical_actions'][slot, i]), 'frozen physical action mismatch')
            require(row['predicted_cost_cowm_b'] == float(pool['cowm_b_costs'][slot, i]) and row['predicted_cost_lewm'] == float(pool['lewm_costs'][slot, i]), 'source score mismatch')
        old.append(rr)
    variants = {}
    for name, h in ref['3b']['branch_file_sha256'].items():
        path = broot / 'branches' / f'{name}.jsonl'
        check(path, h)
        rr = rows(path)
        for slot, row in enumerate(rr):
            identity(old[0][slot], row)
            require(row['variant'] == name, 'variant identity mismatch')
            key = row['variant_action_archive_key']
            arr = np.ascontiguousarray(action[key][slot])
            require(hashlib.sha256(arr.tobytes()).hexdigest() == row['variant_action_sha256'], 'variant action archive mismatch')
        variants[name] = rr
    for method in METHODS:
        baseline_legal = action['candidate0_environment_legalized_normalized']
        updated_legal = action[f'{method}_environment_legalized_normalized']
        targets = np.sqrt(np.mean((updated_legal.astype(np.float64)-baseline_legal.astype(np.float64))**2, axis=(1,2)))
        for j in range(5):
            name = f'{method}_random_{j}'
            for slot, row in enumerate(variants[name]):
                match = row['random_match']
                require(math.isclose(targets[slot], match['target_rms_normalized_after_native_handling'], rel_tol=1e-5, abs_tol=1e-7), 'random target does not match method action displacement')
                if match['status'] == 'matched':
                    actual = float(np.sqrt(np.mean((action[row['variant_action_archive_key']][slot].astype(np.float64)-baseline_legal[slot].astype(np.float64))**2)))
                    require(math.isclose(actual, match['actual_rms_normalized_after_native_handling'], rel_tol=1e-5, abs_tol=1e-7), 'random amplitude record disagrees with archived action')
    # The legacy 3b restore acceptance does not preserve its baseline branch
    # rows. Reconstruct index-0 step-25 cost from accepted 3a and cross-check
    # every persisted per-state true improvement in the 3b summary.
    source_metrics = summary.get('metrics', {})
    baseline_pair_validation = dict(
        source_branches_checked=0,
        valid_pairs_checked=0,
        unavailable_pairs=0,
        max_absolute_delta_difference=0.0,
    )
    for name, branch_rows in variants.items():
        per_state = source_metrics.get(name, {}).get('per_state_true_cost_improvement')
        require(isinstance(per_state, list) and len(per_state) == 50, f'missing v1 per-state true improvements: {name}')
        baseline_pair_validation['source_branches_checked'] += 1
        for slot, row in enumerate(branch_rows):
            original = at25(task, old[0][slot])
            updated = at25(task, row)
            eligible = name in METHODS or row.get('eligible_for_equal_rms_control') is True
            expected = original - updated if eligible and original is not None and updated is not None else None
            recorded = per_state[slot]
            if expected is None:
                require(recorded is None, f'v1 improvement exists without a valid matched step25 pair: {name}/slot={slot}')
                baseline_pair_validation['unavailable_pairs'] += 1
            else:
                require(recorded is not None and math.isfinite(float(recorded)), f'missing v1 paired true improvement: {name}/slot={slot}')
                difference = abs(float(recorded) - expected)
                baseline_pair_validation['max_absolute_delta_difference'] = max(baseline_pair_validation['max_absolute_delta_difference'], difference)
                require(math.isclose(float(recorded), expected, rel_tol=0, abs_tol=1e-6), f'v1 3b baseline/update delta disagrees with 3a candidate0 reconstruction: {name}/slot={slot}')
                baseline_pair_validation['valid_pairs_checked'] += 1
    return pool, old, variants, baseline_pair_validation


def replay_cell(task, seed, path, ref, frozen_sha, refs_sha, old, pool):
    acceptance, summary = accepted(path)
    require(isinstance(acceptance.get('checks'), dict) and all(v is True for v in acceptance['checks'].values()), 'replay acceptance contains a false or missing check')
    check(path / 'frozen_config.json', summary['frozen_config_sha256'])
    f = read(path / 'frozen_config.json')
    require(f['task'] == task and f['evaluation_seed'] == seed, 'replay task/seed mismatch')
    require(f['global_frozen_config_sha256'] == frozen_sha and f['global_source_refs_sha256'] == refs_sha, 'replay global freeze mismatch')
    require(f['source_fixed_pool_sha256'] == ref['3a']['fixed_pool_sha256'], 'replay wrong pool')
    check(path / 'fixed_pool.npz', ref['3a']['fixed_pool_sha256'])
    check(path / 'source_refs.json', f['source_refs_sha256'])
    require(f['restore_check']['status'] == f['candidate_order_check']['status'] == 'pass', 'restore/order check failed')
    for field, source in (('loaded_actor_checkpoint_sha256', 'actor_checkpoint_sha256'), ('loaded_lewm_checkpoint_sha256', 'lewm_checkpoint_sha256')):
        require(f[field] == ref['3a'][source], 'loaded checkpoint mismatch')
    require(f['dataset_sha256'] == ref['3a']['dataset']['sha256'], 'replay dataset mismatch')
    require(f['action_normalizer']['sha256'] == ref['3a']['action_normalizer_sha256'], 'replay action normalizer hash mismatch')
    require(f['protocol']['action_interop_audit']['status'] == 'pass', 'replay normalized/physical action interop audit failed')
    out = []
    for i in range(64):
        name = f'candidate_{i:04d}.jsonl'
        require(summary['branch_file_sha256'][name] == f['branch_file_sha256'][name], 'branch hash chain mismatch')
        check(path / 'branches' / name, f['branch_file_sha256'][name])
        rr = rows(path / 'branches' / name)
        trace_name = name + '.gz'
        require(summary['trace_file_sha256'][trace_name] == f['trace_file_sha256'][trace_name], 'trace hash chain mismatch')
        check(path / 'traces' / trace_name, f['trace_file_sha256'][trace_name])
        traces = rows(path / 'traces' / trace_name)
        for slot, row in enumerate(rr):
            identity(old[i][slot], row)
            require(np.array_equal(row['action'], pool['candidates'][slot, i]), 'replayed action mismatch')
            require(np.array_equal(row['physical_action'], pool['physical_actions'][slot, i]), 'replayed physical action mismatch')
            trace = traces[slot]
            require(trace['state_id'] == row['state_id'] and trace['candidate_index'] == i and trace['episode_id'] == row['episode_id'], 'trace identity mismatch')
            steps = trace['steps']
            require(1 <= len(steps) <= 25 and len(steps) == row['effective_valid_length'] == row['trajectory_step_count'], 'invalid first-episode length')
            require(canonical_sha(steps) == row['trajectory_trace_sha256'], 'canonical trace mismatch')
            require(not any(s['terminated'] or s['truncated'] for s in steps[:-1]), 'postterminal trace steps')
            require(all(b['raw_env_step'] > a['raw_env_step'] for a, b in zip(steps, steps[1:])), 'nonmonotonic trace')
            for s in steps:
                cost(task, s['current'], s['goal'])
                require(np.isfinite(np.asarray(s['action'], dtype=float)).all(), 'nonfinite trace action')
            last = steps[-1]
            require(last['terminated'] or last['truncated'] or len(steps) == 25, 'short nonterminal')
            require(last['current'] == row['endpoint_current'] and last['goal'] == row['endpoint_goal'], 'last state mismatch')
            require(last['terminated'] == row['endpoint_terminated'] and last['truncated'] == row['endpoint_truncated'], 'last flags mismatch')
            require(not (last['terminated'] and last['truncated']), 'simultaneous terminated/truncated endpoint')
            require(last['termination_reason'] == row['endpoint_termination_reason'] and last['termination_reason_source'] == row['endpoint_termination_reason_source'], 'reason mismatch')
            require(row['trajectory_trace_file'] == trace_name, 'trace filename mismatch')
            if len(steps) == 25:
                milestone = row['milestone_state_at_25']
                require(milestone['current'] == steps[24]['current'] and milestone['goal'] == steps[24]['goal'], 'step25 differs from observed trace')
                at25(task, row)
                require(math.isclose(at25(task, row), at25(task, old[i][slot]), rel_tol=0, abs_tol=1e-6), 'v1-v2 actual step25 cost mismatch')
            else:
                require(row['valid_at_25'] is False and row['physical_cost_at_25'] is None, 'early endpoint fabricated step25')
            value = cost(task, last['current'], last['goal'])
            require(math.isclose(value, row['endpoint_cost_recomputed'], rel_tol=0, abs_tol=1e-10), 'endpoint recomputation mismatch')
            require(math.isclose(value, old[i][slot]['terminal_selection_cost'], rel_tol=0, abs_tol=1e-6), 'v1-v2 endpoint mismatch')
            row['_cost'] = value
        out.append(rr)
    return out


def cell_metrics(task, seed, pool, old, replay, variants, tolerance):
    result = {'task': task, 'seed': seed, '3a': [], '3b': [], 'endpoint_types': dict(Counter(r['endpoint_type'] for branch in replay for r in branch)), 'lengths': dict(Counter(r['effective_valid_length'] for branch in replay for r in branch)), 'termination_reasons': dict(Counter(r['endpoint_termination_reason'] for branch in replay for r in branch))}
    for slot in range(50):
        base = old[0][slot]
        item = dict(slot=slot, episode_id=base['episode_id'], state_id=base['state_id'], seed=seed)
        truth = [branch[slot]['_cost'] for branch in replay]
        truth25 = [at25(task, branch[slot]) for branch in replay]
        item['valid_candidates_endpoint'] = 64
        item['valid_candidates_step25'] = sum(v is not None for v in truth25)
        item['endpoint'] = {scorer: ranking(pool[array][slot], truth) for scorer, array in (('cowm_b', 'cowm_b_costs'), ('lewm', 'lewm_costs'))}
        item['step25'] = {scorer: ranking(pool[array][slot], truth25) for scorer, array in (('cowm_b', 'cowm_b_costs'), ('lewm', 'lewm_costs'))} if all(v is not None for v in truth25) else None
        result['3a'].append(item)
        baseline = at25(task, base)
        bi = dict(slot=slot, episode_id=base['episode_id'], state_id=base['state_id'], seed=seed, baseline_cost=baseline, methods={})
        for method in METHODS:
            update = at25(task, variants[method][slot])
            delta = baseline-update if baseline is not None and update is not None else None
            random_delta, match_counts = [], Counter()
            for j in range(5):
                r = variants[f'{method}_random_{j}'][slot]
                match = r['random_match']
                if match['status'] == 'matched':
                    require(abs(match['actual_rms_normalized_after_native_handling']-match['target_rms_normalized_after_native_handling']) <= match['match_tolerance'] + 1e-12, 'random RMS outside frozen tolerance')
                    require(r['eligible_for_equal_rms_control'] is True, 'matched direction eligibility mismatch')
                    require(math.isclose(match['match_tolerance'], max(1e-5, match['target_rms_normalized_after_native_handling']*1e-3), rel_tol=0, abs_tol=1e-12), 'random match tolerance changed')
                    rc = at25(task, r)
                    if baseline is not None and rc is not None:
                        random_delta.append(baseline-rc)
                        match_counts['matched_valid25'] += 1
                    else:
                        match_counts['matched_missing_pair25'] += 1
                else:
                    match_counts['unmatched'] += 1
            random_mean = float(np.mean(random_delta)) if random_delta else None
            bi['methods'][method] = dict(improvement=delta, improved=None if delta is None else float(delta > tolerance),
                original_valid25=baseline is not None, updated_valid25=update is not None,
                updated_effective_length=variants[method][slot]['effective_valid_length'],
                updated_terminal_step=variants[method][slot]['terminal_step'],
                missing_reason=None if delta is not None else 'baseline_early_or_missing25' if baseline is None else 'updated_early_or_missing25',
                random_mean_improvement=random_mean, random_improved_fraction=float(np.mean(np.asarray(random_delta)>tolerance)) if random_delta else None,
                eligible_direction_count=len(random_delta), direction_counts=dict(match_counts),
                method_minus_matched_random=delta-random_mean if delta is not None and random_mean is not None else None)
        x, y = (bi['methods'][m]['improvement'] for m in ('cowm_po', 'lewm_po'))
        bi['cowm_minus_lewm_common_improvement'] = x-y if x is not None and y is not None else None
        result['3b'].append(bi)
    return result


def aggregate(items, key):
    available = [r for r in items if r[key] is not None and math.isfinite(r[key])]
    by_seed = {}
    for r in available:
        by_seed.setdefault(r['seed'], []).append(float(r[key]))
    seed_means = {str(s): float(np.mean(v)) for s, v in sorted(by_seed.items())}
    vals = list(seed_means.values())
    return dict(mean=float(np.mean(vals)) if vals else None, sample_std=float(np.std(vals, ddof=1)) if len(vals)>1 else None,
                estimable_seeds=len(vals), valid_states=len(available), missing_states=len(items)-len(available), seed_means=seed_means)


def bootstrap(items, key, hypothesis=True):
    out = aggregate(items, key)
    valid = [r for r in items if r[key] is not None and math.isfinite(r[key])]
    if not valid:
        return dict(out, ci95=None, p_two_sided=None, source_episode_clusters=0, replicates=B)
    episodes = sorted({r['episode_id'] for r in valid})
    seeds = sorted({r['seed'] for r in valid})
    eidx, sidx = {e:i for i,e in enumerate(episodes)}, {s:i for i,s in enumerate(seeds)}
    sums = np.zeros((len(episodes), len(seeds)))
    counts = np.zeros_like(sums)
    for r in valid:
        i, j = eidx[r['episode_id']], sidx[r['seed']]
        sums[i,j] += r[key]
        counts[i,j] += 1
    rng = np.random.default_rng(RNG_SEED)
    sampled = np.empty(B)
    empty_seed_replicates = 0
    for start in range(0, B, 250):
        n = min(250, B-start)
        weights = rng.multinomial(len(episodes), np.full(len(episodes), 1/len(episodes)), size=n)
        numer, denom = weights @ sums, weights @ counts
        seed_values = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom>0)
        empty_seed_replicates += int(np.any(denom == 0, axis=1).sum())
        sampled[start:start+n] = np.nanmean(seed_values, axis=1)
    estimate = out['mean']
    # Null distribution is bootstrap estimates centered on the observed paired contrast.
    p = float((1 + np.count_nonzero(np.abs(sampled-estimate) >= abs(estimate))) / (B+1)) if hypothesis else None
    return dict(out, ci95=[float(v) for v in np.quantile(sampled, [.025,.975])], p_two_sided=p,
                source_episode_clusters=len(episodes), replicates=B, rng_seed=RNG_SEED,
                bootstrap_replicates_with_unestimable_seed=empty_seed_replicates)


def summaries(cells, tasks):
    summaries, tests = {}, []
    for task in tasks:
        ac = [r for c in cells if c['task']==task for r in c['3a']]
        bc = [r for c in cells if c['task']==task for r in c['3b']]
        out = {'3a': {}, '3b': {}, 'states_processed': len(ac)}
        for endpoint in ('endpoint', 'step25'):
            panel = {}
            for scorer in ('cowm_b','lewm'):
                rr = [dict(seed=r['seed'], episode_id=r['episode_id'], **r[endpoint][scorer]) for r in ac if r[endpoint] is not None]
                panel[scorer] = {metric: bootstrap(rr, metric, hypothesis=False) for metric in ('spearman','hit_at_5','hit_at_1','selection_regret','random_hit_at_5','uniform_expected_regret')}
                panel[scorer]['complete_pool_states'] = len(rr)
                panel[scorer]['incomplete_pool_states'] = len(ac)-len(rr)
                panel[scorer]['total_source_states'] = len(ac)
                panel[scorer]['spearman_undefined_states'] = sum(r['spearman'] is None for r in rr)
                panel[scorer]['true_constant_states'] = sum(r['true_constant'] for r in rr)
                panel[scorer]['predicted_constant_states'] = sum(r['predicted_constant'] for r in rr)
            out['3a'][endpoint] = panel
        for metric in ('spearman','hit_at_5','selection_regret'):
            contrasts = []
            for r in ac:
                x,y = r['endpoint']['cowm_b'][metric], r['endpoint']['lewm'][metric]
                contrasts.append(dict(seed=r['seed'],episode_id=r['episode_id'], difference=x-y if x is not None and y is not None else None))
            tests.append(dict(task=task, comparison=f'3a/cowm_minus_lewm/{metric}', **bootstrap(contrasts,'difference')))
        for method in METHODS:
            rr = [dict(seed=r['seed'],episode_id=r['episode_id'],**r['methods'][method]) for r in bc]
            out['3b'][method] = {metric: bootstrap(rr,metric,hypothesis=False) for metric in ('improvement','improved','random_mean_improvement','random_improved_fraction','method_minus_matched_random')}
            out['3b'][method]['directions_per_state'] = dict(Counter(r['eligible_direction_count'] for r in rr))
            out['3b'][method]['missing_reasons'] = dict(Counter(r['missing_reason'] for r in rr if r['missing_reason']))
            out['3b'][method]['updated_length_distribution'] = dict(Counter(r['updated_effective_length'] for r in rr))
            out['3b'][method]['updated_early_terminal_states'] = sum(r['updated_terminal_step'] is not None and r['updated_terminal_step'] < 25 for r in rr)
            if method == 'cowm_po':
                tests.append(dict(task=task,comparison='3b/cowm_po_minus_matched_random_improvement',**bootstrap(rr,'method_minus_matched_random')))
        out['3b']['cowm_minus_lewm_common_improvement_exploratory'] = bootstrap(bc,'cowm_minus_lewm_common_improvement')
        summaries[task] = out
    estimable = sorted((r for r in tests if r['p_two_sided'] is not None), key=lambda r:r['p_two_sided'])
    running = 0.
    for i, test in enumerate(estimable):
        running = max(running, min(1.,(len(estimable)-i)*test['p_two_sided']))
        test['p_holm'] = running
    for test in tests:
        test.setdefault('p_holm',None)
        test['nonestimable_reason'] = 'no paired valid states' if test['p_two_sided'] is None else None
    return summaries, tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--attempt', default='attempt_001')
    args = parser.parse_args()
    frozen, refs, audit = (read(V2 / n) for n in ('frozen_config.json','source_refs.json','freeze_audit.json'))
    fsha, rsha = sha(V2/'frozen_config.json'), sha(V2/'source_refs.json')
    require(fsha == audit['frozen_config_sha256'] and rsha == audit['source_refs_sha256'] == frozen['source_refs_sha256'], 'global hash chain mismatch')
    analysis_lock = read(ANALYSIS_LOCK)
    require(analysis_lock.get('created_after_pilot_launch') is True, 'analysis lock must disclose post-pilot creation')
    require(analysis_lock.get('parent_frozen_config_sha256') == fsha, 'analysis lock parent freeze mismatch')
    require(analysis_lock.get('parent_source_refs_sha256') == rsha, 'analysis lock source refs mismatch')
    require(analysis_lock.get('analysis_code_path') == 'scripts/cvpr_table3_v2_analysis.py', 'unexpected analysis code path in lock')
    require(analysis_lock.get('analysis_code_sha256') == sha(Path(__file__)), 'analysis code differs from analysis lock')
    require(analysis_lock.get('retrospective_policy_module_path') == 'source/policy/round5_phase1_7.py', 'unexpected retrospective policy path in lock')
    policy_path = ROOT / 'source/policy/round5_phase1_7.py'
    require(analysis_lock.get('retrospective_policy_module_sha256') == sha(policy_path), 'policy module differs from analysis lock')
    require(analysis_lock.get('policy_hash_scope') == 'retrospective_dependency_inventory_not_pilot_launch_frozen', 'policy hash scope is misleading')
    expected_statistics = {
        'bootstrap_replicates': B,
        'rng_seed': RNG_SEED,
        'cluster': 'same source episode jointly resampled across evaluation seeds, retaining all state appearances within a sampled episode',
        'seed_weighting': 'equal mean across estimable evaluation-seed means in each replicate; missing seeds are not imputed',
        'ci': 'percentile 2.5th and 97.5th quantiles of the paired cluster-bootstrap estimate',
        'two_sided_centered_p': '(1 + count(abs(bootstrap_estimate - observed_estimate) >= abs(observed_estimate))) / 10001',
        'holm_family': 'four tasks x four prespecified comparisons (16 planned); adjust the estimable subset and retain nonestimable rows with reasons',
        '3a_oracle_tie': 'all candidates exactly equal to the float64 minimum are true minimizers; no added epsilon',
        '3a_spearman_ties': 'average ranks; constant or undefined rank vector is NA',
        '3a_prediction_tie': 'ascending frozen candidate index',
        '3b_improvement_tolerance': 1e-6,
        '3b_baseline_crosscheck': 'for every v1 refinement/random branch, independently recompute source candidate0 step25 cost minus updated step25 cost and match the archived per-state true improvement within 1e-6',
        'pilot_inference': 'partial descriptive validation only; no full six-seed inference',
    }
    require(analysis_lock.get('statistics') == expected_statistics, 'analysis statistical rules differ from the supplemental lock')
    check(ROOT/frozen['plan_path'],frozen['plan_sha256'])
    for code_path, expected_hash in frozen['implementation_code_sha256'].items():
        check(ROOT/code_path,expected_hash)
    require(set(refs['cells']) == {f'{t}/seed_{s}' for t in frozen['tasks'] for s in frozen['evaluation_seeds']}, 'global cell coverage mismatch')
    selected_seeds = [42] if args.pilot_only else frozen['evaluation_seeds']
    cells, missing = [], []
    for task in frozen['tasks']:
        for seed in selected_seeds:
            key = f'{task}/seed_{seed}'
            path = V2/'3a'/task/f'seed_{seed}'/args.attempt
            if not (path/'acceptance.json').exists():
                if args.pilot_only:
                    missing.append(dict(cell=key,reason='replay acceptance not yet available'))
                    continue
                raise ValueError(f'missing required replay acceptance: {key}')
            ref = refs['cells'][key]
            pool, old, variants, baseline_pair_validation = source_cell(task,seed,ref)
            replay = replay_cell(task,seed,path,ref,fsha,rsha,old,pool)
            tolerance = frozen['3b']['improvement_tolerance_by_task'][task]
            require(tolerance == 1e-6, 'unexpected frozen improvement threshold')
            cell = cell_metrics(task,seed,pool,old,replay,variants,tolerance)
            cell['source_3b_baseline_crosscheck'] = baseline_pair_validation
            cells.append(cell)
    summary, tests = summaries(cells,frozen['tasks'])
    output = dict(schema_version=1, analysis_status='partial_pilot_raw_recomputed' if args.pilot_only else 'all_24_cells_raw_recomputed_pending_main_agent_report_acceptance',
        is_final_acceptance=False, execution_role='main_agent', actual_model='GPT-6 (exact deployment identifier unavailable)', reasoning_effort='not_reported', global_frozen_config_sha256=fsha, global_source_refs_sha256=rsha,
        analysis_lock_sha256=sha(ANALYSIS_LOCK), analysis_lock_created_after_pilot_launch=analysis_lock['created_after_pilot_launch'],
        analysis_code_sha256=sha(Path(__file__)), processed_cells=len(cells), expected_full_cells=24, missing_selected_cells=missing,
        statistics=dict(replicates=B,rng_seed=RNG_SEED, seed_weighting='equal estimable seed means', cluster='same source episode jointly across evaluation seeds; retained state appearance counts',
            p_value='(1 + count(abs(bootstrap_estimate-observed_estimate) >= abs(observed_estimate)))/(10000+1)',
            holm_planned_family_size=16, holm_estimable_comparisons=sum(t['p_two_sided'] is not None for t in tests),
            pilot_inference='descriptive partial analysis; not full six-seed inference' if args.pilot_only else None),
        cells=cells, summaries=summary, primary_tests=tests)
    dest = V2/'analysis'/('pilot.json' if args.pilot_only else 'full.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp = dest.with_suffix('.json.tmp')
    temp.write_text(json.dumps(output,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temp.replace(dest)
    print(json.dumps(dict(status='ok',processed_cells=len(cells),missing_selected_cells=missing,output=str(dest.relative_to(ROOT)))))


if __name__ == '__main__':
    main()
