#!/usr/bin/env python3
"""Summarize the six frozen Reacher baseline prefix experiments."""
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round5/phase5_baseline_horizons'


def paired(a, b):
    keys = lambda r: (r['dataset_episode'], r['start_step'])
    a = {keys(r): bool(r['success']) for r in a['episodes']}
    b = {keys(r): bool(r['success']) for r in b['episodes']}
    assert a.keys() == b.keys() and len(a) == 50
    diffs = np.array([int(b[k]) - int(a[k]) for k in sorted(a)])
    wins, losses = int((diffs > 0).sum()), int((diffs < 0).sum())
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
    rng = np.random.default_rng(42)
    ci = np.quantile(diffs[rng.integers(0, 50, (10000, 50))].mean(1)*100, [.025, .975])
    return {'delta_pp': float(diffs.mean()*100), 'ci95_pp': ci.tolist(),
            'improved': wins, 'regressed': losses, 'mcnemar_p': p}


def main():
    results = {}
    rows, comparisons = [], []
    for method in ['lewm', 'leflow']:
        for steps in [25, 10, 5]:
            d = json.loads((OUT / method / f'{steps}_{steps}' / 'result.json').read_text())
            assert len(d['episodes']) == 50
            results[method, steps] = d
            events = d['horizon_ablation']['planning_events']
            rows.append({'method': method, 'execute_score': f'{steps}/{steps}',
                         'successes': sum(e['success'] for e in d['episodes']), 'n': 50,
                         'planning_calls': len(events),
                         'planning_seconds': sum(e['planning_seconds'] for e in events),
                         'evaluation_seconds': d['evaluation_seconds']})
        for steps in [10, 5]:
            base = results[method, 25]['horizon_ablation']
            other = results[method, steps]['horizon_ablation']
            for key in ['checkpoint_sha256', 'runner_sha256', 'cohort_sha256']:
                assert base[key] == other[key], (method, steps, key)
            comparisons.append({'method': method, 'steps': steps,
                                **paired(results[method, 25], results[method, steps])})
    # Four within-method short-prefix comparisons against 25/25.
    running = 0.
    for rank, i in enumerate(sorted(range(4), key=lambda i: comparisons[i]['mcnemar_p'])):
        running = max(running, min(1., (4-rank)*comparisons[i]['mcnemar_p']))
        comparisons[i]['holm_p_four'] = running
    (OUT / 'analysis.json').write_text(json.dumps({'conditions': rows, 'comparisons': comparisons}, indent=2)+'\n')
    with (OUT / 'conditions.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ['# Reacher：LeWM / LeFlow 执行与评分前缀对照', '',
             '固定已有 checkpoint；50 个 legacy 起点，与 Phase5 pre_report2 的 Reacher cohort SHA 一致；seed=42、目标偏移25步、评测预算50步。', '',
             '所有条件仍生成5个5-step动作块（25步）；仅改变评分所用的前缀和执行前缀。LeWM 保留原生 CEM（300 samples、30 iterations、top-30，沿用默认 warm-start）；LeFlow 保留64条候选、16步flow及原生解码。未新增 candidate clipping，不能与 FastLeWAM 的 cem-clip 解释为仅模型不同的严格消融。', '',
             'LeFlow 的5/5不是生成1-block潜在路径：其现有采样器要求至少2个block，这里始终生成5个block后按首个block的rollout结果评分。LeWM 使用历史本地训练checkpoint；LeFlow使用发布planner及其引用的冻结LeWM，不能将两者差异单独归因于规划算法。', '',
             '| 方法 | execute/score | 成功数 | 规划调用数 | 总规划时间(s) | 总评测时间(s) |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['execute_score']} | {r['successes']}/50 | {r['planning_calls']} | {r['planning_seconds']:.2f} | {r['evaluation_seconds']:.2f} |")
    lines += ['', '差值为短前缀减同方法25/25；按起点配对bootstrap 10,000次。四个McNemar检验统一Holm校正。', '',
              '| 方法 | execute/score | 差值 pp [95% CI] | 改善/退化起点 | 精确p | Holm p |',
              '|---|---|---|---:|---:|---:|']
    for c in comparisons:
        lo, hi = c['ci95_pp']
        lines.append(f"| {c['method']} | {c['steps']}/{c['steps']} | {c['delta_pp']:+.1f} [{lo:+.1f}, {hi:+.1f}] | {c['improved']}/{c['regressed']} | {c['mcnemar_p']:.4g} | {c['holm_p_four']:.4g} |")
    lines += ['',
              '本轮六项均完成，运行时确认所有条件生成5个block，并按配置评分5/2/1个block。LeWM和LeFlow在两个短前缀条件下的成功率点估计均高于自身25/25基线，但四项统一Holm校正后均未达到0.05。bootstrap区间与精确检验可能给出不同的零边界判断，显著性按精确检验及其校正解释。', '',
              '这提示Reacher上的短前缀收益可能跨规划方法出现，并非FastLeWAM独有的现象；当前样本不足以确认稳定提升。相对25/25同时改变了执行和评分长度，不能分别归因于重规划频率或评分窗口。进一步拆分需要10/25、5/25等控制条件。', '',
              'LeWM短前缀增加了规划调用次数，但总规划观测时间从67.31秒降至50.89/53.10秒；LeFlow则从4.40秒增至4.95/7.96秒。缩短评分不保证整段控制计算成本下降；这些共享设备观测不能用于精确效率归因。']
    lines += ['', '单checkpoint、单cohort的探索性结果；不代表重新训练短horizon模型的效果。GPU0/1与其他作业共享，计时只作本次观测，不能作为隔离设备性能。逐条件result.json记录checkpoint/cohort/runner SHA、评分前缀审计和规划事件。', '',
              '产物：`outputs/round5/phase5_baseline_horizons/`；运行入口：`scripts/round5_phase5_baseline_horizons.py`；统计入口：`scripts/round5_phase5_baseline_horizons_analysis.py`。']
    report = ROOT / 'docs/report/round5/round5_phase5_baseline_horizons_report.md'
    report.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
