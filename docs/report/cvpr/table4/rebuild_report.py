"""Rebuild Table 4 from Table 1 artifacts, without running any GPU work."""
from pathlib import Path
import csv
import hashlib
import json
import statistics as stats
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from source.common.cvpr_table1 import cowm_methods

OUT = Path(__file__).resolve().parent
SRC = ROOT / 'outputs/cvpr/table1/v1'
TASKS = ['tworoom', 'pusht', 'reacher', 'cube']
SEEDS = [42, 100, 2026, 3407, 1234, 4444]
METHODS = [m['id'] for m in cowm_methods()]
PRIMARY = ['P0', 'P1', 'P2', 'P3', 'P0-PO-L', 'P0-GF-L', 'P3-PO-refine-L', 'P3-PO-L']

def read_json(p):
    return json.loads(p.read_text())

def sha(p):
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def local(p):
    p = str(p)
    marker = '/outputs/cvpr/table1/v1/'
    return SRC / p.split(marker, 1)[1] if marker in p else ROOT / p

def relative(p):
    return str(p.relative_to(ROOT))

def mid(name, bound):
    return 'cowm_' + name.lower().replace('-', '_') + '__' + bound

def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def fmt(values):
    return f'{stats.mean(values):.2f} ± {stats.stdev(values):.2f}'

state = read_json(SRC / 'status.json')['cells']
refs = {'scope': 'Table 1 aliases only; no new evaluations', 'evaluation_cells': [], 'timing_conditions': [], 'selection_dagger_aliases': [], 'inputs': {}}
for name in ['frozen_config.json', 'freeze.json', 'status.json', 'summary/all_independent_cells.csv', 'summary/timing_conditions.csv', 'timing_summary.json', 'provenance/timing_device_override.json']:
    refs['inputs'][relative(SRC / name)] = sha(SRC / name)
refs['inputs'][relative(ROOT / 'docs/plan/cvpr_table4_plan.md')] = sha(ROOT / 'docs/plan/cvpr_table4_plan.md')
values, episodes, cohorts = {}, {}, {}
per_seed = []
with (SRC / 'summary/all_independent_cells.csv').open() as f:
    cells = [r for r in csv.DictReader(f) if r['family'] == 'cowm']
assert len(cells) == 876
for row in cells:
    path = local(row['result_path'])
    raw = read_json(path)
    ident = raw['cvpr_table1']
    record = state[row['cell_id']]
    assert row['status'] == record['status'] == 'complete'
    assert sha(path) == record['result_sha256']
    assert ident['identity_sha256'] == record['identity_sha256']
    assert ident['cell_id'] == row['cell_id'] and ident['method_id'] == row['method_id']
    trace = local(raw['trace_path'])
    assert sha(trace) == raw['trace_sha256'] == ident['trace_sha256']
    eps = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(eps) == len(raw['summary']['success_vector']) == 50
    flags = [e['success'] for e in eps]
    assert flags == raw['summary']['success_vector']
    rate = sum(flags) / 50
    assert abs(rate - raw['success_rate']) < 1e-10 and abs(rate - float(row['success_rate'])) < 1e-8
    key = (row['method_id'], row['task'], int(row['evaluation_seed']))
    assert key not in values
    values[key] = 100 * rate
    # Store only paired-analysis columns, rather than duplicate the rollout traces.
    episodes[key] = [(e['dataset_episode'], e['row_index'], e['goal_row_index'], bool(e['success'])) for e in eps]
    cohort_key = (row['task'], int(row['evaluation_seed']))
    assert cohorts.setdefault(cohort_key, raw['cohort_sha256']) == raw['cohort_sha256']
    per_seed.append({'method_id': row['method_id'], 'task': row['task'], 'evaluation_seed': row['evaluation_seed'], 'successes': sum(flags), 'episodes': 50, 'success_rate_percent': 100 * rate, 'source_cell_id': row['cell_id']})
    refs['evaluation_cells'].append({'cell_id': row['cell_id'], 'method_id': row['method_id'], 'task': row['task'], 'seed': int(row['evaluation_seed']), 'alias': True, 'result_path': relative(path), 'result_sha256': record['result_sha256'], 'trace_path': relative(trace), 'trace_sha256': raw['trace_sha256'], 'identity_sha256': ident['identity_sha256'], 'checkpoint': relative(local(raw['checkpoint'])), 'cohort_sha256': raw['cohort_sha256']})
print('Verified 876 source evaluation cells and 43,800 episodes.', flush=True)

for bound in ['main', 'crop']:
    for task in TASKS:
        for seed in SEEDS:
            target = mid('selection-dagger', bound) if task == 'reacher' else mid('P3', bound)
            assert (target, task, seed) in values
            refs['selection_dagger_aliases'].append({'variant': bound, 'task': task, 'seed': seed, 'source_method_id': target})

timing = {}
timing_rows = []
with (SRC / 'summary/timing_conditions.csv').open() as f:
    rows = [r for r in csv.DictReader(f) if r['family'] == 'cowm']
assert len(rows) == 146
for row in rows:
    p = local(row['raw_result'])
    raw = read_json(p)
    assert row['status'] == 'complete'
    samples = [s['wall_ms'] for s in raw['raw_samples']]
    assert len(samples) == raw['measurement_count'] == 250
    assert len({s['state_index'] for s in raw['raw_samples']}) == 50
    assert abs(stats.mean(samples) - raw['mean_ms']) < 1e-8
    assert abs(raw['mean_ms'] - float(row['mean_ms'])) < 1e-6
    assert raw['normal_timing_parity']['action']['equal']
    assert raw['normal_timing_parity']['selection']['equal']
    assert raw['gpu_before']['gpu'] == raw['gpu_after']['gpu'] == 6
    assert raw['gpu_before']['gpu_name'] == 'NVIDIA GeForce RTX 4090'
    key = (row['method_id'], row['task'])
    assert key not in timing
    timing[key] = raw
    timing_rows.append({k: row[k] for k in ['method_id', 'task', 'mean_ms', 'p50_ms', 'p95_ms', 'measurement_count', 'unique_states', 'loaded_model_baseline_vram_mib', 'planning_peak_vram_mib']})
    refs['timing_conditions'].append({'condition_id': row['condition_id'], 'alias': True, 'result_path': relative(p), 'result_sha256': sha(p), 'identity_sha256': raw['identity_sha256']})
print('Verified 146 source timing conditions.', flush=True)

def rates(name, bound, task):
    return [values[mid(name, bound), task, seed] for seed in SEEDS]

def macro(name, bound):
    return [stats.mean(values[mid(name, bound), task, seed] for task in TASKS) for seed in SEEDS]

def mean_ms(name, bound):
    return stats.mean(timing[mid(name, bound), task]['mean_ms'] for task in TASKS)

def summary_row(name, bound):
    return {'method': name, 'variant': bound, **{task: fmt(rates(name, bound, task)) for task in TASKS}, 'macro_sr_percent': fmt(macro(name, bound)), 'equal_task_mean_ms': f'{mean_ms(name, bound):.4f}'}
summary = [summary_row(name, bound) for bound in ['main', 'crop'] for name in METHODS]
write_csv('success_per_seed.csv', per_seed)
write_csv('inference_summary.csv', summary)
write_csv('timing_per_task.csv', timing_rows)

# Hierarchical paired bootstrap: resample evaluation seeds and source episodes.
# One task-level source-episode weight is shared across all seeds and repeated
# starts, preserving dependency when a source episode occurs more than once.
B = 20000
rng = np.random.default_rng(20261005)
pair_defs = [('P0-PO-L', 'P0'), ('P0-GF-L', 'P0'), ('P2', 'P1')]
paired, seed_deltas = [], []
for task in TASKS:
    base = [episodes[mid('P0', 'main'), task, s] for s in SEEDS]
    cluster_ids = sorted({e[0] for batch in base for e in batch})
    index = {c: i for i, c in enumerate(cluster_ids)}
    counts = np.zeros((6, len(index)))
    for i, batch in enumerate(base):
        for e in batch:
            counts[i, index[e[0]]] += 1
    weights = rng.multinomial(len(index), np.full(len(index), 1 / len(index)), size=B)
    seed_weights = rng.multinomial(6, np.full(6, 1 / 6), size=B)
    denominator = weights @ counts.T
    assert np.all(denominator > 0)
    for lhs, rhs in pair_defs:
        sums = np.zeros_like(counts)
        deltas = []
        for i, seed in enumerate(SEEDS):
            a = episodes[mid(lhs, 'main'), task, seed]
            b = episodes[mid(rhs, 'main'), task, seed]
            assert [e[:3] for e in a] == [e[:3] for e in b] == [e[:3] for e in base[i]]
            for ea, eb in zip(a, b):
                sums[i, index[ea[0]]] += int(ea[3]) - int(eb[3])
            delta = values[mid(lhs, 'main'), task, seed] - values[mid(rhs, 'main'), task, seed]
            deltas.append(delta)
            seed_deltas.append({'comparison': f'{lhs} − {rhs}', 'task': task, 'evaluation_seed': seed, 'difference_pp': f'{delta:.8f}'})
        boot = ((weights @ sums.T) / denominator * seed_weights).sum(axis=1) / 6 * 100
        lo, hi = np.quantile(boot, [0.025, 0.975])
        paired.append({'comparison': f'{lhs} − {rhs}', 'task': task, 'mean_difference_pp': f'{stats.mean(deltas):.4f}', 'seed_std_pp': f'{stats.stdev(deltas):.4f}', 'ci95_low_pp': f'{lo:.4f}', 'ci95_high_pp': f'{hi:.4f}', 'source_episode_clusters': len(index)})
write_csv('paired_differences_per_seed.csv', seed_deltas)
write_csv('paired_differences.csv', paired)
refs['paired_analysis'] = {'bootstrap_replicates': B, 'rng_seed': 20261005, 'method': 'paired percentile bootstrap; resample six evaluation seeds and task-level source episodes independently; common cluster weights across repeated starts and seeds', 'holm_family': '16 planned tests; P3−Random-64 (4 tests) TBD; no significance claim'}
(OUT / 'source_refs.json').write_text(json.dumps(refs, ensure_ascii=False, indent=2) + '\n')

lines = []
def add(s=''):
    lines.append(s)

def table(headers, rows):
    add('| ' + ' | '.join(headers) + ' |')
    add('| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |')
    for row in rows:
        add('| ' + ' | '.join(map(str, row)) + ' |')
    add()

add('# CVPR Table 4：固定模型下的推理策略报告（Table 1 复用首版）\n')
add('整理日期：2026-10-05。依据 [Table 4 计划](../../../plan/cvpr_table4_plan.md)，复用 [Table 1 报告](../table1/cvpr_table1_report.md)及其原始实验产物。本版仅读取既有结果；未新增训练、闭环评测或计时。Table 1 未做的 Random-64 与同 A+LeWM scorer 桥接实验均保留为 **TBD**。\n')
add('## 完成范围与公共协议\n')
add('已复用 876 个 CoWM 独立成功率单元（43,800 episodes）：main 18×4×6=432、crop 432、Reacher Selection† 短时域 12。另复用 146 个 CoWM 计时条件（main 72、crop 72、Reacher 短时域 2），每条件 250 次测量。Selection† 的另外三任务 alias P3，不增加独立样本。Table 1 的 LeWM、LeFlow、Sub-JEPA 独立 baseline 不代替“同 A+LeWM”桥接。\n')
add('固定每任务 Table 1 R4-AB、训练 seed=3072、epoch=10；Table 2 新权重不纳入。四任务为 TwoRoom、PushT、Reacher、OGBench-Cube。评测 seeds 为 `42,100,2026,3407,1234,4444`，每 seed 50 episodes；每方法每任务共 300 episodes。成功率单位 %，采用六个评测 seeds 的 mean ± sample std（ddof=1）。宏平均先在每 seed 内等权平均四任务，再跨 seed 汇总；误差条不覆盖训练 seed 变化。\n')
add('常规生成/评分/执行跨度为 25/25/25 个环境步，目标 offset=25，总执行预算=50，horizon=5 个 block、每 block=5 步。动作生成采用 Euler S=2；FP32，关闭 autocast/TF32/compile。main 无额外动作投影，保留环境原生动作处理。模型原生 history/padding 沿用 Table 1；goal cache=false、proposal chunk=512、candidate batch=64、solver batch=1；CEM batch=1、warm_start=true，初始测时清空动作/目标缓存和 CEM 残余。冻结协议见 [frozen_config.json](../../../../outputs/cvpr/table1/v1/frozen_config.json)。\n')
add('## 主表：常规 main 协议\n')
add('N 为候选数（CEM 为每轮样本数）；S 为 Euler 步数；I 为 CEM 迭代数；K 为梯度更新次数。P1/P2 elites=30、var_scale=1，P2 初始化缩放=1；梯度步长=0.01、最大 RMS 偏移=0.2。P3-PO-L 对全部候选各做 2 次更新，P3-PO-refine-L 只优化选中的一条。\n')
budget = {'P0': '1/2/—/0', 'P1': '300/—/30/0', 'P2': '300/2/30/0', 'P3': '64/2/—/0', 'P0-PO-L': '1/2/—/2', 'P0-GF-L': '1/2/—/2', 'P3-PO-refine-L': '64/2/—/2（选中一条）', 'P3-PO-L': '64/2/—/2每候选'}
main_rows = []
for name in PRIMARY:
    s = summary_row(name, 'main')
    main_rows.append([name, budget[name], *[s[t] for t in TASKS], s['macro_sr_percent'], s['equal_task_mean_ms']])
main_rows.insert(1, ['Random-64', '64/2/—/0', *['TBD'] * 6])
table(['方法', 'N/S/I/K', 'TwoRoom', 'PushT', 'Reacher', 'OGBench-Cube', '宏平均 SR (%)', '等权平均重规划 mean(ms)'], main_rows)
add('**独立 scorer 桥接面板**：同一 A + LeWM 的四任务 SR、宏平均与计时均为 **TBD**；具体桥接协议及结果沿用 Table 3。其效果解释为更换 scorer 的系统效应。Random-64 必须实际生成全部 64 条再均匀随机选择，选择 RNG 独立于 proposal RNG；本版没有该实验，不能用 P0 补数。\n')
add('## 完整成功率与平均计时附表\n')
add('完整保留 18 种 main 和 18 种 crop。GF-L 共 2 次更新、GF-H 共 10 次；PO-L/PO-refine-L 共 2 次、H 共 5 次。GF 在每个 Euler 步使用局部参考，PO 使用完整生成动作为参考；L/H 名称不表示 GF 与 PO 等计算量。P2-GF/PO 为先修正 A 初始化再 CEM；P3-GF/PO 修正全部候选，P3-PO-refine 先选再修正。\n')
for bound, title in [('main', '主版本 main'), ('crop', '动作裁剪版本 crop')]:
    add('### ' + title + '\n')
    if bound == 'crop':
        add('P0/P3 及变体使用 clip；P1/P2 及变体使用 candidate_clip，候选评分前按真实物理边界换算的归一化边界投影；P3-PO-refine 优化后再次投影。GF/PO 在未投影代价上求梯度、随后投影。\n')
    table(['方法', 'TwoRoom', 'PushT', 'Reacher', 'OGBench-Cube', '宏平均 SR (%)', 'mean(ms)'], [[s['method'], *[s[t] for t in TASKS], s['macro_sr_percent'], s['equal_task_mean_ms']] for s in summary if s['variant'] == bound])
add('### Selection†：Reacher 10/10 部署参考\n')
add('生成跨度仍为 25；仅 Reacher 的评分/执行改为 10/10，目标 offset=25、总执行预算=50 不变。其它三任务引用对应版本 P3。下述宏平均混合时域，单列供部署参考，不进入常规主排名。\n')
dagger_rows = []
for bound in ['main', 'crop']:
    arr = [[values[mid('selection-dagger' if t == 'reacher' else 'P3', bound), t, s] for s in SEEDS] for t in TASKS]
    avg = [stats.mean(a[i] for a in arr) for i in range(6)]
    ms = stats.mean(timing[mid('selection-dagger' if t == 'reacher' else 'P3', bound), t]['mean_ms'] for t in TASKS)
    dagger_rows.append([bound, *[fmt(a) for a in arr], fmt(avg), f'{ms:.4f}'])
table(['版本', 'TwoRoom', 'PushT', 'Reacher', 'OGBench-Cube', '混合时域宏平均 SR (%)', 'mean(ms)'], dagger_rows)
add('## 主要配对：逐 seed 差与区间\n')
add('全部为 Table 1 已观察结果的**事后扩展分析**。逐 seed 差（百分点）见 [paired_differences_per_seed.csv](paired_differences_per_seed.csv)。下表报告 main 的配对均值差、跨 seed 样本标准差及 95% percentile bootstrap 区间：20,000 次重复，RNG seed=20261005；独立重采样评测 seed 与任务内来源 dataset_episode，同来源 episode 的重复起点及跨 seed 出现项共用聚类权重，保持方法配对。区间未做多重比较校正，亦不覆盖训练 seed。\n')
table(['比较', '任务', '均值差 ± seed std (pp)', '配对 95% CI (pp)', '来源 episode 簇数'], [[r['comparison'], r['task'], f"{float(r['mean_difference_pp']):+.2f} ± {float(r['seed_std_pp']):.2f}", f"[{float(r['ci95_low_pp']):+.2f}, {float(r['ci95_high_pp']):+.2f}]", r['source_episode_clusters']] for r in paired])
add('P3−Random-64 的四任务逐 seed 差、配对区间与检验：**TBD**。计划的四组配对×四任务共 16 项 Holm 检验族尚不完整，Holm 校正结果为 **TBD**；本版不作显著性结论。P3−P0 同时改变候选数与选择，只能描述整体推理策略差异。\n')
add('## 计时与显存补表\n')
add('全部引用 Table 1 同批 GPU6、设备报告 NVIDIA GeForce RTX 4090 的既有计时，授权覆盖见 [timing_device_override.json](../../../../outputs/cvpr/table1/v1/provenance/timing_device_override.json)。batch=1 初始完整重规划边界为 CPU 原始输入→CPU 可执行动作，CUDA 同步，包含 encoder/生成/评分/反传/转换，排除环境、模型加载及归档。每任务 10 次预热、50 状态×5 重复=250 次测量，动作 parity atol=1e-6、rtol=1e-5 全通过。mean/P50/P95 单位 ms，显存单位 MiB。平均延迟不是回合总耗时或等时间预算成功率。后续新增设备/会话的条件不能直接与旧计时混合，需同卡同会话补共同测时。\n')
for bound in ['main', 'crop']:
    add('### ' + bound + '：逐任务 mean / P50 / P95 / 显存\n')
    table(['方法', '任务', 'mean', 'P50', 'P95', '加载基线 MiB', '规划峰值 MiB'], [[r['method_id'].removeprefix('cowm_').removesuffix('__' + bound), r['task'], *[f'{float(r[k]):.2f}' for k in ['mean_ms', 'p50_ms', 'p95_ms', 'loaded_model_baseline_vram_mib', 'planning_peak_vram_mib']]] for r in timing_rows if r['method_id'].endswith('__' + bound)])
add('Selection† 此处的显存/时延只新增 Reacher 条件；其它三任务沿用 P3。所有逐任务原始精度数据见 [timing_per_task.csv](timing_per_task.csv)。\n')
add('## 当前结果解读\n')
add('main 中 P3 为 96.25 ± 1.41%，P0 为 94.17 ± 0.61%，整体差为 +2.08 pp，主要来自 Reacher（+7.33 pp）；不能归因为纯评分收益。P2 为 91.67 ± 1.78%，比 P1 的 82.42 ± 2.20% 高 +9.25 pp，但比 P0 低 2.50 pp。生成初始化改善该 CEM 系统，不代表 CEM 必然优于直接生成。\n')
add('单候选轻量修正 P0-PO-L 为 94.50 ± 1.26%，相对 P0 +0.33 pp；P0-GF-L 为 95.58 ± 1.99%，相对 P0 +1.42 pp。P3-PO-L 为 96.42 ± 0.66%，比 P3 +0.17 pp；先选再修正的 P3-PO-refine-L 为 95.92 ± 1.24%。修正对不同任务效果不同，不能保证统一收益。\n')
add('main 的 P3-PO-L 与 P3-PO-H 宏平均均为 96.42%，较大预算没有稳定改善；P3-GF-H 为 95.50%，低于 GF-L 的 96.17%。裁剪同样存在任务依赖：P1 的 Reacher 从 79.00% 升至 90.00%，TwoRoom 从 97.33% 降至 92.33%；P2 的 Reacher 从 76.00% 升至 87.33%，Cube 从 97.67% 降至 92.33%。保留全部预设配置，不按各任务成绩挑选裁剪、时域或方法。\n')
add('TwoRoom/Cube 接近饱和；六个评测 seeds 不是六次训练重复。复用同一 benchmark 的开发观察不能称作未见确认集，本版不推断训练耦合、共享参数或等预算的因果优势。\n')
add('## TBD 清单与状态定义\n')
table(['项目', '状态', '补齐来源/约束'], [['Random-64 四任务 SR/计时', 'TBD（未测）', 'Table 3 桥接；24 单元/1,200 episodes；新方法测时四任务×250调用'], ['同 A+LeWM scorer 四任务 SR/计时', 'TBD（未测）', 'Table 3 同一 A 桥接；24 单元/1,200 episodes；不能替代为 LeWM baseline'], ['P3−Random-64 配对分析', 'TBD（依赖未测实验）', '固定随机选择规则与候选池身份核验后补齐'], ['完整 16 项 Holm 校正', 'TBD（分析待补）', '四组配对全部可用后按原计划校正']])
add('TBD=本版来源 Table 1 未提供实验或其依赖分析待补；失败=已运行但验收不通过（本版复用范围为 0）；—=不适用，例如 P1 的 S。TBD 不填零，不计入宏平均。新增桥接由 Table 3 统一派发，与 Table 5 共用测时清单，避免跨表重复运行。\n')
add('## 可重建产物与核验\n')
add('本次逐一核验 876 个 result 的状态登记 SHA256、cell/method 身份与 identity_sha256，episodes.jsonl SHA256、每单元 50 个回合及逐回合 success、原始 SR 与汇总 CSV 一致，并确认同 task/seed 的 cohort SHA256 一致。146 个计时条件从 250 个 raw_samples 重算 mean，与原始结果及 CSV 一致，核验 50 状态、动作/选择 parity 与设备身份。checkpoint/数据等资产身份沿用 Table 1 已冻结审计链，本版未重新散列大型权重或重新运行动作验收。\n')
add('- [source_refs.json](source_refs.json)：每个复用 result/trace 路径、SHA256、身份、cohort、checkpoint 及 Selection† alias；所有引用为 Table 1 旧单元。\n- [inference_summary.csv](inference_summary.csv)：main/crop 共 36 行 SR 与平均计时。\n- [success_per_seed.csv](success_per_seed.csv)：876 个独立单元的逐 seed SR。\n- [timing_per_task.csv](timing_per_task.csv)：146 个计时条件。\n- [paired_differences_per_seed.csv](paired_differences_per_seed.csv)、[paired_differences.csv](paired_differences.csv)：逐 seed 差、配对区间。\n- [rebuild_report.py](rebuild_report.py)：从仓库根目录运行 `python docs/report/cvpr/table4/rebuild_report.py` 即可重建，需要 NumPy，仅读取旧实验并写入本目录。\n')
add('路径以仓库根目录为基准；原始结果中的历史绝对路径映射回本仓库 `outputs/cvpr/table1/v1/`。本版文档与索引按用户指定统一放在 `docs/report/cvpr/table4/`，不复制原始实验、不伪造 Table 4 新运行。')
(OUT / 'cvpr_table4_report.md').write_text('\n'.join(lines) + '\n')
print('Wrote Table 4 report, source references, and five CSV supplements.', flush=True)
