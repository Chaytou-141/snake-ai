"""5G: compare snapshots within three uninterrupted food13 training trajectories."""

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import tempfile

import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.experiments.feature_comparison import (
    digest, evaluation_rows, prepare_output, seed_cpu, write_csv, write_json,
)
from snake_ai.training.config import TrainingConfig
from snake_ai.training.train import run_training

DEFAULT_PROTOCOL = Path('configs/training_duration_v1.json')
DEFAULT_OUTPUT = Path('results/milestone5g_training_duration')
STEPS = (40000, 60000, 80000, 100000)
SEEDS = (177, 188, 199)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    canonical = Path(__file__).resolve().parents[2]/DEFAULT_PROTOCOL
    defaults = TrainingConfig().to_dict()
    omitted = {'seed','training_seed_start','evaluation_seed_start','evaluation_episodes','random_policy_seed_start'}
    base = {k:v for k,v in defaults.items() if k not in omitted}
    base.update(episodes=100000,max_environment_steps=100000)
    if (protocol != json.loads(canonical.read_text(encoding='utf-8'))
            or protocol['protocol_version'] != '5g-v1' or protocol['observation_mode'] != 'food13'
            or protocol['snapshot_steps'] != list(STEPS) or protocol['primary_comparison'] != [40000,100000]
            or protocol['training_seeds'] != list(SEEDS)
            or protocol['training_seed_starts'] != {'177':1100000,'188':1200000,'199':1300000}
            or protocol['evaluation_seed_start'] != 1500000 or protocol['evaluation_episodes'] != 100
            or protocol['base_training_config'] != base
            or protocol['smoke'] != {'training_seed':211,'training_seed_start':1410000,
                'evaluation_seed_start':1600000,'evaluation_episodes':5,'snapshot_steps':[128,256,384,512],
                'max_environment_steps':512,'learning_starts':32,'batch_size':16}):
        raise ValueError('Protocol differs from the predeclared duration comparison')
    return protocol


def config_for(protocol: dict, seed: int, *, smoke: bool = False) -> TrainingConfig:
    if type(seed) is not int or seed not in ([protocol['smoke']['training_seed']] if smoke else SEEDS):
        raise ValueError('Unknown duration-comparison seed')
    config = TrainingConfig.from_dict({**protocol['base_training_config'], 'seed':seed,
        'observation_mode':'food13','training_seed_start':1410000 if smoke else protocol['training_seed_starts'][str(seed)],
        'evaluation_seed_start':protocol['smoke']['evaluation_seed_start'] if smoke else protocol['evaluation_seed_start'],
        'evaluation_episodes':100})
    if smoke:
        values = protocol['smoke']
        config = replace(config,episodes=values['max_environment_steps'], **{k:values[k] for k in
            ('max_environment_steps','learning_starts','batch_size','evaluation_seed_start','evaluation_episodes')})
    return config


def prepare(output: Path, protocol: dict):
    return prepare_output(output,protocol,extra_source_files=('snake_ai/experiments/training_duration.py',),
                          seed_rule=protocol['seed_rule'])


def verify_training(summary: dict, config: TrainingConfig, steps: tuple[int, ...]) -> None:
    if (summary['status'] != 'environment_step_cap'
            or summary['environment_steps_this_run'] != config.max_environment_steps
            or set(summary.get('snapshots',{})) != {str(s) for s in steps}):
        raise RuntimeError('Incomplete duration training preserved; do not restart or evaluate silently')
    for step in steps:
        record = summary['snapshots'][str(step)]
        if record['environment_steps'] != step or record['optimiser_updates'] != step-config.learning_starts+1:
            raise ValueError('Snapshot counters differ from the fixed protocol')


def run_smoke(protocol: dict, output: Path) -> dict:
    output, metadata = prepare(output,protocol)
    marker = output/'validation.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if not saved['passed'] or saved['protocol_sha256'] != metadata['protocol_sha256']:
            raise ValueError('Invalid development validation marker')
        return saved
    config = config_for(protocol,211,smoke=True)
    steps = tuple(protocol['smoke']['snapshot_steps'])
    findings = []
    with tempfile.TemporaryDirectory(prefix='snakeai-duration-smoke-') as temp:
        directory = Path(temp)/'training'
        seed_cpu(config.seed)
        summary = run_training(config,directory,make_charts=False,snapshot_steps=steps)
        verify_training(summary,config,steps)
        previous = None
        for step in steps:
            checkpoint = directory/f'checkpoints/step-{step:06d}.pt'
            before = digest(checkpoint)
            agent = DQNAgent.load(checkpoint)
            rows, metrics = evaluation_rows(agent,f'{step}steps',config.seed)
            if (agent.config != config or agent.environment_steps != step or not metrics['weights_unchanged']
                    or len(rows)!=5 or digest(checkpoint)!=before):
                raise ValueError('Smoke snapshot/evaluation mismatch')
            weights = [p.detach().clone() for p in agent.online.parameters()]
            if previous is not None and all(torch.equal(a,b) for a,b in zip(previous,weights)):
                raise ValueError('Training did not change snapshot weights')
            previous = weights
            findings.append({'steps':step,'updates':agent.optimiser_steps,'checkpoint_loaded':True,
                             'evaluation_episodes':len(rows),'evaluation_weights_unchanged':True})
        with (directory/'updates.csv').open(newline='') as handle:
            if not all(math.isfinite(float(row['loss'])) for row in csv.DictReader(handle)):
                raise ValueError('Nonfinite smoke loss')
    saved = {'passed':True,'protocol_sha256':metadata['protocol_sha256'],'development_seed':211,
        'development_evaluation_seeds':list(range(1600000,1600005)), 'snapshots':findings,
        'interpretation':'Technical validation only; temporary training files cleaned; not scientific findings'}
    write_json(marker,saved)
    print(json.dumps(saved,indent=2),flush=True)
    return saved


def run_one(protocol: dict, output: Path, seed: int) -> list[dict]:
    config = config_for(protocol,seed)
    output, metadata = prepare(output,protocol)
    marker = output/'validation.json'
    if not marker.exists() or not json.loads(marker.read_text())['passed']:
        raise RuntimeError('Focused tests and smoke validation must precede final training')
    case = output/'runs'/f'seed{seed}'
    complete = case/'complete.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if (saved['protocol_sha256'] != metadata['protocol_sha256'] or any(
                digest(case/name)!=h for name,h in saved['artifact_hashes'].items())):
            raise ValueError('Completed duration run artifacts changed')
        print(f'Preserving completed trajectory {seed}',flush=True)
        return saved['results']
    directory = case/'training'
    if (directory/'summary.json').exists():
        summary = json.loads((directory/'summary.json').read_text())
        manifest = json.loads((directory/'config.json').read_text())
        if manifest['config'] != config.to_dict() or manifest['snapshot_steps'] != list(STEPS):
            raise ValueError('Preserved training configuration differs')
    else:
        seed_cpu(seed)
        print(f'Starting uninterrupted food13 trajectory, seed{seed}',flush=True)
        summary = run_training(config,directory,make_charts=False,snapshot_steps=STEPS)
    verify_training(summary,config,STEPS)
    with (directory/'updates.csv').open(newline='') as handle:
        losses = [float(row['loss']) for row in csv.DictReader(handle)]
    if len(losses)!=99001 or not all(math.isfinite(x) for x in losses):
        raise ValueError('Training update log is incomplete or nonfinite')
    evaluations = case/'evaluations'
    evaluations.mkdir(exist_ok=True)
    results = []
    for step in STEPS:
        checkpoint = directory/f'checkpoints/step-{step:06d}.pt'
        cp_hash = digest(checkpoint)
        agent = DQNAgent.load(checkpoint)
        if (agent.config != config or agent.environment_steps != step
                or agent.optimiser_steps != step-999 or agent.config.input_dim!=13):
            raise ValueError('Snapshot encoder/progress mismatch')
        csv_path = evaluations/f'step-{step:06d}.csv'
        json_path = csv_path.with_suffix('.json')
        if json_path.exists():
            saved = json.loads(json_path.read_text())
            if saved['checkpoint_sha256'] != cp_hash or saved['csv_sha256'] != digest(csv_path):
                raise ValueError('Preserved evaluation differs')
            result = saved['result']
        else:
            if csv_path.exists():
                raise FileExistsError('Partial evaluation preserved for review')
            rows,metrics = evaluation_rows(agent,f'{step//1000}k',seed)
            if len(rows)!=100 or [r['evaluation_seed'] for r in rows]!=list(range(1500000,1500100)):
                raise ValueError('Evaluation did not use all fresh paired seeds')
            for row in rows:
                row['checkpoint_steps'] = step
            record = summary['snapshots'][str(step)]
            result = {'training_seed':seed,'checkpoint_steps':step,**metrics,
                'optimiser_updates':record['optimiser_updates'],'epsilon':record['epsilon'],
                'completed_training_episodes':record['completed_episodes'],
                'training_duration_seconds':record['duration_seconds'],'replay_size':record['replay_size'],
                'checkpoint_sha256':cp_hash}
            write_csv(csv_path,rows)
            write_json(json_path,{'result':result,'checkpoint_sha256':cp_hash,'csv_sha256':digest(csv_path)})
        if digest(checkpoint) != cp_hash:
            raise RuntimeError('Evaluation modified snapshot bytes')
        results.append(result)
        print(json.dumps({k:result[k] for k in ('training_seed','checkpoint_steps','average_score')}),flush=True)
    hashes = {str(p.relative_to(case)):digest(p) for p in sorted(case.rglob('*')) if p.is_file()}
    write_json(complete,{'protocol_sha256':metadata['protocol_sha256'],'results':results,'artifact_hashes':hashes})
    return results


def compare_results(results: list[dict]) -> dict:
    if (len(results)!=12 or {(r['training_seed'],r['checkpoint_steps']) for r in results}
            != {(s,t) for s in SEEDS for t in STEPS}):
        raise ValueError('Exactly three trajectories and four snapshots each are required')
    by_pair = {(r['training_seed'],r['checkpoint_steps']):r['average_score'] for r in results}
    checkpoints = {}
    for step in STEPS:
        values = [by_pair[seed,step] for seed in SEEDS]
        checkpoints[str(step)] = {'mean_of_training_seed_means':statistics.mean(values),
            'sd_across_training_seed_means':statistics.stdev(values),'independent_trajectories':3}
    differences = {str(seed):by_pair[seed,100000]-by_pair[seed,40000] for seed in SEEDS}
    return {'checkpoints':checkpoints,'primary_100k_minus_40k':differences,
        'mean_primary_difference':statistics.mean(differences.values()),
        'sd_primary_differences':statistics.stdev(differences.values()),
        'improved_trajectories':sum(v>0 for v in differences.values()),
        'interpretation':'Three independent training trajectories; correlated within-run snapshots. '
            'Primary comparison is100k minus40k. Intermediate results descriptive; no best-checkpoint selection.'}


def create_charts(output: Path, results: list[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by_pair = {(r['training_seed'],r['checkpoint_steps']):r['average_score'] for r in results}
    fig,ax = plt.subplots(figsize=(8,4.5),layout='constrained')
    for seed in SEEDS:
        ax.plot([s//1000 for s in STEPS],[by_pair[seed,s] for s in STEPS],marker='o',label=f'seed{seed}')
    ax.plot([s//1000 for s in STEPS],[statistics.mean(by_pair[seed,s] for seed in SEEDS) for s in STEPS],
            color='black',linewidth=2,linestyle='--',label='Mean of three trajectories')
    ax.set(xticks=[40,60,80,100],xlabel='Training interactions (thousands)',ylabel='Mean greedy food score',
           title='Evaluation along uninterrupted food13 training trajectories')
    ax.grid(alpha=.2);ax.legend();fig.savefig(output/'evaluation_by_step.png',dpi=150);plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,4.5),layout='constrained')
    for step,offset,color in ((40000,-.18,'#447da8'),(100000,.18,'#d48442')):
        bars = ax.bar([i+offset for i in range(3)],[by_pair[s,step] for s in SEEDS],width=.34,
                      label=f'{step//1000}k snapshot',color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set(xticks=range(3),xticklabels=SEEDS,xlabel='Independent training trajectory seed',
           ylabel='Mean greedy food score',title='Primary comparison: 100k versus 40k within each run')
    ax.margins(y=.2);ax.legend();fig.savefig(output/'primary_comparison.png',dpi=150);plt.close(fig)


def write_report(output: Path, protocol: dict, results: list[dict], comparison: dict) -> None:
    lines = ['# Milestone 5G: food13 training duration','',
        '## Frozen within-run protocol','',protocol['research_question'],protocol['primary_hypothesis'],'',
        'Three fresh independent food13 trajectories, seeds 177/188/199, run continuously to 100,000 moves. '
        'Snapshots at 40k/60k/80k/100k are post-update copies; no replay reset, episode truncation, '
        'RNG draw or evaluation is inserted into training. All snapshots are evaluated only after their '
        '100k trajectory finishes. The 40k snapshot is not a separately trained 40k-budget model. '
        'The primary endpoint is 100k minus 40k; intermediate measurements are descriptive and never select a best model.','',
        'Food13 and 13->128->128->3, rewards, game rules and standard DQN are unchanged. Adam .0005, '
        'gamma .99, replay 50000, batch 64, learning starts 1000, target every 1000 optimizer updates, '
        'clip 10, epsilon 1 to .05 over 50000 moves. Epsilon is .24 at 40k and .05 at 60k/80k/100k. '
        'This tests additional training under the unchanged schedule, not duration independently of exploration.','',
        'CPU one thread, deterministic algorithms where supported. Python/torch seed=s, replay=s+1, '
        'exploration=s+2. Food starts 1100000/1200000/1300000 plus episode index. Greedy evaluation '
        'uses identical fresh seeds 1500000-1500099 for every snapshot. Smoke uses seed 211, food 1410000 '
        'onward and evaluation 1600000-1600004 in temporary directories. Prior seed records were audited '
        'before freezing these ranges. Exact cross-platform/version determinism is not guaranteed.','',
        '## Measured results','',
        '| Seed | Moves | Completed episodes | Updates | Epsilon | Elapsed seconds | Mean | Median | Min | Max | Mean episode moves | Collision | Truncation |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['training_seed']} | {r['checkpoint_steps']} | {r['completed_training_episodes']} | "
            f"{r['optimiser_updates']} | {r['epsilon']:.2f} | {r['training_duration_seconds']:.2f} | "
            f"{r['average_score']:.2f} | {r['median_score']:.2f} | {r['minimum_score']} | {r['maximum_score']} | "
            f"{r['average_episode_length']:.2f} | {r['collision_rate']:.1%} | {r['truncation_rate']:.1%} |")
    lines += ['', '## Independent-trajectory comparison','']
    for step,v in comparison['checkpoints'].items():
        lines.append(f"- {int(step)//1000}k: mean {v['mean_of_training_seed_means']:.2f}; "
                     f"sample SD across three trajectory means {v['sd_across_training_seed_means']:.2f}.")
    lines += ['',f"Primary paired 100k-minus-40k differences: {comparison['primary_100k_minus_40k']}. "
        f"Mean {comparison['mean_primary_difference']:.2f}; improved on {comparison['improved_trajectories']}/3 trajectories.",'',
        'Three trajectories provide preliminary evidence. Four snapshots from a run are correlated repeated '
        'measurements, not four independent training runs. The 100 paired evaluation episodes do not create 100 '
        'independent training replications. No significance tests, tuning, stopping based on evaluation or '
        'retrospective best-checkpoint selection are performed. A higher intermediate score does not replace '
        'the predefined 100k endpoint. Improvement need not be monotonic, and longer survival can reflect '
        'truncation rather than food collection. Fixed snapshots are for evaluation, not exact resume: existing '
        'checkpoint files do not store active-game/replay state; training remains in memory throughout.','',
        '## Artifacts and reproduction','',
        'protocol.json and metadata.json freeze settings/source/software. Each runs/seed*/training directory '
        'has metrics, configurations and ignored checkpoints; evaluations contains 100 raw outcomes per snapshot. '
        'Completion records hash all run files; root per_snapshot_summary.csv and comparison_summary.json '
        'store measured aggregates. Previous experiments/checkpoints are preserved without regeneration.','',
        '`python -m snake_ai.experiments.training_duration --smoke` validates temporary development training. '
        'Run the module without --smoke for all three trajectories, --seed 177 for one missing trajectory '
        'or --aggregate-only to read successful records. Successful runs are reused '
        'without training. Partial failures are preserved for review and are not approximately resumed.','',
        '![Snapshot evaluation scores](evaluation_by_step.png)',
        '![Primary paired comparison](primary_comparison.png)','']
    with (output/'report.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def aggregate(protocol: dict, output: Path) -> dict | None:
    output,_ = prepare(output,protocol)
    if not all((output/f'runs/seed{seed}/complete.json').exists() for seed in SEEDS):
        print('Waiting for all three completed trajectories.',flush=True)
        return None
    results = [r for seed in SEEDS for r in run_one(protocol,output,seed)]
    comparison = compare_results(results)
    summary = output/'comparison_summary.json'
    if summary.exists():
        if json.loads(summary.read_text()) != comparison:
            raise ValueError('Saved duration aggregate differs')
        return comparison
    write_csv(output/'per_snapshot_summary.csv',[{k:v for k,v in r.items() if k!='feature_order'} for r in results])
    create_charts(output,results)
    write_report(output,protocol,results,comparison)
    write_json(summary,comparison)
    print(json.dumps(comparison,indent=2),flush=True)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--aggregate-only',action='store_true')
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.smoke:
        if args.seed is not None or args.aggregate_only:
            parser.error('--smoke is separate from final selection')
        run_smoke(protocol,args.output)
    elif args.aggregate_only:
        aggregate(protocol,args.output)
    else:
        for seed in ([args.seed] if args.seed is not None else SEEDS):
            run_one(protocol,args.output,seed)
        aggregate(protocol,args.output)


if __name__ == '__main__':
    main()
