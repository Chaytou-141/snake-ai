"""Generate a local static evidence dashboard by reading saved results only."""

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil
import statistics
import webbrowser

from snake_ai.demos.catalog import PROJECT_ROOT

STUDIES = (
    ('5d','milestone5d_feature_comparison','Obstacle distances','Original11','Geometry14',
     '500 episodes or 100k moves per run; actual experience counts differ.'),
    ('5e','milestone5e_food_distance','Food displacement','Original11','Food13','40k moves per independent model.'),
    ('5f','milestone5f_body_awareness','Static body awareness','Food13','Body16','40k moves per independent model.'),
    ('5h','milestone5h_double_dqn','Learning target','Standard DQN','Double DQN','100k moves; same food13 representation.'),
)


def load_data(root: Path = PROJECT_ROOT) -> dict:
    root=Path(root);sources={}
    def read(relative):
        path=root/relative;data=path.read_bytes()
        sources[relative]=hashlib.sha256(data).hexdigest()
        return json.loads(data)
    def csv_rows(relative):
        path=root/relative;sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline='',encoding='utf-8') as handle:return list(csv.DictReader(handle))
    initial=read('results/milestone4_evaluation/summary.json')
    initial_rows=csv_rows('results/milestone4_evaluation/episodes.csv')
    baseline={}
    for policy in ('random','dqn_greedy'):
        scores=[int(r['score']) for r in initial_rows if r['policy']==policy]
        value=initial['summaries'][policy]
        if len(scores)!=value['episodes'] or not math.isclose(statistics.mean(scores),value['average_score']):
            raise ValueError('Initial evaluation raw rows differ from saved summary')
        baseline[policy]=value
    studies=[]
    for key,folder,title,label_a,label_b,budget in STUDIES:
        prefix=f'results/{folder}'
        summary=read(f'{prefix}/comparison_summary.json');protocol=read(f'{prefix}/protocol.json')
        rows=csv_rows(f'{prefix}/per_seed_summary.csv');seeds=protocol['training_seeds']
        if len(rows)!=6 or {(r['model'],int(r['training_seed'])) for r in rows}!={(m,s) for m in 'AB' for s in seeds}:
            raise ValueError(f'{key}: expected six independent model/seed results')
        for model in 'AB':
            values=[float(r['average_score']) for r in rows if r['model']==model]
            metrics=summary['variants'][model]
            if not (math.isclose(statistics.mean(values),metrics['mean_of_training_seed_means'])
                    and math.isclose(statistics.stdev(values),metrics['sd_across_training_seed_means'])):
                raise ValueError(f'{key}: aggregate does not match per-seed measurements')
        pairs={str(s):next(float(r['average_score']) for r in rows if r['model']=='B' and int(r['training_seed'])==s)
               -next(float(r['average_score']) for r in rows if r['model']=='A' and int(r['training_seed'])==s) for s in seeds}
        if any(not math.isclose(v,summary['paired_differences_B_minus_A'][s],abs_tol=1e-12) for s,v in pairs.items()):
            raise ValueError(f'{key}: paired differences differ')
        if not math.isclose(statistics.mean(pairs.values()),summary['mean_paired_difference'],abs_tol=1e-12):
            raise ValueError(f'{key}: mean paired difference differs')
        studies.append({'id':key,'folder':folder,'title':title,'labels':{'A':label_a,'B':label_b},
            'budget':budget,'seeds':seeds,'evaluation_seed_start':protocol['evaluation_seed_start'],
            'summary':summary,'rows':rows,'chart':f'{prefix}/average_score.png'})
    duration=read('results/milestone5g_training_duration/comparison_summary.json')
    duration_protocol=read('results/milestone5g_training_duration/protocol.json')
    duration_rows=csv_rows('results/milestone5g_training_duration/per_snapshot_summary.csv')
    if len(duration_rows)!=12 or {(int(r['training_seed']),int(r['checkpoint_steps'])) for r in duration_rows}!= {
            (s,t) for s in duration_protocol['training_seeds'] for t in duration_protocol['snapshot_steps']}:
        raise ValueError('Duration comparison requires three trajectories and four snapshots each')
    for step in duration_protocol['snapshot_steps']:
        vals=[float(r['average_score']) for r in duration_rows if int(r['checkpoint_steps'])==step]
        summary=duration['checkpoints'][str(step)]
        if not (math.isclose(statistics.mean(vals),summary['mean_of_training_seed_means'])
                and math.isclose(statistics.stdev(vals),summary['sd_across_training_seed_means'])):
            raise ValueError('Duration aggregate differs from saved snapshot rows')
    for seed in duration_protocol['training_seeds']:
        pair={int(r['checkpoint_steps']):float(r['average_score']) for r in duration_rows if int(r['training_seed'])==seed}
        if not math.isclose(pair[100000]-pair[40000],duration['primary_100k_minus_40k'][str(seed)],abs_tol=1e-12):
            raise ValueError('Duration paired endpoint difference differs')
    if not math.isclose(statistics.mean(duration['primary_100k_minus_40k'].values()),duration['mean_primary_difference'],abs_tol=1e-12):
        raise ValueError('Duration mean paired difference differs')
    collision=read('results/milestone5b_collision_diagnostics/summary.json')
    traps=read('results/milestone5c_trap_analysis/summary.json')
    diagnostic=read('results/milestone5h_double_dqn/diagnostic_summary.json')
    return {'initial':baseline,'initial_config':initial['config'], 'studies':studies,
        'duration':duration,'duration_protocol':duration_protocol,'duration_rows':duration_rows,
        'collision':{k:collision[k] for k in ('collision_count','collision_categories','reproduction_verified')},
        'traps':traps['analysis'],'diagnostic':{'common_states':diagnostic['common_states'],
            'per_run':diagnostic['per_run'],'interpretation':diagnostic['interpretation']},'source_sha256':sources}


def build_dashboard(root: Path = PROJECT_ROOT, output: Path | None = None) -> Path:
    root=Path(root).resolve();output=Path(output or root/'portfolio/dashboard').resolve()
    if output==root or output.is_relative_to(root/'results'):
        raise ValueError('Dashboard output must not overwrite the project root or historical results')
    data=load_data(root);output.mkdir(parents=True,exist_ok=True);assets=output/'assets';assets.mkdir(exist_ok=True)
    def copy(relative,name):
        shutil.copyfile(root/relative,assets/name)
        return 'assets/'+name
    sections=[]
    for study in data['studies']:
        chart=copy(study['chart'],study['id']+'.png');summary=study['summary'];labels=study['labels']
        body=''.join(f"<tr><th>{escape(labels[m])}</th><td>{summary['variants'][m]['mean_of_training_seed_means']:.2f}</td>"
                     f"<td>{summary['variants'][m]['sd_across_training_seed_means']:.2f}</td></tr>" for m in 'AB')
        notes={'5d':'Static obstacle distances did not improve score on any of the three pairs.',
               '5e':'Food13 improved two pairs, with a tentative average advantage.',
               '5f':'Body16 improved one pair; no consistent benefit was observed.',
               '5h':'Double DQN improved one pair. Its lower observed seed variance needs replication.'}
        start=study['evaluation_seed_start']
        sections.append(f"""<section id="{study['id']}"><div class="study-title"><span>{study['id'].upper()}</span>
<h2>{escape(study['title'])}</h2></div><p class="context">{escape(study['budget'])}
Training seeds {', '.join(map(str,study['seeds']))}; evaluation {start}–{start+99}.</p>
<div class="study-body"><div><table><thead><tr><th>Comparison</th><th>Mean food</th><th>Sample SD*</th></tr></thead>
<tbody>{body}</tbody></table></div><div class="finding"><p>{notes[study['id']]}</p>
<p>Mean paired B − A: <strong>{summary['mean_paired_difference']:+.2f}</strong>.</p></div></div>
<details><summary>View the original comparison chart</summary><img src="{chart}" alt="{escape(study['title'])} across the three training seeds"></details>
</section>""")
    duration_chart=copy('results/milestone5g_training_duration/evaluation_by_step.png','5g.png')
    d=data['duration'];duration_body=''.join(f"<tr><th>{int(step)//1000}k</th><td>{v['mean_of_training_seed_means']:.2f}</td>"
        f"<td>{v['sd_across_training_seed_means']:.2f}</td></tr>" for step,v in d['checkpoints'].items())
    duration_section=f"""<section id="5g"><div class="study-title"><span>5G</span><h2>More training, same trajectory</h2></div>
<p class="context">Food13; three uninterrupted 100k runs. Seeds 177/188/199; evaluation 1500000–1500099.</p>
<div class="study-body"><table><thead><tr><th>Snapshot</th><th>Mean food</th><th>Sample SD*</th></tr></thead><tbody>{duration_body}</tbody></table>
<div class="finding"><p>Mean100k −40k change: <strong>{d['mean_primary_difference']:+.2f}</strong>.
Performance was non-monotonic; the intermediate snapshots never replaced the predefined 100k endpoint.</p>
<p>Snapshots within a run are correlated repeated measurements, not independent training runs.</p></div></div>
<details><summary>View all measured checkpoints</summary><img src="{duration_chart}" alt="Within-run evaluation at 40k, 60k, 80k and 100k"></details></section>"""
    sections.insert(3,duration_section)
    initial=data['initial'];categories=data['collision']['collision_categories'];traps=data['traps']
    preview=''
    if (root/'portfolio/assets/playback.png').is_file():
        image=copy('portfolio/assets/playback.png','playback.png')
        preview=f'<figure><img class="demo" src="{image}" alt="Actual original11 model playback, with live Q estimates"><figcaption>Actual local inference, seed 800000. A demonstration, not a new evaluation experiment.</figcaption></figure>'
    html=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SnakeAI · Experiment ledger</title><style>
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#edf2f6;color:#183147;font:16px/1.6 'Aptos','Segoe UI',sans-serif}}
nav{{position:fixed;inset:0 auto 0 0;width:228px;background:#122a3e;color:#e8f0f5;padding:34px 24px}}nav strong{{font:30px 'Bahnschrift','Segoe UI',sans-serif}}nav p{{font-size:13px;color:#bacbd5}}nav a{{display:block;color:#e8f0f5;text-decoration:none;padding:9px 0;border-bottom:1px solid #294254}}nav a:hover,nav a:focus{{color:#92dbc9}}
main{{margin-left:228px;max-width:1230px;padding:45px 48px 70px}}header{{padding-bottom:28px;border-bottom:3px solid #1a6871}}h1{{font:600 clamp(32px,4vw,52px)/1.12 'Bahnschrift','Segoe UI',sans-serif;letter-spacing:-1.4px;margin:10px 0 22px;max-width:800px}}h2{{font:600 25px/1.3 'Bahnschrift','Segoe UI',sans-serif;margin:0}}p{{max-width:78ch}}.lead{{font-size:20px;color:#375467}}.notice{{background:#dcebea;border-left:4px solid #267678;padding:16px 20px;margin:26px 0}}section{{background:#fff;padding:27px 30px;margin:22px 0;scroll-margin-top:20px}}.study-title{{display:flex;gap:15px;align-items:center}}.study-title span{{font:600 17px 'Bahnschrift','Segoe UI',sans-serif;color:#176c75;border:1px solid #aac4c9;padding:5px 8px}}
.context,figcaption,.footnote{{color:#526b7b;font-size:14px}}.study-body{{display:grid;grid-template-columns:1fr 1fr;gap:25px;align-items:start}}table{{border-collapse:collapse;width:100%;font-size:15px;text-align:left}}th,td{{padding:10px 8px;border-bottom:1px solid #dce4e9}}thead{{color:#4d6576;font-size:13px}}.study-body>*{{min-width:0}}.finding p{{margin-top:6px}}strong{{font-weight:650}}details{{margin-top:16px}}summary{{color:#116979;cursor:pointer;padding:7px 0}}img{{max-width:100%;height:auto}}.demo{{width:100%;max-width:800px}}figure{{margin:16px 0}}a{{color:#116979}}code{{background:#e8edf1;padding:3px 6px;font-size:13px}}.initial{{display:flex;gap:36px;flex-wrap:wrap}}.initial b{{font:600 38px 'Bahnschrift','Segoe UI',sans-serif;color:#156773;display:block}}.initial div{{min-width:170px}}footer{{font-size:14px;color:#526b7b}}@media(max-width:850px){{nav{{position:static;width:auto}}nav a{{display:inline-block;margin-right:15px}}main{{margin:0;padding:22px 15px}}section{{padding:20px 16px}}.study-body{{grid-template-columns:1fr}}table{{font-size:14px}}}}
@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
</style></head><body><nav><strong>SnakeAI</strong><p>Version 1.0<br>Experiment ledger</p><a href="#initial">Initial learning result</a>
<a href="#5d">5D · Obstacle distance</a><a href="#5e">5E · Food displacement</a><a href="#5f">5F · Body awareness</a>
<a href="#5g">5G · Training duration</a><a href="#5h">5H · Double DQN</a><a href="#failures">Failure analysis</a>
<p>Local static report.<br>No training or live evaluation.</p></nav><main><header><h1>What actually improved<br>the agent?</h1>
<p class="lead">A measured study of observation design, training duration and Double DQN in Snake.</p>
<p>One game engine. Reproducible PyTorch experiments. Conclusions that include the negative results.</p></header>
<div class="notice"><strong>Read each comparison in context.</strong> Experiments use different training seeds, evaluation seeds and budgets.
Their scores are not a universal model ranking. *Sample SD describes variation across three training-seed means.</div>
<section id="initial"><h2>Initial learning result</h2><p class="context">Original11; one training seed 7, 500 episodes. Paired 100-episode evaluation 900000–900099.</p>
<div class="initial"><div><b>{initial['random']['average_score']:.2f}</b>Random policy · mean food</div><div><b>{initial['dqn_greedy']['average_score']:.2f}</b>Greedy DQN · mean food</div></div>
<p>This initial result demonstrates learning for one model. It does not establish generalisation across independent training runs; every episode ended in collision.</p>{preview}</section>
{''.join(sections)}
<section id="failures"><h2>Learning did not solve survival</h2><p>In the original model's 100 recorded collisions,
<strong>{categories['B']['count']} had no immediately safe action</strong> at the final move. Avoidable immediate final collisions: {categories['A']['count']}.</p>
<p>Earlier decisions matter. Static reachable-space estimates are snapshots of a moving body, not guarantees of future safety.
The observed Pearson correlation with remaining moves was <strong>{traps['reachable_space_remaining_moves_pearson_r']:.3f}</strong>, computed across correlated moves.</p>
<p>{traps['one_safe_recoveries_to_two_or_three']} of {traps['one_safe_spell_entries']} one-safe-action spells recovered to multiple options.
A constrained state did not always imply imminent death.</p>
<p>Double DQN's Q diagnostic used the same {data['diagnostic']['common_states']} recorded states for all estimators. Online-target disagreement is not proof of true overestimation bias.</p></section>
<footer><p>Generated from saved JSON/CSV measurements. No scores are hand-adjusted. Source paths and SHA-256 hashes:
<a href="data.json">data.json</a>. Checkpoints remain local and excluded from Git.</p>
<p>Launch the demonstration: <code>python -m snake_ai.demo</code>. Three training seeds provide preliminary evidence; no significance tests or best-checkpoint selection were used.</p></footer></main></body></html>"""
    (output/'data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    path=output/'index.html';path.write_text(html,encoding='utf-8')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--open',action='store_true',help='Open the generated local HTML report')
    args=parser.parse_args()
    try:path=build_dashboard(output=args.output)
    except (OSError,ValueError,KeyError) as error:parser.error(f'Cannot build dashboard from saved results: {error}')
    print(path)
    if args.open:webbrowser.open(path.as_uri())


if __name__=='__main__':main()
