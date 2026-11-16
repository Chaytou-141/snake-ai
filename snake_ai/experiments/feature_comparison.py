"""Run fixed observation protocols, validate separately, and preserve successful runs."""

import argparse
import csv
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import statistics
import tempfile

import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.rl import FEATURE_NAMES, SnakeEnv, feature_names
from snake_ai.training.config import TrainingConfig
from snake_ai.training.evaluate import evaluate_agent, summarise
from snake_ai.training.train import run_training

DEFAULT_PROTOCOL = Path("configs/feature_comparison_v1.json")
DEFAULT_OUTPUT = Path("results/milestone5d_feature_comparison")
SOURCE_FILES = ("snake_ai/game/engine.py", "snake_ai/game/geometry.py", "snake_ai/rl/environment.py",
    "snake_ai/models/dqn.py", "snake_ai/agents/dqn_agent.py", "snake_ai/training/config.py",
    "snake_ai/training/replay_buffer.py", "snake_ai/training/train.py", "snake_ai/training/evaluate.py",
    "snake_ai/experiments/feature_comparison.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("protocol_version") == "5f-v1":
        expected = json.loads((Path(__file__).resolve().parents[2]/"configs/body_awareness_v1.json").read_text(encoding="utf-8"))
        defaults = TrainingConfig().to_dict()
        omitted = {"seed","training_seed_start","evaluation_seed_start","evaluation_episodes","random_policy_seed_start"}
        base = {key:value for key,value in defaults.items() if key not in omitted}
        base.update(episodes=40000,max_environment_steps=40000)
        seeds = protocol["training_seeds"]
        if (protocol != expected or protocol["base_training_config"] != base
                or seeds != [111,122,133] or any(type(s) is not int for s in seeds)
                or protocol["variants"] != {"A":"food13","B":"body16"}
                or protocol["evaluation_seed_start"] != 950000 or protocol["evaluation_episodes"] != 100
                or protocol["run_order"] != [[m,s] for s in seeds for m in ("A","B")]
                or protocol["training_seed_starts"] != {str(s):700000+i*50000 for i,s in enumerate(seeds)}
                or protocol["smoke"] != {"training_seed":155,"evaluation_seed_start":960000,
                    "evaluation_episodes":5,"episodes":256,"max_environment_steps":256,"learning_starts":32,"batch_size":16}):
            raise ValueError("Unexpected predeclared body-awareness protocol")
        return protocol
    if protocol.get("protocol_version") == "5e-v1":
        expected = json.loads((Path(__file__).resolve().parents[2]/"configs/food_distance_v1.json").read_text(encoding="utf-8"))
        if protocol != expected:
            raise ValueError("Protocol 5e-v1 must match the predeclared food-distance experiment")
        defaults = TrainingConfig().to_dict()
        omitted = {"seed","training_seed_start","evaluation_seed_start","evaluation_episodes","random_policy_seed_start"}
        base = {key:value for key,value in defaults.items() if key not in omitted}
        base.update(episodes=40000,max_environment_steps=40000)
        if protocol["base_training_config"] != base:
            raise ValueError("Food comparison must preserve hyperparameters and use exactly 40000 moves")
        if (protocol["training_seeds"] != [44,55,66] or protocol["variants"] != {"A":"original11","B":"food13"}
                or protocol["evaluation_seed_start"] != 930000 or protocol["evaluation_episodes"] != 100
                or protocol["run_order"] != [[m,s] for s in (44,55,66) for m in ("A","B")]
                or protocol["smoke"] != {"training_seed":88,"evaluation_seed_start":940000,
                    "evaluation_episodes":5,"episodes":256,"max_environment_steps":256,"learning_starts":32,"batch_size":16}):
            raise ValueError("Unexpected food comparison variants, seeds or smoke protocol")
        return protocol
    if (protocol["protocol_version"] != "5d-v1" or protocol["training_seeds"] != [11,22,33]
            or protocol["variants"] != {"A":"original11", "B":"distance14"}
            or protocol["evaluation_seed_start"] != 910000 or protocol["evaluation_episodes"] != 100
            or protocol["run_order"] != [[model,seed] for seed in (11,22,33) for model in ("A","B")]):
        raise ValueError("Protocol v1 requires the fixed variants, seed pairs, evaluation range and run order")
    base = protocol["base_training_config"]
    defaults = TrainingConfig().to_dict()
    omitted = {"seed","training_seed_start","evaluation_seed_start","evaluation_episodes","random_policy_seed_start"}
    expected = {key:value for key,value in defaults.items() if key not in omitted}
    if base != expected:
        raise ValueError("Protocol v1 training budgets/hyperparameters/rewards must match predefined settings")
    smoke = protocol["smoke"]
    if smoke != {"training_seed":77, "evaluation_seed_start":920000, "evaluation_episodes":5,
                 "episodes":6, "max_environment_steps":256, "learning_starts":32, "batch_size":16}:
        raise ValueError("Unexpected development validation protocol")
    return protocol


def training_config(protocol: dict, model: str, seed: int, *, smoke: bool = False) -> TrainingConfig:
    if model not in protocol["variants"] or type(seed) is not int:
        raise ValueError("Unknown model or invalid training seed")
    if not smoke and seed not in protocol["training_seeds"]:
        raise ValueError("Final training seed is not in the fixed protocol")
    config = TrainingConfig.from_dict({**protocol["base_training_config"], "seed":seed,
        "training_seed_start":((850000 if smoke else protocol["training_seed_starts"][str(seed)])
            if protocol["protocol_version"]=="5f-v1" else
            (300000 + seed*5000 if protocol["protocol_version"]=="5e-v1" else 200000 + seed*1000)), "evaluation_seed_start":protocol["evaluation_seed_start"],
        "evaluation_episodes":protocol["evaluation_episodes"], "observation_mode":protocol["variants"][model]})
    if smoke:
        if seed != protocol["smoke"]["training_seed"]:
            raise ValueError("Smoke training must use its dedicated development seed")
        config = replace(config, **{key:value for key,value in protocol["smoke"].items() if key != "training_seed"})
    return config


def prepare_output(output: Path, protocol: dict, *, extra_source_files: tuple[str, ...] = (),
                   seed_rule: str | None = None) -> tuple[Path, dict]:
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    metadata = {"protocol_sha256": hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
        "source_sha256": {name:digest(repo/name) for name in SOURCE_FILES + extra_source_files},
        "python":platform.python_version(), "platform":platform.platform(), "device":"cpu",
        "dependencies":{name:version(name) for name in ("torch","pygame","matplotlib","numpy")},
        "deterministic_algorithms":True, "cpu_threads":1,
        "seed_rule":"Python and torch: seed; replay: seed+1; exploration: seed+2; food: " +
            ("protocol training_seed_starts[seed]+episode_index; smoke food starts 850000" if protocol["protocol_version"]=="5f-v1" else
             "300000+5000*seed+episode_index" if protocol["protocol_version"]=="5e-v1" else "200000+1000*seed+episode_index")}
    if seed_rule is not None:
        metadata["seed_rule"] = seed_rule
    if output.exists():
        if not (output/"protocol.json").exists() or not (output/"metadata.json").exists():
            raise FileExistsError("Output is not a recognised experiment directory")
        if json.loads((output/"protocol.json").read_text()) != protocol:
            raise ValueError("Stored protocol differs; use a new version/directory before any affected rerun")
        if json.loads((output/"metadata.json").read_text()) != metadata:
            raise ValueError("Source/software changed since protocol was frozen; review before continuing")
    else:
        output.mkdir(parents=True)
        write_json(output/"protocol.json",protocol)
        write_json(output/"metadata.json",metadata)
    return output, metadata


def seed_cpu(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def evaluation_rows(agent: DQNAgent, model: str, seed: int) -> tuple[list[dict], dict]:
    rows, manifest = evaluate_agent(agent, include_random=False)
    values = [row["score"] for row in rows]
    for row in rows:
        row["model"] = model
        row["training_seed"] = seed
        row["evaluation_seed"] = row.pop("environment_seed")
        row["collision"] = row["end_reason"] == "collision"
        row["win"] = row["end_reason"] == "win"
    summary = {**manifest["summaries"]["dqn_greedy"],
               "score_sd_within_evaluation":statistics.stdev(values) if len(values)>1 else 0.,
               "weights_unchanged":manifest["weights_unchanged"], "observation_mode":agent.config.observation_mode,
               "input_dim":agent.config.input_dim,
               "feature_order":list(feature_names(agent.config.observation_mode))}
    return rows, summary


def run_smoke(protocol: dict, output: Path) -> dict:
    output, metadata = prepare_output(output, protocol)
    marker = output/"validation.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["protocol_sha256"] != metadata["protocol_sha256"] or not saved["passed"]:
            raise ValueError("Invalid previous validation")
        return saved
    findings = {}
    with tempfile.TemporaryDirectory(prefix="snakeai-feature-smoke-") as temp:
        for model in ("A","B"):
            config = training_config(protocol,model,protocol["smoke"]["training_seed"],smoke=True)
            seed_cpu(config.seed)
            run_dir = Path(temp)/model
            training = run_training(config,run_dir,make_charts=False)
            if (training["status"] not in ("episode_budget","environment_step_cap")
                    or training["optimiser_updates_this_run"] < 1 or not training["weights_changed"]):
                raise RuntimeError("Smoke training did not complete valid finite updates")
            loaded = DQNAgent.load(run_dir/"checkpoints/final.pt")
            rows, metrics = evaluation_rows(loaded,model,config.seed)
            env = SnakeEnv(config.environment,config.rewards,observation_mode=config.observation_mode)
            obs,_ = env.reset(seed=config.evaluation_seed_start)
            assert len(obs)==config.input_dim and loaded.predict(obs).shape==(3,)
            assert all(torch.isfinite(loaded.predict(obs))) and metrics["weights_unchanged"]
            with (run_dir/"updates.csv").open(newline="") as handle:
                assert all(float(row["loss"]) < float("inf") for row in csv.DictReader(handle))
            findings[model] = {"input_dim":config.input_dim, "updates":training["optimiser_updates_this_run"],
                "weights_changed":True, "checkpoint_loaded":True, "evaluation_episodes":len(rows),
                "evaluation_weights_unchanged":True}
    saved = {"passed":True, "protocol_sha256":metadata["protocol_sha256"], "development_seed":protocol["smoke"]["training_seed"],
             "development_evaluation_seeds":list(range(protocol["smoke"]["evaluation_seed_start"],
                 protocol["smoke"]["evaluation_seed_start"]+protocol["smoke"]["evaluation_episodes"])), "findings":findings,
             "interpretation":"Technical validation only; not a scientific performance result"}
    write_json(marker,saved)
    print(json.dumps(saved,indent=2),flush=True)
    return saved


def run_one(protocol: dict, output: Path, model: str, seed: int) -> dict:
    config = training_config(protocol,model,seed)
    output, metadata = prepare_output(output,protocol)
    if not (output/"validation.json").exists() or not json.loads((output/"validation.json").read_text())["passed"]:
        raise RuntimeError("Run focused tests and the development smoke test before final training")
    case = output/"runs"/f"{model}_seed{seed}"
    complete = case/"complete.json"
    if complete.exists():
        cached = json.loads(complete.read_text())
        if cached["protocol_sha256"] != metadata["protocol_sha256"] or any(
                digest(case/name)!=expected for name,expected in cached["artifact_hashes"].items()):
            raise ValueError("Completed run artifacts changed")
        print(f"Preserving completed run {model}, seed {seed}",flush=True)
        return cached["result"]
    seed_cpu(seed)
    training_dir = case/"training"
    if (training_dir/"summary.json").exists():
        training = json.loads((training_dir/"summary.json").read_text())
        if json.loads((training_dir/"config.json").read_text())["config"] != config.to_dict():
            raise ValueError("Existing training configuration differs")
    else:
        print(f"Starting {model} ({config.input_dim} inputs), training seed {seed}",flush=True)
        training = run_training(config,training_dir,make_charts=False)
    if training["status"] not in ("episode_budget","environment_step_cap"):
        raise RuntimeError(f"Preserved incomplete training {model}/{seed}: {training['status']}")
    if (training["environment_steps_this_run"]>config.max_environment_steps
            or training["completed_episodes_this_run"]>config.episodes):
        raise ValueError("Training budget exceeded")
    if protocol["protocol_version"] in ("5e-v1","5f-v1") and training["environment_steps_this_run"]!=40000:
        raise RuntimeError("Incomplete fixed-step run preserved; do not evaluate or silently restart")
    with (training_dir/"updates.csv").open(newline="") as handle:
        losses = [float(row["loss"]) for row in csv.DictReader(handle)]
    if not losses or not all(torch.isfinite(torch.tensor(losses))):
        raise ValueError("Training loss is missing or nonfinite")
    checkpoint = training_dir/"checkpoints/final.pt"
    checkpoint_hash = digest(checkpoint)
    agent = DQNAgent.load(checkpoint)
    if agent.config != config:
        raise ValueError("Final checkpoint configuration mismatch")
    evaluation_file = case/"evaluation.json"
    if evaluation_file.exists():
        saved_eval = json.loads(evaluation_file.read_text())
        if saved_eval["checkpoint_sha256"] != checkpoint_hash or saved_eval["csv_sha256"] != digest(case/"evaluation.csv"):
            raise ValueError("Existing evaluation metadata mismatch")
        metrics = saved_eval["metrics"]
    else:
        if (case/"evaluation.csv").exists():
            raise FileExistsError("Incomplete evaluation output preserved; review rather than overwrite")
        eval_rows, metrics = evaluation_rows(agent,model,seed)
        if len(eval_rows)!=100 or [r["evaluation_seed"] for r in eval_rows] != list(range(config.evaluation_seed_start,config.evaluation_seed_start+config.evaluation_episodes)):
            raise ValueError("Evaluation did not use all 100 reserved seeds")
        if digest(checkpoint) != checkpoint_hash:
            raise RuntimeError("Evaluation changed checkpoint bytes")
        write_csv(case/"evaluation.csv",eval_rows)
        write_json(evaluation_file,{"checkpoint_sha256":checkpoint_hash,
            "csv_sha256":digest(case/"evaluation.csv"), "metrics":metrics})
    result = {"model":model, "training_seed":seed, **metrics,
        "completed_training_episodes":training["completed_episodes_this_run"],
        "training_environment_steps":training["environment_steps_this_run"],
        "optimiser_updates":training["optimiser_updates_this_run"], "training_duration_seconds":training["duration_seconds"],
        "training_status":training["status"], "checkpoint_sha256":checkpoint_hash,
        "parameter_count":sum(p.numel() for p in agent.online.parameters())}
    artifacts = {str(path.relative_to(case)):digest(path) for path in sorted(case.rglob("*")) if path.is_file()}
    write_json(complete,{"protocol_sha256":metadata["protocol_sha256"], "result":result, "artifact_hashes":artifacts})
    print(json.dumps({key:result[key] for key in ("model","training_seed","average_score",
        "training_environment_steps","optimiser_updates")}),flush=True)
    return result


def compare_results(results: list[dict], seeds=(11,22,33)) -> dict:
    expected = {(model,seed) for model in ("A","B") for seed in seeds}
    if len(results)!=6 or {(row["model"],row["training_seed"]) for row in results} != expected:
        raise ValueError("Comparison requires exactly all six completed model/seed pairs")
    grouped = {model:{row["training_seed"]:row["average_score"] for row in results if row["model"]==model}
               for model in ("A","B")}
    variants = {model:{"mean_of_training_seed_means":statistics.mean(values.values()),
                      "sd_across_training_seed_means":statistics.stdev(values.values()), "training_replications":3}
                for model,values in grouped.items()}
    differences = {str(seed):grouped["B"][seed]-grouped["A"][seed] for seed in seeds}
    values = list(differences.values())
    return {"variants":variants, "paired_differences_B_minus_A":differences,
        "mean_paired_difference":statistics.mean(values),
        "B_higher_all_three_seeds":all(value>0 for value in values),
        "A_higher_all_three_seeds":all(value<0 for value in values),
        "direction_consistent_nonzero":all(value>0 for value in values) or all(value<0 for value in values),
        "interpretation":"Three independent training seeds per variant; preliminary descriptive comparison, no significance tests. 300 evaluation episodes are not 300 training replications."}


def create_charts(output: Path, results: list[dict], protocol: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = protocol["training_seeds"] if protocol else [11,22,33]
    food = protocol is not None and protocol["protocol_version"]=="5e-v1"
    body = protocol is not None and protocol["protocol_version"]=="5f-v1"
    by_pair = {(row["model"],row["training_seed"]):row for row in results}
    colors = {"A":"#447da8","B":"#d48442"}
    for filename,metric,title,ylabel in (("average_score","average_score","Greedy score on 100 fresh seeds","Mean food score"),
        ("episode_length","average_episode_length","Greedy episode length on 100 fresh seeds","Mean moves")):
        fig,ax = plt.subplots(figsize=(8,4.5),constrained_layout=True)
        for model,offset in (("A",-0.18),("B",0.18)):
            bars=ax.bar([index+offset for index in range(3)],[by_pair[model,seed][metric] for seed in seeds],
                        width=.34,color=colors[model],label=f"{model}: {len(feature_names(protocol['variants'][model])) if protocol else (11 if model=='A' else 14)} features")
            ax.bar_label(bars,fmt="%.2f",padding=3)
        ax.set(xticks=range(3),xticklabels=seeds,xlabel="Independent training seed",ylabel=ylabel,title=title)
        ax.margins(y=.2);ax.legend();fig.savefig(output/f"{filename}.png",dpi=150);plt.close(fig)
    if body:
        return
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis,model in zip(axes,("A","B")):
        for seed in seeds:
            with (output/"runs"/f"{model}_seed{seed}"/"training/episodes.csv").open(newline="") as handle:
                rows=[row for row in csv.DictReader(handle) if row["episode_complete"]=="True"]
            scores=[float(row["score"]) for row in rows]
            rolling=[statistics.mean(scores[max(0,index-24):index+1]) for index in range(len(scores))]
            axis.plot([int(row["environment_steps"]) for row in rows] if food else range(1,len(rolling)+1),rolling,label=f"seed {seed}")
        axis.set(title=f"Model {model}",xlabel="Environment interactions" if food else "Completed training episode",ylabel="Rolling mean food score (up to 25 episodes)")
        axis.legend();axis.grid(alpha=.2)
    fig.suptitle("Matched 40,000-step training; rolling scores exclude incomplete episodes" if food else
                 "Training includes exploration; equal episode budgets do not imply equal steps")
    fig.savefig(output/"training_performance.png",dpi=150);plt.close(fig)
    if food:
        return
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    for index,model in enumerate(("A","B")):
        means=[by_pair[model,seed]["average_score"] for seed in seeds]
        ax.scatter([index-.06,index,index+.06],means,color=colors[model],label=f"{model}: individual training seeds")
        ax.errorbar(index,statistics.mean(means),yerr=statistics.stdev(means),fmt="_",color="black",capsize=8)
    ax.set(xticks=[0,1],xticklabels=["A: 11 features","B: 14 features"],ylabel="Evaluation mean score per training seed",
           title="Between-training-seed variability (mean ± sample SD; not a confidence interval)")
    ax.legend(fontsize=8);fig.savefig(output/"performance_variability.png",dpi=150);plt.close(fig)


def write_report(output: Path, results: list[dict], comparison: dict) -> None:
    lines=["# Milestone 5D — fixed feature comparison", "", "## Protocol", "",
        "Research question: does appending three static obstacle distances improve mean food score? "
        "The hypothesis was specified before training; benefit was not assumed. Protocol 5d-v1 uses "
        "training seeds 11/22/33, final checkpoints only, 500 episodes or 100,000 moves per run, and "
        "fresh evaluation seeds 910000–910099 shared by every model. No tuning or model selection.", "",
        "A uses 11→128→128→3; B uses 14→128→128→3. The first 11 features are identical. "
        "B appends static free-cell distances straight/right/left divided by 19 on the 20×20 board. "
        "The tail counts as occupied for distance, even when immediate collision rules allow it to depart. "
        "Distances describe geometry, not future safety. Parameter counts are 18,435 and 18,819.", "",
        "All other settings match: Adam 0.0005, gamma 0.99, replay 50,000, batch 64, warmup 1,000, "
        "target copy every 1,000 updates, gradient norm cap 10, epsilon 1→0.05 over 50,000 moves. "
        "Rewards and episode limits are unchanged. See protocol.json and metadata.json for complete settings, software and source hashes.", "",
        "## Per-training-seed results", "",
        "| Model | Training seed | Train episodes | Train moves | Updates | Eval mean | Eval median | Eval score SD | Mean eval length | Collision / truncation / win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in results:
        lines.append(f"| {row['model']} | {row['training_seed']} | {row['completed_training_episodes']} | "
            f"{row['training_environment_steps']} | {row['optimiser_updates']} | {row['average_score']:.2f} | "
            f"{row['median_score']} | {row['score_sd_within_evaluation']:.3f} | {row['average_episode_length']:.2f} | "
            f"{row['collision_rate']:.0%} / {row['truncation_rate']:.0%} / {row['win_rate']:.0%} |")
    lines += ["", "## Across independent training seeds", "", "| Model | Mean of seed means | Sample SD across seed means |",
              "| --- | ---: | ---: |"]
    for model,item in comparison["variants"].items():
        lines.append(f"| {model} | {item['mean_of_training_seed_means']:.6f} | {item['sd_across_training_seed_means']:.6f} |")
    lines += ["", f"Paired B−A differences by training seed: {comparison['paired_differences_B_minus_A']}. "
        f"Mean paired difference: **{comparison['mean_paired_difference']:.6f}**. "
        f"B higher on all three seeds: **{comparison['B_higher_all_three_seeds']}**. "
        f"Direction consistently nonzero: **{comparison['direction_consistent_nonzero']}**.", "",
        "## Fairness and limitations", "",
        "The independent training runs, not the 300 evaluation episodes per variant, are the replications. "
        "Three seeds provide preliminary descriptive evidence; no significance tests or broad superiority claim "
        "are made. The earlier seed-7 model and its already-used evaluation range were not part of this experiment.", "",
        "For seed s, both variants use food seeds 200000+1000*s+episode index. Python/torch use s, replay s+1, "
        "exploration s+2. CPU uses one thread and deterministic algorithms. Same food seeds provide paired "
        "initial conditions, but different trajectories consume randomness differently. Input size changes "
        "initialisation draws, so equal seeds do not give identical hidden parameters. B necessarily adds 384 "
        "input-layer parameters. Equal episode caps allow unequal experience counts and epsilon progress; "
        "actual budgets are reported. Cross-platform/version determinism is not guaranteed.", "",
        "Technical smoke validation uses seed 77 and evaluation seeds 920000–920004 in temporary directories; "
        "those results are not scientific findings. Each completed final run is hashed and never retrained on "
        "rerun. An interrupted but complete training phase can be reused for evaluation; incomplete training "
        "is preserved for review rather than silently restarted or approximately resumed.", "",
        "## Files and reproduction", "",
        "`runs/<model>_seed<seed>/training/` holds configuration, episode/update CSVs, summary, and ignored checkpoints. "
        "Each run's `evaluation.csv` records all 100 outcomes including model, training/evaluation seed, score, "
        "reward, moves, collision/truncation/win flags. `evaluation.json` stores metrics and hashes; `complete.json` "
        "records completion and artifact integrity. Root per_seed_summary.csv and comparison_summary.json "
        "contain the reported results. The four PNGs show per-seed mean scores, rolling training performance, "
        "episode lengths, and training-seed variability.", "",
        "Run `python -m snake_ai.experiments.feature_comparison --smoke` after focused tests; then run the same "
        "module without --smoke. Use `--model A --seed 11` for one pair, preserving completed others. "
        "`--aggregate-only` reads completed runs without training. Existing root configuration/source hashes "
        "must match. Protocol changes require a new version/directory and documented reason before affected reruns.", "",
        "Next work should be a separately approved, pre-registered experiment informed by variability and failures. "
        "Do not infer that more features always help or tune further on these now-inspected evaluation seeds.", ""]
    with (output/"report.md").open("x",encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_food_report(output: Path, results: list[dict], comparison: dict) -> None:
    lines = ["# Milestone 5E: controlled food-distance experiment", "",
        "## Predeclared protocol", "",
        "Primary hypothesis: adding normalised food displacement improves mean greedy food score. "
        "Secondary hypotheses: improved median score/survival and lower between-training-seed score variability. "
        "These hypotheses were frozen before training, with no tuning or checkpoint selection.", "",
        "A: original11, 11 -> 128 -> 128 -> 3. B: food13, 13 -> 128 -> 128 -> 3. "
        "The first 11 features and the existing distance14 mode are unchanged. B appends "
        "dx=(food_x-head_x)/(width-1), dy=(food_y-head_y)/(height-1). Positive means right/down; "
        "float32 offsets lie in [-1,1]. Only current visible food is used; absent food yields zeros.", "",
        "Training seeds 44/55/66; exactly 40,000 interactions per fresh CPU run. The existing episode "
        "loop uses a 40,000-episode safety ceiling which cannot end a run before 40,000 moves. "
        "Incomplete last episodes are excluded from completed-episode statistics. Adam 0.0005, gamma .99, "
        "batch64, replay50000, warmup1000, target every1000 optimiser updates, clip10, epsilon1 to .05 "
        "over50000 moves; rewards and termination rules unchanged.", "",
        "All final checkpoints are evaluated greedily on 100 paired seeds 930000-930099. "
        "Before experiments, the user approved replacing the requested 920000 range because five of "
        "those seeds were already used by the 5D smoke test. Development smoke uses seed88 and "
        "940000-940004 in temporary directories, separate from final findings.", "",
        "## Actual run budgets and primary results", "",
        "| Model | Seed | Moves | Completed episodes | Updates | Seconds | Mean score |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['training_seed']} | {r['training_environment_steps']} | "
            f"{r['completed_training_episodes']} | {r['optimiser_updates']} | {r['training_duration_seconds']:.2f} | "
            f"{r['average_score']:.2f} |")
    lines += ["", "## Secondary outcomes", "",
        "| Model | Seed | Median | Min | Max | Mean moves | Score SD within eval | Collision | Truncation | Win |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['training_seed']} | {r['median_score']:.2f} | {r['minimum_score']} | "
            f"{r['maximum_score']} | {r['average_episode_length']:.2f} | {r['score_sd_within_evaluation']:.2f} | "
            f"{r['collision_rate']:.1%} | {r['truncation_rate']:.1%} | {r['win_rate']:.1%} |")
    lines += ["", "## Independent training-seed comparison", ""]
    for model,v in comparison['variants'].items():
        lines.append(f"- {model}: {v['mean_of_training_seed_means']:.2f} mean score; "
                     f"{v['sd_across_training_seed_means']:.2f} sample SD across three training-seed means.")
    differences = comparison['paired_differences_B_minus_A']
    higher = sum(value>0 for value in differences.values())
    lines += ["", f"Paired B minus A: {differences}. Mean paired difference: "
        f"{comparison['mean_paired_difference']:.2f}. B outperformed A on {higher}/3 training seeds.", "",
        "Evidence is descriptively consistent with the primary hypothesis in this protocol." if higher==3 else
        "The primary hypothesis was not consistently supported across all three training seeds.", "",
        "Secondary hypothesis summaries (means across the three training runs):", ""]
    for model in ('A','B'):
        group=[r for r in results if r['model']==model]
        lines.append(f"- {model}: mean of evaluation medians {statistics.mean(r['median_score'] for r in group):.2f}; "
                     f"mean episode length {statistics.mean(r['average_episode_length'] for r in group):.2f} moves.")
    lines += ["", "## Reproducibility and limits", "",
        "Python/torch seed=s, replay=s+1, exploration=s+2; food episode seed=300000+5000*s+episode_index. "
        "Each training seed has its own disjoint possible food-seed block; A/B share initial conditions "
        "but consume randomness differently along different trajectories. Deterministic CPU algorithms "
        "and one thread are used; exact cross-platform/version reproducibility is not guaranteed. "
        "Input size changes initialisation draws; B adds 256 parameters (18,691 vs A's 18,435). "
        "Matched moves, updates and epsilon progress do not imply identical wall-clock time or replay experience.", "",
        "Three independent training seeds provide exploratory evidence only. The 300 evaluation episodes "
        "per variant are not 300 independent training replications. No significance tests, tuning or best-seed "
        "selection are performed. Survival alone can reflect starvation/truncation rather than food efficiency. "
        "No universal superiority claim is justified. Future work requires a separately predeclared experiment "
        "with independent seeds; these inspected evaluation outcomes must not become tuning data.", "",
        "## Artifacts and reproduction", "",
        "Each runs/<model>_seed<seed>/ directory contains configuration, training CSVs/summary, ignored "
        "checkpoints, all 100 evaluation rows, and integrity hashes. Root protocol.json and metadata.json "
        "freeze methodology/software/source; per_seed_summary.csv and comparison_summary.json preserve aggregates. "
        "Earlier artifacts are preserved without regeneration.", "",
        "Run focused tests, then `python -m snake_ai.experiments.feature_comparison "
        "--protocol configs/food_distance_v1.json --smoke`; remove --smoke for all final runs. "
        "Use --model A --seed 44 for an individual run, or --aggregate-only to read completed runs. "
        "Successful runs are hashed and reused, never retrained; partial failures are preserved for review. "
        "Use the recorded source version/software for historical reproduction; source drift is rejected.", "",
        "![Average score](average_score.png)", "![Training curves](training_performance.png)",
        "![Episode length](episode_length.png)", ""]
    with (output/'report.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def write_body_report(output: Path, results: list[dict], comparison: dict, protocol: dict) -> None:
    seeds = protocol['training_seeds']
    lines = ["# Milestone 5F: controlled body-awareness experiment", "", "## Frozen hypothesis and protocol", "",
        "Primary hypothesis: appending three static body-distance features to food13 improves mean food score "
        "under matched training-step budgets. Secondary outcomes are median score, episode length, collision, "
        "truncation and between-training-seed variability. Improvement was not assumed.", "",
        f"A uses food13 (13 -> 128 -> 128 -> 3); B uses body16 (16 -> 128 -> 128 -> 3). "
        f"Both are fresh models trained with seeds {seeds}, exactly 40,000 moves, 39,001 updates and epsilon 0.24. "
        "All final checkpoints receive 100 greedy evaluations on seeds 950000-950099, without selection or tuning.", "",
        "The first 13 features are unchanged. Body rays are straight/right/left relative to movement. "
        "For a visible body hit, count free cells before that segment, excluding the head and including the tail, "
        "then divide by max(width,height)-1 (19 on the 20x20 board). An adjacent body gives 0. "
        "If no body is visible before the boundary, return 1: a fixed maximum-range sentinel, independent "
        "of distance to the wall. Walls are not body segments. Original danger features still detect walls. "
        "All values are float32 in [0,1]. The sentinel does not mean 19 free cells physically exist along a short ray. "
        "These features measure current layout, not moving-body trajectories or guaranteed future safety.", "",
        f"Seed audit/amendment before training: {protocol['seed_audit_note']}", "",
        "Training food starts are frozen in training_seed_starts; episode_index is added within each pair. "
        "Blocks 700000,750000,800000 each allow 40000 disjoint possible episode seeds. Smoke uses model seed 155, "
        "food seeds 850000 onward and evaluation 960000-960004 in temporary directories. Smoke outcomes are "
        "technical validation only. Final evaluation and smoke ranges were checked against earlier experiment records.", "",
        "CPU, one thread, deterministic algorithms where supported; Python/torch use model seed, replay seed+1, "
        "exploration seed+2. A/B share episode-food seeds but different trajectories consume randomness differently. "
        "Input-size differences also change initialisation draws. B has 384 extra input parameters (19,075 vs 18,691). "
        "Exact cross-version/platform determinism and equal wall-clock cost are not claimed.", "",
        "Adam 0.0005, gamma 0.99, replay 50000, batch 64, warmup 1000, target every 1000 updates, clip 10, "
        "epsilon 1 to 0.05 over 50000 moves. Rewards, game rules and episode limits are unchanged. "
        "The existing trainer's 40000-episode safety ceiling cannot stop before 40000 moves, because every "
        "episode requires at least one move. Final incomplete episodes are logged separately and excluded "
        "from completed-episode statistics.", "", "## Measured budgets and evaluation", "",
        "| Model | Seed | Moves | Completed episodes | Updates | Seconds | Mean score | Median | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['training_seed']} | {r['training_environment_steps']} | "
            f"{r['completed_training_episodes']} | {r['optimiser_updates']} | {r['training_duration_seconds']:.2f} | "
            f"{r['average_score']:.2f} | {r['median_score']:.2f} | {r['minimum_score']} | {r['maximum_score']} |")
    lines += ["", "| Model | Seed | Mean episode moves | Score SD within eval | Collision | Truncation | Win |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['training_seed']} | {r['average_episode_length']:.2f} | "
            f"{r['score_sd_within_evaluation']:.2f} | {r['collision_rate']:.1%} | {r['truncation_rate']:.1%} | {r['win_rate']:.1%} |")
    lines += ["", "## Training-seed-level comparison", ""]
    for model,v in comparison['variants'].items():
        lines.append(f"- {model}: {v['mean_of_training_seed_means']:.2f} mean score; "
                     f"sample SD across three training-seed means {v['sd_across_training_seed_means']:.2f}.")
    diffs = comparison['paired_differences_B_minus_A']
    higher = sum(x>0 for x in diffs.values())
    lines += ["", f"Paired B-minus-A differences: {diffs}. Mean paired difference: "
        f"{comparison['mean_paired_difference']:.2f}. B was higher on {higher}/3 training seeds.", "",
        "Three independent training seeds provide preliminary descriptive evidence only. The 100 evaluation "
        "episodes per model are not 100 independent training replications. No significance tests, hyperparameter "
        "search, best-checkpoint or best-seed selection were performed. Survival can be prolonged by "
        "truncation without efficient food collection. Static body proximity does not establish long-term "
        "planning capability; the tail moves and cells can become free. Further work needs a separately "
        "predeclared hypothesis and fresh seeds, rather than tuning on these now-inspected outcomes.", "",
        "## Artifacts and reproduction", "",
        "protocol.json and metadata.json freeze settings, software versions and source hashes. Each "
        "runs/<model>_seed<seed>/ directory contains training config/CSV/summary, ignored checkpoints, "
        "100 raw evaluation outcomes and hashed completion records. Root per_seed_summary.csv and "
        "comparison_summary.json contain aggregates. Previous artifacts are preserved without regeneration.", "",
        "After focused tests, run `python -m snake_ai.experiments.feature_comparison "
        "--protocol configs/body_awareness_v1.json --smoke`, then remove --smoke for final runs. "
        "Select --model/--seed for one missing run. Completed runs are verified and reused, never retrained. "
        "Incomplete failures are preserved for review; source/software changes are rejected. Use "
        "--aggregate-only to read completed runs without training. For historical reproduction, use "
        "the corresponding recorded source/software versions.", "",
        "![Mean score comparison](average_score.png)", "![Episode length](episode_length.png)", ""]
    with (output/'report.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def aggregate_experiment(protocol: dict, output: Path) -> dict | None:
    output,metadata=prepare_output(output,protocol)
    if not all((output/"runs"/f"{model}_seed{seed}"/"complete.json").exists() for model,seed in protocol["run_order"]):
        print("Comparison waits for all six completed runs.",flush=True)
        return None
    results=[run_one(protocol,output,model,seed) for model,seed in protocol["run_order"]]
    comparison=compare_results(results,protocol["training_seeds"])
    if (output/"comparison_summary.json").exists():
        if json.loads((output/"comparison_summary.json").read_text()) != comparison:
            raise ValueError("Stored aggregate differs")
        return comparison
    write_csv(output/"per_seed_summary.csv",[{key:value for key,value in row.items() if key!='feature_order'} for row in results])
    create_charts(output,results,protocol)
    if protocol["protocol_version"]=="5f-v1":
        write_body_report(output,results,comparison,protocol)
    elif protocol["protocol_version"]=="5e-v1":
        write_food_report(output,results,comparison)
    else:
        write_report(output,results,comparison)
    write_json(output/"comparison_summary.json",comparison)
    print(json.dumps(comparison,indent=2),flush=True)
    return comparison


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--model",choices=["A","B"])
    parser.add_argument("--seed",type=int)
    parser.add_argument("--aggregate-only",action="store_true")
    args=parser.parse_args()
    protocol=load_protocol(args.protocol)
    if args.output is None:
        args.output=({"5f-v1":Path("results/milestone5f_body_awareness"),
                      "5e-v1":Path("results/milestone5e_food_distance")}.get(protocol["protocol_version"],DEFAULT_OUTPUT))
    if args.smoke:
        if args.model is not None or args.seed is not None or args.aggregate_only:
            parser.error("--smoke is separate from final model/seed selection")
        run_smoke(protocol,args.output)
        return
    if args.aggregate_only:
        aggregate_experiment(protocol,args.output)
        return
    if (args.model is None)!=(args.seed is None):
        parser.error("Provide both --model and --seed, or neither")
    pairs=[(args.model,args.seed)] if args.model else protocol["run_order"]
    for model,seed in pairs:
        run_one(protocol,args.output,model,seed)
    aggregate_experiment(protocol,args.output)


if __name__=="__main__":
    main()
