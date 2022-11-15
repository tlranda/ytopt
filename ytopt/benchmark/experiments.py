import os, subprocess, argparse, configparser
from collections import OrderedDict
HERE = os.path.dirname(os.path.abspath(__file__))

valid_run_status = ["check", "check_resumable", "run", "override", "sanity", "announce"]
runnable = ["run", "override"]
checkable = ["check", "check_resumable", "sanity", "announce"]
always_announce = ["sanity", "announce"]

def sanity_check(checkname):
    pass

def run(cmd, prelude, args):
    if prelude != "":
        print(f"-- {prelude} --")
    print(cmd)
    if not args.eval_lock:
        status = subprocess.run(cmd, shell=True)

def output_check(checkname, prelude, expect, args, can_rm=True):
    if os.path.exists(checkname):
        try:
            with open(checkname, 'r') as f:
                linecount = len(f.readlines())
            if prelude != "":
                print(f"-- {prelude} --")
            print(f"| {linecount} lines in {checkname} |")
            if can_rm and linecount < expect:
                print(f"!! Remove bad output {checkname} (Expected {expect}) !!")
                subprocess.run(f"rm -f {checkname}", shell=True)
        except UnicodeDecodeError:
            print(f"[] Plot {checkname} exists []")
    else:
        if prelude != "":
            print(f"-! {prelude} !-")
        print(f"!! did not find {checkname} !!")

def verify_output(checkname, runstatus, invoke, expect, args, resumable=None, can_rm=True):
    if runstatus not in valid_run_status:
        raise ValueError(f"Runstatus must be in {valid_run_status}")
    if args.never_remove:
        can_rm = False
    r = 0
    b = 0
    if os.path.exists(checkname):
        if runstatus == "override":
            run(invoke, runstatus, args)
            r = 1
        elif runstatus != "run":
            b = 1
        output_check(checkname, runstatus.upper(), expect, args, can_rm)
        if runstatus in always_announce:
            print(invoke)
    else:
        found = False
        for backup in args.backup:
            if not backup.endswith('/'):
                backup += "/"
            if os.path.exists(backup+checkname):
                if runstatus == "override":
                    run(invoke, runstatus, args)
                    r = 1
                    output_check(checkname, f"{runstatus.upper()} OVERRIDE", expect, args, can_rm)
                else:
                    output_check(backup+checkname, f"{runstatus.upper()} BACKUP @{backup}", expect, args, can_rm)
                    b = 1
                    if runstatus in always_announce:
                        print(invoke)
                found = True
                break
        if not found:
            if runstatus == "check":
                warn = "!! No file"
                if args.backup == []:
                    warn += ", no backup given,"
                else:
                    warn += f" or backup @{args.backup}"
                print(warn+f" for {checkname} !!")
                print(invoke)
                b = 1
            elif runstatus != "check_resumable":
                bonus = f"; No backup @{args.backup}" if args.backup is not None else "; No backup given"
                if runstatus in always_announce:
                    print(invoke)
                if runstatus in runnable:
                    run(invoke, runstatus+bonus, args)
                    r = 1
                output_check(checkname, "CHECK NEW RUN", expect, args, can_rm)
    if runstatus == "sanity" or r == 1:
        sanity_check(checkname)
    # Short recursion on resumable
    if resumable is not None:
        verify_output(resumable, "check_resumable", invoke, expect, args, can_rm=False)
    return r, b

def build_test_suite(experiment, runtype, args, key, problem_sizes=None):
    # Get in the experiment directory
    os.chdir(f"{HERE}/{experiment}_exp")
    print(f"<< BEGIN {key} for {experiment}  >>")
    sect = args.cfg[key]
    try:
        expect = sect['expect']
    except KeyError:
        expect = 0
    # Determine parallelism
    parallel = args.parallel_id is not None and args.n_parallel is not None
    # Fetch the problem sizes
    if problem_sizes is None:
        problem_sizes = subprocess.run("python -m ytopt.benchmark.size_lookup --p "+" ".join([f"problem.{s}" for s in sect['sizes']]), shell=True, stdout=subprocess.PIPE)
        problem_sizes = dict((k, int(v)) for (k,v) in zip(sect['sizes'], problem_sizes.stdout.decode('utf-8').split()))
    # GIANT SWITCH on experiment types
    calls = 0
    bluffs = 0
    verifications = 0
    # DATA COLLECTION TYPES
    if key == 'OFFLINE':
        nontargets = [_ for _ in sect['sizes'] if _ not in sect['targets']]
        for loopct, problem in enumerate(nontargets):
            if parallel and loopct % args.n_parallel != args.parallel_id:
                continue
            out_name = f"results_rf_{problem.lower()}_{experiment}.csv"
            resume = f"results_{problem_sizes[problem]}.csv"
            invoke = f"python -m ytopt.search.ambs --problem problem.{problem} --evaluator {sect['evaluator']} "+\
                     f"--max-evals={sect['evals']} --learner {sect['learner']} --set-KAPPA {sect['kappa']} "+\
                     f"--acq-func {sect['acqfn']} --set-SEED {sect['offline_seed']} --resume {resume}; "+\
                     f"mv {resume} {out_name}"
            info = verify_output(out_name, runtype, invoke, expect, args, resumable=resume)
            calls += info[0]
            bluffs += info[1]
            verifications += 1
    elif key == 'O3':
        for loopct, target in enumerate(sect['targets']):
            if parallel and loopct % args.n_parallel != args.parallel_id:
                continue
            invoke = f"python -c \"import pandas as pd; from problem import {target}; "+\
                     f"obj = {target}.O3(); "+\
                     "pd.DataFrame({'objective': [obj], 'elapsed_time': [obj]})"+\
                     f".to_csv('data/DEFAULT_{target.upper()}.csv', index=False)\""
            info = verify_output(f"data/DEFAULT_{target.upper()}.csv", runtype, invoke, expect, args)
            calls += info[0]
            bluffs += info[1]
            verifications += 1
    # DERIVATIVE DATA COLLECTION
    elif key == 'ONLINE':
        problem_prefix = sect['problem_prefix']
        for target in sect['targets']:
            for model in sect['models']:
                for loopct, seed in enumerate(sect['seeds']):
                    if parallel and loopct % args.n_parallel != args.parallel_id:
                        continue
                    # No Refit
                    resume = f"{experiment}_NO_REFIT_{model}_{target}_{seed}_ALL.csv"
                    invoke = "python -m ytopt.benchmark.base_online_tl --n-refit 0 "+\
                             f"--max-evals {sect['evals']} --seed {seed} --top {sect['top']} "+\
                             f"--inputs {' '.join([problem_prefix+'.'+i for i in sect['inputs']])} "+\
                             f"--targets {problem_prefix}.{target} --model {model} --no-log-obj "+\
                             f"--output-prefix {experiment}_NO_REFIT_{model}_{target}_{seed} "+\
                             f"--resume {resume}"
                    info = verify_output(resume, runtype, invoke, expect, args)
                    calls += info[0]
                    bluffs += info[1]
                    verifications += 1
    elif key == 'LEARN':
        problem_prefix = sect['problem_prefix']
        for target in sect['targets']:
            for model in sect['models']:
                for loopct, inputs in enumerate(sect['inputs']):
                    if parallel and loopct % args.n_parallel != args.parallel_id:
                        continue
                    seed = sect['seeds'][0]
                    #inputs = sect['inputs']
                    inp_str = '#'.join(inputs)
                    refit = sect['refits']
                    top = sect['top']
                    refit_str = "NO_REFIT" if refit == 0 else f"REFIT_{refit}"
                    resume = f"{experiment}_{refit_str}_TOP_{int(10*top)}_INP_{inp_str}_TARG_{target}_SEED_{seed}_ALL.csv"
                    invoke = f"python -m ytopt.benchmark.base_online_tl --n-refit {refit} "+\
                             f"--max-evals {sect['evals']} --seed {seed} --top {top} "+\
                             f"--inputs {' '.join([problem_prefix+'.'+i for i in inputs])} "+\
                             f"--targets {problem_prefix}.{target} --model {model} --no-log-obj "+\
                             f"--output-prefix {resume[:-8]} --resume {resume}"
                    info = verify_output(resume, runtype, invoke, expect, args)
                    calls += info[0]
                    bluffs += info[1]
                    verifications += 1
    elif key == 'REFIT':
        problem_prefix = sect['problem_prefix']
        for target in sect['targets']:
            for model in sect['models']:
                for loopct, seed in enumerate(sect['seeds']):
                    if parallel and loopct % args.n_parallel != args.parallel_id:
                        continue
                    # Refit
                    resume = f"{experiment}_REFIT_{sect['refits']}_{model}_{target}_{seed}_ALL.csv"
                    invoke = f"python -m ytopt.benchmark.base_online_tl --n-refit {sect['refits']} "+\
                             f"--max-evals {sect['evals']} --seed {seed} --top {sect['top']} "+\
                             f"--inputs {' '.join([problem_prefix+'.'+i for i in sect['inputs']])} "+\
                             f"--targets {problem_prefix}.{target} --model {model} --no-log-obj "+\
                             f"--output-prefix {experiment}_REFIT_{sect['refits']}_{model}_{target}_{seed} "+\
                             f"--resume {resume} --resume-fit -1"
                    info = verify_output(resume, runtype, invoke, expect, args)
                    calls += info[0]
                    bluffs += info[1]
                    verifications += 1
    elif key == 'BOOTSTRAP':
        problem_prefix = sect['problem_prefix']
        for target in sect['targets']:
            for model in sect['models']:
                for loopct, seed in enumerate(sect['seeds']):
                    if parallel and loopct % args.n_parallel != args.parallel_id:
                        continue
                    # Bootstrap
                    resume = f"results_{problem_sizes[target]}.csv"
                    outfile = f"{experiment}_BOOTSTRAP_{sect['bootstrap']}_{model}_{target}_{seed}_ALL.csv"
                    invoke = f"python -m ytopt.benchmark.search.ambs --problem {problem_prefix}.{target} --max-evals "+\
                             f"{sect['evals']} --n-generate {sect['bootstrap']} --top {sect['top']} "+\
                             f"--inputs {' '.join([problem_prefix+'.'+i for i in sect['inputs']])} "+\
                             f"--model {model} --evaluator {sect['evaluator']} --learner {sect['learner']} "+\
                             f"--set-KAPPA {sect['kappa']} --acq-func {sect['acqfn']} "+\
                             f"--set-SEED {seed} --set-NI {sect['ni']} --resume {resume}; "+\
                             f"mv {resume} {outfile}"
                    info = verify_output(outfile, runtype, invoke, expect, args, resumable=resume)
                    calls += info[0]
                    bluffs += info[1]
                    verifications += 1
    # DERIVATIVE^2 DATA COLLECTION (Analysis Data)
    elif key == "XFER":
        invoke = "python -m ytopt.benchmark.force_transfer --inputs "+\
                 f"{' '.join([problem_prefix+'.'+_ for _ in sect['sizes']])} "+\
                 "--targets "+\
                 f"{' '.join([problem_prefix+'.'+_ for _ in sect['inputs']])} "
        if sect['backup'] is not None and len(sect['backup']) > 0:
            invoke += f" --backups {' '.join(sect['backup'])}"
        info = verify_output(f"xfer_results_{experiment.lstrip('_')}.csv", runtype, invoke, expect, args)
        calls += info[0]
        bluffs += info[1]
        verifications += 1
    elif key == "INFERENCE":
        problem_prefix = sect['problem_prefix']
        for target in sect['targets']:
            for model in sect['models']:
                for seed in sect['seeds']:
                    invoke = f"python -m ytopt.benchmark.inference_test --n-refit {sect['refits']} --max-evals "+\
                             f"{sect['evals']} --seed {seed} --top {sect['top']} --inputs "+\
                             f"{' '.join([problem_prefix+'.'+_ for _ in sect['inputs']])} "+\
                             f"--target {problem_prefix}.{target} --model {model} --no-log-obj "
                    info = verify_output(f"inference_{experiment.lstrip('_')}.csv", runtype, invoke, expect, args)
                    calls += info[0]
                    bluffs += info[1]
                    verifications += 1
    elif key == 'REJECTION':
        problem_prefix = sect['problem_prefix']
        for target in sect['targets']:
            target = problem_prefix+'.'+target
            for model in sect['models']:
                for loopct, seed in enumerate(sect['seeds']):
                    if parallel and loopct % args.n_parallel != args.parallel_id:
                        continue
                    outfile = f"REJECT_{model}_{target}_{seed}"
                    invoke = f"python -m ytopt.benchmark.base_online_tl --max-evals {sect['evals']} --n-refit 0 "+\
                             f"--seed {seed} --top {sect['top']} --targets {target} --model {model} --inputs "+\
                             f"{' '.join([problem_prefix+'.'+i for i in sect['inputs']])} --skip-evals "+\
                             f"--output-prefix {outfile}"
                    info = verify_output(outfile+'_ALL.csv', runtype, invoke, expect, args)
                    calls += info[0]
                    bluffs += info[1]
                    verifications += 1
    # PLOT TYPES
    elif key == 'COMPETITIVE':
        experiment_dir = args.backup if sect['use_backup'] and args.backup is not None else './'
        if type(experiment_dir) is list:
            if len(experiment_dir) == 1:
                experiment_dir = experiment_dir[0]
            else:
                raise ValueError(f"{key} section parsing does not support multiple backups")
        for target in sect['targets']:
            # SM, ML, and XL
            # data/thomas_experiments/*_{target.upper()}_*.csv
            invoke = f"python -m ytopt.benchmark.plot_analysis --output {experiment}_{target.lower()}_configs "+\
                     f"--input *_{target.upper()}_*.csv data/thomas_experiments/*_{target.upper()}_*.csv "+\
                     f"data/jaehoon_experiments/results_rf_{target.lower()}_*.csv data/gptune_experiments/"+\
                     f"results_gptune_*{target.lower()}* --x-axis walltime --unname results --trim .csv "+\
                     "--legend best --synchronous --drop-extension --fig-dims 8 4 --top 0.9 "+\
                     "--ignore data/jaehoon_experiments/*200eval* data/thomas_experiments/*1337*.csv "+\
                     "data/thomas_experiments/*5555*.csv --no-text --global-top"
            if sect['as_speedup']:
                invoke += f" --as-speedup-vs data/DEFAULT_{target.upper()}.csv --max-objective "
            if sect['show']:
                invoke += " --show"
            info = verify_output(f"{experiment}_{target.lower()}_configs_competitive.png", runtype,
                                    invoke, expect, args)
            calls += info[0]
            bluffs += info[1]
            verifications += 1
    elif key in ['WALLTIME', 'EVALUATION']:
        experiment_dir = args.backup if sect['use_backup'] and args.backup is not None else './'
        if type(experiment_dir) is list:
            if len(experiment_dir) == 1:
                experiment_dir = experiment_dir[0]
            else:
                raise ValueError(f"{key} section parsing does not support multiple backups")
        for target in sect['targets']:
            # Raw performance with evaluation or wall-time x-axes
            for axis in ["walltime", "evaluation"]:
                if key.lower() != axis:
                    continue
                invoke = f"python -m ytopt.benchmark.plot_analysis --output {experiment}_{target.lower()}_{axis} "+\
                         f"--best {experiment_dir}*_{target.upper()}_*.csv "+\
                         f"data/jaehoon_experiments/results_rf_{target.lower()}_*.csv data/gptune_experiments/"+\
                         f"results_gptune_*{target.lower()}* data/thomas_experiments/*_{target.upper()}_*.csv "#+\
                         #f"--baseline data/results_rf_{target.lower()}_{experiment.lstrip('_')}.csv "
                if sect['as_speedup']:
                    invoke += f"--as-speedup-vs data/DEFAULT_{target.upper()}.csv --max-objective "
                else:
                    invoke += "data/DEFAULT.csv --log-y "
                invoke += f"--x-axis {axis} --log-x --unname {experiment_dir}_ "+\
                         f"--trim data/results_{problem_sizes[target]}.csv --legend best --synchronous "+\
                         "--ignore data/jaehoon_experiments/*200eval* data/thomas_experiments/*1337*.csv "+\
                         "data/thomas_experiments/*5555*.csv --no-text --drop-overhead --clean-names"
                if sect['show']:
                    invoke += " --show"
                info = verify_output(f"{experiment}_{target.lower()}_{axis}_plot.png", runtype, invoke, expect, args)
                calls += info[0]
                bluffs += info[1]
                verifications += 1
    elif key == "PCA":
        experiment_dir = args.backup if sect['use_backup'] and args.backup is not None else './'
        if type(experiment_dir) is list:
            if len(experiment_dir) == 1:
                experiment_dir = experiment_dir[0]
            else:
                raise ValueError(f"{key} section parsing does not support multiple backups")
        # PCA plots
        # LINE 2 WAS TWICE? #data/*{sect['pca']}*.csv "+\
        invoke = f"python -m ytopt.benchmark.plot_analysis --output {experiment} "+\
                 f"--pca data/*{sect['pca']}*.csv "+\
                 f"--pca-algorithm {sect['pca_algorithm']} --pca-points {sect['pca_points']} "+\
                 f"--pca-problem problem.Problem --legend best --no-text"
        if sect['show']:
            invoke += " --show"
        info = verify_output(f"{experiment}_{sect['pca_algorithm']}.png", runtype, invoke, expect, args)
        calls += info[0]
        bluffs += info[1]
        verifications += 1
    elif key == "TSNE":
        invoke = "python -m ytopt.benchmark.tsne_figure --problem problem.S --convert "+\
                 f"{' '.join(sect['convert'])} --quantile {' '.join([str(_) for _ in sect['quantile']])} --output "+\
                 f"{experiment}_TSNE.png"
        if sect['rank']:
            invoke += " --rank-color"
        info = verify_output(f"{experiment}_TSNE.png", runtype, invoke, expect, args)
        calls += info[0]
        bluffs += info[1]
        verifications += 1
    # ANALYSIS CALLS
    # TBD
    else:
        raise ValueError(f"Unknown section {key}")
    print(f"<< CONCLUDE {key} for {experiment}. {calls} calls made & {bluffs} calls bluffed. {verifications} attempted verifies >>")
    return problem_sizes

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--eval-lock', action='store_true', help="Prevent statements from being executed")
    prs.add_argument('--config-file', type=str, required=True, help="File to read configuration from")
    prs.add_argument('--experiments', type=str, nargs='+', required=True, help="Experiments to run")
    prs.add_argument('--runstatus', type=str, nargs='*', default=[], help="Way to run experiments")
    prs.add_argument('--backup', type=str, nargs='*', help="Directory path(s) to look for backup files")
    prs.add_argument('--skip', type=str, nargs='*', default=[], help="Config sections to skip")
    prs.add_argument('--only', type=str, nargs='*', default=[], help="ONLY run these config sections")
    prs.add_argument('--section-sizecache', action='store_true', help="Cache problem size values between sections")
    prs.add_argument('--experiment-sizecache', action='store_true', help="Cache problem size values between experiments")
    prs.add_argument('--parallel-id', type=int, help="Parallel identifier for this process (should be 0-max inclusive)")
    prs.add_argument('--n-parallel', type=int, help="Total number of parallel tasks available for coordination")
    prs.add_argument('--never-remove', action='store_true', help="Never delete partial results")
    return prs

def config_bind(cfg):
    # Basically evaluate the config file into nested dictionaries
    cp = configparser.ConfigParser()
    cp.read(cfg)
    cfg_dict = dict()
    for s in cp:
        if s == 'DEFAULT':
            continue
        cfg_dict[s] = dict()
        for p in cp[s]:
            try:
                cfg_dict[s][p] = eval(cp[s][p])
            except SyntaxError:
                if len(cp[s][p]) > 0:
                    print(f"Warning! {cfg} [{s}][{p}] may have incorrect python syntax")
                cfg_dict[s][p] = ""
    # Apply priority to all elements (zero-priority == always demote to last)
    last_priority = 0
    update_keys = []
    for s in cfg_dict.keys():
        if 'priority' not in cfg_dict[s].keys():
            continue
        elif cfg_dict[s]['priority'] == 0:
            update_keys.append(s)
        else:
            last_priority = max(last_priority, cfg_dict[s]['priority'])
    last_priority += 1 # Guarantee it runs later
    for s in update_keys:
        cfg_dict[s]['priority'] = last_priority
    # Use OrderedDict for proper ordering
    def sortfn(kk):
        if 'priority' in cfg_dict[kk].keys():
            return cfg_dict[kk]['priority']
        else:
            return last_priority+1
    cfg_dict = OrderedDict([(k, cfg_dict[k]) for k in sorted(cfg_dict.keys(), key=sortfn)])
    # Recursive loading
    for k,v in cfg_dict.items():
        if 'inherit' in v.keys():
            for default in v['inherit']:
                for kk, vv in cfg_dict[default].items():
                    if kk == 'inherit':
                        continue
                    cfg_dict[k].setdefault(kk, vv)
    # Drop inheritable keys
    drop = [k for k in cfg_dict.keys() if k.startswith('INHERIT')]
    for k in drop:
        del cfg_dict[k]
    # Re-prioritize if needed
    return OrderedDict([(k, cfg_dict[k]) for k in sorted(cfg_dict.keys(), key=sortfn)])

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # Warning on improper parallel call
    from operator import xor
    if xor(args.parallel_id is not None, args.n_parallel is not None):
        raise ValueError("BOTH --parallel-id and --n-parallel must be supplied to enable parallelization")
    del xor
    # Load data from config and bind to args
    args.cfg = config_bind(args.config_file)
    # Fix endings of backup directories
    backup = []
    if args.backup is not None:
        for b in args.backup:
            if not b.endswith('/'):
                b += "/"
            backup.append(b)
    args.backup = backup
    if args.runstatus == []:
        args.runstatus = ["run"]
    # Repeat last known experiment runstatus to fill in blanks
    while len(args.runstatus) < len(args.experiments):
        args.runstatus.append(args.runstatus[-1])
    return args

if __name__ == '__main__':
    args = parse(build())
    print(args)
    problem_sizes = None
    #sort_dict = dict((k,v['priority']) for (k,v) in args.cfg.items())
    # For each experiment, run all run-type things as a test suite
    for experiment, runtype in zip(args.experiments, args.runstatus):
        for section in args.cfg.keys():
        #for section in sorted(args.cfg.keys(), key=lambda key: sort_dict[key]):
            if section in args.skip:
                continue
            if len(args.only) > 0 and section not in args.only:
                continue
            # Allow runstatus to define remove behavior
            revert_remove = False
            if runtype in checkable:
                revert_remove = True
                old_remove = args.never_remove
                args.never_remove = True
            # Allow config to define backup
            revert_backup = False
            if 'backup' in args.cfg[section].keys():
                revert_backup = True
                old_backup = args.backup
                args.backup = args.cfg[section]['backup']
            # Execute and cache results if indicated
            problem_sizes = build_test_suite(experiment, runtype, args, section, problem_sizes)
            if not args.section_sizecache:
                problem_sizes = None
            # Restorations
            if revert_backup:
                args.backup = old_backup
            if revert_remove:
                args.never_remove = old_remove
        if not args.experiment_sizecache:
            problem_sizes = None

