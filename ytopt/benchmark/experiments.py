import os, subprocess, argparse, configparser
HERE = os.path.dirname(os.path.abspath(__file__))

valid_run_status = ["check", "run", "override", "sanity", "announce"]
always_announce = ["sanity", "announce"]

def sanity_check(checkname):
    pass

def run(cmd, prelude, args):
    if prelude != "":
        print(f"-- {prelude} --")
    print(cmd)
    if not args.eval_lock:
        status = subprocess.run(cmd, shell=True)

def output_check(checkname, prelude, expect, args):
    if os.path.exists(checkname):
        try:
            with open(checkname, 'r') as f:
                linecount = len(f.readlines())
            if prelude != "":
                print(f"-- {prelude} --")
            print(f"| {linecount} lines in {checkname} |")
            if linecount < expect:
                print(f"!! Remove bad output {checkname} (Expected {expect}) !!")
                subprocess.run(f"rm -f {checkname}", shell=True)
        except UnicodeDecodeError:
            print(f"[] Plot {checkname} exists []")
    else:
        if prelude != "":
            print(f"-! {prelude} !-")
        print(f"!! did not find {checkname} !!")

def verify_output(checkname, runstatus, invoke, expect, args):
    if runstatus not in valid_run_status:
        raise ValueError(f"Runstatus must be in {valid_run_status}")
    r = 0
    b = 0
    if os.path.exists(checkname):
        if runstatus == "override":
            run(invoke, runstatus, args)
            r = 1
        elif runstatus != "run":
            b = 1
        output_check(checkname, "CHECK", expect, args)
        if runstatus in always_announce:
            print(invoke)
    elif args.backup is not None and os.path.exists(args.backup+checkname):
        if runstatus == "override":
            run(invoke, runstatus, args)
            r = 1
            output_check(checkname, "CHECK OVERRIDE", expect, args)
        else:
            output_check(args.backup+checkname, f"CHECK BACKUP @{args.backup}", expect, args)
            b = 1
            if runstatus in always_announce:
                print(invoke)
    else:
        if runstatus == "check":
            warn = "!! No file"
            if args.backup is None:
                warn += ", no backup given,"
            else:
                warn += f" or backup @{args.backup}"
            print(warn+f" for {checkname} !!")
            print(invoke)
            b = 1
        else:
            bonus = f"; No backup @{args.backup}" if args.backup is not None else "; No backup given"
            run(invoke, runstatus+bonus, args)
            r = 1
            output_check(checkname, "CHECK NEW RUN", expect, args)
    if runstatus == "sanity" or r == 1:
        sanity_check(checkname)
    return r, b

def build_test_suite(experiment, runtype, args, key):
    # Get in the experiment directory
    os.chdir(f"{HERE}/{experiment}_exp")
    print(f"<< BEGIN {key} for {experiment}  >>")
    sect = args.cfg[key]
    expect = sect['expect']
    # Fetch the problem sizes
    problem_sizes = subprocess.run("python -m ytopt.benchmark.size_lookup --p "+" ".join([f"problem.{s}" for s in sect['sizes']]), shell=True, stdout=subprocess.PIPE)
    problem_sizes = dict((k, int(v)) for (k,v) in zip(sect['sizes'], problem_sizes.stdout.decode('utf-8').split()))
    calls = 0
    bluffs = 0
    if key == 'OFFLINE':
        for problem in sect['sizes']:
            out_name = f"results_rf_{problem.lower()}_{experiment}.csv"
            invoke = f"python -m ytopt.search.ambs --problem problem.{problem} --evaluator {sect['evaluator']} "+\
                     f"--max-evals={sect['evals']} --learner {sect['learner']} --set-KAPPA {sect['kappa']} "+\
                     f"--acq-func {sect['acqfn']} --set-SEED {sect['offline_seed']}; "+\
                     f"mv results_{problem_sizes[problem]}.csv {out_name}"
            info = verify_output(out_name, runtype, invoke, expect, args)
            calls += info[0]
            bluffs += info[1]
    elif key == 'ONLINE':
        problem_prefix = sect['problem_prefix']
        for target in sect['targets']:
            for model in sect['models']:
                for seed in sect['seeds']:
                    # No Refit
                    invoke = "python -m ytopt.benchmark.base_online_tl --n-refit 0 "+\
                             f"--max-evals {sect['evals']} --seed {seed} --top {sect['top']} "+\
                             f"--inputs {' '.join([problem_prefix+'.'+i for i in sect['inputs']])} "+\
                             f"--targets {problem_prefix}.{target} --model {model} --unique --no-log-obj "+\
                             f"--output-prefix {experiment}_NO_REFIT_{model}_{target}_{seed}"
                    info = verify_output(f"{experiment}_NO_REFIT_{model}_{target}_{seed}_ALL.csv", runtype,
                                  invoke, expect, args)
                    calls += info[0]
                    bluffs += info[1]
                    # Refit
                    invoke = f"python -m ytopt.benchmark.base_online_tl --n-refit {sect['refits']} "+\
                             f"--max-evals {sect['evals']} --seed {seed} --top {sect['top']} "+\
                             f"--inputs {' '.join([problem_prefix+'.'+i for i in sect['inputs']])} "+\
                             f"--targets {problem_prefix}.{target} --model {model} --unique --no-log-obj "+\
                             f"--output-prefix {experiment}_REFIT_{sect['refits']}_{model}_{target}_{seed}"
                    info = verify_output(f"{experiment}_REFIT_{sect['refits']}_{model}_{target}_{seed}_ALL.csv",
                                  runtype, invoke, expect, args)
                    calls += info[0]
                    bluffs += info[1]
                    # Bootstrap
                    invoke = f"python -m ytopt.benchmark.search.ambs --problem {problem_prefix}.{target} --max-evals "+\
                             f"{sect['evals']} --n-generate {sect['bootstrap']} --top {sect['bootstrap_top'] } "+\
                             f"--inputs {' '.join([problem_prefix+'.'+i for i in sect['inputs']])} "+\
                             f"--model {model} --evaluator {sect['evaluator']} --learner {sect['learner']} "+\
                             f"--set-KAPPA {sect['kappa']} --acq-func {sect['acqfn']} "+\
                             f"--set-SEED {seed} --set-NI {sect['ni']}; "+\
                             f"mv results_{problem_sizes[target]}.csv {experiment}_BOOTSTRAP_"+\
                             f"{sect['bootstrap']}_{model}_{target}_{seed}_ALL.csv"
                    info = verify_output(f"{experiment}_BOOTSTRAP_{sect['bootstrap']}_{model}_{target}_{seed}_ALL.csv",
                                  runtype, invoke, expect, args)
                    calls += info[0]
                    bluffs += info[1]
    elif key == 'PLOTS':
        experiment_dir = args.backup if sect['use_backup'] and args.backup is not None else './'
        if len(experiment_dir) > 0 and not experiment_dir.endswith('/'):
            experiment_dir += '/'
        for target in sect['targets']:
            # Raw performance with evaluation or wall-time x-axes
            for axis in ["walltime", "evaluation"]:
                invoke = f"python -m ytopt.benchmark.plot_analysis --output {experiment}_{target.lower()}_{axis} "+\
                         f"--best {experiment_dir}*_{target.upper()}_*.csv "+\
                         f"data/jaehoon_experiments/results_rf_{target.lower()}_*.csv data/gptune_experiments/"+\
                         f"results_gptune_*{target.lower()}* "+\
                         f"--baseline data/results_rf_{target.lower()}_{experiment.lstrip('_')}.csv "
                if sect['as_speedup']:
                    invoke += f"--as-speedup-vs data/DEFAULT_{target.upper()}.csv --max-objective "
                else:
                    invoke += "data/DEFAULT.csv --log-y "
                invoke += f"--x-axis {axis} --log-x --unname {experiment_dir}_ "+\
                         f"--trim data/results_{problem_sizes[target]}.csv --legend best --synchronous "+\
                         "--ignore data/jaehoon_experiments/*200eval* data/thomas_experiments/*1337*.csv "+\
                         "data/thomas_experiments/*5555*.csv --no-text --drop-overhead"
                if sect['show']:
                    invoke += " --show"
                info = verify_output(f"{experiment}_{target.lower()}_{axis}_plot.png", runtype, invoke, expect, args)
                calls += info[0]
                bluffs += info[1]
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
    elif key == 'O3':
        for target in sect['targets']:
            invoke = f"python -c \"import pandas as pd; from problem import {target}; "+\
                     f"obj = {target}.O3(); "+\
                     "pd.DataFrame({'objective': [obj], 'elapsed_time': [obj]})"+\
                     f".to_csv('data/DEFAULT_{target.upper()}.csv', index=False)\""
            info = verify_output(f"data/DEFAULT_{target.upper()}.csv", runtype, invoke, expect, args)
            calls += info[0]
            bluffs += info[1]
    else:
        raise ValueError(f"Unknown section {key}")
    print(f"<< CONCLUDE {key} for {experiment}. {calls} calls made & {bluffs} calls bluffed >>")

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--eval-lock', action='store_true', help="Prevent statements from being executed")
    prs.add_argument('--config-file', type=str, required=True, help="File to read configuration from")
    prs.add_argument('--experiments', type=str, nargs='+', required=True, help="Experiments to run")
    prs.add_argument('--runstatus', type=str, nargs='*', default=[], help="Way to run experiments")
    prs.add_argument('--backup', type=str, default=None, help="Directory path to look for backup files")
    prs.add_argument('--skip', type=str, nargs='*', default=[], help="Config sections to skip")
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
    return cfg_dict

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # Load data from config and bind to args
    args.cfg = config_bind(args.config_file)
    if args.runstatus == []:
        args.runstatus = ["run"]
    # Repeat last known experiment runstatus to fill in blanks
    while len(args.runstatus) < len(args.experiments):
        args.runstatus.append(args.runstatus[-1])
    return args

if __name__ == '__main__':
    args = parse(build())
    print(args)
    # For each experiment, run all run-type things as a test suite
    for experiment, runtype in zip(args.experiments, args.runstatus):
        for section in sorted(args.cfg.keys()):
            if section in args.skip:
                continue
            # Allow config to define backup
            revert_backup = False
            if args.backup is None and 'backup' in args.cfg[section].keys():
                revert_backup = True
                args.backup = args.cfg[section]['backup']
            if args.backup is not None and not args.backup.endswith('/'):
                args.backup += "/"
            build_test_suite(experiment, runtype, args, section)
            if revert_backup:
                args.backup = None

