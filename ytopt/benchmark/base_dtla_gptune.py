from autotune.space import Space, Integer, Real
from autotune.problem import TuningProblem
from GPTune.gptune import GPTune, BuildSurrogateModel_tl
from GPTune.computer import Computer
from GPTune.data import Categoricalnorm, Data
from GPTune.database import HistoryDB, GetMachineConfiguration
from GPTune.options import Options

import openturns as ot
import argparse, sys, os
import pandas as pd, json, datetime, uuid
from ytopt.benchmark.base_problem import ecp_problem_builder, polybench_problem_builder
from pprint import pprint

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('-benchmark', type=str, required=True, help='Benchmark name')
    parser.add_argument('-inputs', type=str, nargs='+', required=True, help='Problem sizes as predefined knowledge')
    parser.add_argument('-target', type=str, required=True, help='Target task to train on')
    parser.add_argument('-nrun', type=int, default=2, help='Number of runs per task')
    parser.add_argument('-seed', type=int, default=1234, help='Set seed')
    parser.add_argument('-builder', choices=['polybench', 'ecp'], default='polybench', help='Problem builder')
    return parser

def parse(parser, args=None):
    if args is None:
        args = parser.parse_args()
    # Rebind to correct factory object
    if args.builder == 'polybench':
        args.builder = polybench_problem_builder
    elif args.builder == 'ecp':
        args.builder = ecp_problem_builder
    else:
        raise ValueError(f"Unsupported problem builder {args.builder}")
    return args

def localized_load(benchmark):
    real_names = [_ for _ in os.listdir() if os.path.isdir(_) and _.endswith('_exp')]
    look_names = [_.lstrip('_')[:-4] for _ in real_names]
    if benchmark not in ['.', '..']:
        try:
            bench_dir = real_names[look_names.index(benchmark)]
        except ValueError:
            raise ValueError(f"Could not locate {benchmark} -- available: {look_names}")
        HERE = os.path.dirname(os.path.abspath(__file__))+'/'+bench_dir
    else:
        HERE = os.path.dirname(os.path.abspath(__file__))+'/'+benchmark
    sys.path.insert(0, HERE)
    from problem import input_space, lookup_ival
    kwargs = {}
    if benchmark == 'sw4lite':
        from problem import SW4Lite_Plopper
        kwargs.update({'plopper_class': SW4Lite_Plopper,
                       'sourcefile': HERE+'/mmp_new.C',})
    elif benchmark == 'amg':
        from problem import AMG_Plopper
        kwargs.update({'plopper_class': AMG_Plopper,})
    elif benchmark == 'xsbench':
        from problem import XSBench_Plopper
        kwargs.update({'plopper_class': XSBench_Plopper,})
    elif benchmark == 'rsbench':
        from problem import RSBench_Plopper
        kwargs.update({'plopper_class': RSBench_Plopper,})
    elif benchmark == 'floyd_warshall':
        from problem import Floyd_Warshall_Plopper
        kwargs.update({'plopper_class': Floyd_Warshall_Plopper,})
    os.chdir(HERE)
    print(f"Benchmark {benchmark} loaded")
    return HERE, input_space, lookup_ival, kwargs

def infer_size(fname, lookup_ival):
    og_fname = fname
    fname = os.path.basename(fname) # Drop directories
    fname = fname.rsplit('.',1)[0] # Drop file extension
    if fname.startswith('results'):
        fname = fname[8:]
    if fname.startswith('rf'):
        fname = fname[3:]
    if len(fname.split('_')) == 2:
        fname = fname.split('_')[0].upper()
    else:
        raise ValueError(f"No support for this input file name {og_fname} (parse: {fname})")
    inv_lookup = dict((v[0], k) for (k,v) in lookup_ival.items())
    return inv_lookup[fname]

def csvs_to_gptune(fnames, tuning_metadata, lookup_ival):
    # Top-level JSON info, func_eval will be filled based on data
    json_dict = {'tuning_problem_name': tuning_metadata['tuning_problem_name'],
                 'tuning_problem_category': None,
                 'surrogate_model': [],
                 'func_eval': [],
                }
    # Template for a function evaluation
    func_template = {'constants': {},
                     'machine_configuration': tuning_metadata['machine_configuration'],
                     'software_configuration': tuning_metadata['software_configuration'],
                     'additional_output': {},
                     'source': 'measure',
                    }
    # Loop safety
    parameters = None
    if type(fnames) is str:
        fnames = [fnames]
    # Prepare return structures
    sizes = []
    dicts = []
    for fname in fnames:
        # Make basic copy
        gptune_dict = dict((k,v) for (k,v) in json_dict.items())
        csv = pd.read_csv(fname)
        # Only set parameters once -- they'll be consistent throughout different files
        if parameters is None:
            parameters = [_ for _ in csv.columns if _.startswith('p') and _ != 'predicted']
        for index, row in csv.iterrows():
            new_eval = dict((k,v) for (k,v) in func_template.items())
            try:
                new_eval['task_parameter'] = {'isize': row['isize']}
            except KeyError:
                new_eval['task_parameter'] = {'isize': infer_size(fname, lookup_ival)}
            # SINGLE update per task size
            if index == 0:
                sizes.append(new_eval['task_parameter']['isize'])
            new_eval['tuning_parameter'] = dict((col, str(row[col])) for col in parameters)
            new_eval['evaluation_result'] = {'time': row['objective']}
            new_eval['evaluation_detail'] = {'time': {'evaluations': row['objective'],
                                                      'objective_scheme': 'average'}}
            # Time data maybe not needed?
            """
            now = datetime.datetime.now()
            new_eval['time']: {'tm_year': now.year,
                               'tm_mon': now.month,
                               'tm_mday': now.day,
                               'tm_hour': now.hour,
                               'tm_min': now.minute,
                               'tm_sec': now.second,
                               'tm_wday': now.weekday(),
                               'tm_yday': now.toordinal() - datetime.date(now.year, 1, 1).toordinal() + 1,
                               'tm_isdst': -1,
                              }
            """
            new_eval['uid'] = uuid.uuid4()
            gptune_dict['func_eval'].append(new_eval)
        dicts.append(gptune_dict)
        print(f"GPTune-ified {fname}")
    return dicts, sizes

# Return either the sequence or choice attribute based on what 'obj' actually defines
def seqchoice(obj):
    if hasattr(obj, 'sequence') and obj.sequence is not None:
        return obj.sequence
    elif hasattr(obj, 'choices') and obj.choices is not None:
        return obj.choices
    raise ValueError(f"Object {obj} lacks or has NONE for sequences and choices")

def main():
    args = parse(build())
    ot.RandomGenerator.SetSeed(args.seed)

    # Move into directory and fetch the relevant input space and problem description, perhaps other relevant
    # kwargs for the specified benchmark
    HERE, input_space, lookup_ival, kwargs = localized_load(args.benchmark)
    # FIRST, indirect lookup the factory builder in GPTune mode
    problem_lookup = args.builder(lookup_ival,
                                  input_space,
                                  HERE,
                                  name=args.benchmark+"_Problem",
                                  returnmode='GPTune',
                                  selflog=HERE+'/results.csv',
                                  **kwargs)
    # Next build the actual instance for evaluating the target problem
    target_problem = problem_lookup(args.target.upper())
    print(f"Target problem {args.target} constructed")

    # *S are passed to GPTune objects directly
    # *_space are used to build surrogate models and MOSTLY share kwargs
    # As such the *S_options define common options when this saves re-specification

    # Steal the parameter names / values from Problem object's input space
    PS_options = [{'name': x,
                   'transform': 'onehot',
                   'categories': seqchoice(target_problem.input_space[x])
                  } for x in target_problem.input_space.get_hyperparameter_names()]
    PS = Space([Categoricalnorm(**options) for options in PS_options])
    parameter_space = []
    # Parameter space requires some alteration due to inconsistencies
    for options in PS_options:
        options['transformer'] = options.pop('transform') # REALLY?! Keyname alteration
        options['type'] = 'categorical' # Bonus key
        # Categories key MAY need to become list instead of tuple
        # options['categories'] = list(options['categories'])
        parameter_space.append(options)

    # Able to steal this entirely from Problem object API
    OS = target_problem.output_space
    output_space = [{'name': 'time',
                     'type': 'real',
                     'transformer': 'identity',
                     'lower_bound': float(0.0),
                     'upper_bound': float('Inf')}]

    # Steal input space limits from Problem object API
    input_space = [{'name': 'isize',
                    'type': 'int',
                    'transformer': 'normalize',
                    'lower_bound': min(target_problem.dataset_lookup.keys()),
                    'upper_bound': max(target_problem.dataset_lookup.keys())}]
    IS = Space([Integer(low=input_space[0]['lower_bound'],
                        high=input_space[0]['upper_bound'],
                        transform='normalize',
                        name='isize')])

    # Meta Dicts are part of building surrogate models for each input, but have a lot of common
    # specification templated here
    base_meta_dict = {'tuning_problem_name': target_problem.name.split('Problem')[0][:-1],
                      'modeler': 'Model_GPy_LCM',
                      'input_space': input_space,
                      'output_space': output_space,
                      'parameter_space': parameter_space,
                      'loadable_machine_configurations': {'swing': {'intel': {'nodes': 1, 'cores': 128}}},
                      'loadable_software_configurations': {}
                     }
    constraints = {}
    objectives = target_problem.objective
    problem = TuningProblem(IS,PS,OS, objectives, constraints, None) # None = models (dict of names : func(point_dict) -> list(outputs)

    # Used to have consistent machine definition
    tuning_metadata = {
        "tuning_problem_name": base_meta_dict['tuning_problem_name'],
        "use_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": "swing",
            "intel": { "nodes": 1, "cores": 128 }
        },
        "software_configuration": {},
        "loadable_machine_configurations": base_meta_dict['loadable_machine_configurations'],
        "loadable_software_configurations": base_meta_dict['loadable_software_configurations'],
    }
    machine, processor, nodes, cores = GetMachineConfiguration(meta_dict=tuning_metadata)
    print(f"Machine: {machine} | Processor: {processor} | Num_Nodes: {nodes} | Num_Cores: {cores}")

    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    data = Data(problem)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    options  = Options()
    # These options inherited from Jaehoon's script
    options.update({'model_restarts': 1,
                    'distributed_memory_parallelism': False,
                    'shared_memory_parallelism': False,
                    'objective_evaluation_parallelism': False,
                    'objective_multisample_threads': 1,
                    'objective_multisample_Processes': 1,
                    'objective_nprocmax': 1,
                    'model_processes': 1,
                    'model_class': 'Model_GPy_LCM',
                    'verbose': False, # True
                    'sample_class': 'SampleOpenTURNS',
                    #'model_max_iters': args.nrun,
                   })
    options.validate(computer=computer)
    # Create the GPTune object
    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb)

    # Load prior evaluations in GPTune-ready format
    prior_traces, prior_sizes = csvs_to_gptune(args.inputs, tuning_metadata, lookup_ival)

    # Use surrogate models to show the prior data to GPTune
    #model_functions = {}
    #for prior_data, prior_task in zip(prior_traces, prior_sizes):
    #    meta_dict = dict((k,v) for (k,v) in base_meta_dict.items())
    #    if type(prior_task) is not list:
    #        prior_task = [prior_task]
    #    meta_dict['task_parameter'] = [prior_task]
    #    model_functions[tuple(prior_task)] = BuildSurrogateModel_tl(metadata_path=None,
    #                                                                metadata=meta_dict,
    #                                                                function_evaluations=prior_data['func_eval'])

    # ALTERNATIVE SPEC FOR ABOVE LOOP BLOCK
    func_evals = []
    for prior_data in prior_traces:
        func_evals.extend(prior_data['func_eval'])
    models, model_functions = gt.GenSurrogateModel([[s] for s in prior_sizes], func_evals)
    # gptune.data is properly set by the above call

    # Set up the actual transfer learning task
    transfer_task = [[target_problem.problem_class]]
    # Normalized = True is NOT a typical argument -- I modified GPTune.TLA1() to use this to
    # SKIP the normalization on self.data.I and self.data.P because the surrogate function already
    # normalizes this
    data, modeler, stats = gt.TLA1(transfer_task, args.nrun, normalized=True)
    #data, modeler, stats = gt.TLA(args.nrun, Igiven=transfer_task, models_transfer=models)
    print(f"Stats: {stats}")

if __name__ == "__main__":
    main()

