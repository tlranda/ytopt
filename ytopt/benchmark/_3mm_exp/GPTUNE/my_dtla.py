from autotune.space import Space, Integer, Real
from autotune.problem import TuningProblem
from GPTune.gptune import GPTune
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
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-seed', type=int, default=1234, help='Set seed')
    parser.add_argument('-builder', choices=['polybench', 'ecp'], default='polybench', help='Problem builder')
    return parser

def parse(parser, args=None):
    if args is None:
        args = parser.parse_args()
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
    return HERE, input_space, lookup_ival, kwargs

def main():
    args = parse(build())
    ot.RandomGenerator.SetSeed(args.seed)

    if args.builder == 'polybench':
        problem_lookup = polybench_problem_builder
    elif args.builder == 'ecp':
        problem_lookup = ecp_problem_builder
    else:
        raise ValueError(f"Unsupported problem builder {args.builder}")
    HERE, input_space, lookup_ival, kwargs = localized_load(args.benchmark)
    problem_lookup = problem_lookup(lookup_ival, input_space, HERE, name=args.benchmark+"_Problem",
                                    returnmode='GPTune', selflog=HERE+'/results.csv', **kwargs)
    target_problem = problem_lookup(args.target.upper())
    objectives = target_problem.objective
    def seqchoice(obj):
        if hasattr(obj, 'sequence') and obj.sequence is not None:
            return obj.sequence
        elif hasattr(obj, 'choices') and obj.choices is not None:
            return obj.choices
        raise ValueError(f"Object {obj} lacks or has NONE for sequences and choices")
    PS = Space([Categoricalnorm(seqchoice(target_problem.input_space[x]), transform='onehot', name=x) \
                for x in target_problem.input_space.get_hyperparameter_names()])
    OS = target_problem.output_space
    IS = Space([Integer(min(target_problem.dataset_lookup.keys()), max(target_problem.dataset_lookup.keys()),
                        transform='normalize', name='isize')])
    constraints = {}
    problem = TuningProblem(IS,PS,OS, objectives, constraints, None)

    tuning_metadata = {
        "tuning_problem_name": target_problem.name.split('Problem')[0][:-1],
        "use_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": "swing",
            "intel": { "nodes": 1, "cores": 128 }
        },
        "software_configuration": {},
        "loadable_machine_configurations": {
            "swing": {
                "intel": { "nodes": 1, "cores": 128 }
            }
        },
        "loadable_software_configurations": {}
    }

    machine, processor, nodes, cores = GetMachineConfiguration(meta_dict = tuning_metadata)
    print(f"Machine: {machine} | Processor: {processor} | Num_Nodes: {nodes} | Num_Cores: {cores}")
    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options  = Options()
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
                   })
    options.validate(computer=computer)

    prior_data = csvs_to_gptune(args.inputs, tuning_metadata, lookup_ival)
    model_functions[1] = BuildSurrogateModel(metadata_path=None, metadata=meta_dict, function_evals=prior_data['func_eval'])
    #giventask = [[target_problem.problem_class]]
    #data = Data(problem)
    #historydb = HistoryDB(meta_dict = tuning_metadata)
    #historydb.load_func_eval = False # ?
    #gt = GPTune(problem, computer=computer, data=data, options=options,
    #            historydb=historydb, driverabspath=os.path.abspath(__file__))
    #data, modeler, stats = gt.MLA(NS=args.nrun, Igiven=giventask, NI=1, NS1=int(max(args.nrun//2,1)))
    #print(f"Stats: {stats}")

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
    for fname in fnames:
        csv = pd.read_csv(fname)
        if parameters is None:
            parameters = [_ for _ in csv.columns if _.startswith('p') and _ != 'predicted']
            for index, row in csv.iterrows():
                new_eval = dict((k,v) for (k,v) in func_template.items())
                try:
                    new_eval['task_parameter'] = row['isize']
                except KeyError:
                    new_eval['task_parameter'] = infer_size(fname, lookup_ival)
                new_eval['tuning_parameter'] = dict((col, str(row[col])) for col in parameters)
                new_eval['evaluation_result'] = {'y': row['objective']}
                new_eval['evaluation_detail'] = {'y': {'evaluations': row['objective'], 'objective_scheme': 'average'}}
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
                new_eval['uid'] = uuid.uuid4()
                json_dict['func_eval'].append(new_eval)
    return json_dict

if __name__ == "__main__":
    main()

    """
    model_functions = {}
    for i, ss in zip(range(1,len(giventask),1),source_task):
        tvalue_ = giventask[i]
        print ('======================================',tvalue_)
        i_space = []
        for i, t_name in enumerate(task_name):
            i_space.append({"name":t_name,"type":"int","transformer":"normalize","lower_bound":i_min[i],"upper_bound":i_max[i]})
        meta_dict = {
            "tuning_problem_name":str(Kernel_name),
            "modeler":"Model_GPy_LCM",
            "task_parameter":[tvalue_],
            "input_space":i_space,
            "parameter_space": [
        {"name": "p0","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "]},
        {"name": "p1","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "]},
        {"name": "p2","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "]},
        {"name": "p3","transformer": "onehot","type": "categorical","categories": ['4','8','16','20','32','50','64','80','96','100','128']},
        {"name": "p4","transformer": "onehot","type": "categorical","categories": ['4','8','16','20','32','50','64','80','100','128','2048']},
        {"name": "p5","transformer": "onehot","type": "categorical","categories": ['4','8','16','20','32','50','64','80','100','128','256']},
        {"name": "p6","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(j2) pack array(C) allocate(malloc)", " "]},
        {"name": "p7","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(i1) pack array(D) allocate(malloc)", " "]},
        {"name": "p8","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(j2) pack array(E) allocate(malloc)", " "]},
        {"name": "p9","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(i1) pack array(F) allocate(malloc)", " "]},
            ],
            "output_space": [{"name":"y","type":"real","transformer":"identity","lower_bound":float(0.0),"upper_bound":float('Inf')}],
            "loadable_machine_configurations":{"swing":{"amd":{"nodes":[1],"cores":128}}},
            "loadable_software_configurations":{}
        }
        f = open(f"TLA_experiments/SLA-GPTune-{ss}-200/{Kernel_name}.json")
        func_evals = json.load(f)
        f.close()
        print (f"..........................TLA_experiments/SLA-GPTune-{ss}-200/{Kernel_name}.json")
#         print (func_evals['func_eval'])
        model_functions[tuple(tvalue_)] = BuildSurrogateModel(metadata_path=None,metadata=meta_dict,function_evaluations=func_evals['func_eval'])
        print (model_functions)

    NI=len(giventask)  ## number of tasks
    NS=nrun ## number of runs

    newtask = [[400]]
    (aprxopts, objval, stats) = gt.TLA1(newtask, NS=None)
    print("stats: ", stats)

    "" Print the optimal parameters and function evaluations ""
    for tid in range(len(newtask)):
        print("new task: %s" % (newtask[tid]))
        print('    predicted Popt: ', aprxopts[tid], ' objval: ', objval[tid])
    """

