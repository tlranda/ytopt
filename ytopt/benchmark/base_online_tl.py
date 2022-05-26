import numpy as np, pandas as pd
#from autotune.space import *
import os, sys, time, argparse
from csv import writer
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
# Will only use one of these, make a selection dictionary
sdv_models = {'GaussianCopula': GaussianCopula,
              'CopulaGAN': CopulaGAN,
              'CTGAN': CTGAN,
              'TVAE': TVAE,
              'random': None}
from sdv.constraints import CustomConstraint, Between
from sdv.sampling.tabular import Condition
from ytopt.search.util import load_from_file

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_evals', type=int, default=10,
                        help='maximum number of evaluations')
    parser.add_argument('--n_refit', type=int, default=0,
                        help='refit the model')
    parser.add_argument('--seed', type=int, default=1234,
                        help='set seed')
    parser.add_argument('--top', type=float, default=0.1,
                        help='how much to train')
    parser.add_argument('--inputs', type=str, nargs='+',
                        help='problems to use as input')
    parser.add_argument('--targets', type=str, nargs='+',
                        help='problems to use as target tasks')
    parser.add_argument('--model', choices=list(sdv_models.keys()),
                        default='GaussianCopula', help='SDV model')
    parser.add_argument('--retries', type=int, default=1000,
                        help='#retries given to SDV row generation')
    parser.add_argument('--single-target', action='store_true',
                        help='Treat each target as a unique problem (default: solve all targets at once)')
    parser.add_argument('--unique', action='store_true',
                        help='Do not re-evaluate points seen since the last dataset generation')
    parser.add_argument('--output-prefix', type=str, default='results_sdv',
                        help='Output files are created using this prefix (default: [results_sdv]*.csv)')
    return parser

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

def bind_from_args(args):
    return args.model, args.retries, args.single_target, args.unique, \
           args.max_evals, args.n_refit, args.top, args.seed

time_start = time.time()

def param_type(k, problem):
    v = problem.problem_params[k]
    if v == 'categorical':
        if hasattr(problem, 'categorical_cast'):
            v = problem.categorical_cast[k]
        else:
            v = 'str'
    if v == 'integer':
        v = 'int64'
    return v

def online(targets, data, inputs, args, fname):
    global time_start
    sdv_model, max_retries, _, unique, \
               MAX_EVALS, N_REFIT, TOP, _ = bind_from_args(args)

    # All problems (input and target alike) must utilize the same parameters or this is not going to work
    param_names = set(targets[0].params)
    for target_problem in targets[1:]:
        other_names = set(target_problem.params)
        if len(param_names.difference(other_names)) > 0:
            raise ValueError(f"Targets {targets[0].name} and "
                             f"{target_problem.name} utilize different parameters")
    for input_problem in inputs:
        other_names = set(input_problem.params)
        if len(param_names.difference(other_names)) > 0:
            raise ValueError(f"Target {targets[0].name} and "
                             f"{input_problem.name} utilize different parameters")
    param_names = sorted(param_names)
    n_params = len(param_names)

    if sdv_model != 'random':
        model = sdv_models[sdv_model](
                  field_names = ['input']+param_names+['runtime'],
                  field_transformers = targets[0].problem_params,
                  constraints=targets[0].constraints,
                  min_value = None,
                  max_value = None
                )
    else:
        model = None
    csv_fields = param_names+['objective','predicted','elapsed_sec']*len(targets)
    # writing to csv file
    with open(fname, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = writer(csvfile)
        # writing the fields
        csvwriter.writerow(csv_fields)

        # Make conditions for each target
        conditions = []
        for target_problem in targets:
            conditions.append(Condition({'input': int(target_problem.problem_class)},
                                        num_rows=max(100, MAX_EVALS)))
        evals_infer = []
        eval_master = 0
        # Initial fit
        if model is not None:
            model.fit(data)
        while eval_master < MAX_EVALS:
            # Generate prospective points
            if sdv_model == 'GaussianCopula':
                ss1 = model.sample_conditions(conditions)
            elif sdv_model != 'random':
                # Reject sampling means you may have to repeatedly try
                # in order to generate the requested number of rows
                max_attempts = 100
                attempts = 0
                old_ss_len = 0
                new_ss_len = 0
                # Should be a way to sample all conditions at once until
                # they have all met their row requirement, but IDK so we're
                # going to do them one at a time for now
                for cond in conditions:
                    while attempts < max_attempts and cond.num_rows > 0 and new_ss_len < cond.num_rows:
                        attempts += 1
                        print(f"Reject strategy attempt {attempts}. {cond.num_rows} data to be retrieved")
                        ss = model.sample_conditions([cond], max_tries=max_retries)
                        new_ss_len += len(ss.index)
                        if attempts == 1:
                            ss1 = ss
                        else:
                            ss1 = ss1.append(ss, ignore_index=True)
                        cond.num_rows -= new_ss_len - old_ss_len
                        old_ss_len = new_ss_len
            else:
                # random model is achieved by sampling configurations from the target problem's input space
                columns = ['input']+param_names+['runtime']
                dtypes = [(k,param_type(k, targets[0])) for k in columns]
                random_data = []
                for idx, cond in enumerate(conditions):
                    for _ in range(cond.num_rows):
                        # Generate a random valid sample in the parameter space
                        random_params = targets[idx].input_space.sample_configuration().get_dictionary()
                        random_params = [random_params[k] for k in param_names]
                        # Generate the runtime estimate
                        inference = 1.0
                        random_data.append(tuple([cond.column_values['input']]+random_params+[inference]))
                ss1 = np.array(random_data, dtype=dtypes)
                ss1 = pd.DataFrame(ss1, columns=columns)
                # Make dataframe from calling targets[i].input_space.sample_configuration.get_dictionary()
            # Don't evaluate the exact same parameter configuration multiple times in a fitting round
            ss1 = ss1.drop_duplicates(subset=param_names, keep="first")
            ss = ss1.sort_values(by='runtime')#, ascending=False)
            new_sdv = ss[:MAX_EVALS]
            eval_update = 0
            stop = False
            while not stop:
                for row in new_sdv.iterrows():
                    if eval_update == N_REFIT:
                        # update model
                        if model is not None:
                            model.fit(data)
                        stop = True
                        break
                    sample_point_val = row[1].values[1:]
                    sample_point = dict((pp,vv) for (pp,vv) in zip(param_names, sample_point_val))
                    # Search to see if this combination of parameter values AND
                    # problem class already exist in the data frame.
                    # If we're unique, skip it, otherwise we will replace it with new value
                    problem_tuple = tuple(param_names+['input'])
                    search_equals = tuple(row[1].values[1:1+n_params].tolist()+[row[1].values[0]])
                    n_matching_columns = (data[list(problem_tuple)] == search_equals).sum(1)
                    full_match_idx = np.where(n_matching_columns == n_params+1)[0]
                    matching_data = data.iloc[full_match_idx]
                    if matching_data.empty or not unique:
                        ss = []
                        for target_problem in targets:
                            # Use the target problem's .objective() call to generate an evaluation
                            evals_infer.append(target_problem.objective(sample_point))
                            print(target_problem.name, sample_point, evals_infer[-1])
                            now = time.time()
                            elapsed = now - time_start
                            if ss == []:
                                ss = [sample_point[k] for k in param_names]
                                ss += [evals_infer[-1]]
                                ss += [sample_point_val[-1]]
                                ss += [elapsed]
                            else:
                                # Append new target data to the CSV row
                                ss2 = [evals_infer[-1]]
                                ss2 += [sample_point_val[-1]]
                                ss2 += [elapsed]
                                ss.extend(ss2)
                            # Basically: problem parameters, np.log(time), problem_class
                            evaluated = list(search_equals)
                            # Insert runtime before the problem class size
                            evaluated.insert(-1, float(np.log(evals_infer[-1])))
                            # For each problem we want to denote the actual result in
                            # our dataset to improve future data generation
                            if matching_data.empty:
                                data.loc[max(data.index)+1] = evaluated
                            else:
                                # Replace results -- should be mostly the same
                                # Have to wrap `evaluated` in a list for pandas to be
                                # happy with the overwrite
                                data.loc[matching_data.index] = [evaluated]
                        # Record in CSV and update iteration
                        csvwriter.writerow(ss)
                        csvfile.flush()
                        eval_update += 1
                        eval_master += 1
                if unique:
                    stop = True
    csvfile.close()


def main(args=None):
    args = parse(build(), args)
    sdv_model, max_retries, one_target, _, \
               MAX_EVALS, N_REFIT, TOP, RANDOM_SEED = bind_from_args(args)
    output_prefix = args.output_prefix
    print(f"USING {sdv_model} for constraints with {max_retries} allotted retries")
    print('max_evals', MAX_EVALS, 'number of refit', N_REFIT, 'how much to train', TOP,
          'seed', RANDOM_SEED)
    # Seed control
    np.random.seed(RANDOM_SEED)
    global time_start
    time_start = time.time()

    X_opt = []
    print ('----------------------------- how much data to use?', TOP)

    # Load the input and target problems
    inputs, targets, frames = [], [], []
    # Fetch the target problem(s)'s plopper
    for idx, problemName in enumerate(args.inputs):
        # NOTE: When specified as 'filename.attribute', the second argument 'Problem'
        # is effectively ignored. If only the filename is given (ie: 'filename.py'),
        # defaults to finding the 'Problem' attribute in that file
        if problemName.endswith('.py'):
            attr = 'Problem'
        else:
            pName, attr = problemName.split('.')
            pName += '.py'
        inputs.append(load_from_file(pName, attr))
        # Load the best top x%
        results_file = inputs[-1].plopper.kernel_dir+"/results_"+str(inputs[-1].problem_class)+".csv"
        if not os.path.exists(results_file):
            # Execute the input problem and move its results files to the above directory
            raise ValueError(f"Could not find {results_file} for '{problemName}' "
                             f"[{inputs[-1].name}]"
                             "\nYou may need to run this problem or rename its output "
                             "as above for the script to locate it")
        dataframe = pd.read_csv(results_file)
        dataframe['runtime'] = np.log(dataframe['objective']) # log(run time)
        dataframe['input'] = pd.Series(int(inputs[-1].problem_class) for _ in range(len(dataframe.index)))
        q_10_s = np.quantile(dataframe.runtime.values, TOP)
        real_df = dataframe.loc[dataframe['runtime'] <= q_10_s]
        real_data = real_df.drop(columns=['elapsed_sec'])
        real_data = real_data.drop(columns=['objective'])
        frames.append(real_data)
    real_data = pd.concat(frames)

    for problemName in args.targets:
        if problemName.endswith('.py'):
            attr = 'Problem'
        else:
            pName, attr = problemName.split('.')
            pName += '.py'
        targets.append(load_from_file(pName, attr))
        # make target evaluations silent as we'll report them on our own
        targets[-1].silent = True
        # Seed control
        targets[-1].seed(RANDOM_SEED)
        # Single-target mode
        if one_target:
            online([targets[-1]], real_data, inputs, args, f"{output_prefix}_{targets[-1].name}.csv")
    if not one_target:
        online(targets, real_data, inputs, args, f"{output_prefix}_ALL.csv")

if __name__ == '__main__':
    main()

