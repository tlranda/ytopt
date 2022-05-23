import numpy as np, pandas as pd
#from autotune.space import *
import os, sys, time, argparse
from csv import writer
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
# Will only use one of these, make a selection dictionary
sdv_models = {'GaussianCopula': GaussianCopula,
              'CopulaGAN': CopulaGAN,
              'CTGAN': CTGAN,
              'TVAE': TVAE}
from sdv.constraints import CustomConstraint, Between
from sdv.sampling.tabular import Condition
from ytopt.search.util import generic_loader

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_evals', type=int, default=10, help='maximum number of evaluations')
    parser.add_argument('--n_refit', type=int, default=0, help='refit the model')
    parser.add_argument('--seed', type=int, default=1234, help='set seed')
    parser.add_argument('--top', type=float, default=0.1, help='how much to train')
    parser.add_argument('--inputs', type=str, nargs='+', help='problems to use as input')
    parser.add_argument('--targets', type=int, nargs='+', help='problems to use as target tasks')
    parser.add_argument('--model', choices=list(sdv_models.keys()), default='GaussianCopula', help='SDV model')
    parser.add_argument('--retries', type=int, default=1000, help='#retries given to SDV row generation')
    return parser

def parse(prs, args=None):
    if args is None:
        args = parser.parse_args()
    return args

def main(args=None):
    args = parse(build(), args)
    sdv_model   = args.model
    max_retries = args.retries
    print(f"USING {sdv_model} for constraints with {max_retries} allotted retries")
    MAX_EVALS   = args.max_evals
    N_REFIT     = args.n_refit
    TOP         = args.top
    RANDOM_SEED = args.seed
    TARGET_task = args.target
    print('max_evals', MAX_EVALS, 'number of refit', N_REFIT, 'how much to train', TOP,
          'seed', RANDOM_SEED, 'target task', TARGET_task)
    np.random.seed(RANDOM_SEED)

    Time_start = time.time()
    print ('time...now', Time_start)

    X_opt = []
    print ('----------------------------- how much data to use?', TOP)

    # Load the input and target problems
    inputs, targets, frames = [], [], []
    # Fetch the target problem(s)'s plopper
    for problemName in args.inputs:
        # NOTE: When specified as 'filename.attribute', the second argument 'Problem' is effectively ignored
        # If only the filename is given (ie: 'filename.py'), defaults to finding the 'Problem' attribute in that file
        inputs.append(generic_loader(problemName, 'Problem'))
        # Load the best top x%
        dataframe = pd.read_csv(inputs[-1].plopper.kernel_dir+"/results_rf_"+str(inputs[-1].problem_class)+".csv")
        dataframe['runtime'] = np.log(dataframe['objective']) # log(run time)
        dataframe['input'] = pd.Series(int(inputs[-1].problem_class) for _ in range(len(dataframe.index)))
        q_10_s = np.quantile(dataframe.runtime.values, TOP)
        real_df = dataframe.loc[dataframe['runtime'] <= q_10_s]
        real_data = real_df.drop(columns=['elapsed_sec'])
        real_data = real_data.drop(columns=['objective'])
        frames.append(real_data)
    real_data = pd.concat(frames)
    for problemName in args.targets:
        targets.append(generic_loader(problemName, 'Problem'))

    # Steal from Problem.x1
    param_names = x1
    n_param = len(param_names)

    model = sdv_models[sdv_model](
                field_names = ['input']+targets[-1].params+['runtime'],
                # Similarly a bit tricky, may require problem to help define this
                field_transformers = targets[-1].field_transformers,
                constraints=[targets[-1].constraint],
                min_value = None,
                max_value = None
        )

    filename = "results_sdv.csv"
    fields   = targets[-1].params+['exe_time','predicted','elapsed_sec']
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)

        evals_infer = []
        eval_master = 0
        while eval_master < MAX_EVALS:
            # update model
            model.fit(real_data)
            conditions = Condition({'input': int(targets[-1].problem_class)}, num_rows=max(100, MAX_EVALS))
            if sdv_model == 'GaussianCopula':
                ss1 = model.sample_conditions([conditions])
            else:
                # Reject sampling means you may have to repeatedly try in order to generate the requested number of rows
                old_ss_len = 0
                new_ss_len = 0
                max_attempts = 100
                attempts = 0
                while attempts < max_attempts and conditions.num_rows > 0 and new_ss_len < conditions.num_rows:
                    attempts += 1
                    print(f"Reject strategy attempt {attempts}. {conditions.num_rows} data to be retrieved")
                    ss = model.sample_conditions([conditions], max_tries=max_retries)
                    new_ss_len += len(ss.index)
                    if attempts == 1:
                        ss1 = ss
                    else:
                        ss1 = ss1.append(ss, ignore_index=True)
                    conditions.num_rows -= new_ss_len - old_ss_len
                    old_ss_len = new_ss_len
            # Special handling function may be necessary here to do this kind of thing
            #ss1 = ss1.drop_duplicates(subset='BLOCK_SIZE', keep="first")
            ss  = ss1.sort_values(by='runtime')#, ascending=False)
            new_sdv = ss[:MAX_EVALS]
            max_evals = N_REFIT
            eval_update = 0
            stop = False
            while stop == False:
                for row in new_sdv.iterrows():
                    if eval_update == max_evals:
                        stop = True
                        break
                    # May not be safe to index like this lol
                    sample_point_val = row[1].values[1:]
                    # Should be a chain expression or something
                    sample_point = {x1[0]:sample_point_val[0]}
                    # Use the target problem's .objective() call
                    res          = targets[-1].objective(sample_point)
                    print (sample_point, res)
                    evals_infer.append(res)
                    now = time.time()
                    elapsed = now - Time_start
                    # Similar chain or something
                    ss = [sample_point['BLOCK_SIZE']]+[res]+[sample_point_val[-1]]+[elapsed]
                    csvwriter.writerow(ss)
                    csvfile.flush()
                    row_prev = row
                    evaluated = row[1].values[1:]
                    evaluated[-1] = float(np.log(res))
                    evaluated = np.append(evaluated,row[1].values[0])
                    real_data.loc[max(real_data.index)+1] = evaluated # real_data = [
                    eval_update += 1
                    eval_master += 1
    csvfile.close()


if __name__ == '__main__':
    main()

