import sdv
# Uncomment ONE of these to switch which model you use
# Non-GaussianCopula are NOT verified to work yet
from sdv.tabular import GaussianCopula as sdv_model
#from sdv.tabular import CopulaGAN as sdv_model
#from sdv.tabular import CTGAN as sdv_model
#from sdv.tabular import TVAE as sdv_model
from sdv.constraints import Between
from sdv.sampling import Condition
import pandas as pd, numpy as np
from copy import deepcopy as dcpy
import pdb

# Parameters for test() function
seed = 1234        # RNG seed
rows = 8           # Training data rows
cols = 5           # Features in training data
n_constraints = 3  # First-N columns are Constrained columns
n_rows = 5         # Rows to generate per seed
abs_min = 0        # Minimum INTEGER value for any constrained column
abs_max = 100      # Maximum INTEGER value for any constrained column
pre_float = False  # Whether training data is initially expressed as float or not

# Params for running script as main
TRACE = False
SCAN_SEED = 1
SCAN = False
N_SEEDS = 10
N_SKIP = 0
BAD_SEED = 9850

assert n_constraints <= cols # Doesn't make sense otherwise

# This class will somewhat automatically fix data to go from integer to float domain and back
# SDV SHOULD do this for you but it has caveats/bugs that aren't fixed yet
# Expected usage:
"""
    my_constraints = [sdv.constraints.Between(...), ...]
    #! NOTE! ONLY supports the Between constraint for now. Other constraints are ValueErrors
    #! NOTE! These constraints should NOT be applied to your SDV model!
        # Replace EACH of them with Between(column, low=0, high=1) when using the model constructor

    my_workaround = sdv_workaround(my_constraints)
    ...
    my_data = pd.DataFrame(...) # Should be clearly INT type for constrained columns
    my_model = sdv.tabular.GaussianCopula(...) # ! Corresponding constraints must be Between 0,1
    my_model.fit(my_workaround(my_data))
    ...
    my_original_condition = {column: data}
    transformed_data = my_workaround(pd.DataFrame(my_original_condition))
    my_condition = sdv.sampling.Condition(transformed_data)
    generated = my_model.sample_conditions([my_condition])
    my_generated = my_workaround(generated) # ! Now back to integer domain and usable as if SDV has no bugs
"""
class sdv_workaround:
    # Static information you don't need to recalculate
    eps_min = 2*np.finfo(float).eps
    eps_max = 1.0 - eps_min

    def __init__(self, constraints):
        if type(constraints) != list:
            constraints = [constraints]
        # Only support Between
        for cons in constraints:
            if type(cons) != Between:
                raise ValueError
        self.cols = [_.constraint_column for _ in constraints]
        self.lows = [_._low for _ in constraints]
        self.highs = [_._high for _ in constraints]
        self.ranges = [h-l for (l,h) in zip(self.lows, self.highs)]

    def __call__(self, data, direction=None):
        # Automate transformation, can give hint to limit work
        if type(data) != pd.DataFrame:
            raise ValueError
        # Automatic approximate type check for affected data
        dtype = data.dtypes[self.cols].all().kind
        if dtype == 'i':
            if direction is None or direction == 'float':
                return self.to_float(data)
            else:
                return data
        elif dtype == 'f':
            if direction is None or direction == 'int':
                return self.from_float(data)
            else:
                return data
        else:
            raise ValueError

    def __str__(self):
        per_cons = []
        for c, l, h, r in zip(self.cols, self.lows, self.highs, self.ranges):
            con_repr = ", ".join([f"Column: {c}", f"Low: {l}", f"High: {h}", f"Range: {r}"])
            per_cons.append("["+con_repr+"]")
        return "\n".join(per_cons)

    def to_float(self, data):
        if type(data) != pd.DataFrame:
            raise ValueError
        else:
            data = dcpy(data)
            # Should be per data column
            for c, l, r in zip(self.cols, self.lows, self.ranges):
                data[c] = (data[c] - l) / r
                # Fix SDV hating an exact match on the lower bound regardless of strictness
                for idx, val in enumerate(data[c]):
                    if val == 0.0:
                        data.loc[idx, c] = self.eps_min
                    if val == 1.0:
                        data.loc[idx, c] = self.eps_max
                # Explicit type fix
                data[c] = data[c].astype(float)
            return data

    def from_float(self, data):
        if type(data) != pd.DataFrame:
            raise ValueError
        else:
            # Should be per data column
            data = dcpy(data)
            for c, l, r in zip(self.cols, self.lows, self.ranges):
                # Soft rounding for very small floating-point errors
                data[c] = round((data[c]*r)+l, 0)
                # Explicit type fix
                data[c] = data[c].astype(int)
            return data

    # BEYOND THIS POINT IS NOT GENERALLY NECESSARY -- USED FOR TESTING/VALIDATION

    def clean(self, data):
        # Used to ensure randomly generated data is valid under constraints
        # NOTE THIS WILL REPLACE ILL-CONSTRAINED DATA WITH RANDOM VALID DATA
        if type(data) is not pd.DataFrame:
            raise ValueError
        data = dcpy(data)
        data_kind = data.dtypes[self.cols].all().kind
        if data_kind not in ['i', 'f']:
            raise ValueError
        for c, l, h, r in zip(self.cols, self.lows, self.highs, self.ranges):
            if c in data.columns:
                new_c = data[c]
                for idx, val in enumerate(new_c):
                    # Find bad data and replace with random good data
                    if val < l or val > h:
                        if data_kind == 'i':
                            new_c[idx] = np.random.randint(l,h)
                        else:
                            # Cast rand (0,1) to int and back to specific float range within (0,1)
                            new_c[idx] = ((np.random.rand()*r+l)-l)/r
                data[c] = new_c
        return data

    def sample_ints(self):
        ints = {}
        for c, l, h in zip(self.cols, self.lows, self.highs):
            ints[c] = [np.random.randint(l,h)]
        return pd.DataFrame(ints)

    def sample_floats(self):
        return self(self.sample_ints())


# Make integer 0+ --> alphabetical string base A-Z
# So DataFrame columns have nice appearance / references
def charify(i):
    if i == 0:
        return "A"
    else:
        s = ""
        while i >= 0:
            s = chr(65+i%26)+s # Append current letter (built back-to-front)
            i = i//26-1 # Re-zero index for next letter
        return s

# Automate making an arbitrary constraint to enforce
def between_maker(column):
    vals = None
    while vals is None or vals[0] == vals[1]:
        vals = np.random.randint(abs_min, abs_max, size=2)
    return Between(column, low=min(vals), high=max(vals))

# Automate testing a bit of data using this random seed
def test(seed, silent=False):
    np.random.seed(seed)
    # What we actually want the INTEGER constraint to be
    sdv_constraints = [between_maker(charify(_)) for _ in range(n_constraints)]
    # The lie we'll tell the model (a float between 0 and 1)
    model_constraints = [Between(charify(_), low=0, high=1, strict=True) for _ in range(n_constraints)]
    sdv_fitter = sdv_workaround(sdv_constraints)
    if not silent:
        print("Constraints")
        print(sdv_fitter)

    # Some arbitrary data to constrained sample against
    if pre_float:
        predata = [[np.random.rand() for _ in range(rows)] for col in range(cols)]
    else:
        predata = [[np.random.randint(abs_min, abs_max) for _ in range(rows)] for col in range(cols)]
    data = pd.DataFrame(dict((charify(col), predata[col]) for col in range(cols)))
    # This step is for testing verification only -- to ensure random data is OK in constraints
    data = sdv_fitter.clean(data)
    if not silent:
        print("Training Data")
        print(data.head())

    # Model to utilize
    model = sdv_model(
            field_names = data.columns.tolist(),
            # Note that the type MUST be float for ALL constrained columns until SDV fixes integer
            # stability on their end, else may fail to produce samples for some valid inputs
            field_transformers = dict((c, 'float') for c in data.columns),
            constraints = model_constraints,
            min_value = None,
            max_value = None)
    fit_data = sdv_fitter(data, direction='float')
    model.fit(fit_data)

    # Make conditions
    conditions = []
    # Usually you don't need to make constraints this way, but it helps with validation testing
    sample_targets = sdv_fitter.sample_floats()
    if not silent:
        print("Forced values")
        print(sdv_fitter(sample_targets))
    conditions.append(Condition(dict((col.constraint_column, target) for (col, target) in zip(sdv_constraints, sample_targets.values[0])), num_rows=n_rows))
    out = model.sample_conditions(conditions)
    if not silent:
        print("RAW conditional samples")
        print(out)
        print("Transformed conditional samples")
        print(sdv_fitter(out))

if __name__ == '__main__':
    if SCAN:
        # Search for seeds that exhibit poor behavior
        np.random.seed(SCAN_SEED)
        import tqdm
        bar = tqdm.tqdm([np.random.randint(1,99999) for _ in range(N_SEEDS)][N_SKIP:])
        bar.set_description("Seed")
        for seed in bar:
            try:
                test(seed, silent=True)
            except:
                print("\n"+f"BROKEN on seed {seed}")
                break
    else:
        # Display behavior on a particular seed
        if TRACE:
            pdb.set_trace()
        test(BAD_SEED)

