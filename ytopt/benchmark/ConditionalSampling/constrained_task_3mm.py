import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys
import pdb

# Fetch 3mm data from the problem
PROBLEM_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../_3mm_exp/")
if PROBLEM_PATH not in sys.path:
    sys.path.insert(1,PROBLEM_PATH)
from problem import lookup_ival

# Replicate the constraint
constraint_high = max(lookup_ival.keys())
constraint_low = min(lookup_ival.keys())
x_range = np.arange(constraint_low,constraint_high,1)[1:]
sizes = np.asarray(sorted(lookup_ival.keys()))

def sdv_constrained(x, low, high):
    x = (x - low) / (high - low)
    x = (x * 0.95) + 0.025
    return np.log(x / (1.0 - x))

fig,ax = plt.subplots()
for size in sizes[1:-1]:
    name = lookup_ival[size][1]
    if len(name) > 2:
        name = name.lower().capitalize()
    constrained = sdv_constrained(size, constraint_low, constraint_high)
    ax.scatter(size, constrained, label=f"{name} Task")
    print(f"{name} --> {constrained}")
ax.legend(loc='upper left')
ax.plot(x_range, sdv_constrained(x_range, constraint_low, constraint_high), label="Logit Values", zorder=-1)
#ax.set_xscale('log')
ax.set_xlabel('Task Size')
ax.set_ylabel('Constraint Value')
ax.set_title(f"3mm constraints between {constraint_low} and {constraint_high}")
fig.savefig(os.path.dirname(__file__)+"/Assets/constrained_values.png", format='png')

