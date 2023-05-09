import numpy as np, pandas as pd
from scipy import stats
from sdv.tabular import GaussianCopula
from sdv.constraints import CustomConstraint, Between
from sdv.sampling.tabular import Condition
import matplotlib
import matplotlib.pyplot as plt
import operator
import itertools
from copy import deepcopy as dcpy

##########################################
#                                        #
#   DEFINING AN ARBITRARY RELATIONSHIP   #
#     TO MODEL WITH GAUSSIAN COPULA      #
#                                        #
##########################################

# Construct a simple relationship f(x*)
# Uses randomly generated terms to form polynomials of the form (ax+b) * (cx+d) * ...
def create_arbitrary_relationship(tick_range=(0,50,0.5), n_vars=2, rounding=2, seed=1234):
    if seed is not None:
        np.random.seed(seed)
    ticks = np.arange(*tick_range)
    # Rescaled to limit range
    coeffs = np.round(np.random.rand(n_vars) / 2,rounding)
    # Rescaled to permit +/- and absolute value up to 10
    biases = np.round((np.random.rand(n_vars) - 0.5) * 20,rounding)
    # Measure function as f(x*) = (c0 * x + b0) * (c1 * x + b1) * ...
    f_xs = operator.mul(*[coeffs[_] * ticks + biases[_] for _ in range(n_vars)])
    # Report function as distance from minimum
    f_xs = np.abs(f_xs.min() - f_xs)
    return f_xs, coeffs, biases, ticks

# Adjust an existing relationship to create a correlated trend by altering its coefficient/bias terms
def create_correlated_relationship(coeffs, biases, ticks, coeff_nudge=None, bias_nudge=None, rounding=2):
    if coeff_nudge is None:
        # Coeffs nudge in the same direction, but a slightly larger nudge for later elements
        coeff_nudge = ((np.random.rand(1) - 0.5) / 2) * np.ones(len(coeffs))
        # Diminishing change for larger elements
        for idx in range(1,len(coeffs)):
            coeff_nudge[idx] += (0.01 ** idx) * coeff_nudge[idx-1]
        # Coeff nudge should NOT change the sign of any coefficient
        sign_adjust = np.where(np.sign(coeffs) - np.sign(coeff_nudge) != 0)[0]
        coeff_nudge[sign_adjust] *= -1/2
    coeffs = np.round(coeffs + coeff_nudge, rounding)
    if bias_nudge is None:
        # Bias nudge in ANY direction with absolute value up to 10
        bias_nudge = (np.random.rand(len(biases)) - 0.5) * 20
    biases = np.round(biases + bias_nudge, rounding)
    # Measure function as f(x*) = (c0 * x + b0) * (c1 * x + b1) * ...
    f_xs = operator.mul(*[coeffs[_] * ticks + biases[_] for _ in range(len(coeffs))])
    # Report function as distance from minimum
    f_xs = np.abs(f_xs.min() - f_xs)
    return f_xs, coeffs, biases, coeff_nudge, bias_nudge, ticks

# Source relationship 1
f_x1, coeffs, biases, ticks = create_arbitrary_relationship()
# Correlated relationship 1
f_x2, rel_coeffs, rel_biases, coeff_nudge, bias_nudge, _ = create_correlated_relationship(dcpy(coeffs), dcpy(biases), dcpy(ticks))
# Transfer relationship is perfectly correlated with this shift
f_tl, tl_coeffs, tl_biases, *_ = create_correlated_relationship(dcpy(rel_coeffs), dcpy(rel_biases), dcpy(ticks), coeff_nudge, bias_nudge)

# Plot initial relationships
fig,ax = plt.subplots()
ax.plot(ticks, f_x1, label=f"f(src_1)")# ~ {' * '.join(['('+str(c)+'x + '+str(b)+')' for c,b in zip(coeffs, biases)])}")
ax.plot(ticks, f_x2, label=f"f(src_2)")# ~ {' * '.join(['('+str(c)+'x + '+str(b)+')' for c,b in zip(rel_coeffs, rel_biases)])}")
ax.plot(ticks, f_tl, label=f"f(tgt_1)")# ~ {' * '.join(['('+str(c)+'x + '+str(b)+')' for c,b in zip(tl_coeffs, tl_biases)])}")
ax.legend()
ax.axhline(y=0, color='k', zorder=-1)
ax.axvline(x=0, color='k', zorder=-1)
fig.savefig("Assets/GC_funcs.png", format="png")
fig.clear()

##########################################
#                                        #
# FIT A GAUSSIAN COPULA TO SOURCES FROM  #
#         THE AVAILABLE DATA             #
#                                        #
##########################################

# Prepare datasets for Gaussian Copula
source_data = pd.concat([pd.DataFrame(dict(
                            ((f"x{idx}", ticks) for idx in range(len(coeffs))),
                            **{'size': t*np.ones(len(y)),'y': y}))
                            for y,t in zip([f_x1, f_x2], range(1,3))])
target_truth = pd.DataFrame(dict(((f"x{idx}", ticks) for idx in range(len(tl_coeffs))),
                                    **{'size': 3*np.ones(len(f_tl)), 'y': f_tl}))

# Prepare a Gaussian Copula model
constraint_low, constraint_high = (1,3)
constraints = [Between(column='size', low=constraint_low, high=constraint_high)]
transformers = dict((k, 'float') if k == 'y' else (k, 'categorical') for k in source_data.columns)
model = GaussianCopula(field_names=transformers.keys(),
                       field_transformers=transformers,
                       constraints=constraints,
                       min_value=None,
                       max_value=None,
                       )
model.fit(source_data)

n_samples = 10

#################################################
#                                               #
#      MIMIC SDV-STYLE UNCONDITIONAL SAMPLING   #
#             WITH THE GAUSSIAN COPULA          #
#                   IN 3 STEPS                  #
# ie: output = model.sample(num_rows=n_samples) #
#                                               #
#################################################

'''**********************************************
*                                               *
*   STEP (1): EXTRACT MODEL DATA AND METADATA   *
*     ie: already performed by model.fit()      *
*                                               *
**********************************************'''

# Model components: Extract covariance and univariates
model_covariance = model._model.covariance
model_univariates = model._model.univariates

# ABUSE OF KNOWLEDGE: All categories for x* have EQUAL OCCURRENCE COUNTS
# This means that SDV->RDT->Pandas sorting will be unstable (uses QuickSort)
# And the actual order of categories that it produces are essentially random
# As such, we can MANUALLY build a sorted order of categories that will DIFFER
# from what SDV utilizes, but not mathematically diverge
category_means = model._metadata._hyper_transformer._transformers_sequence[0].means
category_means = pd.Series(category_means.values, index=sorted(category_means.index))
n_categories = len(category_means)

'''******************************************************
*                                                       *
*          STEP (2): INITIAL SAMPLING FROM GC           *
*                                                       *
*       MIMIC = unconditional sampling using GC         *
*  ie: mimic = model._model.sample(num_rows=n_samples)  *
*                                                       *
******************************************************'''

# Use covariance and ZERO-means to pick a number of samples
# Identify the CDF for all sampled data
def unconditional_GC_sampling(n_samples, n_columns, covariance):
    random_samples = np.random.multivariate_normal(np.zeros(n_columns), covariance, size=n_samples)
    return stats.norm.cdf(random_samples)

base_random_cdf = unconditional_GC_sampling(n_samples, len(source_data.columns), model_covariance)

# Use the percent point for each univariate to utilize GC distribution
def make_sampled_frame(columns, univariates, cdfs):
    mimic = {}
    for column, univariate, cdf_axis in zip(columns, univariates, itertools.count()):
        # Add '.value' for SDV compatibility
        mimic[column+'.value'] = univariate.percent_point(cdfs[:,cdf_axis])
    return pd.DataFrame(mimic)

# Alter columns for SDV compatibility
columns = source_data.columns.tolist()
columns[columns.index('size')] = 'size#1#3'
mimic = make_sampled_frame(columns, model_univariates, base_random_cdf)

'''******************************************************
*                                                       *
*         STEP (3): REVERSE TRANSFORM SAMPLES           *
*                                                       *
*   OUTPUT = reversed data from GC --> resemble source  *
* ie: output = model._metadata.reverse_transform(mimic) *
*                                                       *
******************************************************'''

def reverse_constraint(data, constraint_range):
    data = 1.0 / (1.0 + np.exp(-data))
    data = (data - 0.025) / 0.95
    data = data * (constraint_range[1] - constraint_range[0]) + constraint_range[0]
    return data.clip(constraint_range[0], constraint_range[1])

def reverse_transform(mimicry, n_categories, category_means, constraint_range):
    output = {}
    for column, data in mimicry.items():
        # Rename to drop the '.value' intermediate column name
        column = column.split('.',1)[0]
        if column.startswith('x'):
            # This is SDV's more "memory intensive" technique (no batching) -- suitable for demonstration
            # Broadcast shapes
            broadcast_data = np.broadcast_to(data.to_numpy(), (n_categories, len(data))).T
            broadcast_means = np.broadcast_to(category_means.to_numpy(), (len(data), n_categories))
            # Measure distance
            diffs = np.abs(broadcast_data - broadcast_means)
            # Find closest
            indices = np.argmin(diffs,axis=1)
            # Fetch category
            indexer = list(category_means.index).__getitem__
            data = pd.Series(indices).apply(indexer)
        elif '#' in column:
            # Undo the Between constraint
            column = column.split('#',1)[0]
            data = reverse_constraint(data, constraint_range)
        # The 'y' column and other numeric / non-categorical data are not transformed
        output[column] = data
    return pd.DataFrame(output)
output = reverse_transform(mimic, n_categories, category_means, (constraint_low, constraint_high))

########################################################################
#                                                                      #
#                 MIMIC SDV-STYLE CONDITIONAL SAMPLING                 #
#                       WITH THE GAUSSIAN COPULA                       #
#                              IN 3 STEPS                              #
# ie: output = model.sample(num_rows=n_samples, conditions=conditions) #
#                                                                      #
########################################################################

# Prepare sampling conditions
condition_value = 3
conditions = [Condition({'size': condition_value}, num_rows=10)]

import pdb
pdb.set_trace()

# Sample some data
sampled = model.sample_conditions(conditions)

'''*************************************************************************
*                                                                          *
*                     STEP (1): TRANSFORM THE CONDITION                    *
* ie: model._metadata.transform(pd.DataFrame({'size': [condition_value]})) *
*                                                                          *
*************************************************************************'''

def forward_constraint(data, constraint_range):
    data = (data - constraint_range[0]) / (constraint_range[1] - constraint_range[0])
    data = (data * 0.95) + 0.025
    return np.log(data / (1.0 - data))

manual_condition = forward_constraint(condition_value, (constraint_low, constraint_high))

'''***************************************************************************
*                                                                            *
*                    STEP (2): INITIAL SAMPLING FROM GC                      *
*                                                                            *
*                  MIMIC = conditional sampling using GC                     *
* ie: mimic = model._model.sample(num_rows=n_samples, conditions=conditions) *
*                                                                            *
***************************************************************************'''

# Identify the CDF for all sampled data
def conditional_GC_sampling(n_samples, n_columns, covariance, transformed_condition):
    random_samples = np.random.multivariate_normal(np.zeros(n_columns), covariance, size=n_samples)
    return stats.norm.cdf(random_samples)

conditional_random_cdf = conditional_GC_sampling(n_samples, len(source_data.columns), model_covariance, transformed_condition)

# Alter columns for SDV compatibility
columns = source_data.columns.tolist()
columns[columns.index('size')] = 'size#1#3'
mimic = make_sampled_frame(columns, model_univariates, conditional_random_cdf)

'''******************************************************
*                                                       *
*         STEP (3): REVERSE TRANSFORM SAMPLES           *
*                                                       *
*   OUTPUT = reversed data from GC --> resemble source  *
* ie: output = model._metadata.reverse_transform(mimic) *
*                                                       *
*             SAME AS UNCONDITIONAL SAMPLING            *
*                                                       *
******************************************************'''

output = reverse_transform(mimic, n_categories, category_means, (constraint_low, constraint_high))

