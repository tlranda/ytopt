# Conditional Sampling for Gaussian Copulas

## What is Conditional Sampling?

Conditional sampling is a means of distribution sampling that _guarantees_ specified conditions will be simultaneously held for all generated data, if possible.
This is performed through two mechanisms: _constraints_ and _conditions_.

_Constraints_ are defined limitations that all data must adhere to at all times.
In theory, any constraint relevant to the sampled distribution can be specified, but we refer to the [Synthetic Data Vault's constraints](https://docs.sdv.dev/sdv/reference/constraint-logic/predefined-constraint-classes) as concrete implementations of a broad variety of useful constraints.
These constraints include requiring values to be either positive or negative, fixing upper and/or lower bounds to generated ranges, one-hot encoding over multiple columns, etc.

For our work, we focus on constraining a range of values, such as the range of task sizes, _t_ that can be generated (at the time of development, these were called "Between" constraints; at the time of writing this guide, they are now referred to as "ScalarRange" constraints).
SDV also provides the capability to [define your own constraints](https://docs.sdv.dev/sdv/reference/constraint-logic/custom-logic) if existing APIs do not suit your needs.
Constraints ensure that proper bounds are known for model fitting so that training data does not over-specify model behavior -- especially when transferring to out-of-distribution (extrapolated) trends.

_Conditions_ are more specific constrained requirements imposed during sampling; they only apply to a particular instance of synthetic data generation.
In SDV, this is accomplished by [defining a Condition object and a number of instances that must adhere to the specified condition](https://docs.sdv.dev/sdv/single-table-data/sampling#simulate-scenarios).

In our work, conditions are used to specify an _exact_ task to be sampled during model inference.

## Implementation Specifics: Constraints

### Defining Constraints for a Benchmark

Using SDV, we can specify a constraint on the task column that limits the range of generated data between an upper and lower bound.
This requires us to have some notion of a "greatest" and "least" task, which in our work typically corresponds to the scale of input provided during autotuning.
The specific values of this range are manually defined to encapsulate all tasks we intend to use as either transfer-learning sources or transfer-learning targets.

As a concrete example, refer to [the 3mm benchmark's .h file](benchmark/_3mm_exp/3mm.h).
Each "dataset" defines the _I,J,K,L,_ and _M_ sizes for array dimensions in this kernel, ranging from "Mini" (16,18,20,22,24) to "Huge" (3200,3600,4000,4400,4800).
The input sizes do not have linear scaling and the work done by the kernel has cubic scaling with input size, so we do not use categorical encoding for the task sizes.
Rather, we arbitrarily pick the size _I_ as an integer indication of the work to be done, such that our constraint specifies that tasks have values inclusively between 16 and 3200.
The sizes for each dataset are represented in [the problem specification for each benchmark](https://github.com/tlranda/ytopt/blob/08c81ba62b5c2209ef6f30b6a772d1053f234463/ytopt/benchmark/_3mm_exp/problem.py#L59).
In our code, this dictionary maps the integer size to a tuple of string names that can be used to identify the size.
Therefore the SDV constraints for a problem are defined by the minimum and maximum for task sizes, as we've done for our [Polybench problem factory](https://github.com/tlranda/ytopt/blob/08c81ba62b5c2209ef6f30b6a772d1053f234463/ytopt/benchmark/base_problem.py#L315).

### Satisfying Constraints During Model Training

Constraints have to satisfied at all times, including model training, so we gather constraints before fitting to data and [use them in the SDV object constructor](https://github.com/tlranda/ytopt/blob/08c81ba62b5c2209ef6f30b6a772d1053f234463/ytopt/benchmark/base_online_tl.py#L304).
As such, when we represent data to the Gaussian Copula model, every record _must_ satisfy all constraints.

We add the integer size for _t_ to each collected sample indicating the source task scale.
Since our factories are capable of representing the task size in their assembled objects, we load an appropriate object and [reference its data to locate previous autotuning records from disk](https://github.com/tlranda/ytopt/blob/08c81ba62b5c2209ef6f30b6a772d1053f234463/ytopt/benchmark/base_online_tl.py#L538).
After acquiring the source tuning data, we add the `problem_class` attribute [to each loaded record](https://github.com/tlranda/ytopt/blob/08c81ba62b5c2209ef6f30b6a772d1053f234463/ytopt/benchmark/base_online_tl.py#L574), so each source task is properly annotated based on the same definitions used to collect the data.

Since a constraint is applied to the task size, the Gaussian Copula will not learn directly from these input records.
SDV will automatically transform all data based on the constraint according to a reversible logit transform:

$data = 0.95 \times \frac{data-low}{high-low} + 0.025$

$data = ln \lparen\frac{data}{1 - data}\rparen$

This nonlinear representation selects the following points for 3mm task sizes:
