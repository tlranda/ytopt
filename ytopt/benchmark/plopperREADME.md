# Refactored Plopper API

The refactored plopper API is based on this directory's `base_plopper.py`.
Features of the rework are detailed below, with usage examples and patterns detailed throughout.

Chapters:

* [Plopper Rework](plopperREADME.md#plopper-rework)
  + [SIMPLIFIED DEVELOPMENT](plopperREADME.md#simplified-development)
  + [BENEFITS](plopperREADME.md#benefits)
  + [FULLY-IMPLEMENTED EXAMPLES](plopperREADME.md#fully-implemented-examples)
* [Problem Rework](plopperREADME.md#problem-rework)
* [Automatic Online Experiments](plopperREADME.md#automatic-online-experiments)
* [Plotting/Analysis](plopperREADME.md#plotting-analysis)

# Plopper Rework

## SIMPLIFIED DEVELOPMENT

Most of the heavy lifting is done by the Plopper class, so just subclass it and change the few parts you need.

### Parameterization of Source Files

When changing values in a source file, use a `findReplaceRegex` object and bind it to your class instances after the `super().__init__()` call
```python
# Source file contains strings to replaced that look like '#P0', '#P1', etc
# The group is used to form the parameter value that informs which of the problem parameters replaces this string in the file's text
find = r"#(P[0-9]+)"
# Here, the parameters are expressed as 'P0', 'P1', etc

# If there are extra parts of the matched string as a prefix or suffix, specify how they should be found and replaced here
# The first string should be what you originally find in the source file, the second should be its replacement
prefix_transform = tuple(["#", ""])
# Here, the extra '#' character is simply removed when we make a replacement
# There isn't any leftover suffix, so we can omit the suffix argument, but it is explicitly shown below
regexObject = base_plopper.findReplaceRegex(find, prefix=prefix_transform, suffix=None)

...

class myPlopper(base_plopper.Plopper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.findReplace is None:
            self.findReplace = regexObject
        ...
```

### Objective Values Using Python, Perl (Legacy behavior), or Anything Else

By default, uses the `time` library in Python to time the subprocess execution time as the objective value.
If the process's stdout or stderr produce a float-interpretable value, this becomes the objective value instead.
You can override the `getTime()` method to determine what overriding objective value is used.
```python
def getTime(self, process: subprocess.CompletedProcess, dictVal, *args, **kwargs):
    # Return None to default to Python's timing of the subprocess
    # Return another value to override the objective
```
On a similar note, you can change how repeated trials choose an overall objective value by overriding `metric()`.
The default behavior is to return the minimum objective value.
```python
def metric(self, timing_list):
    # Use mean instead of minimum
    return np.mean(timing_list)
```

### Simply Compile (if needed) and Run

Define a `compileString()` method if you need to utilize compilation.
Default behavior skips a compilation attempt unless the method returns a string.
```python
def compileString(self, outfile, dictVal, *args, **kwargs):
    # May reference dictVal parameterization or other args as needed to customize compilation
    return 'gcc ' + outfile + ' -o ' + outfile[:-len(self.output_extension)]
```

Define a `runString()` method to show what is executed to get the objective value.
```python
def runString(self, outfile, dictVal, *args, **kwargs):
    # Since the above used '-o' with a trimmed outfile name, we should replicate that change here
    binary = outfile[:-len(self.output_extension)]
    if 'runtime_args' in kwargs.keys():
        return binary + kwargs['runtime_args']
    else:
        return binary
```

### Optional Persistence

Derive a plopper from the LazyPlopper rather than the base Plopper object to automatically gain access to persistence behavior: recalling prior evaluations and immediately reporting the previously noted objective value rather than re-evaluating them.
There are a few optional parameters to control serialization frequency and what cache file is loaded (allowing separate caches ie: for different systems)

To use, simply take your existing plopper derivative:
```python
class myPlopper(base_plopper.Plopper):
    ...
```

and use the LazyPlopper instead:
```python
class myPlopper(base_plopper.LazyPlopper):
    ...
```

This feature can also serve as secondary backup for checkpoint-restart-like behavior with the YTOPT search operation, albeit with some overhead from the Ray processes etc.

## BENEFITS

* Most refactored plopper files are shorter than their original, while making the specifics of their operation clearer
* Abstraction is powerful enough to remove the need for Perl timing script variants or Plopper variants. One Plopper can be sufficient to handle all variations of an autotuning application
* Use of the refactored plopper is nearly 100% transparent to problem definitions. For most problems, using a refactored plopper derivative requires two changes:
    1. Change the imported plopper to the derivative class.
    2. Add any initialization arguments you defined for your derived plopper to the object's instantiation call.

## FULLY-IMPLEMENTED EXAMPLES

mmm-block-tl: [OLD_PLOPPER](mmm-block-tl/plopper/plopper.py) --> [NEW\_PLOPPER](mmm-block-tl/plopper/newPlopper.py)
* Simple plopper that just uses compile-time definitions to alter files, but the files are passed in as additional \*args values

xsbench-omp-tl: [OLD\_PLOPPER](xsbench-omp-tl/plopper/plopper.py) --> [NEW\_PLOPPER](xsbench-omp-tl/plopper/newPlopper.py)
* More advanced plopper that makes source code changes and optionally avoids the need to write per-problem-size `exe*.pl` scripts (maintains optional backwards compatiblity with such scripts)
* Note that this plopper is capable of replacing _ALL_ of the plopper variants formerly used in this benchmark, merely by changing the problem size argument during object instantiation.

# Problem Rework

To make things even easier, similar work has been done for problems.
Problem classes can derive from BaseProblem (to be integrated in the autotune package overriding the TuningProblem specification).

In addition to the normal TuningProblem arguments, the problem rework requires:
* Params: A dictionary defining the ConfigSpace transformations for each parameter in the input space
  + Use `BaseProblem.configure_space()` to assist in creating ConfigSpace objects if desired
* Problem Class: An integer defining the scale of this problem instance

Optionally, one may specify:
* silent: Suppress objective print statements
* use\_capital\_params: For when the placeholder in files use capitalized param names but elsewhere they are lowercase

Reworked problems can automatically define useful output names, result destinations, etc.
Typically, a subclass problem only needs to define its input/parameter/output spaces, models, class size and constraints to pass into the initializer.

NOTE: Any constraint expressed should relate to the problem class, and should be VALID for any valid problem class. As such, when deriving from base\_problem, it is recommended to supply a class-based constraint to apply to all class instances.

For examples, refer to [mmm-block-tl](mmm-block-tl/mmm_problem/NEW_s.py) and [xsbench-omp-tl](xsbench-omp-tl/xsbench/NEW_s.py) examples.
The latter includes a special case objective override that is instructive if your problem makes use of the \*args, \*\*kwargs pattern for its objective.

# Automatic Online Experiments

Using all of the above, you can easily convert any set of previously executed offline problems into an online experiment using `base_online_tl.py`.
Target problems do not need to be trained offline, and can be completely cold when evaluated.
For the script to work, problems need to be compatible with BaseProblem, as internal data are referenced to ensure the online experiment is set up correctly.

NOTE: If your problems are not available as python-installed modules, you may need to execute from the directory containing your modules in order to load them correctly.

# Plotting Analysis

The `plot_analysis.py` script interprets the _online_ traces of different experiments.
`python plot_analysis.py --help` is instructive for running the script, but a few other handy items are detailed below.

The script can calculate means/standard deviations automatically for experiments if they follow the same naming convention except for an integer near the end of the string (hard-coded to be before extension or '\_ALL' and the extension. Adding your seed value to output names for online learning will make this run much easier immediately after your experiment.

