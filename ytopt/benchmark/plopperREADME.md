# Refactored Plopper API

The refactored plopper API is based on this directory's `base\_plopper.py`.
Features of the rework are detailed below, with usage examples and patterns detailed throughout.

## SIMPLIFIED DEVELOPMENT

Most of the heavy lifting is done by the Plopper class, so just subclass it and change the few parts you need.

### Initialization, One-and-Done Checks

If your problem needs to make some initialization changes, it is recommended to make them by overriding `initChecks()`:
```python
class myPlopper(base_plopper.Plopper):
    def initChecks(self, **kwargs):
        if 'myAttr' in kwargs.keys():
            # Do things based on this attribute passed to __init__()
        # Can also do one-off environment checks here
        import os
        try:
            self.device = 'cuda:'+os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
        except KeyError:
            self.device = 'cpu'
        ...
```

### Parameterization of Source Files

When changing values in a source file, use a `findReplaceRegex` object and bind it to your class instances during the `initChecks()` call
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
    def initChecks(self, **kwargs):
        self.findReplace = regexObject
        ...
```

### Objective Values Using Python, Perl (Legacy behavior), or Anything Else

By default, uses the `time` library in Python to time the subprocess execution time as the objective value.
If the process's stdout or stderr produce a float-interpretable value, this becomes the objective value instead.
You can override the `getTime()` method to determine what overriding objective value is used.
```python
def getTime(self, process, dictVal, *args, **kwargs):
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
Default behavior skips a compilation attempt unless the method returns a nonempty string.
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

Derive a plopper from the LazyPlopper rather than the base Plopper object to automatically gain access to powerful persistence behavior.
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
* Use of the refactored plopper is nearly 100% transparent. Change the imported plopper to the refactored one. Add any initialization arguments you defined to the object's instantiation call. Done.

## FULLY-IMPLEMENTED EXAMPLES

mmm-block-tl: [OLD\ PLOPPER](https://github.com/tlranda/ytopt/blob/plopper_refactor/ytopt/benchmark/mmm-block-tl/plopper/plopper.py) --> [NEW\_PLOPPER](https://github.com/tlranda/ytopt/blob/plopper_refactor/ytopt/benchmark/mmm-block-tl/plopper/newPlopper.py)
* Simple plopper that just uses compile-time definitions to alter files, but the files are passed in as additional \*args values

xsbench-omp-tl: [OLD\_PLOPPER](https://github.com/tlranda/ytopt/blob/plopper_refactor/ytopt/benchmark/xsbench-omp-tl/plopper/plopper.py) --> [NEW\_PLOPPER](https://github.com/tlranda/ytopt/blob/plopper_refactor/ytopt/benchmark/xsbench-omp-tl/plopper/newPlopper.py)
* More advanced plopper that makes source code changes and optionally avoids the need to write per-problem-size `exe*.pl` scripts (maintains optional backwards compatiblity with such scripts)


