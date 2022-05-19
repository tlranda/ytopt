import os, uuid, re, time, subprocess
import torch # Currently used for serialization

"""
    Expected usage:
    * Override the runString() call to produce the string command that evaluates the outputfile
        + If your benchmark reports its own measurement, override getTime() call to extract it from the process information
            * If you don't want to use the best-case time as objective value, override metric()
    * If your benchmark requires compilation, subclass this and override the compileString() call
        + If you do not use compilation but need to make use of plotValues(), set force_plot=True on initialization
        + Create a findReplaceRegex object to handle plotting values and pass them in for initialization
            * find should be a list of regexes to match
            * prefix|suffix should be a list of tuples of equal length, where each tuple has 2 strings
                + As of now, only static strings are supported (not regexes)
                + The first string is matched from the original input (removed)
                + The second string is replaced in the new output (substitution for removed)
    * If your benchmark needs to make startup checks for any of the following, override initChecks():
        + Host or GPU architecture
        + Presence of CUDA in source code (cache in self.buffer if you don't mind)
        + Compiler availability or dependent default flags for the compiler

    * Note that (args, kwargs) propagation pattern in findRuntime() allows for access to additional parameters as such:
      + compileString()
      + execute()
        * getTime()
        * runString()
      As such, these functions should have UNIQUE (args, kwargs) unless it is INTENDED for them to be shared across
      some or all of the above. This is intended for transparency if these functions are overridden and need additional
      information that is not tracked via the Plopper's self
"""

class findReplaceRegex:
    empty_from_to = [tuple(["",""])]
    """
        * find should be a list of regexes to match
        * prefix|suffix should be a list of tuples of equal length, where each tuple has 2 strings
            + As of now, only static strings are supported (not regexes)
            + The first string is matched from the original input (removed)
            + The second string is replaced in the new output (substitution for removed)
    """

    def __init__(self, find, prefix=None, suffix=None):
        if type(find) is str:
            find = tuple([find])
        self.find = find
        self.nitems = len(self.find)
        REQUIRED_ELEMS = 2*self.nitems

        # Repeated code for each of these attributes
        for attrName, passedValue in zip(['prefix', 'suffix'], [prefix, suffix]):
            # Be nice about wrapping/replacing default values
            if passedValue is None:
                passedValue = findReplaceRegex.empty_from_to
            elif type(passedValue) is tuple and type(passedValue[0]) is str:
                passedValue = [passedValue]
            # Validation of required length for each find-regex
            nAttrItems = sum(map(len, passedValue))
            if nAttrItems != REQUIRED_ELEMS:
                raise ValueError(f"{attrName} must have 2-element tuple per element in the find regex list (got {nAttrItems}, needed {REQUIRED_ELEMS})")
            else:
                self.__setattr__(attrName, passedValue)

        # Magic variables that can try to predict common use patterns and ease function paramaterization
        self.iter_idx = None
        self.invert_direction = 0

    def __iter__(self):
        # Enumeration just to set up the magic variable
        for idx, regex in enumerate(self.find):
            self.iter_idx = idx
            yield regex

    def replace(self, match, to, string):
        # Automatically handle the expected replacement patterns
        if to is None or to == "":
            return re.sub(self.wrap(match, noInvert=True), "", string)
        else:
            return re.sub(self.wrap(match), self.wrap(to), string)

    def wrap(self, wrap, direction=None, idx=None, noInvert=False):
        # When direction|idx are None, attempt to use magic variables to predict correct output
        # Actual values passed to direction may be ['from'==0,'to'==1]
        # Actual values passed to idx may be in the range of values in self.find
        if direction is None:
            direction = self.invert_direction
        if type(direction) is str:
            if direction.lower() == 'from':
                direction = 0
            elif direction.lower() == 'to':
                direction = 1
        if direction not in [0, 1]:
            raise ValueError(f"Could not parse direction '{direction}', must be in ['from', 'to'] or [0, 1]")
        if idx is None:
            if self.iter_idx is None:
                if self.nitems == 1:
                    # Only case where these can both be None and we unambiguously match user expectations
                    idx = 0
                else:
                    raise ValueError(f"Index to wrap is poorly defined! Please define an index")
            else:
                idx = self.iter_idx
        # Magic updates for next call to match (usually expect opposite direction)
        if not noInvert:
            self.invert_direction = int(not direction)
        return self.prefix[idx][direction] + wrap + self.suffix[idx][direction]


class Plopper:
    def __init__(self, sourcefile, outputdir=None, output_extension='.tmp',
                 evaluation_tries=3, retries=0, findReplace=None,
                 infinity=1, force_plot=False, **kwargs):
        self.sourcefile = sourcefile # Basis for runtime / plotting values
        self.kernel_dir = os.path.abspath(self.sourcefile[:self.sourcefile.rfind('/')])

        if outputdir is None:
            # Use CWD as basis
            outputdir = os.path.abspath(".")
        self.outputdir = outputdir+"/tmp_files" # Where temporary files will be generated
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        self.output_extension = output_extension # In case compilers are VERY picky about the extension on your intermediate files
        self.evaluation_tries = evaluation_tries # Number of executions to average amongst
        self.retries = retries # Number of failed evaluations to re-attempt before giving up
        if findreplace is not None and type(findReplace) is not findReplaceRegex:
            raise ValueError("Only support findReplaceRegex type for the findReplace attribute at this time")
        self.findReplace = findReplace # findReplaceRegex object
        self.infinity = infinity # Very large value to return on failure to compile or execute
        self.force_pot = force_plot # Always utilize plotValues() even if there is no compilation string

        self.buffer = None
        # If your plopper requires additional startup checks or attribute initialization, just override
        # the initChecks() call below with any necessary arguments coming from keyword arguments
        # This may include things such as checking for CUDA in the source file (go ahead and cache it in self.buffer if so),
        # tracking the host architecture, GPU architecture, or determining what compiler or basic compiler options to use
        self.initChecks(**kwargs)

    def initChecks(self, **kwargs):
        pass

    def compileString(self, outfile, *args, **kwargs):
        # Return empty string to skip compilation
        # Override with compiling string rules to make a particular compilation occur (includes plotValues)
        # Final executable MUST be written to `outfile`
        return ""

    def runString(self, outfile, *args, **kwargs):
        # Return the string used to execute the attempt
        # outfile is the temporary filename that is generated for this particular instance, ignore it if no compilation/plotted values were used
        # Override as needed
        return self.kernel_dir + outfile

    # PLANNED CHANGES:
    # Use regex for all changes (similar to commented out section), requires API change to supply REGEX from/to somewhere
    # Replace the Markers in the source file with the corresponding values
    def plotValues(self, dictVal, outputfile, findReplace=None):
        if findReplace is None:
            if self.findReplace is None:
                raise ValueError("PlotValues behavior not defined by findReplaceRegex!")
            findReplace = self.findReplace

        # Use cache to save I/O time on repeated plots
        if self.buffer is None:
            with open(self.sourcefile, "r") as f1:
                self.buffer = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in self.buffer:
                # For each regex in the findReplaceRegex object
                for idx, find in enumerate(findReplace):
                    # While it matches in the line
                    while re.search(find, line):
                        # Cache substitutions as they may appear multiple times in a line, but will all be handled on first encounter
                        foundGroups = []
                        for m in re.finditer(find, line):
                            match = m.group(1)
                            if match in foundGroups:
                                continue
                            line = findReplace.replace(match, dictval[match], line)
                            foundGroups.append(match)
                f2.write(line)

    def getTime(self, process, *args, **kwargs):
        # Define how to recover self-attributed objective values from the subprocess object
        # Return None to default to the python-based time library's timing of the event
        try:
            return float(process.stdout.decode('utf-8'))
        except ValueError:
            try:
                return float(process.stderr.decode('utf-8'))
            except ValueError:
                return None

    def metric(self, timing_list):
        # Allows for different interpretations of repeated events
        # Defaults to best-case scenario
        return min(timing_list)

    def execute(self, outfile, *args, **kwargs):
        times = []
        failues = 0
        while failures <= self.retries and len(times) < self.evaluation_tries:
            start = time.time()
            execution_status = subprocess.run(self.runString(outfile, *args, **kwargs), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            duration = time.time() - start
            # Find the execution time
            derived_time = self.getTime(execution_status, *args, **kwargs)
            if derived_time is not None:
                duration = derived_time
            if duration == 0.0:
                failures += 1
            else:
                times.append(duration)
        # Unable to evaluate this execution
        if failures > self.retries:
            return self.infinity
        return self.metric(times)

    # Function to find the execution time of the interim file, and return the execution time as cost to the search module
    # Additional args provided here will propagate to:
    # * compileString
    # * execute
    #   + getTime
    #   + runString
    def findRuntime(self, x, params, *args, **kwargs):
        # Generate non-colliding name to write outputs to:
        interimfile = self.outputdir+"/"+str(uuid.uuid4())+self.output_extension

        # Generate intermediate file
        dictVal = dict((k,v) for (k,v) in zip(params, x))
        # If there is a compiling string, we need to run plotValues
        compile_str = self.compileString(interimfile, *args, **kwargs)
        if self.force_plot or compile_str != "":
            self.plotValues(dictVal, interimfile, *args, **kwargs)
            # Compilation
            if compile_str != "":
                compilation_status = subprocess.run(compile_str, shell=True, stderr=subprocess.PIPE)
                # Find execution time ONLY when the compiler return code is zero, else return infinity
                if compilation_status.returncode != 0:
                # and len(compilation_status.stderr) == 0: # Second condition is to check for warnings
                    print(compilation_status.stderr)
                    print("Compile failed")
                    return self.infinity
        # Evaluation
        return self.execute(interimfile, *args, **kwargs)

class LazyPlopper(Plopper):
    def __init__(self, *args, cachefile=None, lazySaveInterval=10, **kwargs):
        super().__init__(*args, **kwargs)
        # Make cache available
        if cachefile is None:
            cachefile = "lazyplopper_cache_"+str(uuid.uuid4())+".cache"
        self.cachefile = cachefile
        # Load cache
        if os.path.exists(self.cachefile):
            self.load()
        else:
            self.cache = dict()
        # Define a checkpoint interval to save new information at prior to object deletion
        self.lazySaveInterval = lazySaveInterval
        self.lazySaveCounter = 0

    def __del__(self):
        self.save()

    # Currently implemented using Pytorch serialization. Override these functions to use something else
    def load(self):
        torch.load(self.cachefile)
    def save(self):
        torch.save(self.cache, self.cachefile)

    def findRuntime(self, x, params, *args, **kwargs):
        searchtup = ([x[0], params[0], *args]+[v for (k,v) in kwargs.items()])
        # Lazy evaluation doesn't call findRuntime() when it has seen the runtime before
        if searchtup in self.cache.keys():
            return self.cache[searchtup]
        else:
            rval = super().findRuntime(x, params, *args, **kwargs)
            self.cache[searchtup] = rval
            # Checkpoint new save values every interval to avoid catastrophic loss
            self.lazySaveCounter += 1
            if self.lazySaveCounter >= self.lazySaveInterval:
                self.lazySaveCounter = 0
                self.save()
            return rval

