from ytopt.benchmark import base_plopper

# Params take the form 'P#' (ie: P0, P1...) but are expressed in template code as '#P0', '#P1', ....
# So the P should be part of the capturing group to lookup values, and the template # should be removed as a prefix
find = r"#(P[0-9]+)"
prefix_transform = tuple(["#", ""])
regexObject = base_plopper.findReplaceRegex(find, prefix=prefix_transform)

# This information used to be put in different exe.pl files
# Note that below you can still automatically call the exe*.pl file of your choice when evaluation_tries == 1
runtime_args = {'_s': " -m event -l 100000 >/dev/null 2>&1",
                '_m': " -m event -l 1000000 >/dev/null 2>&1",
                '_l': " -m event -l 5000000 >/dev/null 2>&1",
                '_xl': "-m event -l 10000000 >/dev/null 2>&1",
                '': " >/dev/null 2&>1"}

class XSBench_Plopper(base_plopper.Plopper):
    """
        Call to findRuntime should be:
            (x, params)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Automatically load the desired findReplaceRegex we created above unless overridden
        if self.findReplace is None:
            self.findReplace = regexObject
        # Track the size for the right completion to the perl script name
        self.use_exe_perl = self.evaluation_tries == 1
        if 'exe_size' in kwargs.keys():
            self.exe_size = kwargs['exe_size']
        else:
            self.exe_size = ''
        # Use the same logic to locate the exact run configuration if timing using Python instead of a perl script
        self.runtime_args = runtime_args[self.exe_size]

    def compileString(self, outfile, dictVal, *args, **kwargs):
        gcc_cmd = "gcc -std=gnu99 -Wall -flto  -fopenmp -DOPENMP -O3 "
        # Drop the extension for output file
        gcc_cmd += " -o " + outfile[:-len(self.output_extension)] + " "
        gcc_cmd += outfile + " " + self.kernel_dir + "/Materials.c "
        gcc_cmd += self.kernel_dir + "/XSutils.c " + " -I"
        gcc_cmd += self.kernel_dir + " -lm -L${CONDA_PREFIX}/lib"
        return gcc_cmd

    def runString(self, outfile, *args, **kwargs):
        # Extension was dropped for executable
        binary = outfile[:-len(self.output_extension)]
        if self.use_exe_perl:
            return self.kernel_dir + "/exe" + self.exe_size + ".pl " + binary
        else:
            return binary + self.runtime_args

