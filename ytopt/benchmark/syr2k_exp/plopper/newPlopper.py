import os, sys
FETCH_PLOPPER = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../../")
if FETCH_PLOPPER not in sys.path:
    sys.path.append(FETCH_PLOPPER)
import base_plopper

# Params take the form 'p#' (ie: p0, p1)... and are expressed in template code as '#P0', '#P1', '#P2'...
# So the P should be part of the capturing group to lookup values, and the template # should be removed as a prefix
find = r"#(P[0-9]+)"
prefix_transform = tuple(["#", ""])
regexObject = base_plopper.findReplaceRegex(find, prefix=prefix_transform)

# No runtime args

class Syr2k_Plopper(base_plopper.Plopper):
    """
        Call to findRuntime should be:
            (x, params, d_size)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.findReplace is None:
            self.findReplace = regexObject

    def compileString(self, outfile, dictVal, *args, **kwargs):
        d_size = args[0]
        # Drop the extension for output file
        clang_cmd = "clang "+outfile+" "+self.kernel_dir+"/polybench.c "+\
                    "-I"+self.kernel_dir+" "+str(d_size)+" -DPOLYBENCH_TIME "+\
                    "-std=c99 -fno-unroll-loops -O3 -mllvm -polly "+\
                    "-mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names "+\
                    "-ffast-math -march=native -o "+outfile[:-len(self.output_extension)]
        return clang_cmd

    def runString(self, outfile, dictVal, *args, **kwargs):
        # Extension was dropped for executable name
        # Base version was set to go via srun
        return "srun -n 1 "+outfile[:-len(self.output_extension)]

    def getTime(self, process, dictVal, *args, **kwargs):
        # Base version extracted objective from the process itself
        try:
            exe_time_run = [float(s) for s in process.stdout.decode('utf-8').split('\n')[-4:-1]]
        except (IndexError, ValueError):
            print(f"Failed to get time for {dictVal}")
            return None
        # Seems likely to be bad this shouldn't be a list right lol
        return exe_time_run

