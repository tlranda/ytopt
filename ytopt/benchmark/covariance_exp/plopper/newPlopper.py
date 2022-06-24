from ytopt.benchmark import base_plopper
# Params take the form 'p#' (ie: p0, p1)... and are expressed in template code as '#P0', '#P1', '#P2'...
# So the P should be part of the capturing group to lookup values, and the template # should be removed as a prefix
regexObject = base_plopper.findReplaceRegex(r"#(P[0-9]+)", prefix=tuple(["#",""]))
# No runtime args

class Covariance_Plopper(base_plopper.Plopper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.findReplace is None:
            self.findReplace = regexObject

    def compileString(self, outfile, dictVal, *args, **kwargs):
        """ Call to findRuntime should be: (x, params, d_size) """
        d_size = args[0]
        # Drop extension in the output file name to prevent clobber
        clang_cmd = "clang "+outfile+" "+self.kernel_dir+"/polybench.c "+\
                    "-I"+self.kernel_dir+" "+str(d_size)+" -DPOLYBENCH_TIME "+\
                    "-std=c99 -fno-unroll-loops -O3 -mllvm -polly "+\
                    "-mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names "+\
                    "-ffast-math -march=native -o "+outfile[:-len(self.output_extension)]
        return clang_cmd

    def runString(self, outfile, dictVal, *args, **kwargs):
        # Extension was dropped for executable, execute using srun
        return outfile[:-len(self.output_extension)]

    def getTime(self, process, dictVal, *arg, **kwargs):
        exe_time_run = [float(s) for s in process.stdout.decode('utf-8').split('\n')[-4:-1]]
        return exe_time_run

