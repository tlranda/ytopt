from ytopt.benchmark import base_plopper

class _3MM_Plopper(base_plopper.Plopper):
    """ Call to findRuntime should be: (x, params, d_size) """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.findReplace is None:
            self.findReplace = base_plopper.findReplaceRegex(r"#(P[0-9]+)", prefix=tuple(["#",""]))

    def compileString(self, outfile, dictVal, *args, **kwargs):
        d_size = args[0]
        # Drop extension on output file
        clang_cmd = "clang "+outfile+" "+self.kernel_dir+"/polybench.c "+"-I"+self.kernel_dir+\
                    " "+str(d_size)+" -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 "+\
                    "-mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names "+\
                    "-ffast-math -march=native -o "+outfile[:-len(self.output_extension)]
        return clang_cmd

    def runString(self, outfile, dictVal, *args, **kwargs):
        return outfile[:-len(self.output_extension)]

    def getTime(self, process, dictVal, *args, **kwargs):
        return [float(s) for s in process.stdout.decode('utf-8').split('\n')[-4:-1]]

