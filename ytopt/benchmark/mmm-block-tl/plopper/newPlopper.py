import os, sys
FETCH_PLOPPER = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../../")
if FETCH_PLOPPER not in sys.path:
    sys.path.append(FETCH_PLOPPER)
import base_plopper

class MMM_Plopper(base_plopper.LazyPlopper):
    """
        Call to findRuntime should be:
            (x, params, d_size)
    """
    def initChecks(self, **kwargs):
        self.use_exe_perl = self.evaluation_tries == 1

    def compileString(self, outfile, dictVal, *args, **kwargs):
        d_size = args[0]

        gcc_cmd = 'g++ '+self.kernel_dir+d_size
        gcc_cmd += f' -DBLOCK_SIZE={dictVal["BLOCK_SIZE"]}'
        gcc_cmd += ' -o ' + outfile
        return gcc_cmd

    def runString(self, outfile, *args, **kwargs):
        if self.use_exe_perl:
            return self.kernel_dir + "/exe.pl " + outfile
        else:
            return outfile

