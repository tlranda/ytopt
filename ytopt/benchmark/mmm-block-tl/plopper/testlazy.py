import plopper, os, time

dirpath = os.path.abspath('../mmm_problem')

NCALLS = 30
SETUP = (dirpath+'/mmm_block_100.cpp', dirpath)
PROBLEM = ([30], ["BLOCK_SIZE"], '/mmm_block_100.cpp')

# Basic plopper calls
print("Test basic plopper")
bpsr = time.time()
obj = plopper.Plopper(*SETUP)
times = []
for _ in range(NCALLS):
    times.append(obj.findRuntime(*PROBLEM))
bpsp = time.time()
print(f"Basic plopper finds in {bpsp-bpsr} seconds: {times}")

# Lazy plopper calls
print(f"Test lazy plopper")
lpsr = time.time()
obj2 = plopper.LazyPlopper(*SETUP)
times = []
for _ in range(NCALLS):
    times.append(obj2.findRuntime(*PROBLEM))
lpsp = time.time()
print(f"Lazy plopper finds in {lpsp-lpsr} seconds: {times}")

