import argparse
import os
import pathlib
import re
import subprocess

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("remote_hostname", help="Remote hostname to transfer documents to (can be SSH alias)")
    prs.add_argument("remote_path", type=pathlib.Path, help="Path on remote host to deposit documents at")
    prs.add_argument("benchmark", type=pathlib.Path, help="Benchmark that has {benchmark}_collated.csv and {benchmark}/mmp_#.* to transfer")
    prs.add_argument("from_", metavar="from", type=int, help="Minimum mmp_# to transfer (all >= this are copied)")
    prs.add_argument("--test-list", action="store_true", help="Only list the local files found matching benchmark + from criteria")
    prs.add_argument("--test-scp", action="store_true", help="Only show what the SCP command may look like with possibly truncated file list")
    prs.add_argument("--test-trunc", type=int, default=None, help="Truncate test outputs after this many entries to make them easier to read")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    args.remote_path = pathlib.Path(f"{args.remote_hostname}:{args.remote_path}")
    return args

def main():
    args = parse()

    # Have individual benchmarks in different case names now
    try:
        base_benchmark = [_ for _ in pathlib.Path('.').iterdir()
                            if _.is_dir() and _.stem.lower() == str(args.benchmark).lower()+'_jobs'][0]
    except:
        raise ValueError("Could not find JOBS directory for benchmark")
    os.chdir(base_benchmark)

    # Fetch the collation CSV
    collation = pathlib.Path(f"{args.benchmark}_collated.csv")
    if not collation.exists():
        raise ValueError(f"Expected collation CSV '{collation}', but did not find it")
    # Fetch the >= MMP files
    mmps = list()
    if not (args.benchmark.exists() and args.benchmark.is_dir()):
        raise ValueError(f"Expected benchmark directory '{relative_to}', but not found or not directory")
    get_numeric = re.compile(r"(?:\D*)(\d+)(?:.*)")
    for fname in args.benchmark.iterdir():
        numeric_match = re.match(get_numeric, fname.stem)
        if numeric_match is None:
            continue
        if int(numeric_match.groups()[0]) >= args.from_:
            mmps.append(fname)
    mmps = sorted(mmps)
    command0 = ['scp',str(collation),str(args.remote_path)]
    command1 = ['scp']+[str(_) for _ in mmps]+[str(args.remote_path / args.benchmark)]
    if args.test_list:
        print(command0[1])
        print("\n".join(command1[1:-1][:args.test_trunc]))
        return
    if args.test_scp:
        print(" ".join(command0))
        print(" ".join(command1[:-1][:(args.test_trunc+1) if args.test_trunc is not None else None]+[command1[-1]]))
        return
    print(" ".join(command0))
    subprocess.run(command0, shell=True)
    print(" ".join(command1))
    subprocess.run(command1, shell=True)

if __name__ == '__main__':
    main()

