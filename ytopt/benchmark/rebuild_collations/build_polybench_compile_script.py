import os
import subprocess
import argparse
import pathlib

import pandas as pd
import tqdm

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--collation-reference', required=True, help="CSV that defines collation sources and information")
    prs.add_argument('--include-base', default="/home/trandall/ytune_2022/ytopt_tlranda/ytopt/benchmark/", help="Include directories to utilize in compilation (will be auto-suffixed with '{collation_reference.stem.split('_collated',1)[0]}_exp'")
    prs.add_argument('--clang', default='/lcrc/project/EE-ECP/jkoo/sw/clang13.2/release_pragma-clang-loop/bin/clang', help="Clang to use (default: Custom Swing system path)")
    prs.add_argument('--IR', action='store_true', help="Only build IR's")
    prs.add_argument('--AS', action='store_true', help="Only do IR assembly")
    prs.add_argument('--DIS', action='store_true', help="Only do Bitcode Disassembly")
    prs.add_argument('--with-opt', action='store_true', help="Add optimization to Bitcode Disassembly")
    prs.add_argument('--dry-script', action='store_true', help="Generated script only checks if files exist or not")
    prs.add_argument('--optimization-levels', choices=['3','2','1','0'], default=None, nargs='*', help="Optimization level(s) to build for (default: O3)")
    return prs

def parse(prs=None, args=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # Don't do more than one of these at once
    exclusives = ["IR", "AS", "DIS"]
    if sum([getattr(args, e) for e in exclusives]) > 1:
        multipleExclusiveOptions = f"Can only specify one of {', '.join(['--'+e for e in exclusives])} at a time"
        raise ValueError(multipleExclusiveOptions)
    # Need to be a Path objects
    args.collation_reference = pathlib.Path(args.collation_reference)
    args.clang = pathlib.Path(args.clang)
    args.include_base = pathlib.Path(args.include_base)
    # Keep relative path
    if args.AS:
        args.clang = args.clang.with_name("llvm-as")
    elif args.DIS:
        args.clang = args.clang.with_name("llvm-dis")
    if args.optimization_levels is None:
        args.optimization_levels = '3'
    args.optimization_levels = [f"-O{l}" for l in args.optimization_levels]
    return args

def lookup_size(csv, name, args):
    sizes = {'S': 'SMALL',
             'M': 'MEDIUM',
             'L': 'LARGE',
             'SM': 'SM',
             'ML': 'ML',
             'XL': 'EXTRALARGE'}
    # There can be duplicate IDs, but the size will be the same so just pick the first
    namesize = csv.loc[csv['id'] == str(name.resolve()),'size'].tolist()[0]
    return f'-D{sizes[namesize]}_DATASET'

def main(args=None):
    args = parse(args=args)
    if args.IR:
        cmd_template = "{} {} {} -I{} -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops {} {} "+\
                       "-mllvm -polly -mllvm -polly-process-unprofitable "+\
                       "-mllvm -polly-use-llvm-names -ffast-math -march=native -S -emit-llvm"
    elif args.AS or args.DIS:
        cmd_template = "{} {} -o {}"
    else:
        cmd_template = "{} {} {} -I{} -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops {} "+\
                       "-mllvm -polly -mllvm -polly-process-unprofitable "+\
                       "-mllvm -polly-use-llvm-names -ffast-math -march=native {} -o {}"

    basic_path = args.collation_reference.with_name(args.collation_reference.stem.split('_collated',1)[0])
    print("Load CSV", args.collation_reference)
    collation = pd.read_csv(args.collation_reference)
    print(len(collation), "records loaded")
    output_path = basic_path.with_name(basic_path.stem+'_compile.sh')
    with open(output_path, 'w') as f:
        for fname in tqdm.tqdm(sorted(filter(lambda bp: len(bp.stem.split('_')) == 2, basic_path.iterdir()), key=lambda p: int(p.stem.split('_',1)[1]))):
        #for fname in tqdm.tqdm(sorted(pathlib.Path('Heat3d_JOBS/heat3d_missed').iterdir(), key=lambda p: int(p.stem.split('_',1)[1]))):
            if fname.suffix != '.c':
                continue
            try:
                size = lookup_size(collation, fname, args)
            except:
                if 'JOBS' in fname.parts[0]:
                    fname = fname.relative_to(fname.parts[0])
                try:
                    size = lookup_size(collation,fname, args)
                except:
                    print(f"No CSV record for file", fname)
                    continue
            if 'JOBS' in fname.parts[0]:
                fname = fname.relative_to(fname.parts[0])
            for opt_level in args.optimization_levels:
                if args.AS:
                    cmd = cmd_template.format(args.clang,
                                              fname.with_suffix('.ll'),
                                              fname.with_suffix('.bc'))
                    expect = fname.with_suffix('.bc')
                elif args.DIS:
                    cmd = cmd_template.format(args.clang,
                                              fname.with_suffix('.bc'),
                                              fname.with_name(fname.stem+'_reassembled.ll'))
                    expect = fname.with_name(fname.stem+'_reassembled.ll')
                    if args.with_opt:
                        cmd += f"; {args.clang.with_name('opt')} -S -O3 {fname.with_name(fname.stem+'_reassembled.ll')} -o {fname.with_name(fname.stem+'_optimized.ll')}"
                        expect = fname.with_name(fname.stem+'_optimized.ll')
                else: # normal, args.IR
                    special_include = args.include_base / (args.collation_reference.stem.split('_collated',1)[0]+"_exp/")
                    out_name = fname.with_suffix('')
                    if opt_level != '-O3':
                        out_name = fname.parent / (fname.stem + "_" + opt_level[1:].lower() + fname.suffix)
                    cmd = cmd_template.format(args.clang,
                                              fname,
                                              special_include / "polybench.c",
                                              special_include,
                                              opt_level,
                                              size,
                                              out_name)
                    # Make this drop into bench/.ll
                    expect = out_name.with_suffix('.ll' if args.IR else '')
                f.write(f"if [ -f '{expect}' ]; then\n")
                f.write(f"   echo '{expect} exists';\n")
                f.write( "else\n")
                if args.dry_script:
                    f.write(f"     echo '!! {expect} does NOT exist';\n")
                else:
                    f.write(f'    echo "{cmd}"'+"\n")
                    f.write( '    '+cmd+"\n")
                    if args.IR:
                        f.write(f"    if [ $? -ne 0 ]; then exit; else rm -f polybench.ll; mv *.ll {expect}; fi;\n")
                    else:
                        f.write("    if [ $? -ne 0 ]; then exit; fi;\n")
                f.write( 'fi\n')
                if args.AS or args.DIS:
                    # These ones don't need to loop optimization levels and I'm not rewriting the structure to fix that
                    break
    print("Script written to", output_path)

if __name__ == '__main__':
    main()

