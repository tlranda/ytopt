import os
import subprocess
import argparse
import pathlib

import pandas as pd

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--collation-reference', required=True, help="CSV that defines collation sources and information")
    prs.add_argument('--include-base', default="/home/trandall/ytune_2022/ytopt_tlranda/ytopt/benchmark/", help="Include directories to utilize in compilation (will be auto-suffixed with '{collation_reference.stem.split('_collated',1)[0]}_exp'")
    prs.add_argument('--clang', default='/lcrc/project/EE-ECP/jkoo/sw/clang13.2/release_pragma-clang-loop/bin/clang', help="Clang to use (default: Custom Swing system path)")
    prs.add_argument('--IR', action='store_true', help="Only build IR's")
    prs.add_argument('--AS', action='store_true', help="Only do IR assembly")
    prs.add_argument('--DIS', action='store_true', help="Only do Bitcode Disassembly")
    prs.add_argument('--with-opt', action='store_true', help="Add optimization to Bitcode Disassembly")
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
    return args

def lookup_size(csv, name):
    sizes = {'S': 'SMALL',
             'M': 'MEDIUM',
             'L': 'LARGE',
             'SM': 'SM',
             'ML': 'ML',
             'XL': 'XL'}
    # There can be duplicate IDs, but the size will be the same so just pick the first
    namesize = csv.loc[csv['id'] == str(name.resolve()),'size'].tolist()[0]
    return f'-D{sizes[namesize]}_DATASIZE'

def main(args=None):
    args = parse(args=args)
    if args.IR:
        cmd_template = "{} {} {} -I{} -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 "+\
                       "-mllvm -polly -mllvm -polly-process-unprofitable "+\
                       "-mllvm -polly-use-llvm-names -ffast-math -march=native -S -emit-llvm"
    elif args.AS or args.DIS:
        cmd_template = "{} {} -o {}"
    else:
        cmd_template = "{} {} {} -I{} -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 "+\
                       "-mllvm -polly -mllvm -polly-process-unprofitable "+\
                       "-mllvm -polly-use-llvm-names -ffast-math -march=native {} -o {}"

    basic_path = args.collation_reference.with_name(args.collation_reference.stem.split('_collated',1)[0])
    collation = pd.read_csv(args.collation_reference)
    with open(basic_path.with_name(basic_path.stem+'_compile.sh'), 'w') as f:
        for fname in sorted(basic_path.iterdir(), key=lambda p: int(p.stem.split('_',1)[1])):
            if fname.suffix != '.c':
                continue
            size = lookup_size(collation,fname)
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
            else:
                special_include = args.include_base / (args.collation_reference.stem.split('_collated',1)[0]+"_exp/")
                cmd = cmd_template.format(args.clang,
                                          fname,
                                          special_include / "polybench.c",
                                          special_include,
                                          size,
                                          fname.with_suffix(''))
                expect = fname.with_suffix('.ll' if args.IR else '')
            f.write(f"if [ -f '{expect}' ]; then\n")
            f.write(f"   echo '{expect} exists';\n")
            f.write( "else\n")
            f.write(f'    echo "{cmd}"'+"\n")
            f.write( '    '+cmd+"\n")
            if args.IR:
                f.write(f"    if [ $? -ne 0 ]; then exit; else rm -f polybench.ll; mv *.ll {fname.with_suffix('.ll')}; fi;\n")
            else:
                f.write("    if [ $? -ne 0 ]; then exit; fi;\n")
            f.write( 'fi\n')

if __name__ == '__main__':
    main()

