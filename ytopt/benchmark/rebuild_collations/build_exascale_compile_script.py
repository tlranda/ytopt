import os
import subprocess
import argparse
import pathlib

import pandas as pd

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--collation-reference', required=True, help="CSV that defines collation sources and information")
    prs.add_argument('--include-base', default="/home/trandall/ytune_2022/ytopt_tlranda/ytopt/benchmark/", help="Include directories to utilize in compilation (will be auto-suffixed with '{collation_reference.stem.split('_collated',1)[0]}_exp'")
    prs.add_argument("--mode", choices=['AMG','RSBench','XSBench','SW4Lite'], help="Which exascale compiler statements to use for building")
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
        if args.mode == 'SW4Lite':
            cmd_template = "nvcc -O3 -x cu -I{kernel_dir}/src -c -dc -arch=sm_80 -DSM4_CROUTINES "+\
            "-DSW4_CUDA -DSW4_NONBLOCKING -ccbin mpicxx -Xptxas -v -Xcompiler -fopenmp "+\
            "-DSW4_OPENMP -I{kernel_dir}/src/double -Xcompiler -std=c++11 -c {template} "+\
            "-Xcompiler -S -Xcompiler -emit-llvm"

            # For full compilation, we needed to link everything together, but I don't think that
            # will be strictly necessary for our use case?
            #
            #"nvcc -Xcompiler -fopenmp -Xlinker -ccbin mpicxx -o {output} main.o {template_o} "+\
            #"Source.o SuperGrid.o GridPointSource.o time_functions_cu.o ew-cfromfort.o EW_cuda.o "+\
            #"Sarray.o device-routines.o EWCuda.o CheckPoint.o Parallel_IO.o EW-dg.o "+\
            #"MaterialData.o MaterialBlock.o Polynomial.o SecondOrderSection.o TimeSeries.o "+\
            #"sacsubc.o curvilinear-c.o -lcuda -lnvpumath -lnvc -lcudart -llapack -lm -lblas -lgfortran"
        elif args.mode in ['XSBench','RSBench']:
            cmd_template = "{clang} {template} {kernel_dir}/material.c {kernel_dir}/utils.c "+\
            "-I{kernel_dir} -fopenmp -DOPENMP -fno-unroll-loops -O3 -mllvm -polly "+\
            "-mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math -lm "+\
            "-march=native -I/lcrc/project/EE-ECP/jkoo/sw/clang13.2/llvm-project/"+\
            "release_pragma-clang-loop/projects/openmp/runtime/src -S -emit-llvm"
        elif args.mode == "AMG":
            cmd_template = "mpicc -fopenmp -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm "+\
                      "-polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math "+\
                      "-march=native {template} -I{kernel_dir}/ -I{kernel_dir}/struct_mv "+\
                      "-I{kernel_dir}/sstruct_mv -I{kernel_dir}/IJ_mv -I{kernel_dir}/seq_mv "+\
                      "-I{kernel_dir}/parcsr_mv -I{kernel_dir}/utilities -I{kernel_dir}/parcsr_ls "+\
                      "-I{kernel_dir}/krylov -DTIMER_USE_MPI -DHYPRE_USING_OPENMP "+\
                      "-DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -DHYPRE_USING_PERSISTENT_COMM "+\
                      "-DHYPRE_HOPSCOTCH -DHYPRE_BIGINT -DHYPRE_TIMING -L{kernel_dir}/parcsr_ls "+\
                      "-L{kernel_dir}/parcsr_mv -L{kernel_dir}/IJ_mv -L{kernel_dir}/seq_mv "+\
                      "-L{kernel_dir}/sstruct_mv -L{kernel_dir}/struct_mv -L{kernel_dir}/krylov "+\
                      "-L{kernel_dir}/utilities -lparcsr_ls -lparcsr_mv -lseq_mv -lsstruct_mv "+\
                      "-lIJ_mv -lHYPRE_struct_mv -lkrylov -lHYPRE_utilities -lm -S -emit-llvm"
    elif args.AS or args.DIS:
        cmd_template = "{} {} -o {}"
    else:
        raise NotImplemented
    basic_path = args.collation_reference.with_name(args.collation_reference.stem.split('_collated',1)[0])
    collation = pd.read_csv(args.collation_reference)
    with open(basic_path.with_name(basic_path.stem+'_compile.sh'), 'w') as f:
        for fname in sorted(basic_path.iterdir()):
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
                expect = fname.with_suffix('.ll' if args.IR else '')
                special_include = args.include_base / (args.collation_reference.stem.split('_collated',1)[0]+"_exp/")
                if args.mode in ['XSBench','RSBench']:
                    cmd = cmd_template.format(clang=args.clang,
                                              kernel_dir=special_include,
                                              template=fname)
                elif args.mode in ['AMG', 'SW4Lite']:
                    cmd = cmd_template.format(kernel_dir=special_include,
                                              template=fname)
                else:
                    raise NotImplemented
                    cmd = cmd_template.format(args.clang,
                                              fname,
                                              special_include / "polybench.c",
                                              special_include,
                                              size,
                                              fname.with_suffix(''))
            f.write(f"if [ -f '{expect}' ]; then\n")
            f.write(f"   echo '{expect} exists';\n")
            f.write( "else\n")
            f.write(f'    echo "{cmd}"'+"\n")
            f.write( '    '+cmd+"\n")
            if args.IR:
                if args.mode == 'SW4Lite':
                    f.write(f"    if [ $? -ne 0 ]; then exit; else mv *.o {fname.with_suffix('.ll')}; fi;\n")
                else:
                    f.write(f"    if [ $? -ne 0 ]; then exit; else ( mv mmp_*.ll {fname.with_suffix('.ll')}; rm -f *.ll ) fi;\n")
            else:
                f.write("    if [ $? -ne 0 ]; then exit; fi;\n")
            f.write( 'fi\n')

if __name__ == '__main__':
    main()

