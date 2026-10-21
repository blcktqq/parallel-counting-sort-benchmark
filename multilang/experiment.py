"""Build and compare five counting-sort implementations on identical i32 files.

--quick is a smoke run. The default full run uses 8 scenarios and 5 repetitions.
No network or package installation is performed by this script.
"""
from array import array
from contextlib import ExitStack
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE))
import counting_sort as cs

def tool(name):
    value=os.environ.get({'g++':'CXX','rustc':'RUSTC','go':'GO','node':'NODE'}[name],name)
    path=shutil.which(value)
    if path is None:
        raise RuntimeError(f'Missing compiler {name}. Install the required toolchain or set its environment variable (CXX, RUSTC, GO, NODE).')
    return path

def build():
    binary=ROOT/'bin';binary.mkdir(exist_ok=True)
    commands={
        'C++':[tool('g++'),'-O3','-std=c++17','-pthread',str(ROOT/'count_cpp.cpp'),'-o',str(binary/'count_cpp')],
        'Rust':[tool('rustc'),'-O','--edition=2021',str(ROOT/'count_rust.rs'),'-o',str(binary/'count_rust')],
        'Go':[tool('go'),'build','-o',str(binary/'count_go'),str(ROOT/'count_go.go')],
    }
    versions={}
    for language,command in commands.items():
        flag='version' if language=='Go' else '--version'
        versions[language]=subprocess.check_output([command[0],flag],text=True).splitlines()[0]
        subprocess.run(command,check=True)
    versions['Python']=sys.version
    versions['Node.js']=subprocess.check_output([tool('node'),'--version'],text=True).strip()
    return versions,commands

def python_benchmark(path,lo,k,repeats,ps):
    data=array('i');data.frombytes(path.read_bytes())
    if sys.byteorder!='little':data.byteswap()
    a=data.tolist();del data
    expected=sorted(a)
    rows=[]
    with ExitStack() as stack:
        pools={p:stack.enter_context(cs.ParallelCountingSort(p)) for p in ps}
        configs=[('seq',1)]+[(mode,p) for p in ps for mode in ('warm','cold')]
        def invoke(mode,p):
            if mode=='seq':return cs.counting_sort(a,lo,k)
            if mode=='warm':return pools[p].sort(a,lo,k)
            with cs.ParallelCountingSort(p) as pool:
                return pool.sort(a,lo,k)
        for mode,p in configs:assert invoke(mode,p)==expected
        rng=random.Random(42)
        for rep in range(1,repeats+1):
            rng.shuffle(configs)
            for mode,p in configs:
                start=time.perf_counter();out=invoke(mode,p)
                ms=(time.perf_counter()-start)*1000
                assert out==expected
                rows.append(dict(language='Python',mode=mode,p=p,repeat=rep,ms=ms))
                del out
    return rows

def write_csv(path,records):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)

def run(quick=False):
    versions,commands=build()
    results=ROOT/('smoke-results' if quick else 'results')
    results.mkdir(exist_ok=True)
    data_dir=ROOT/'data';data_dir.mkdir(exist_ok=True)
    # Keep numerical-library threads out of child process measurements.
    child_env=os.environ.copy()
    child_env['GOMAXPROCS']=str(max(1,math.floor(cs.cpu_capacity())))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        child_env[key]='1'
    executables={'C++':[str(ROOT/'bin/count_cpp')],'Rust':[str(ROOT/'bin/count_rust')],
                 'Go':[str(ROOT/'bin/count_go')],'Node.js':[tool('node'),str(ROOT/'count_node.mjs')]}
    verification={}
    for language,command in executables.items():
        verification[language]=subprocess.check_output([*command,'--self-test'],text=True,env=child_env).strip()
        print(verification[language],flush=True)
    cs.run_self_tests()
    sizes=(1000,10000) if quick else (10000,100000,1000000,10000000)
    ks=(7,256) if quick else (256,65536)
    repeats=3 if quick else 5
    ps=(1,2,4)
    records=[];manifest=[]
    ordering=random.Random(20261021)
    for k in ks:
        for n in sizes:
            lo=-k//2
            rng=random.Random(42)
            values=array('i',(rng.randrange(lo,lo+k) for _ in range(n)))
            assert values.itemsize==4
            if sys.byteorder!='little':values.byteswap()
            path=data_dir/f'n{n}_k{k}.i32'
            path.write_bytes(values.tobytes());del values
            manifest.append(dict(n=n,k=k,lo=lo,seed=42,bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),format='little-endian int32'))
            languages=list(executables)+['Python'];ordering.shuffle(languages)
            for language in languages:
                if language=='Python':
                    part=python_benchmark(path,lo,k,repeats,ps)
                else:
                    out=results/f'{language.replace("+","p")}_n{n}_k{k}.csv'
                    subprocess.run([*executables[language],str(path),str(lo),str(k),
                                    str(repeats),str(out),','.join(map(str,ps))],check=True,env=child_env)
                    with out.open() as stream:part=list(csv.DictReader(stream))
                    out.unlink()
                for row in part:
                    row.update(n=n,k=k,distribution='uniform')
                    row['p']=int(row['p']);row['repeat']=int(row['repeat']);row['ms']=float(row['ms'])
                    records.append(row)
                print(f'{language}: verified n={n:,}, K={k:,}',flush=True)
            # Checkpoint after every complete scenario.
            write_csv(results/'raw.csv',records)
            (results/'inputs.json').write_text(json.dumps(manifest,indent=2))
    summary=[]
    for item in manifest:
        n,k=item['n'],item['k']
        for language in ['Python','C++','Rust','Go','Node.js']:
            subset=[r for r in records if (r['n'],r['k'],r['language'])==(n,k,language)]
            base=statistics.median(r['ms'] for r in subset if r['mode']=='seq')
            for mode,p in [('seq',1)]+[(m,p) for m in ('warm','cold') for p in ps]:
                times=[r['ms'] for r in subset if (r['mode'],r['p'])==(mode,p)]
                median=statistics.median(times)
                summary.append(dict(language=language,n=n,k=k,mode=mode,p=p,
                    median_ms=median,min_ms=min(times),max_ms=max(times),
                    speedup=base/median,efficiency=base/median/p))
    write_csv(results/'summary.csv',summary)
    environment=dict(versions=versions,build_commands=commands,
        platform=platform.platform(),cpu_count=os.cpu_count(),cpu_capacity=cs.cpu_capacity(),
        processes=list(ps),repeats=repeats,seed=42,GOMAXPROCS=child_env['GOMAXPROCS'],
        verification=verification,all_timed_results_checked=True,quick=quick,
        node_command=executables['Node.js'],node_warmups={'seq':10,'warm':10,'cold':1},
        node_memory='Int32Array keys, Float64Array integer frequencies exact at tested n. '
                    'Copied blocks and histograms use transferable ArrayBuffers, no SharedArrayBuffer.',
        memory='All parallel variants copy input blocks; each task owns a private histogram. '
               'Python transfers pickled data between processes; compiled programs use threads/goroutines; '
               'Node.js transfers owned copied buffers between worker_threads.',
        timing='Read/generation, compilation, reference sorting, equality checks and application startup excluded. '
               'Allocation, block copies, dispatch, counting, reduction and output construction included. '
               'Cold additionally includes worker creation and shutdown; Go goroutines may reuse OS threads. '
               'Go and Node.js use default GC; collections occurring inside a sample are included. '
               'Temporary-buffer reclamation may occur at different times across implementations.')
    (results/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    print(f'PASS: {len(records)} samples, all five languages, results at {results}',flush=True)
    return summary

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    run(args.quick)
