"""Practical 2: counting sort, Python multiprocessing, no shared memory.

Only the standard library is needed. Worker functions live in this importable
module so that worker processes can import them with the spawn start method.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time


def cpu_capacity():
    """Best-effort capacity: logical CPUs, affinity and Linux cgroup v2 quota."""
    limits = [float(os.cpu_count() or 1)]
    if hasattr(os, "sched_getaffinity"):
        limits.append(float(len(os.sched_getaffinity(0))))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(int(quota) / int(period))
    except (OSError, ValueError):
        pass
    return min(limits)


def histogram(a, lo, k):
    """Count integer keys in [lo, lo+K-1]. Supports negative keys via a shift."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("K must be a positive integer")
    counts = [0] * k
    for x in a:
        if not isinstance(x, int):
            raise ValueError("Every key must be an integer")
        index = x - lo
        if index < 0 or index >= k:
            raise ValueError("A key is outside [lo, lo+K-1]")
        counts[index] += 1
    return counts


def restore(counts, lo):
    """Reconstruct an integer list from frequencies in ascending key order."""
    out = []
    for index, count in enumerate(counts):
        out.extend([lo + index] * count)
    return out


def counting_sort(a, lo, k):
    return restore(histogram(a, lo, k), lo)


def count_task(task):
    block, lo, k = task
    return histogram(block, lo, k)


def _ready(queue, start_gate):
    queue.put(os.getpid())
    # Keep every worker busy in initialization until the full team has started.
    # Otherwise a very fast ping can let the executor reuse an idle worker
    # instead of creating all requested processes.
    start_gate.wait()


def _ping():
    return os.getpid()


class ParallelCountingSort:
    """Reusable process pool; every sort call copies and serializes its blocks."""
    def __init__(self, processes):
        if not isinstance(processes, int) or processes < 1:
            raise ValueError("processes must be a positive integer")
        self.p = processes
        ctx = mp.get_context("spawn")
        ready = ctx.Queue()
        start_gate = ctx.Event()
        started = time.perf_counter()
        self.pool = ProcessPoolExecutor(max_workers=processes, mp_context=ctx,
                                       initializer=_ready, initargs=(ready, start_gate))
        try:
            futures = [self.pool.submit(_ping) for _ in range(processes)]
            for _ in range(processes):
                ready.get(timeout=120)
            start_gate.set()
            for future in futures:
                future.result(timeout=120)
        except BaseException:
            start_gate.set()
            self.pool.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            ready.close()
            ready.join_thread()
        self.startup_s = time.perf_counter() - started

    def sort(self, a, lo, k):
        if not isinstance(k, int) or k < 1:
            raise ValueError("K must be a positive integer")
        n, p = len(a), self.p
        # Copying the slices is deliberately INSIDE the measured operation.
        tasks = [(a[n*t//p:n*(t+1)//p], lo, k) for t in range(p)]
        partial = list(self.pool.map(count_task, tasks, chunksize=1))
        counts = [0] * k
        for local in partial:
            for index, count in enumerate(local):
                counts[index] += count
        return restore(counts, lo)

    def close(self):
        self.pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_self_tests():
    rng = random.Random(42)
    passed = 0
    # Three workers also exercise n<p and a non-power-of-two worker count.
    with ParallelCountingSort(3) as sorter:
        for n in (0, 1, 2, 7, 100):
            for k in (1, 2, 7, 31):
                lo = -k // 2
                a = [rng.randrange(lo, lo+k) for _ in range(n)]
                for source in (a, sorted(a), sorted(a, reverse=True), [lo]*n):
                    expected = sorted(source)
                    assert counting_sort(source, lo, k) == expected
                    assert sorter.sort(source, lo, k) == expected
                    passed += 1
        for bad in ([3], [-1], [1.5]):
            for sort in (counting_sort, sorter.sort):
                try:
                    sort(bad, 0, 3)
                except ValueError:
                    passed += 1
                else:
                    raise AssertionError("Invalid input was accepted")
    print(f"PASS: 80 datasets in both implementations; 6 invalid-input checks")
    return passed


def run_benchmarks(output_dir="results", repeats=5,
                   sizes=(10000, 100000, 1000000, 5000000),
                   processes=(1, 2, 4), sensitivity=True):
    if repeats < 3 or repeats % 2 != 1:
        raise ValueError("Use an odd number of repetitions >= 3")
    dest = Path(output_dir)
    dest.mkdir(exist_ok=True, parents=True)
    cases = [(int(n), 256, "uniform") for n in sizes]
    if sensitivity:
        cases += [(1000000, k, "uniform") for k in (16, 4096, 65536)]
        cases += [(1000000, 256, "constant")]
    rows, setup = [], []
    shuffle = random.Random(20261021)
    with ExitStack() as stack:
        sorters = {p: stack.enter_context(ParallelCountingSort(p)) for p in processes}
        setup = [{"p": p, "pool_startup_ms": s.startup_s*1000}
                 for p, s in sorters.items()]
        for n, k, distribution in cases:
            lo = -k // 2
            rng = random.Random(42)
            a = ([rng.randrange(lo, lo+k) for _ in range(n)]
                 if distribution == "uniform" else [lo] * n)
            expected = sorted(a)  # Reference only, outside every timed region.
            methods = [("sequential", 1)] + [("parallel", p) for p in processes]
            def invoke(method, p):
                return (counting_sort(a,lo,k) if method == "sequential"
                        else sorters[p].sort(a,lo,k))
            for method, p in methods:
                assert invoke(method, p) == expected
            for repeat in range(1,repeats+1):
                order = methods.copy()
                shuffle.shuffle(order)
                for method, p in order:
                    # Discard previous output before starting the next timer.
                    start = time.perf_counter()
                    out = invoke(method,p)
                    ms = (time.perf_counter()-start)*1000
                    assert out == expected
                    rows.append(dict(n=n,k=k,distribution=distribution,
                                     method=method,p=p,repeat=repeat,ms=ms))
                    del out
            print(f"Verified n={n:,}, K={k:,}, {distribution}",flush=True)
    summary=[]
    for n,k,distribution in cases:
        subset=[r for r in rows if (r['n'],r['k'],r['distribution'])==(n,k,distribution)]
        base=statistics.median(r['ms'] for r in subset if r['method']=='sequential')
        for method,p in [("sequential",1)]+[("parallel",p) for p in processes]:
            times=[r['ms'] for r in subset if (r['method'],r['p'])==(method,p)]
            med=statistics.median(times)
            summary.append(dict(n=n,k=k,distribution=distribution,method=method,p=p,
                median_ms=med,min_ms=min(times),max_ms=max(times),
                speedup=base/med,efficiency=base/med/p))
    for name,records in (("raw.csv",rows),("summary.csv",summary)):
        with (dest/name).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    environment=dict(python=sys.version,platform=platform.platform(),
        cpu_count=os.cpu_count(),cpu_capacity=cpu_capacity(),start_method="spawn",
        repeats=repeats,seed=42,processes=list(processes),pool_startup=setup,
        timing="Warm pools; includes input slicing, IPC/pickle, local histograms, "
               "sequential reduction and output construction. Excludes pool startup, "
               "input generation, reference sorting, validation and plotting.")
    (dest/"environment.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--output",default="results")
    parser.add_argument("--repeats",type=int,default=5)
    args=parser.parse_args()
    if args.self_test:
        run_self_tests()
    if args.benchmark:
        run_benchmarks(args.output,args.repeats)
    if not args.self_test and not args.benchmark:
        print(counting_sort([4,-1,2,4,0,-1,3,2],-1,6))
