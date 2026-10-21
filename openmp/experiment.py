"""Verify and measure Cython/OpenMP on the same inputs as the five-language run.

Run from repo root: python openmp/experiment.py [--quick] [--self-test]
An installed/built omp_counting extension is required. No compiler runs here.
"""
from array import array
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import omp_counting
from counting_sort import cpu_capacity


def self_test():
    checks = 0
    for n in (0, 1, 2, 7, 100):
        for k in (1, 7, 256):
            lo = -k // 2
            rng = random.Random(42)
            values = [rng.randrange(lo, lo + k) for _ in range(n)]
            for data in (values, sorted(values), sorted(values, reverse=True), [lo] * n):
                a = array('i', data)
                for p in (0, 1, 2, 4):
                    out, actual = omp_counting.sort(a, lo, k, p)
                    assert out.tolist() == sorted(data)
                    assert a.tolist() == data
                    assert 1 <= actual <= (p or 1)
                checks += 1
    invalid = [([1], 0, 1, 2), ([-1], 0, 1, 2), ([], 0, 0, 2),
               ([], 0, 1, -1), ([], 2147483647, 2, 2)]
    for data, lo, k, p in invalid:
        try:
            omp_counting.sort(array('i', data), lo, k, p)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    # Check exact handling near both int32 endpoints and extreme differences.
    for data, lo, k in [([-2147483648, -2147483647], -2147483648, 2),
                        ([2147483647, 2147483646], 2147483646, 2)]:
        for p in (0, 4):
            out, _ = omp_counting.sort(array('i', data), lo, k, p)
            assert out.tolist() == sorted(data)
    try:
        omp_counting.sort(array('i', [2147483647]), -2147483648, 2, 4)
    except ValueError:
        pass
    else:
        raise AssertionError('Overflowing key offset accepted')
    print(f'OpenMP PASS: {checks} datasets x 4 configurations, 6 invalid inputs, int32 endpoints', flush=True)
    return checks


def write_csv(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(quick=False):
    self_test()
    target = ROOT / 'openmp' / ('smoke-results' if quick else 'results')
    target.mkdir(exist_ok=True)
    sizes = (1000, 10000) if quick else (10000, 100000, 1000000, 10000000)
    ks = (7, 256) if quick else (256, 65536)
    repeats = 3 if quick else 5
    records, inputs = [], []
    for k in ks:
        for n in sizes:
            lo = -k // 2
            rng = random.Random(42)
            a = array('i', (rng.randrange(lo, lo + k) for _ in range(n)))
            encoded = array('i', a)
            if sys.byteorder != 'little':
                encoded.byteswap()
            sha = hashlib.sha256(encoded.tobytes()).hexdigest()
            inputs.append(dict(n=n, k=k, lo=lo, seed=42, sha256=sha))
            expected = array('i', sorted(a))
            configs = [('seq', 0)] + [('parallel', p) for p in (1, 2, 4)]
            for _, p in configs:
                assert omp_counting.sort(a, lo, k, p)[0] == expected
            order = random.Random(42)
            for repeat in range(1, repeats + 1):
                order.shuffle(configs)
                for mode, p in configs:
                    start = time.perf_counter()
                    out, actual = omp_counting.sort(a, lo, k, p)
                    ms = (time.perf_counter() - start) * 1000
                    assert out == expected
                    records.append(dict(language='Cython+OpenMP', n=n, k=k,
                                        mode=mode, p=p or 1, actual_threads=actual,
                                        repeat=repeat, ms=ms))
                    del out
            print(f'OpenMP verified n={n:,}, K={k:,}', flush=True)
    summary = []
    for item in inputs:
        part = [r for r in records if (r['n'], r['k']) == (item['n'], item['k'])]
        baseline = statistics.median(r['ms'] for r in part if r['mode'] == 'seq')
        for mode, p in [('seq', 1)] + [('parallel', p) for p in (1, 2, 4)]:
            rows = [r for r in part if (r['mode'], r['p']) == (mode, p)]
            times = [r['ms'] for r in rows]
            med = statistics.median(times)
            summary.append(dict(language='Cython+OpenMP', n=item['n'], k=item['k'],
                                mode=mode, p=p, actual_threads=rows[0]['actual_threads'],
                                median_ms=med, min_ms=min(times), max_ms=max(times),
                                speedup=baseline / med, efficiency=baseline / med / p))
    write_csv(target / 'raw.csv', records)
    write_csv(target / 'summary.csv', summary)
    (target / 'inputs.json').write_text(json.dumps(inputs, indent=2))
    env = dict(python=sys.version, platform=platform.platform(), cpu_capacity=cpu_capacity(),
               extension=Path(omp_counting.__file__).name, repeats=repeats, warmups=1,
               actual_team_sizes=sorted({r['actual_threads'] for r in records}),
               all_timed_results_checked=True,
               OMP_THREAD_LIMIT=os.environ.get('OMP_THREAD_LIMIT'),
               memory='Shared address space; copied input with disjoint block ranges; '
                      'private uint64 histogram rows; sequential merge and restore.',
               timing='Allocation, sequential copy on parallel path, counting, merge, '
                      'restore and scope cleanup included; generation, reference, '
                      'validation, build and application startup excluded. '
                      'One warmup per configuration. Each parallel call enters a new '
                      'OpenMP region; the runtime may retain OS threads. No cold '
                      'worker-pool equivalent is reported.')
    (target / 'environment.json').write_text(json.dumps(env, indent=2))
    print(f'PASS: {len(records)} OpenMP samples, actual teams {env["actual_team_sizes"]}', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.quick)
