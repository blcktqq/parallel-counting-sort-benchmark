"""Counting sort in Python syntax, compiled by Cython to C with OpenMP.

Build with: python openmp/setup.py build_ext --inplace
Input/output: contiguous signed 32-bit integer buffers (e.g. array('i')).
The parallel path copies the input, assigns disjoint blocks and histograms,
waits at the implicit prange barrier, then reduces and restores sequentially.
"""
from array import array
import cython
from cython.parallel import prange
from cython.cimports.openmp import omp_get_num_threads, omp_set_dynamic
from cython.cimports.libc.string import memcpy


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _count_block(a: cython.const[cython.int][::1], first: cython.Py_ssize_t,
                 last: cython.Py_ssize_t, h: cython.ulonglong[::1],
                 offset: cython.Py_ssize_t, lo: cython.int,
                 k: cython.Py_ssize_t) -> cython.int:
    i: cython.Py_ssize_t
    key: cython.longlong
    for i in range(first, last):
        key = cython.cast(cython.longlong, a[i]) - lo
        if key < 0 or key >= k:
            return 1
        h[offset + key] += 1
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _restore(h: cython.ulonglong[::1], k: cython.Py_ssize_t,
             lo: cython.int, out: cython.int[::1]) -> cython.void:
    key: cython.Py_ssize_t
    pos: cython.Py_ssize_t = 0
    stop: cython.Py_ssize_t
    for key in range(k):
        stop = pos + h[key]
        while pos < stop:
            out[pos] = lo + key
            pos += 1


def sort(a: cython.const[cython.int][::1], lo: cython.int,
         k: cython.Py_ssize_t, workers: cython.int = 0):
    """Return (sorted array('i'), actual OpenMP team size).

    workers=0 is the sequential baseline; workers>=1 uses OpenMP.
    Only the caller thread may call this function. Do not mutate a concurrently.
    """
    if not cython.compiled:
        raise RuntimeError('Compile this module with Cython and OpenMP first')
    if k < 1:
        raise ValueError('K must be positive')
    if workers < 0:
        raise ValueError('workers must be zero (sequential) or positive')
    if cython.cast(cython.longlong, lo) + k - 1 > 2147483647:
        raise ValueError('The key range must fit signed int32')
    if array('i').itemsize != 4 or array('Q').itemsize != 8:
        raise RuntimeError('This build needs 32-bit int and 64-bit counters')

    n: cython.Py_ssize_t = a.shape[0]
    p: cython.int = workers if workers else 1
    counts = array('Q', [0]) * (p * k)
    total = array('Q', [0]) * k
    output = array('i', [0]) * n
    errors = array('i', [0]) * p
    teams = array('i', [1]) * p
    h: cython.ulonglong[::1] = counts
    merged: cython.ulonglong[::1] = total
    out: cython.int[::1] = output
    invalid: cython.int[::1] = errors
    team_sizes: cython.int[::1] = teams
    copied: cython.int[::1]
    t: cython.Py_ssize_t
    key: cython.Py_ssize_t

    if workers:
        # One contiguous allocation holds p disjoint private block ranges.
        # Copying is sequential and included in the caller's measured interval.
        blocks = array('i', [0]) * n
        copied = blocks
        if n:
            with cython.nogil:
                memcpy(cython.address(copied[0]), cython.address(a[0]),
                       n * cython.sizeof(cython.int))
        omp_set_dynamic(0)
        for t in prange(p, nogil=True, schedule='static', num_threads=p):
            team_sizes[t] = omp_get_num_threads()
            invalid[t] = _count_block(copied, n * t // p, n * (t + 1) // p,
                                      h, t * k, lo, k)
        # prange has an implicit barrier: all histograms are complete here.
    else:
        with cython.nogil:
            invalid[0] = _count_block(a, 0, n, h, 0, lo, k)

    if any(errors):
        raise ValueError('A key is outside [lo, lo+K-1]')
    with cython.nogil:
        for t in range(p):
            for key in range(k):
                merged[key] += h[t * k + key]
        _restore(merged, k, lo, out)
    return output, max(teams)
