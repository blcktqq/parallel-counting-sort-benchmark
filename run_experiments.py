"""Run the original five implementations, optionally followed by OpenMP."""
import argparse
from multilang.experiment import run as run_five

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--with-openmp', action='store_true')
    args = parser.parse_args()
    if args.with_openmp:
        # Check before starting a long benchmark, rather than after it.
        import omp_counting
    run_five(args.quick)
    if args.with_openmp:
        from openmp.experiment import run as run_openmp
        run_openmp(args.quick)
