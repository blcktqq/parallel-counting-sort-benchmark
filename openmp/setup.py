"""Linux/GCC build: Python syntax -> C (not C++) -> OpenMP extension."""
from pathlib import Path
import os
from setuptools import Extension, setup
from Cython.Build import cythonize

# Some Python distributions remember a compiler unavailable on this machine.
# This build targets GCC on Linux; callers may override CC explicitly.
os.environ.setdefault('CC', 'gcc')
source = Path(__file__).resolve().parent / 'omp_counting.py'
setup(
    name='counting-sort-openmp',
    version='1.0.0',
    ext_modules=cythonize(
        [Extension('omp_counting', [str(source)], language='c',
                   extra_compile_args=['-O3', '-fopenmp'],
                   extra_link_args=['-fopenmp'])],
        compiler_directives={'language_level': 3, 'boundscheck': False,
                             'wraparound': False, 'initializedcheck': False,
                             'cdivision': True},
    ),
)
