from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="monotonic_align",
    ext_modules=cythonize("core.pyx", language_level="3"),
    include_dirs=[numpy.get_include()]
)
