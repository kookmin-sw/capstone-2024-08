from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

# Set output directory and ensure it exists
output_dir = "tts/monotonic_align/"
os.makedirs(output_dir, exist_ok=True)

# Set up the extension to compile
ext_modules = cythonize(
    "core.pyx",
    compiler_directives={'language_level': "3"}
)

setup(
    name='monotonic_align',
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()]
)
