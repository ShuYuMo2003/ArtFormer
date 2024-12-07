from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from Cython.Distutils import build_ext
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension

mise_module = Extension(
    'mise',
    sources=[
        'mise.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[np.get_include()]
)

setup(
    ext_modules=cythonize([mise_module]),
    cmdclass={
        'build_ext': BuildExtension
    }
)