from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from Cython.Distutils import build_ext
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension

simplify_mesh_module = Extension(
    'simplify_mesh',
    sources=[
        'simplify_mesh.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[np.get_include()]
)

setup(
    ext_modules=cythonize([simplify_mesh_module]),
    cmdclass={
        'build_ext': BuildExtension
    }
)