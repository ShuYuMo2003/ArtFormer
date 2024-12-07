from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from Cython.Distutils import build_ext
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension

mcubes_module = Extension(
    'mcubes',
    sources=[
        'mcubes.pyx',
        'pywrapper.cpp',
        'marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[np.get_include()]
)

setup(
    ext_modules=cythonize([mcubes_module]),
    cmdclass={
        'build_ext': BuildExtension
    }
)
