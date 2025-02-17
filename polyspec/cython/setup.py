from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# Compile pyx files requiring WignerSymbols and C++
modules = ['tauNL_utils','fNL_utils']
for module in modules:
    ext_modules = [
        Extension(
            module,
            [module+'.pyx','wignerSymbols-cpp.cpp'],
            language="c++",
            libraries=["stdc++"],
            extra_compile_args=["-fopenmp","-O3", "-ffast-math", "-march=broadwell"],
            extra_link_args=["-fopenmp"],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            )
    ]
    setup(
        name=module,
        ext_modules=cythonize(ext_modules),
        include_dirs=[numpy.get_include()]
    )

# Compile remaining pyx files with C
modules = ['k_integrals','ideal_fisher']
for module in modules:
    ext_modules = [
        Extension(
            module,
            [module+'.pyx'],
            libraries=['gsl','gslcblas','mvec','m'],
            extra_compile_args=["-fopenmp","-O3", "-ffast-math", "-march=broadwell"],
            extra_link_args=["-fopenmp"],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            )
    ]
    setup(
        name=module,
        ext_modules=cythonize(ext_modules),
        include_dirs=[numpy.get_include()]
    )