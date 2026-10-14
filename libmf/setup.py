# -*- coding: utf-8 -*-
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_module = Extension(
    "blocktree",
    ["cppblocktree.cpp", "blocktree.pyx"],
    language='c++',
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-std=c++0x', '-fopenmp'],
    extra_link_args=['-fopenmp',],
)

setup(
    name = 'Flash Blocktree Module',
    #cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize([ext_module]),
)
