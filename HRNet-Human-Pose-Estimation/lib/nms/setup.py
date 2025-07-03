from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='cpu_nms',
    ext_modules=cythonize([
        Extension(
            name='nms.cpu_nms',
            sources=['cpu_nms.pyx'],
            include_dirs=[np.get_include()],
            language='c++'
        )
    ]),
)