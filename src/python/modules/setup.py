from distutils.core import setup, Extension
import numpy

setup(
	name="functiongenerator",
	version="0.0",
	ext_modules = [Extension("functiongenerator", ["functiongenerator.c"],include_dirs = [numpy.get_include()])]
)

