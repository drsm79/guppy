#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.FunctionGenerator import FunctionGenerator
import numpy
import random

class FunctionGenerator_t(unittest.TestCase):

	def setUp(self):
		self.globalsize = (32,1)
                self.grid = numpy.array([[0,10,10],[0,10,10],[0,10,10], [0,10,10], [0,10,10],[0,10,10]])

	def test_FG(self):
		self.function_string = 'dim[0]+dim[1]*dim[2];'
		FG = FunctionGenerator(self.function_string, 6)	
		result = FG(globalsize=self.globalsize, grid=self.grid)
		print result[0,0,0,0,0,0]
		print result[3,3,3,0,0,0]
		print result[6,6,6,0,0,0]
		print result[9,9,9,3,0,0]

if __name__ == '__main__':
	unittest.main()
