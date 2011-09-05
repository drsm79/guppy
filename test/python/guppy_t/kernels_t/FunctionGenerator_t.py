#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.FunctionGenerator import FunctionGenerator
import numpy
from guppy_t import TimedTest


class Random_t(TimedTest):

	def test_time(self):
	#Run the kernel
		self.globalsize = (32,32)
		self.grid = numpy.arange(18000).reshape(6,5,2,10,30)	
		self.function_string = 'dim[1]*dim[2]+dim[26];'
		Int = FunctionGenerator(self.function_string, 30)	
       		self.start_timer()
		self.result = Int(globalsize=self.globalsize, grid=self.grid)
		self.stop_timer()
		print self.result
		
if __name__ == '__main__':
	unittest.main()
