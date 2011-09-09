#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.BuiltinRandom import BuiltinRandom
from guppy.kernels.RandomGaussian import RandomGaussian
from guppy.kernels.RandomFlat import RandomFlat
import pyopencl
import numpy
import random
from guppy_t import TimedTest

def rand_ints(size, mean, sd):
	return []
	#return [random.gauss(mean, sd) for i in range(size)]

class RandomTime_t(TimedTest):

	n_rands = 10000000
	threadcount = 4096
	mean = 0.0
	sd = 1.0

	def test_time_pyopencl(self):
        	rand = BuiltinRandom()
		self.start_timer()
        	result = rand(shape=self.n_rands)
        	print '\n'
		self.stop_timer()
	def test_time_numpy(self):
		self.start_timer()
		numpy_arr = numpy.random.normal(loc=self.mean, scale=self.sd, size=self.n_rands)
		print '\n'
		self.stop_timer()
		
	def test_time_homegrownGaussian(self):
		rand = RandomGaussian()
		self.start_timer()
		result = rand(mean=self.mean, sd=self.sd, threadcount=self.threadcount, size = self.n_rands)
		self.stop_timer()
	def test_time_homegrownFlat(self):
                rand = RandomFlat()
		self.start_timer()
                result = rand(threadcount=self.threadcount, size = self.n_rands)
		print '\n'
                self.stop_timer()

	def test_time_python(self):
		self.start_timer()
		result = rand_ints(self.n_rands, self.mean, self.sd)
		print '\n'
		self.stop_timer()


if __name__ == '__main__':
        unittest.main()


