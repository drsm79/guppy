#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.RandomGaussian import RandomGaussian
import numpy

class Random_t(unittest.TestCase):

	def setUp(self):
	#Run the kernel
		self.n_rands = 100000
		self.n_threads = 10
		self.mean=0.
		self.sd=1
		
		rand = RandomGaussian()	
       		self.result = rand(threadcount=self.n_threads,size=self.n_rands, mean=self.mean, sd=self.sd)

	#test the Size of the Output
	def test_size(self):
		self.assertEquals(self.n_rands, self.result.size)
	
	#test the mean and standard deviation
	def test_meanstd(self):

		#Check the output mean is consistent with the input mean 
		result_mean = numpy.mean(self.result)
		result_sd = numpy.std(self.result)
		diff = numpy.abs(result_mean-float(self.mean))
		self.assertAlmostEqual(self.mean, result_mean, 2)
		self.assertAlmostEqual(self.sd, result_sd, 2)

if __name__ == '__main__':
	unittest.main()
