#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.Gaussian1DIntegrator import Gaussian1DIntegrator
import numpy
from guppy_t import TimedTest


class Random_t(TimedTest):

	def test_time(self):
	#Run the kernel
		self.trials = 25000000
		self.mean=0.0
		self.sd=1.0
		self.low=-1.0
		self.high=1.0
		self.globalsize = 8192
		self.bincount = 1
		Int = Gaussian1DIntegrator()	
       		self.start_timer()
		self.result = Int(trials=self.trials, globalsize=self.globalsize, bincount =self.bincount, \
				  mean=self.mean, sd=self.sd, low=self.low, high=self.high)
		self.stop_timer()
		print self.result
		
if __name__ == '__main__':
	unittest.main()
