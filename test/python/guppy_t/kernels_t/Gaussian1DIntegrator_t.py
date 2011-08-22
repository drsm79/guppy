#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.Gaussian1DIntegrator import Gaussian1DIntegrator
import numpy
from guppy_t import TimedTest


class Random_t(TimedTest):

	def test_time(self):
	#Run the kernel
		self.n_rands = 220000 #this causes errors if too large
		self.mean=0.0
		self.sd=1.0
		self.low=-1.0
		self.high=0.5
		self.blocks = 32
		self.threads = 64
		self.shape=(self.blocks,)	
		Int = Gaussian1DIntegrator()	
       		self.start_timer()
		self.result = Int(blocks=self.blocks, threads=self.threads, block_rands=self.n_rands, mean=self.mean, sd=self.sd, low=self.low, high=self.high)
		print self.result
		self.stop_timer()
	
	
if __name__ == '__main__':
	unittest.main()
