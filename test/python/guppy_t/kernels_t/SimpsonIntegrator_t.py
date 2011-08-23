#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.SimpsonIntegrator import SimpsonIntegrator
import numpy
from guppy_t import TimedTest


class Random_t(TimedTest):

	def test_time(self):
	#Run the kernel
		self.mean=0.0
		self.sd=1.0
		self.low=0.0
		self.high=0.52
		self.blocks = 128
		self.threads = 256
		self.shape=(self.blocks,)	
		Int = SimpsonIntegrator()	
       		self.start_timer()
		self.result = Int(blocks=self.blocks, threads=self.threads, mean=self.mean, sd=self.sd, low=self.low, high=self.high)
		self.stop_timer()
		print self.result
	
	
if __name__ == '__main__':
	unittest.main()
