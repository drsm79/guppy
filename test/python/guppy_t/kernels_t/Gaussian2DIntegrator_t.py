#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.Gaussian2DIntegrator import Gaussian2DIntegrator
import numpy
from guppy_t import TimedTest


class Random_t(TimedTest):

	def test_time(self):
	#Run the kernel
		self.n_rands = 30000
		self.xmean=0.0
		self.xsd=1.0
		self.xlow=-0.6
		self.xhigh=0.4
		self.ymean=0.0
		self.ysd=1.0
		self.ylow=-0.8
		self.yhigh=0.1

		self.blocks = 32
		self.threads = 256
		self.shape=(self.blocks,)	
		Int = Gaussian2DIntegrator()	
       		self.start_timer()
		self.result = Int(blocks=self.blocks, threads=self.threads, block_rands=self.n_rands, \
		 		 xmean=self.xmean, xsd=self.xsd, xlow=self.xlow, xhigh=self.xhigh,     \
				 ymean=self.ymean, ysd=self.ysd, ylow=self.ylow, yhigh=self.yhigh)
		self.stop_timer()
		print self.result
		
if __name__ == '__main__':
	unittest.main()
