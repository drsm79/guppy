#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.Gaussian2DIntegrator import Gaussian2DIntegrator
import numpy
from guppy_t import TimedTest


class Random_t(TimedTest):

	def test_time(self):
	#Run the kernel
		self.trials = 1000000
		self.xmean=0.0
		self.xsd=1.0
		self.xlow=-1.0
		self.xhigh=1.0
		self.ymean=0.0
		self.ysd=1.0
		self.ylow=-1.0
		self.yhigh=1.0
		self.globalsize = (32,32)
		self.bincount = (10,10)

		Int = Gaussian2DIntegrator()	
       		self.start_timer()
		self.result = Int(globalsize=self.globalsize, bincount=self.bincount, trials=self.trials, \
		 		 xmean=self.xmean, xsd=self.xsd, xlow=self.xlow, xhigh=self.xhigh,     \
				 ymean=self.ymean, ysd=self.ysd, ylow=self.ylow, yhigh=self.yhigh)
		self.stop_timer()
		print self.result
		
if __name__ == '__main__':
	unittest.main()
