#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.Test import Test
import numpy

class Test_t(unittest.TestCase):

	def test_float64(self):
		#Run the kernel
        	print '\ntesting float64...\n'
		a = numpy.random.rand(50000).astype(numpy.float64)
        	b = numpy.random.rand(50000).astype(numpy.float64)
        	t = Test()
        	self.assertEquals(0, t(host_buffers = {'a': a, 'b': b}))
	def test_float32(self):
   		#Run the kernel
                print '\ntesting float32...\n'
                a = numpy.random.rand(50000).astype(numpy.float32)
                b = numpy.random.rand(50000).astype(numpy.float32)
                t = Test()
                self.assertEquals(0, t(host_buffers = {'a': a, 'b': b}))
	def test_int32(self):
		#Run the kernel
                print '\ntesting int32...\n'
                a = numpy.array(range(0,50000)).astype(numpy.int32)
                b = numpy.array(range(0,50000)).astype(numpy.int32)
                t = Test()
                self.assertEquals(0, t(host_buffers = {'a': a, 'b': b}))

if __name__ == '__main__':
	unittest.main()
