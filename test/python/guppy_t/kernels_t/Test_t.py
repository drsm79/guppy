#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.Test import Test
import numpy

class Test_t(unittest.TestCase):
    def testCall(self):
        a = numpy.random.rand(50000).astype(numpy.float32)
        b = numpy.random.rand(50000).astype(numpy.float32)

        t = Test()
        self.assertEquals(0, t(host_buffers = {'a': a, 'b': b}))

if __name__ == '__main__':
	unittest.main()