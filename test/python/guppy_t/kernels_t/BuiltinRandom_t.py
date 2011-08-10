#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.BuiltinRandom import BuiltinRandom
import pyopencl
import numpy
from guppy_t import TimedTest

class BuiltinRandom_t(TimedTest):
    def testCall(self):
        n_rands = 10000000
        rand = BuiltinRandom()
        self.start_timer()
        result = rand(shape=n_rands)
        self.stop_timer()

        self.assertEquals(n_rands, result.size)

        self.assertEquals(pyopencl.array.max(result).get(), numpy.amax(result.get()))
        self.assertEquals(pyopencl.array.min(result).get(), numpy.amin(result.get()))

if __name__ == '__main__':
	unittest.main()