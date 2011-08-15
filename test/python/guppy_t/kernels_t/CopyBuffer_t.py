#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.CopyBuffer import CopyBuffer
import numpy
from guppy_t import TimedTest

class CopyBuffer_t(unittest.TestCase):
    def testCall(self):
        buffer_cp = CopyBuffer()

        input_array = numpy.array([1,2,3,4,5], dtype=numpy.float32)

        result = buffer_cp(input_array)
        print result

if __name__ == '__main__':
	unittest.main()
