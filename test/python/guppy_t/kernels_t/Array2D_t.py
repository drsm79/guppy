#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.Array2D import Array2D
import numpy
from guppy_t import TimedTest

class AddBuffer_t(unittest.TestCase):
    def testCall(self):
        arr = Array2D()
        result = arr()
        print result

if __name__ == '__main__':
	unittest.main()
