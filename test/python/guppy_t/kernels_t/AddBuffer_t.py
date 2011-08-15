#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.AddBuffer import AddBuffer
import numpy
from guppy_t import TimedTest

class AddBuffer_t(unittest.TestCase):
    def testCall(self):
        buffer_and = AddBuffer()
        result = buffer_and()
        print result

if __name__ == '__main__':
	unittest.main()
