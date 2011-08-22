#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.LocalMem import LocalMem
import numpy
from guppy_t import TimedTest

class AddBuffer_t(unittest.TestCase):
    def testCall(self):
        lmem = LocalMem()
        result = lmem()
        print result

if __name__ == '__main__':
	unittest.main()
