#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.ThreadNames import ThreadNames
import numpy

class Test_t(unittest.TestCase):
    def testCall(self):
        t = ThreadNames()
        print t()

if __name__ == '__main__':
	unittest.main()