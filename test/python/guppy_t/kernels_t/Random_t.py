#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.Random import Random
import numpy
from guppy_t import TimedTest

class Random_t(unittest.TestCase):
    def testCall(self):
        n_rands = 10
        rand = Random()
        result = rand(size=n_rands)

        self.assertEquals(n_rands, result.size)


if __name__ == '__main__':
	unittest.main()