#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.RandomFlat import RandomFlat
import numpy
from guppy_t import TimedTest


class Random_t(unittest.TestCase):
    def testCall(self):
        n_rands = 777777
	n_threads = 256
        rand = RandomFlat()
        result = rand(threadcount=n_threads,size=n_rands)
	print result
        self.assertEquals(n_rands, result.size)


if __name__ == '__main__':
	unittest.main()
