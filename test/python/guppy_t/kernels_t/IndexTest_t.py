#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.IndexTest import IndexTest
import numpy
from guppy_t import TimedTest

class CopyBuffer_t(unittest.TestCase):
    def testCall(self):
	self.block_x = 32
	self.block_y = 32
	self.thread_x = 16
	self.thread_y = 8
        index_test = IndexTest()
        result = index_test(block_x = self.block_x, block_y = self.block_y, thread_x = self.thread_x, thread_y = self.thread_y)
        print result

if __name__ == '__main__':
	unittest.main()
