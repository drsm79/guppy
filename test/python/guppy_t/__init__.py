#!/usr/bin/env python
# encoding: utf-8
"""
__init__.py

Created by Simon Metson on 2011-08-10.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import unittest
import inspect
from time import time

class TimedTest(unittest.TestCase):
    def setUp(self):
        self.start_time = 0
        self.duration = 0
        self.test_name = self.id().split('.')[-1]

    def tearDown(self):
        print 'Test %s ran for %s' % (self.test_name, self.duration)

    def start_timer(self):
        self.start_time = time()

    def stop_timer(self):
        self.duration = time() - self.start_time

if __name__ == "__main__":
    import guppy_t.kernels_t as kernel_tests

    suite = unittest.TestSuite()
    print inspect.getmembers(kernel_tests)
    for test in kernel_tests.__dict__.values():
        print test
    #for test in kernel_tests.__all__:
     #   print test
#suite.addTest(CMSCouchTest(sys.argv[1]))
#    unittest.TextTestRunner().run(suite)