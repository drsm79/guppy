#!/usr/bin/env python
# encoding: utf-8

import numpy
from pyopencl.clrandom import rand
from guppy.kernels import Kernel

class BuiltinRandom(Kernel):
    """
    Reuse pyOpenCl's random number generator. This means we don't need to
    compile a kernel ourselves, or worry about buffer management.
    """
    kernel_file = ""

    def __call__(self, *args, **kwargs):
        """
        Because we don't need to manage buffers or compile kernel code we
        override the __call__ and just call the pyOpenCl code. Returns a
        pyopencl.array.Array.
        """
        return rand(self.queue, kwargs['shape'], numpy.float32)