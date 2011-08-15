#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy

class CopyBuffer(Kernel):
    """
    A simple test kernel, doesn't load the code from a file
    """
    kernel_file = "copyBuffer.cl"

    def _create_buffers(self, input_array):
        """
        Copy a buffer to another buffer
        """
        mf = cl.mem_flags

        self.dtype = input_array.dtype
        self.global_size = input_array.shape
        print self.dtype, self.global_size

        input_one = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_array)
        self.buffers.append(input_one)

        self.buffers.append(cl.Buffer(self.ctx, mf.WRITE_ONLY, input_array.nbytes))

    def _run_kernel(self):
        """
        Run the kernel (self.program).
        """
        # This is broken
        self.program.sum(self.queue, self.global_size, None, *self.buffers)
        # Make an empty array to copy data into
        a_plus_b = numpy.empty(self.global_size, dtype=self.dtype)
        # Enqueue the copy
        cl.enqueue_copy(self.queue, a_plus_b, self.buffers[1])
        return a_plus_b