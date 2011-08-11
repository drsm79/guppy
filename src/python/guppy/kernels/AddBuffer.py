#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy

class AddBuffer(Kernel):
    """
    A simple test kernel, doesn't load the code from a file
    """
    kernel_file = "addBuffer.cl"

    def _create_buffers(self):
        """
        Add up two 5-element arrays
        """
        mf = cl.mem_flags

        input_arr = numpy.array(range(0,100))

        parallelism = numpy.long(1)
        self.buffers.append((parallelism,))

        self.dtype = input_arr.dtype
        self.global_size = input_arr.shape
        input_one = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_arr)
        self.buffers.append(input_one)
        i_two = input_arr[::-1]
        input_two = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numpy.array(range(0,100)))
        self.buffers.append(input_two)

        self.buffers.append(cl.Buffer(self.ctx, mf.WRITE_ONLY, input_arr.nbytes))

    def _run_kernel(self):
        """
        Run the kernel (self.program).
        """
        # This is broken
        print self.global_size
        self.program.sum(self.queue, self.global_size, *self.buffers, g_times_l=True)
        # Make an empty array to copy data into
        a_plus_b = numpy.empty(100, dtype=self.dtype)
        # Enqueue the copy
        cl.enqueue_copy(self.queue, a_plus_b, self.buffers[-1])
        print a_plus_b