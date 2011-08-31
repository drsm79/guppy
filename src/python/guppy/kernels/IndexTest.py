#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy

class IndexTest(Kernel):
    """
    A simple test kernel, doesn't load the code from a file
    """
    kernel_file = "IndexTest.cl"

    def _create_buffers(self, block_x, block_y, thread_x, thread_y):
        """
        Copy a buffer to another buffer
        """
        mf = cl.mem_flags

        self.dtype = numpy.float32
        self.global_size = (block_x*thread_x, block_y*thread_y)
	self.local_size = (thread_x, thread_y)
	self.resultshape = (block_x*block_y*thread_x*thread_y)
	output_arr = numpy.empty(shape=self.resultshape, dtype=self.dtype)
        self.buffers.append(cl.Buffer(self.ctx, mf.WRITE_ONLY, output_arr.nbytes))

    def _run_kernel(self):
        """
        Run the kernel (self.program).
        """
        # This is broken
        self.program.IndexTest(self.queue, self.global_size, self.local_size, *self.buffers)
        # Make an empty array to copy data into
        result_arr = numpy.empty(self.resultshape, dtype=self.dtype)
	# Enqueue the copy
        cl.enqueue_copy(self.queue, result_arr, self.buffers[-1])
        print result_arr.size
	return result_arr
	
