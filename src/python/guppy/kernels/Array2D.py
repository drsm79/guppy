#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy

class Array2D(Kernel):
    """
    A simple test kernel, doesn't load the code from a file
    """
    kernel_file = "Array2D.cl"

    def _create_buffers(self):
        """
        Add up two 5-element arrays
        """
        mf = cl.mem_flags
        input_arr = numpy.arange(10).reshape(5,2).astype(numpy.float32)
        print input_arr
	self.dtype = input_arr.dtype
        self.global_size = (5,)

	#Create the buffers and append to a list
	input_one = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_arr)
	self.buffers.append(input_one)
	out_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY,size=input_arr.nbytes)
        self.buffers.append(out_buff)

    def _run_kernel(self):
        """
        Run the kernel (self.program).
        """
	self.program.Array2D(self.queue, self.global_size, None, *self.buffers)
        # Make an empty array to copy data into
	result_arr = numpy.empty(shape=(5,2), dtype=self.dtype)
     # Enqueue the copy
  	cl.enqueue_copy(self.queue, result_arr, self.buffers[-1])
	return result_arr
