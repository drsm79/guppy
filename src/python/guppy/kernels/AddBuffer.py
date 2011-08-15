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
        input_arr = numpy.array(range(0,10000), dtype=numpy.float32)
	i_two = numpy.array(range(10000,0,-1), dtype=input_arr.dtype)

        self.dtype = input_arr.dtype
        self.global_size = input_arr.shape

	#Create the buffers and append to a list
	input_one = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_arr)
	self.buffers.append(input_one)
        input_two = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=i_two)
        self.buffers.append(input_two)
	out_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY,size=input_arr.nbytes)
        self.buffers.append(out_buff)

    def _run_kernel(self):
        """
        Run the kernel (self.program).
        """
	self.program.sum(self.queue, self.global_size, None, *self.buffers)
	print "these are read back"
        # Make an empty array to copy data into
	result_arr = numpy.empty(self.global_size, dtype=self.dtype)
     # Enqueue the copy
  	cl.enqueue_copy(self.queue, result_arr, self.buffers[-1])
	return result_arr
