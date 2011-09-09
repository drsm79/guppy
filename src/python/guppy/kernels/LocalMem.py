#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy

class LocalMem(Kernel):
    """
    A simple test kernel, doesn't load the code from a file
    """
    kernel_file = "LocalMem.cl"

    def _create_buffers(self):
        """
        Add up two 5-element arrays
        """
        mf = cl.mem_flags

	self.blocks = 32
	self.threads = 128
        self.global_size = (self.threads*self.blocks,)
	self.local_size = (self.threads,)

 	input_arr = numpy.array(range(0,self.threads*self.blocks), dtype=numpy.float32)

        self.dtype = input_arr.dtype

	#Create the buffers and append to a list
	input_one = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_arr)
	self.buffers.append(input_one)

	out_arr = numpy.empty(shape=(self.blocks,), dtype = numpy.float32)
	out_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=out_arr)
        self.buffers.append(out_buff)

	local_arr = numpy.empty(shape=(self.threads,), dtype = numpy.float32)
	#local_buff = cl.Buffer(self.ctx, mf.READ_WRITE, size=local_arr.nbytes)
	self.buffers.append(cl.LocalMemory(local_arr.nbytes))

    def _run_kernel(self):
        """
        Run the kernel (self.program).
        """
	self.program.sum(self.queue, self.global_size, self.local_size, *self.buffers)
	print "these are read back"
        # Make an empty array to copy data into
	result_arr = numpy.empty(self.blocks, dtype=self.dtype)
     # Enqueue the copy
  	cl.enqueue_copy(self.queue, result_arr, self.buffers[1])
	return result_arr
