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

        input_arr = numpy.array(range(0,10), dtype=numpy.float32)
	i_two = numpy.array(range(10,0,-1), dtype=input_arr.dtype)


       # parallelism = numpy.long(1)
       # self.buffers.append((parallelism,))

        self.dtype = input_arr.dtype
        self.global_size = input_arr.shape
      	self.local_size = self.global_size


	input_one = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_arr)
        
	self.buffers.append(input_one)

        input_two = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=i_two)
        self.buffers.append(input_two)
	
	out_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY,size=input_arr.nbytes)
        self.buffers.append(out_buff)

	print "called in Create Buffers"
	print self.global_size
	print self.dtype
	print self.buffers[0].get_host_array(shape=self.global_size, dtype=self.dtype)
	print self.buffers[1].get_host_array(shape=self.global_size, dtype=self.dtype)
	print self.buffers[2].get_host_array(shape=self.global_size, dtype=self.dtype)

    def _run_kernel(self):
        """
        Run the kernel (self.program).
        """
        # This is broken
        print "These are the parameters that are sent:"
        print self.global_size
        print self.dtype
        print self.buffers[0].get_host_array(shape=self.global_size, dtype=self.dtype)
        print self.buffers[1].get_host_array(shape=self.global_size, dtype=self.dtype)
        print self.buffers[2].get_host_array(shape=self.global_size, dtype=self.dtype)

	self.program.sum(self.queue, self.global_size, None, *self.buffers)
	print "these are read back"
        # Make an empty array to copy data into
	a = numpy.zeros(self.global_size, dtype=self.dtype)   
	b = numpy.zeros(self.global_size, dtype=self.dtype)
	c = numpy.zeros(self.global_size, dtype=self.dtype)
	print c
     # Enqueue the copy
  	cl.enqueue_copy(self.queue, a, self.buffers[0])
	cl.enqueue_copy(self.queue, b,self.buffers[1])
      	cl.enqueue_copy(self.queue, c,self.buffers[2])

	print self.global_size
        print self.dtype
	print a
	print b
	print c
	#return c
