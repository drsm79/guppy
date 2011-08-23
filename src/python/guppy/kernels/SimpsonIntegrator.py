#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy
import math

class SimpsonIntegrator(Kernel):
	"""
    	A simple test kernel, doesn't load the code from a file
    	"""
	kernel_file = "SimpsonIntegrator.cl"

    	def _create_buffers(self, blocks=64, threads=128, mean=0, sd=1, low=-1, high=1, seeds=None):
		"""
        	Define the necessary input and ouptut Buffer objects for the Kernel,
        	store in self.input_buffers and self.output_buffers. Subclasses should
        	overwrite this.
       	 	"""
        	mf = cl.mem_flags

        	# The GPUrand function takes four inputs:
        	#
        	#       d_Result = the output array of random numbers
        	#       d_seed = four random numbers in an array
		#	float* args: list of necessary parameters bundled together 
        	#       args[0]: nPerThread = number of random numbers to generate per thread
        	#       args[1]: Blocks = number of blocks used
		#	args[2]: Threads = number of threads per block
		#	args[3]: Mean
		#	args[4]: Standard Deviation
		#	args[5]: Lower Limit
		#	args[6]: Upper Limit
		#
	
		#Calculate kernel variables
	       	self.global_size = (threads*blocks,)
		self.local_size = (threads,)
		self.resultshape = (blocks*threads,)	
		self.h = float(high-low)/float(blocks*threads)	

		#set the writeable output array to hold the random numbers
		self.result = numpy.empty(shape=self.resultshape, dtype=numpy.float32)
		output_buff = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result)
	 	self.buffers.append(output_buff)

	        arg_arr = numpy.array([mean, sd, low, high], dtype=numpy.float32)
		arg_bufr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arg_arr)
	        self.buffers.append(arg_bufr)

		#Assign  Memory Arrays
		x_arr = numpy.arange(low, high, self.h).astype(numpy.float32)
		x_buff = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x_arr)
		self.buffers.append(x_buff)

	def _run_kernel(self):
        	
		"""
        	Run the kernel (self.program). Subclasses should overwrite this.
        	"""
        	self.program.SimpsonIntegrator(self.queue, self.global_size, self.local_size, *self.buffers)
		#Create empty array to output the random numbers result from device-->host
		result_arr = numpy.empty(self.resultshape, dtype=numpy.float32)
		cl.enqueue_copy(self.queue, result_arr, self.buffers[0])		
		arr_sum = sum(result_arr)
	
		return arr_sum*self.h/3
