#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy
import math

class Gaussian2DIntegrator(Kernel):
	"""
    	A simple test kernel, doesn't load the code from a file
    	"""
	kernel_file = "Gaussian2DIntegrator.cl"

    	def _create_buffers(self, block_rands=120000,  blocks=(64,), threads=(128,) , xmean=0, xsd=1, xlow=-1, xhigh=1, \
			    ymean=0, ysd=1, ylow=-1, yhigh=1,seeds=None):
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
		#Hardcode thread sizes for debugging
		threads = 32
		blocks = 16

		nPerThread = math.ceil(float(block_rands)/float(threads))
	       	self.global_size = (threads*blocks,threads*blocks)
		self.local_size = (threads,threads)
		self.resultshape = self.global_size	
	
		#set the writeable output array to hold the random numbers
		self.result = numpy.empty(shape=self.resultshape, dtype=numpy.float32)
		output_buff = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result)
	 	self.buffers.append(output_buff)

		#Generate seeds, required to be >180 by HybridTaus algorithm 
	        self.seeds1 = numpy.random.randint(180, 1000000, (threads*blocks*threads*blocks,)).astype(numpy.uint32)
        	seed1_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.seeds1)
        	print self.seeds1.size
		self.buffers.append(seed1_buffer)

		self.seeds2 = numpy.random.randint(180, 1000000, (threads*blocks*threads*blocks,)).astype(numpy.uint32)
                seed2_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.seeds2)
                self.buffers.append(seed2_buffer)

		self.seeds3 = numpy.random.randint(180, 1000000, (threads*blocks*threads*blocks,)).astype(numpy.uint32)
                seed3_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.seeds3)
                self.buffers.append(seed3_buffer)

                self.seeds4 = numpy.random.randint(180, 1000000, (threads*blocks*threads*blocks,)).astype(numpy.uint32)
                seed4_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.seeds4)
                self.buffers.append(seed4_buffer)


		#write arguments to buffer as nPerRNG

	        arg_arr = numpy.array([nPerThread, xmean, xsd, xlow, xhigh, ymean, ysd, ylow, yhigh], dtype=numpy.float32)
		print arg_arr
		arg_bufr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arg_arr)
	        self.buffers.append(arg_bufr)

	def _run_kernel(self):
        	
		"""
        	Run the kernel (self.program). Subclasses should overwrite this.
        	"""
        	self.program.Gaussian2DIntegrator(self.queue, self.global_size, None, *self.buffers)
		#Create empty array to output the random numbers result from device-->host
		result_arr = numpy.empty(self.resultshape, dtype=numpy.float32)
		cl.enqueue_copy(self.queue, result_arr, self.buffers[0])		
		print result_arr.size
		print result_arr
		return sum(sum(result_arr))
