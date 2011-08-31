#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy
import math

class Gaussian1DIntegrator(Kernel):
	"""
    	A simple test kernel, doesn't load the code from a file
    	"""
	kernel_file = "Gaussian1DIntegrator.cl"

    	def _create_buffers(self, trials=1200, globalsize=128, bincount=100, mean=0, sd=1, low=-1, high=1, seeds=None):
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
        	#       args[0]: TrialsPerThread = number of random numbers to generate per thread
        	#       args[1]: BinCount = number of bins
		#	args[2]: TargetTrials
		#	args[3]: Mean
		#	args[4]: Standard Deviation
		#	args[5]: Lower Limit
		#	args[6]: Upper Limit
		#
	
		TrialsPerThread = math.ceil(float(trials)/float(globalsize))
		self.resultshape = (bincount*globalsize, 3)	
		self.globalsize = (globalsize,)
		self.binwidth = float(high-low)/float(bincount)
		
		#set the writeable output array to hold the random numbers
		self.result = numpy.empty(shape=self.resultshape, dtype=numpy.float32)
		output_buff = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result)
	 	self.buffers.append(output_buff)

		#Generate seeds, required to be >180 by HybridTaus algorithm 
  	      	if not seeds:
	            	self.seeds = numpy.random.randint(180, 1000000, (globalsize, 4)).astype(numpy.uint32)
       	 	else:
            		self.seeds = seeds
        	seed_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.seeds)
        	self.buffers.append(seed_buffer)

		#write arguments to buffer as nPerRNG

	        arg_arr = numpy.array([TrialsPerThread, bincount,trials,  mean, sd, low, high], dtype=numpy.float32)
		arg_bufr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arg_arr)
	        self.buffers.append(arg_bufr)

	def _run_kernel(self):
        	
		"""
        	Run the kernel (self.program). Subclasses should overwrite this.
        	"""
        	self.program.Gaussian1DIntegrator(self.queue, self.globalsize, None, *self.buffers)
		#Create empty array to output the random numbers result from device-->host
		result_arr = numpy.empty(self.resultshape, dtype=numpy.float32)
		cl.enqueue_copy(self.queue, result_arr, self.buffers[0])		
		print result_arr
		return sum(result_arr)[2]*self.binwidth/self.globalsize[0]
		
