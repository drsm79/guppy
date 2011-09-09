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

    	def _create_buffers(self, trials=1200, globalsize=(16,16), bincount=(10,10), xmean=0, xsd=1, xlow=-1, xhigh=1, \
			    ymean=0, ysd=1, ylow=-1, yhigh=1,seeds=None):
		"""
        	Define the necessary input and ouptut Buffer objects for the Kernel,
        	store in self.input_buffers and self.output_buffers. Subclasses should
        	overwrite this.
       	 	"""
        	mf = cl.mem_flags

        	# The Gaussian 2D integrator function takes the following parameters in its argument vector:
        	#
        	#       d_Result = the output array of random numbers
        	#       d_seed = four random numbers in an array
		#	float* args: list of necessary parameters bundled together 
        	#       args[0]: TrialsPerThread = number of random numbers to generate per thread
        	#       args[1]: x mean
		#	args[2]: x standard deviation
		#	args[3]: x low
		#	args[4]: x high
		#	args[5]: y mean
		#	args[6]: y standard deviation
		#	args[7]: y low
		#	args[8]: y high
		#	args[9]: x bincount
		#	args[10]: y bincount	

		#Ensure that globalsize and bincount tuples are the right length
		if len(bincount) < 2:
			bincount = (bincount[0], bincount[0])
		if len(globalsize) < 2:
			globalsize = (globalsize[0], globalsize[0])

	
		TrialsPerThread = math.ceil(float(trials)/float(globalsize[0]*globalsize[1]))
	       	self.global_size = (globalsize[0],globalsize[1])
		self.resultlength = globalsize[0]*globalsize[1]*bincount[0]*bincount[1]
		self.resultshape = (self.resultlength,3)	
		self.binarea = ((xhigh-xlow)*(yhigh-ylow))/(bincount[0]*bincount[1])
		
		#set the writeable output array to hold the random numbers
		self.result = numpy.empty(shape=self.resultshape, dtype=numpy.float32)
		output_buff = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result)
	 	self.buffers.append(output_buff)

		#Generate seeds, required to be >180 by HybridTaus algorithm 
	        self.seeds = numpy.random.randint(180, 1000000, size=(self.resultlength, 4)).astype(numpy.uint32)
        	seed_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.seeds)
		self.buffers.append(seed_buffer)
		
		#Write argument array to global memory
	        arg_arr = numpy.array([TrialsPerThread, xmean, xsd, xlow, xhigh, ymean, ysd, ylow, yhigh, bincount[0], bincount[1]], dtype=numpy.float32)
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
		
		#Integrate the result and return single value to the caller
		return sum(result_arr)[2] * self.binarea/(self.global_size[0]*self.global_size[1])
