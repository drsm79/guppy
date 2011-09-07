#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy
import math
from operator import mul
import functiongenerator

class FunctionGenerator(Kernel):

	kernel_file = "FunctionGenerator.cl"
	function_string = ""
	dim_string = ""

	def __init__(self, function_string, dim):
		#Inherit the kernel constructor
		Kernel.__init__(self)
		if dim:
			self.dimstring = "#define DIMCOUNT "+str(dim)+"\n"
			self.dim = dim;
		if function_string:
			self.function_string = self.dimstring+"float fn(float *dim) \n { \n return "+function_string+"\n } \n"				
    	
	def _create_buffers(self, grid, globalsize=(16,16)):
        	mf = cl.mem_flags
		input_arr = functiongenerator.creategrid(grid)
		self.output_shape = functiongenerator.getoutputshape(grid)
		self.global_size = globalsize
		self.Threads = reduce(mul, globalsize)
		self.dtype = numpy.float32		

		#Input array, flattened to a 1D array
		self.input = numpy.array(input_arr).astype(self.dtype)
		self.input_size = reduce(mul, self.input.shape)
		#Empty Output array to read results back to
                self.output_size = self.input_size/self.dim
		self.output = numpy.empty(shape=(self.output_size,), dtype = self.dtype)

		#Write Input to Buffer
                input_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.input)
                self.buffers.append(input_buff)

		#Write Args to Buffer
		EvalsPerThread  =  math.ceil(float(self.output_size)/float(self.Threads))

		self.args = numpy.array([EvalsPerThread, self.output_size]).astype(self.dtype)
		arg_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.args)
		self.buffers.append(arg_buff)

		output_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.output.nbytes)
		self.buffers.append(output_buff)

	def _build_program(self):

		#Insert the generic function into the kernel code 
		self.kernel_string = self.function_string+self.kernel_string
		self.program = cl.Program(self.ctx, self.kernel_string).build()

	def _run_kernel(self):
        	
        	self.program.FunctionGenerator(self.queue, self.global_size, None, *self.buffers)
		
		#Create empty array to output the random numbers result from device-->host
		cl.enqueue_copy(self.queue, self.output, self.buffers[-1])		
		result = self.output.reshape(self.output_shape)
		return result		
