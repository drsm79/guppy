#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy
import math
from operator import mul




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

		#get the order, shape of the grid
		self.old_shape = grid.shape
                #get the new shape, check for dimension agreement
		shapelist = list(grid.shape)
		
		dimcheck = shapelist.pop()
		if dimcheck-self.dim != 0:
			raise Exception, "Dimension mismatch: expected: "+str(self.dim)+", found "+str(dimcheck)

		self.new_shape = tuple(shapelist)
		self.Evals = reduce(mul, self.new_shape)
		self.Threads = reduce(mul, globalsize)
		
		#Input array, flattened to a 1D array
		self.input = grid.flatten().astype(numpy.float32)
		#Empty Output array to read results back to
		self.output = numpy.empty(shape=self.new_shape, dtype = numpy.float32)

		#Write Input to Buffer
                input_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.input)
                self.buffers.append(input_buff)

		#Write Args to Buffer
		EvalsPerThread  =  math.ceil(float(self.Evals)/float(self.Threads))
		self.args = numpy.array([EvalsPerThread, self.Evals]).astype(numpy.float32)
		arg_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.args)
		self.buffers.append(arg_buff)

		output_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.output.nbytes)
		self.buffers.append(output_buff)

		#calculate the number of map operations each thread has to calculate
	       	self.global_size = globalsize

	def _build_program(self):

		#Insert the generic function into the kernel code 
		self.kernel_string = self.function_string+self.kernel_string
		self.program = cl.Program(self.ctx, self.kernel_string).build()

	def _run_kernel(self):
        	
        	self.program.FunctionGenerator(self.queue, self.global_size, None, *self.buffers)
		
		#Create empty array to output the random numbers result from device-->host
		cl.enqueue_copy(self.queue, self.output, self.buffers[-1])		
		return self.output
		
