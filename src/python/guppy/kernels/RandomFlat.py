#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy
import math

class RandomFlat(Kernel):
    """
    A simple test kernel, doesn't load the code from a file
    """
    kernel_file = "rndFlat.cl"

    def _create_buffers(self, size, threadcount, seeds=None):
        """
        Define the necessary input and ouptut Buffer objects for the Kernel,
        store in self.input_buffers and self.output_buffers. Subclasses should
        overwrite this.
        """
        mf = cl.mem_flags

        # The GPUrand function takes four inputs:
        #
        #       d_Rand = the output array of random numbers
        #       d_seed = four random numbers in an array
        #       nPerRNG = number of random numbers to generate per thread
        #       RNG_COUNT = number of threads used
	#	Target = the number of samples requested.
	#
	#	Example call:  rand(threadcount=256,size=777777)
	#	requests 777777 random numbers to be generated over 256 threads.
	#	777777 is clearly not a multiple of 256 so we say that nPerRNG ceil(777777.0/256.0) = 3039
	#	i.e. each of the 256 thread generates 3038 numbers each, giving a total of 3039*256= 777984, 
	#	and the kernels break their loops when the required 777777 have been calculated. This is why
	#	'nPerRNG' (for-loop variable), RNG_COUNT (ie. num of threads, required to index the output)
	# 	and Target (required size, to determine when the kernel can terminate early)


	#Calculate kernel variables
	nPerRNG = math.ceil(float(size)/float(threadcount))
        self.global_size = (threadcount,)
	self.size = size

	#set the writeable output array to hold the random numbers
        output_size = numpy.float32(size).nbytes * size
        self.buffers.append(cl.Buffer(self.ctx, mf.WRITE_ONLY, size=output_size))

	#Generate seeds, required to be >180 by HybridTaus algorithm 
        if not seeds:
            self.seeds = numpy.random.randint(180, 1000000, (threadcount, 4)).astype(numpy.uint32)
        else:
            self.seeds = seeds
        seed_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.seeds)
        self.buffers.append(seed_buffer)

	#write nPerRNG to buffer as nPerRNG
        n_arr = numpy.array(numpy.int(nPerRNG))
        n_bufr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n_arr)
        self.buffers.append(n_bufr)

	#write threadcount to buffer as RNG_COUNT
        tc_arr = numpy.array(numpy.int(threadcount))
        tc_bufr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tc_arr)
        self.buffers.append(tc_bufr)

	#Write size to buffer as TARGET
	t_arr = numpy.array(numpy.int(size))
	t_bufr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_arr)
	self.buffers.append(t_bufr)

    def _run_kernel(self):
        """
        Run the kernel (self.program). Subclasses should overwrite this.
        """
        self.program.rndFlat(self.queue, self.global_size, None, *self.buffers)
	#Create empty array to output the random numbers result from device-->host
	result_arr = numpy.empty(self.size, dtype=numpy.float32)
	cl.enqueue_copy(self.queue, result_arr, self.buffers[0])
	return result_arr
