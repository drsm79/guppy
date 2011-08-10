#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy

class Random(Kernel):
    """
    A simple test kernel, doesn't load the code from a file
    """
    kernel_file = "oclRandomKernel.cl"

    def _create_buffers(self, size, parallelism=1, seeds=None):
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


        self.parallelism = numpy.int32(parallelism)
        self.buffers.append(self.parallelism)
        #print numpy.zeros(size).nbytes
        #print numpy.float32(1).nbytes * size * 2
        # NOT SURE THIS IS RIGHT
        self.global_size = numpy.float32(1).nbytes * size * 2
        self.buffers.append(cl.Buffer(self.ctx, mf.WRITE_ONLY, self.global_size))

        if not seeds:
            self.seeds = numpy.random.rand(4).astype(numpy.float32)
        else:
            self.seeds = seeds
        seed_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.seeds)
        self.buffers.append(seed_buffer)

        n_arr = numpy.int32(size/parallelism)
        n_bufr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n_arr)
        self.buffers.append(n_bufr)

    #
    # def _build_program(self):
    #     TODO: set compile parameters to use include files, if necessary
    #   self.program = cl.Program(self.ctx, self.kernel_string).build(options=[])
    #

    def _run_kernel(self):
        """
        Run the kernel (self.program). Subclasses should overwrite this.
        """
        # This is broken
        self.program.GPUrand(self.queue, self.global_size, *self.buffers)
        return self.buffers[0]