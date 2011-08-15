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
        #       RNG_COUNT = number of threads used

        self.parallelism = numpy.long(parallelism)
        #self.buffers.append((self.parallelism,))
        self.global_size = (size,)

        output_size = numpy.float32(size).nbytes * size * 20
        self.buffers.append(cl.Buffer(self.ctx, mf.WRITE_ONLY, output_size))

        if not seeds:
            self.seeds = numpy.random.rand(size * 4).astype(numpy.float32)
        else:
            self.seeds = seeds
        seed_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.seeds)
        self.buffers.append(seed_buffer)

        # TODO: check why an array is needed (something about buffer...) and look at alternatives
        n_arr = numpy.array(numpy.long(size/parallelism))
        n_bufr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n_arr)
        self.buffers.append(n_bufr)

        p_arr = numpy.array(numpy.long(parallelism))
        p_bufr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_arr)
        self.buffers.append(p_bufr)

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
        self.program.GPUrand(self.queue, self.global_size, None, *self.buffers, g_times_l=True)
        # print len(self.buffers)
        # print self.buffers[1].hostbuf
        # print self.buffers[1].get_info()
        # print self.buffers[1].get_host_array()
        # return self.buffers[1].get_host_array()