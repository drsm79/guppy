#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy

class ThreadNames(Kernel):
    """
    A simple test kernel.
    """
    kernel_file = "threadNames.cl"

    def _create_buffers(self, parallelism=5):
        """
        Define the necessary input and ouptut Buffer objects for the Kernel,
        store in self.input_buffers and self.output_buffers. Subclasses should
        overwrite this.
        """
        mf = cl.mem_flags

        self.size = parallelism

        parallelism = numpy.long(parallelism)
        #self.buffers.append((parallelism,))
        # local_size = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=parallelism)
        # self.buffers.append(local_size)

        output = numpy.zeros(parallelism)
        self.buffers.append(cl.Buffer(self.ctx, mf.WRITE_ONLY, output.nbytes))

    def _run_kernel(self):
        """
        Run the kernel (self.program). Subclasses should overwrite this.
        """
        self.program.thread_names(self.queue, self.global_size, None, *self.buffers)
        result = numpy.empty(self.size, dtype = numpy.float32)
        # Internal? copy of output from gpu
        cl.enqueue_copy(self.queue, result, self.buffers[-1])

        return result