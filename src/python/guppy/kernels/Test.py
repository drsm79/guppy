#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
from guppy.kernels import Kernel
import numpy
import numpy.linalg as la

class Test(Kernel):
    """
    A simple test kernel.
    """
    kernel_file = "test.cl"

    def _create_buffers(self, host_buffers):
        """
        Define the necessary input and ouptut Buffer objects for the Kernel,
        store in self.input_buffers and self.output_buffers. Subclasses should
        overwrite this.
        """
        mf = cl.mem_flags
        self.host_buffers = host_buffers
        a = self.host_buffers['a']
        b = self.host_buffers['b']

        self.global_size = a.shape
        self.dtype = a.dtype

        self.a_plus_b = self.host_buffers['a'] + self.host_buffers['b']

        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

        self.buffers.append(a_buf)
        self.buffers.append(b_buf)
        self.buffers.append(cl.Buffer(self.ctx, mf.WRITE_ONLY, b.nbytes))

    def _run_kernel(self):
        """
        Run the kernel (self.program). Subclasses should overwrite this.
        """
        self.program.sum(self.queue, self.global_size, None, *self.buffers)
        a_plus_b = numpy.empty(self.global_size, dtype = self.dtype)
        # Internal? copy of output from gpu
        cl.enqueue_copy(self.queue, a_plus_b, self.buffers[-1])
        return la.norm(a_plus_b - self.a_plus_b)
