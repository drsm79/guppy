#!/usr/bin/env python
# encoding: utf-8
import pyopencl as cl
import logging
import os

__all__ = ['Gauss', 'Test', 'Random']

class Kernel:
    """
    A Kernel holds a string that contains the C code to compile with pyOpenCL
    and manages running that kernel.
    """
    kernel_file = ''

    def __init__(self, cl_file_location='../../../c/kernels'):
        """
        Initialising the Kernel loads the C code from kernel_file and sets up
        the necessary CommandQueue and context.
        """
        #TODO: replace print with logger
        #TODO: investigate precompilation of kernels
        #TODO: set dType uniformly
        # log here print cl.version.VERSION_TEXT
        if self.kernel_file:
            base = __file__.rsplit('/', 1)[0]
            filename = os.path.join(base, cl_file_location, self.kernel_file)
            # log here print os.path.abspath(filename)
            self.kernel_string = open(filename).read()
        else:
            self.kernel_string = ""
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.buffers = []
        # This is a tuple describing the dimensions of the output
        self.global_size = (0,)

    def __call__(self, *args, **kwargs):
        """
        Substitute in any changes into the kernel_string and create the
        necessary Buffer and Program objects.
        """
        self._create_buffers(*args, **kwargs)
        self._build_program()
        return self._run_kernel()

    def _create_buffers(self, args, kwargs):
        """
        Define the necessary input and ouptut Buffer objects for the Kernel,
        store in self.input_buffers and self.output_buffers. Subclasses should
        overwrite this.

        TODO: Handling of buffers needs significant improvement
        """
        raise NotImplementedError

    def _build_program(self):
        """
        Separated out so individual Kernels can define special compile options.
        Default is none.
        """
        self.program = cl.Program(self.ctx, self.kernel_string).build()

    def _run_kernel(self):
        """
        Run the kernel (self.program). Subclasses should overwrite this. Takes
        no arguments, everything should be held in self.
        """
        raise NotImplementedError
