========================================
Project layout
========================================

Python code
========================================

C code
========================================

Kernel
----------------------------------------
Kernels are short pieces of C code which are compiled via OpenCL and run on a
GPU.


Library
----------------------------------------
Library contains simple C functions that Kernels may include and use. The code
in library is compiled via pystats setup.py and made available to the kernels.