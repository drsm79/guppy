GupPy 'Kernel' class
Currently, all Python classes descend from a generic 'Kernel' base class, and should be saved in the github/guppy/src/python/guppy/kernels folder within the GupPy directory hierarchy. Here, there should be a '__init_.py' file which contains the declaration of the 'Kernel' class, and its member functions. The Kernel base class is responsible for all the 'boilerplate' code necessary to set up an OpenCL kernel: it reads the GPU kernel from a separate file, creates a OpenCL context to run on a GPU device, creates buffers to read and write arrays of data between the host and the device, builds the program and finally runs the kernel. Because most of the code required to configure a GPU device is contained within the Kernel class, it is relatively simple to derive Sub-classes from 'Kernel' - see the Gaussian Integrator examples later. All that is required to write a sub-class is:
1.	The OpenCL kernel (usually saved as a .cl file) has to be saved in the github/guppy/src/c/kernels folder, and its name passed to the Python class through the 'kernel_file' property.
2.	The arguments of the OpenCL kernel have to be created in memory by creating buffers. Sub classes override the __create_buffers function, typically in a manner as follows: 
o	An 'empty' Numpy array of correct shape and datatype (usually float32) is created
o	This array is buffered to memory using the cl.Buffer() method. This requires the relevant array to be passed as the 'hostbuf' parameter, and flags to say whether the buffer is read only, write only or both read-write. This method allocates global memory only: if __local memory is needed then use the cl.LocalMemory method instead. See PyOpenCL documentation for more information in both cases.
o	The buffer is appended to a list (mainly to preserve code readability), and this list is passed to the kernel call later on. It is important that the buffers are created in the same order that the are declared in the __kernel arguments within the '.cl' file.   
3.	The __run_kernel function is overwritten by subclasses, for example: 
o	The kernel is executed by calling self.program.<kernel name> where <kernel name> should be replaced with the name of the __kernel function within the OpenCL '.cl' kernel file.
o	A local Numpy array of the correct shape/datatype is created to output the kernel results to. The required PyOpenCL function for this is 'enqueue_copy()' e.g.     'cl.enqueue_copy(self.queue, local_arr, gpu_buffer[0])'  would read the contents of the first gpu_buffer into a local_arr Numpy Array. 
o	A 'return' result should be provided. This could simply be the raw Numpy array itself, or else one could do something like Integrate this Numpy array and return a single value, as is the case in the Gaussian Integrators examples. This would probably differ depending on the nature of your kernel. 
This is essentially all that is required to create a Python kernel class. Python unit tests can be created to test, time and validate the kernels; each kernel test is saved in the github/guppy/test/python/guppy_t/kernels_t folder, where the '<kernel name>_t.py'  naming convention indicates that it is a test script for the <kernel name> kernel.  Within the test, the kernel can be imported using the line 'from guppy.kernels.<Python Kernel Name> import <Python Kernel Name>' where <Python Kernel Name> is the name of the Python sub-class written, as per the method above. The unit test can be written as normal (see the Python documentation for more), and the kernel called, using the 1D Gaussian as an example, by:
?
Int = Gaussian1DIntegrator()
self.start_timer()
self.result = Int(trials=self.trials, globalsize=self.globalsize, bincount =self.bincount, \
mean=self.mean, sd=self.sd, low=self.low, high=self.high)
self.stop_timer()
print self.result
It should be noted that any timings of kernels should be around the 'self.result = ' line, and not include the 'Int = Gaussian1DIntegrator' line, otherwise the timing will include all the time taken to initialise the OpenCL context, command queue, platform etc, and not the kernel execution time as required.   
Gaussian 1D Integrator class
As an example of creating a kernel class using the methods outlined above, a one-dimensional Gaussian Integrator class was created. This example uses three files: 'Gaussian1DIntegrator.cl' in the source folder for OpenCL kernels (location: github/guppy/src/c/kernels), the Python kernel class 'Gaussian1DIntegrator.py' (location: github/guppy/src/python/guppy/kernels), and a unit test class 'Gaussian1DIntegrator_t.py' (location: github/guppy/test/python/guppy_t/kernels_t).
Gaussian1DIntegrator.cl
The OpenCL file contains the code to run on the GPU, which is essentially 'c' code with a few OpenCL functions/keywords. It contains the actual kernel function that is executed (Gaussian1DIntegrator, identified with the __kernel prefix), and a number of functions to generate the random numbers, gaussian function etc. The functions 'TausStep', 'LCGStep' and 'HybridTaus' functions are all required to generate a random number between [0,1], using what is essentially the Hybrid Tausworth algorithm (see: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html ). The function 'ScaleRange' maps a [0,1] random number, to a number across the range [low, high] where low and high are user specified variables representing the lower and upper limits of integration. The function 'Gauss' is pretty self explanatory, it merely contains the normalized gaussian function requiring an x value, the mean and standard deviation of the distribution as arguments. Note that the maths of the function was explicitly written out, largely because the use of built-in 'pow' functions occasionally returned strange values: (this could have been due to human error rather than more fundamental bugs with OpenCL builtin math functions, but it is worth mentioning in case similar behaviour is observed in the future).
__kernel Gaussian1DIntegrator requires 3 buffers as its arguments: a *d_Results[3] array to hold the results of the integration, *d_Seeds[4] to hold unsigned values to seed the random number generator in each thread (4 values per thread, each value > 180) and finally a *args array to hold additional argument parameters e.g. the mean, standard deviation, integration limits etc. The kernel effectively generates a user-specified number of random trials, and for each value the corresponding Gaussian value is calculated and placed in the appropriate bin in the d_Results array. After kernel execution, the bins can be copied to the CPU and integrated by taking a summation of values (see Numerical Integration techniques). A single value can be output, and validated if required within the test class.
Gaussian1DIntegrator.py
This is the Python kernel class, descending from the generic 'Kernel' class. The kernel_file property is set to point to the Gaussian1D.cl kernel file: e.g. kernel_file = "Gaussian1DIntegrator.cl". The create buffers method is overwritten, and indicates the arguments required at the user end, namely the globalsize (total number of threads), the bincount (number of bins), the number of trials (number of randomly generated floats, spread across all concurrent threads) and finally the Gaussian function parameters such as Mean, Standard Deviation, and Integration limits. The buffers are created for the Results, Seeds and Argument Vector (see internal code comments for more details), and appended to a list of buffers. 
The kernel is executed on the GPU device by overwritten the '__run_kernel' method. The call self.program.Gaussian1DIntegrator performs all the operations on the GPU, and the results can be read from the device back to the host using enqueue_copy. It should be noted that whilst the globalsize of the kernel must be user specified, localsize can be - and in the current examples, is - left as None, letting PyOpenCL itself calculate it for you. Due to hardware differences, leaving it as None allows for convenience, but potentially at the expense of performance, if PyOpenCL did not return a suitable localsize parameter. Finally, after reading the kernel results, the summation of the bins * binwidth is a numerical estimate for the Integral, and is returned as the result of the kernel class.
Gaussian1DIntegrator_t.py
This is the test class, from which the kernel is called, and any necessary tests on the results are performed. The tests are executed by running this Python script through the command line in the usual manner, i.e. python Gaussian1DIntegrator_t.py. At present the only test is timing the kernel run time, and printing the output result, but future validation tests could easily be appended.
Gaussian 2D Integrator class
Gaussian2DIntegrator.cl
The Gaussian 2D Integrator class extends the functionality of the Gaussian 1D Integrator to allow for 2-dimensional integration in x and y. The kernel requires the same arguments as before (*d_Results[3] for the results, *d_Seeds[4] to seed the random number generators and *args for additional parameters), but there are some fundamental differences between the 1D and 2D versions which are worth addressing. Firstly, 2D integration naturally implies that the globalsize and localsize tuples are extended to 2 dimensions, but unfortunately OpenCL restricts the use of pointers to pointers in kernel arguments. This obviously complicates the nature of the problem because you cannot pass a variable length 2-dimensional array as an argument, but this can be addressed in two different ways. Firstly, one could attempt to use Mako templates to set the dimensions of arrays in kernel arguments, but this method wasnt pursued sufficiently to vouch for its viability - further investigation would be required. Instead, a method of 'flattening' a 2 dimensional array into a 1 dimension array was implemented which would allow for the use of single pointers, albeit with slightly more complicated index arithmetic in the kernel itself. The index to global memory could be calculated as follows:
1.	The 2D Integrator requires tuples for globalsize and bincount. For example globalsize=(32,32) and bincount=(10,10) would create 8192 threads each of 100 bin, only spread across a 2-dimensional grid. Thus, the 1-dimensional equivalent would be 8192*100 = 819200 items long. It would then require logic to calculate a) the appropriate thread and then b) the correct bin within that thread.
2.	To get the unique 1D index of each thread on a 2-dimensional grid of threads on a GPU, all that is required is the line 
?
int gid = get_global_size(0)*get_global_id(1) + get_global_id(0);
where get_global_size gets the total length of the x dimension, and get_global_id gets the unique x, y coordinates of the specific thread. It will return a number between 0 and globalsize-1, i.e. 0-8191 in the (32,32) example given above.
3.	A similar method is required to calculate the 1D bin index from a 2D configuration. The kernel loops over the x and y directions using the iBin_x, and iBin_y for loop variables and, once the correct bin has been located, the bin index is calculated by 
?
binindex  = (iBin_y*bincount_x)+iBin_x;
So, in the bincount=(10,10) example, iBin_x and iBin_y both loop over the range 0-9 to return all the bins with coordinates between (0,0) and (9,9). This is mapped to a 1D index, binindex, between 0-99.
4.	The 1D thread index gid has been identified, and the bin to update within that thread has been calculated with binindex. To continue the globalsize=(32,32), bincount=(10,10) example,  gid is found across the range 0-8191, and binindex similarly in the 0-99 range. The *d_Results array is 32*32*10*10=819200 items long, and can be considered to be dimensioned so that elements 0-99 are the bins for thread0, 100-199 are the bins for thread1 etc, all the way to 819100-819199 being the bins for thread8191. Once this becomes apparent, it is simple to get a globalindex by 
?
index = (gid*bincount_x*bincount_y) + binindex;
where bin_count_x and bin_count_y are constants representing the dimensions of the bins: (10,10) in the example. gid and binindex are calculated as demonstrated above
Gaussian2DIntegrator.py
This is the Python kernel class, and differs only slightly from the 1-dimensional equivalent. The main changes are that more arguments are required whenever creating an instance of Gaussian2DIntegrator: parameters for the mean, standard deviation and integration limits are required for the y dimension, in addition to the x variables from before. Similarly, globalsize and bincount must be passed as tuples, with both x and y dimensions explicitly provided, e.g globalsize=(32,32) has the same number of threads as the 1-dimensional (8192,) equivalent, only it distributes them over 2 dimensions, not one. If a single value is passed e.g. globalsize=(32,) this is intepreted by the Gaussian2DIntegrator.py as being (32,32), ie. if only one dimension is provided, it copies this size to the y dimension too, in order to create a symmetrical grid. Other than this, the Gaussian2DIntegrator.py class and Gaussian2DIntegrator_t.py behave almost identically to the 1-dimensional Integrator.
Function Generator class
The intention was to create a kernel that would be able to map a generic function onto a n-dimensional grid of points, and output the results as a numpy array. This is currently incomplete and functionality is not completely stable: it currently performs well for simple functions with few dimensions, but performance is poor/doesn't work once the size of the output array reaches a certain size. Here is a description of initial attempts to write the class.
Currently, the user must initialise the FunctionGenerator with a string containing a c function written in <math.h>, and an integer containing the number of dimensions required. Once initialised, a call can be made to the FunctionGenerator instance specifying both the number of GPU threads to run over, and a numpy array of size (n,3) containing the limits of each dimension. For example:
?
self.globalsize = (32,32)
self.grid = numpy.array([[0,10,10],[0,10,10],[0,10,10]])
self.function_string = 'dim[0]+dim[1]*dim[2];'
FG = FunctionGenerator(self.function_string, 3)
result = FG(globalsize=self.globalsize, grid=self.grid)
Here, the FunctionGenerator is provided a 3-dimensional function (each dimension must be named dim[0], dim[1] etc) as a string (remember to append the function with a semicolon), and the  number of dimensions, 3. self.grid is a numpy array of shape (N,3) where N is the number of dimensions; for the k'th dimension, grid[k,0] contains the lower bound, grid[k,1] the upper bound, and grid[k,2] is the number of divisions. In this example, we have 3 dimensions each at integer divisions between 0 and 10, which would require 1000 function evaluations on the GPU: fn(0,0,0), fn(0,0,1)....fn(9,9,9). It would then be possible to index the 'result' array to find the function value at a given point on the grid, e.g. result[0,5,6] would be the result at dim[0]=0, dim[1]=5, dim[2] = 6.
FunctionGenerator.cl
This file contains the kernel code that is run be each thread on the GPU. As arguments, it requires an array d_params, which is essentially a flattened equivalent of the 'grid' parameters mentioned earlier; an array args which group together variables required by each thread; and finally d_output which is a flattened array to hold the evaluated function results.
The kernel code, at time of writing is as follows:
?
__kernel void FunctionGenerator(
   __global const float *d_params,
   __global const float *args,
   __global float *d_output)
{
float dim[DIMCOUNT];                                                    //private array to store all dimensions
int idim, iEval;                                                        //for loop variables
float res=0;                                                            //holds the result of the function evaluation
int gid = get_global_id(1)*get_global_size(0)+get_global_id(0);         //1D global id
int gsize = get_global_size(0)*get_global_size(1);                      //global size
int EvalsPerThread = args[0];                                           //number of function evaluations performed by each thread
int Size = args[1];
int index = 0;
float dimlow,dimhigh;
int dimcount;
int a,b;
int dimindex;
 
 for (iEval=0; iEval<Size; iEval++)
        {
        index = gid*EvalsPerThread+iEval;
        if (index < Size)
                {
                //Populate the dimension values
                //Get dimension indexes from index
                b = Size;
                a = index;
                for (idim=0; idim < DIMCOUNT; idim++)
                        {
                        dimlow = d_params[idim*3];
                        dimhigh = d_params[idim*3+1];
                        dimcount = (int)d_params[idim*3+2];
 
                        b /=dimcount;
                        dimindex = a/b;
                        a %= b;
                        //getdimvalue from dimindex;                 
                        dim[idim] = dimindex*(dimhigh-dimlow)*(1.0f/dimcount) +dimlow;
                        }
                //Evaluate function from dimensions
                res = fn(dim);
                d_output[index] = res;
                }
        }
}
The thread asks for its unique global id, gid, and then iterates over the number of function evaluations it is required to do (for example, 1000 points over 32 threads, would require each thread to do at least 1000/32 evaluations each, before checking whether the necessary 'size' has been exceeded). 'Index' is a 1D index of the position on the grid of points that is needed to be evaluated. The kernel knows that it is evaluating a specific location on the grid (at point 'index'), and needs to calculate the values of each dimension at this point. To do so, it iterates over all the dimension (idim < DIMCOUNT), reads the boundaries from the d_params array and calculates the dimension index for each dimension. This assumes that dim[0] is the slowest changing dimension, and that dim[N-1] is the quickest changing. For example:
Take the 3 dimensional [0,10,10],[0,10,10],[0,10,10] example from before. If dim[0] is the slowest changing, and dim[2] is the quickest, then a 1D array of all points would be (0,0,0),(0,0,1),(0,0,2)...(0,0,9),(0,1,0),(0,1,1)....(9,9,8),(9,9,9). The indexes for each dimension, dimindex is calculated from the temporary variables a and b, and the value of that dimension at position dimindex is given by dimindex*(RANGE OF DIMENSION)/(NUMBER OF DIVISIONS) + (LOWER DIMENSION LIMIT). This value is added to the local array dim[], and after completed it can be passed to the fn() function to evaluate the function at that given set of points. This result is ouput to the 1D d_output array, which can then be reshaped by the Python Class on completion.
  


