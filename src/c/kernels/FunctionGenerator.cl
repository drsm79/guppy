__kernel void FunctionGenerator( 
   __global const float *d_input,
   __global const float *args,
   __global float *d_output)
{

float dim[DIMCOUNT];							//private array to store all dimensions
int idim, iEval;							//for loop variables
float res=0;								//holds the result of the function evaluation
int gid = get_global_id(1)*get_global_size(0)+get_global_id(0);		//1D global id 
int gsize = get_global_size(0)*get_global_size(1);			//global size 
int EvalsPerThread = args[0];						//number of function evaluations performed by each thread
int TotalEvals = args[1];
int index = 0;

for (iEval=0; iEval<EvalsPerThread; iEval++)
	{
	index = gid*EvalsPerThread+iEval;
	if (index < TotalEvals)
		{
		//Populate the dimension values
		for (idim=0; idim < DIMCOUNT; idim++)
			dim[idim] = d_input[index*DIMCOUNT+idim];
		res = fn(dim);
		//write result to output array;
		d_output[index] = res;
		}
	}
}

