__kernel void FunctionGenerator( 
   __global const float *d_input,
   __global const float *args,
   __global float *d_output)
{

float dim[DIMCOUNT];

int idim, iEval;
float res=0;
int gid = get_global_id(1)*get_global_size(0)+get_global_id(0);
int gsize = get_global_size(0)*get_global_size(1);
int EvalsPerThread = args[0];
int TotalEvals = args[1];
int index = 0;
//Read values from d_input, map the predefined function, then update the output array

for (iEval=0; iEval<EvalsPerThread; iEval++)
	{
	index = gid*EvalsPerThread+iEval;
	//Populate the dimension values
	for (idim=0; idim < DIMCOUNT; idim++)
		{
		if (index < TotalEvals)
			dim[idim] = d_input[index*DIMCOUNT+idim];
		}
	res = fn(dim);
	//write result to output array;
	if (index < TotalEvals)
		d_output[index] = res;
	}

}

