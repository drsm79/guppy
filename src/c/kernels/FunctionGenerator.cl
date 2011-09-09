__kernel void FunctionGenerator( 
   __global const float *d_params,
   __global const float *args,
   __global float *d_output)
{

float dim[DIMCOUNT];							//private array to store all dimensions
int idim, iEval;							//for loop variables
float res=0;								//holds the result of the function evaluation
int gid = get_global_id(1)*get_global_size(0)+get_global_id(0);		//1D global id 
int gsize = get_global_size(0)*get_global_size(1);			//global size 
int EvalsPerThread = args[0];						//number of function evaluations performed by each thread
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

