float fnGauss(float x, const float mean, const float sd)
{
float pwr, temp;
const float PI = 3.14159265358979f;

pwr = pow(x-mean, 2)/(-2*pow(sd,2));
temp = exp(pwr);
temp /= sqrt(2*PI*pow(sd,2));
return temp;
}

__kernel void SimpsonIntegrator( 
   __global float * d_Results,
   __global const float* args,
   __global float *x)

{
	int globalID = get_global_id(0);			//global id for this thread
	float l_MEAN =args[0];					//local mean and standard deviation of the Gaussian
	float l_SD = args[1];	
	float c;

	if ((globalID == 0 ) || (globalID == get_global_size(0)-1))
		{
		c = 0;
		}
	else if (globalID % 2 == 1)
		{
		c = 2.0f;
		}
	else
		{
		c = 4.0f;
		}

	d_Results[globalID] = fnGauss(x[globalID], l_MEAN, l_SD)*c;

}

