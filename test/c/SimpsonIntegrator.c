
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time()
{
static int sec = -1;
struct timeval t1;

gettimeofday(&t1, NULL);
if (sec < 0 ) sec = t1.tv_sec;
return (1.0e-6*t1.tv_usec)+(t1.tv_sec - sec);
}


//scales a random number [0,1] to the range [low, high]
float ScaleRange(float u1, float low, float high)
{
float range;
range = high-low;
return (u1*range)+low;
}

//Gaussian function
float fnGauss(float x, const float mean, const float sd)
{
float pwr, temp;
const float PI = 3.14159265358979f;

pwr = pow(x-mean, 2)/(-2*pow(sd,2));
temp = exp(pwr);
temp /= sqrt(2*PI*pow(sd,2));
return temp;
}

void SimpsonIntegrator( 
   	float * results, 
	float * x,
   	float * c,
   	int groupID, int localID,
   	const float* args)
{
//get the globalID;
float l_x;
float l_mean = args[0];
float l_sd = args[1];
float threads = args[2];

int globalID=groupID*threads+localID;
l_x = x[globalID];


results[globalID] = fnGauss(l_x, l_mean, l_sd)*c[globalID];


}


int main(int argc, char* argv[])
	{
	float *results;
	float *args;
	int groupID, localID,  i;
	int threads = 512;			//change the threads per block
	int blocks =256;				//change the number of blocks
	float low = -1.0;			//lower limit
	float high = 0.4;			//upper limit
	int N = blocks*threads;
	float mean = 0.0;
	float sd = 1.0;
	float *x;
	float *c;
	float h = (high-low)/N;
	static int coef = 0;
	//Create xi array
	x = (float *)malloc(N*sizeof(float));
	c = (float *)malloc(N*sizeof(float));
	for (i = 0; i<N; i++)
		{
		x[i] = low+i*h;
		if ((i==0) || (i == N-1))
			c[i] = 1;
		else if (coef == 0)
			c[i] = 2;
		else
			c[i] = 4;
		coef = 1-coef;
		}

	//set the arguments
	args = (float *)malloc(sizeof(float)*3);
	args[0] = mean;
	args[1] = sd;
	args[2] = threads;

	float sum=0.0;
	double time1, time2;	
	//loop over each block
	results = (float *)malloc(sizeof(float)*N);
	time1= get_time();
	for (groupID = 0; groupID < blocks; groupID++)
	{
		for (localID = 0; localID < threads; localID++)
			{
			SimpsonIntegrator(results, x, c, groupID, localID, args);
			}
	}

	for (i=0; i<N; i++)
		{
		sum += results[i];
		printf("results are %f\n", results[i]);
		}
	sum *=h/3;
	time2 = get_time();
	free(results);
		
free(args);
free(x);
free(c);
printf("Integral between %f and %f is %f, calculated in %f seconds\n", low, high, sum, time2);
return 1;
}
