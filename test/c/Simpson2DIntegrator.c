\
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
   	int * c,
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

int MatrixValue(int i1, int i2, int i3, int index, int size)
{
if ((index==0) || (index ==size-1))
	return i1;
else if (index % 2 == 0)
	return i2;
else
	return i3;

} 

int main(int argc, char* argv[])
	{
	float *results;
	float *args;
	int groupID, localID,  i,j;
	int threadsX = 12, threadsY = 12;			//change the threads per block
	int blocksX =2, blocksY =2;
	float xlow = -1.0;			//lower limit
	float xhigh = 1.0;			//upper limit
	float ylow = -2.0;
	float yhigh = 2.0;

	int Nx = blocksX*threadsX;
	int Ny = blocksY*threadsY;
	float mean = 0.0;
	float sd = 1.0;
	float *x;
	int **c;
	float h,k;

	static int coef = 0;
	//Create xi array

	//Create the Coefficients memory
	c = (int **)malloc(Ny*sizeof(int*));
	for (i=0; i<Ny; i++)
		{
		c[i] = (int *)malloc(sizeof(int)*Nx);
		}
	
	//Populate grid
	for (i=0; i<Ny; i++)
		{
		for (j=0; j<Nx; j++)
			{
			if ((i==0) || (i==Ny-1))
				c[i][j] = MatrixValue(1,2,4,j,Nx);
			else if (i % 2 == 0)
				//print a 2,8,4,8...2 row
				c[i][j] = MatrixValue(2,8,4,j,Nx);
			else
				//print a 4, 16, 8, 16....4 row
				c[i][j] = MatrixValue(4,16,8,j,Nx);
			printf("  %d", c[i][j]);
			}
		printf("\n");
		}

for (i=0; i<Ny; i++)
	{
	free((void *)c[i]);
	}
free((void *)c);
//printf("Integral between %f and %f is %f, calculated in %f seconds\n", low, high, sum, time2);
return 1;
}
