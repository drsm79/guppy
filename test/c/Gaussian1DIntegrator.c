
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

unsigned TausStep(unsigned *z, int S1, int S2, int S3, unsigned M)
{
   unsigned b=(((*z << S1) ^ *z) >> S2);
   return *z = (((*z & M) << S3) ^ b);
 }

unsigned LCGStep(unsigned *z, unsigned A, unsigned C)
{
return *z=(A**z+C);
}

//random number generator
float HybridTaus(unsigned *z1,unsigned *z2,unsigned *z3,unsigned *z4)
 {
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  return 2.3283064365387e-10 * (              // Periods
     TausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1
    TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1
       TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1
     LCGStep(z4, 1664525, 1013904223UL)        // p4=2^32
     );
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

pwr = x-mean;
pwr *= pwr;
pwr /= -2*sd*sd;
temp = exp(pwr);
temp /= sd*sqrt(2*PI);
return temp;
}

void Gaussian1DIntegrator( 
   float * d_Results,
   int groupID, int localID,
   unsigned int d_Seed[4],
   const float* args,
   float (*d_Bins)[3])
{
	int globalID;
	int iRand, iBin;					//for loop variables
   	float rf;						//random float (rf)
   	int l_NPERTHREAD = args[0];				//local, number of randoms produced by this thread
   	int   l_BLOCKS = args[1];                               //Number of concurrent workgroups
        int   l_THREADS = args[2];                              //Number of threads within the workgroup
	float l_MEAN =args[3];					//local mean and standard deviation of the Gaussian
	float l_SD = args[4];	
 	float l_LOW = args[5];					//local lower and upper limits to integrate between
	float   l_HIGH = args[6];
   	float th_low, th_high;					//local limits of the block
	float l_BLOCKWIDTH = (l_HIGH-l_LOW)/l_BLOCKS;		//width of the block
   	float l_THREADWIDTH = l_BLOCKWIDTH/l_THREADS;		//width of a bin within the block
	unsigned z1,z2,z3,z4;					//local seeds for random number generator, unique to thread
	float temp;

	//initialise seeds from global memory
	z1 = d_Seed[0];
   	z1 = d_Seed[1];
   	z2 = d_Seed[2];
   	z3 = d_Seed[3];

	globalID = groupID*l_THREADS+localID;
	//Get the dimensions of this workgroup;
	th_low = (globalID*l_THREADWIDTH)+l_LOW;
	th_high = ((globalID+1)*l_THREADWIDTH)+l_LOW;

	//loop over the random number generation
	for (iRand=0;iRand<l_NPERTHREAD;iRand+=1)
		{
		//Get uniform random number, and scale to range [l_LOW, l_HIGH]
		rf = HybridTaus(&z1,&z2,&z3,&z4);
		rf = ScaleRange(rf, th_low, th_high);
		//Bin the number
		if ((rf>th_low) && (rf<=th_high))
					{
					//bin the number
					d_Bins[localID][0] += 1;
					d_Bins[localID][1] += fnGauss(rf, l_MEAN, l_SD);
					d_Bins[localID][2] = d_Bins[localID][1]/d_Bins[localID][0];
					}
		}
}

int main(int argc, char* argv[])
	{
	float *results;
	float *args;
	unsigned  seeds[4];
	int groupID, localID,  i;
	int threads = 1024;			//change the threads per block
	int blocks =1;				//change the number of blocks
	float low = -1.0;			//lower limit
	float high = 1.0;			//upper limit

	//set the arguments
	args = (float *)malloc(sizeof(float)*7);
	args[0] = 1500;
        args[1] =blocks;                               //Number of concurrent workgroups
        args[2] =threads;                              //Number of threads within the workgroup
        args[3] = 0.0f;                                  //local mean and standard deviation of the Gaussian
        args[4] =1.0f;
        args[5] =  low;                                  //local lower and upper limits to integrate between
        args[6] =  high;
	float sum=0.0;
	double time1, time2;	
	//loop over each block
	results = (float *)malloc(sizeof(float)*blocks);
	time1= get_time();
	for (groupID = 0; groupID < blocks; groupID++)
	{
		float d_Bins[threads][3];
		for (localID =0; localID < threads; localID++)
			{
			d_Bins[localID][0]=0;
			d_Bins[localID][1] =0;
			d_Bins[localID][2] = 0;
			}
		float blocksum=0;
		for (localID = 0; localID < threads; localID++)
			{
			//set the seeds
			seeds[0] = 180+rand();
        		seeds[1] = 180+rand();
        		seeds[2] = 180+rand();
        		seeds[3] = 180+rand();
			Gaussian1DIntegrator(results, groupID, localID,seeds, args, d_Bins);
			blocksum += d_Bins[localID][2];
			}
		results[groupID] = blocksum*(high-low)/(threads*blocks);
		sum += results[groupID];
	}
	time2 = get_time();
	free(results);
		
free(args);
printf("Integral between %f and %f is %f, calculated in %f seconds across %d threads\n", low, high, sum, time2, blocks*threads);
return 1;
}
