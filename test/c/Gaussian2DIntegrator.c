
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct
        {
        float mean;
        float sd;
        float high;
        float low;
        } GaussVars_t;

typedef struct
        {
        float x;
        float y;
        float xlength;
        float ylength;
        } ThreadDims_t;


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

float Gauss2D(float x, float y, const GaussVars_t xGauss, const GaussVars_t yGauss)
{
float a, b, temp;
const float PI = 3.14159265358979f;
a = x-xGauss.mean;
a *= a;
a /= -2*xGauss.sd*xGauss.sd;
b = y-yGauss.mean;
b *= b;
b /= -2*yGauss.sd*yGauss.sd;
a +=b;
temp = exp(a);
temp /= xGauss.sd*yGauss.sd*2*PI;
return temp;
}

void Gaussian2DIntegrator( 
   float * d_Results,
   int globalID,
   int localxID,
   int localyID,
   int groupxID,
   int groupyID,
   unsigned int d_Seed[4],
   const float* args)
{	
        GaussVars_t xGauss, yGauss;
        ThreadDims_t ThreadDims;
  	int   nPerThread = args[0];                             //local, number of randoms produced by this thread
        int   blocks_x = args[1];                               //Number of concurrent workgroups
        int   blocks_y = args[2];                              //Number of threads within the workgroup
        int   threads_x = args[3];
	int   threads_y = args[4];
	//Gaussian Variables
        xGauss.mean = args[5];
        xGauss.sd = args[6];  
        xGauss.low = args[7];
        xGauss.high = args[8];
        yGauss.mean = args[9];
        yGauss.sd = args[10];
        yGauss.low = args[11];
        yGauss.high = args[12];
	
	float globalxID = groupxID*blocks_x*threads_x+localxID;
	float globalyID = groupyID*blocks_y*threads_y+localyID;
        ThreadDims.xlength = (xGauss.high-xGauss.low)/(blocks_x*threads_x);               //width of the block
        ThreadDims.ylength = (yGauss.high-yGauss.low)/(blocks_x*threads_y);
        ThreadDims.x = globalxID*ThreadDims.xlength+xGauss.low;
        ThreadDims.y = globalyID*ThreadDims.ylength+yGauss.low;

	int iRand, iBin;					//for loop variables
   	float rfx, rfy;						//random float (rf)

	unsigned z1,z2,z3,z4;					//local seeds for random number generator, unique to thread
	//initialise seeds from global memory
	z1 = d_Seed[0];
   	z1 = d_Seed[1];
   	z2 = d_Seed[2];
   	z3 = d_Seed[3];

      //loop over the random number generation
        float sum = 0.0f;
        int n = 0;

        for (iRand=0;iRand<nPerThread;iRand+=1)
                {
                //Get uniform random number, and scale to range [l_LOW, l_HIGH]
                rfx = HybridTaus(&z1,&z2,&z3,&z4);
                rfx = ScaleRange(rfx, ThreadDims.x, ThreadDims.xlength);
                rfy = HybridTaus(&z1, &z2, &z3, &z4);
                rfy = ScaleRange(rfy, ThreadDims.y, ThreadDims.ylength);
                n++;
                sum += Gauss2D(rfx, rfy, xGauss, yGauss);
                }
	d_Results[globalID] = (sum/n)*ThreadDims.xlength*ThreadDims.ylength;
	}


int main(int argc, char* argv[])
	{
	float *results;
	float *args;
	unsigned  seeds[4];
	int groupID_x, localID_x, groupID_y, localID_y,  i;
	int threads_x = 16;
	int threads_y = 16;			//change the threads per block
	int blocks_x =1;
	int blocks_y = 1;				//change the number of blocks
	
	float xmean = 0.0f;
	float xsd = 1.0f;
	float xlow = -1.0;			//lower limit
	float xhigh = 1.0;			//upper limit
	float ymean = 0.0f;
        float ysd = 1.0f;
        float ylow = -1.0;                      //lower limit
        float yhigh = 1.0;                      //upper limit
        
	//set the arguments
	args = (float *)malloc(sizeof(float)*13);
	args[0] = 300;
        args[1] =blocks_x;                               //Number of concurrent workgroups
        args[2] =blocks_y;                              //Number of threads within the workgroup
        args[3] = threads_x;                                  //local mean and standard deviation of the Gaussian
        args[4] =threads_y;
        args[5] = xmean;                                  //local lower and upper limits to integrate between
        args[6] =  xsd;
	args[7] = xlow;
	args[8] = xhigh;
	args[9] = ymean;
	args[10] = ysd;
	args[11] = ylow;
	args[12] = yhigh;
	float sum=0.0;
	double time1, time2;	
	int localIndex, blockIndex, globalIndex;
	//loop over each block
	results = (float *)malloc(sizeof(float)*blocks_x*blocks_y*threads_x*threads_y);
	time1= get_time();
	for (groupID_x = 0; groupID_x < blocks_x; groupID_x++)
		for (groupID_y = 0; groupID_y < blocks_y; groupID_y++)
			for (localID_x = 0; localID_x < threads_x; localID_x++)
				for (localID_y = 0; localID_y < threads_y; localID_y++)
					{
					//set the seeds
					seeds[0] = 180+rand();
        				seeds[1] = 180+rand();
        				seeds[2] = 180+rand();
        				seeds[3] = 180+rand();
					//Get the "globalID"
					localIndex = localID_y*threads_x + localID_x;
					blockIndex = groupID_y*threads_x*threads_y + localIndex;
					globalIndex = groupID_x*blocks_y*threads_x*threads_y + blockIndex;
					printf("localIndex = %d, blockIndex = %d, globalIndex =%d\n", localIndex, blockIndex, globalIndex);
					Gaussian2DIntegrator(results, globalIndex, localID_x, localID_y, groupID_x, groupID_y, seeds, args);
					sum += results[globalIndex];
					}
	time2 = get_time();
	free(results);
		
free(args);
printf("Integral between %f and %f is %f, calculated in %f seconds\n", xlow, xhigh, sum, time2);
return 1;
}
