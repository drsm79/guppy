	////STRUCTS

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

unsigned TausStep(unsigned *z, int S1, int S2, int S3, unsigned M)
{
   unsigned b=(((*z << S1) ^ *z) >> S2);
   return *z = (((*z & M) << S3) ^ b);
 }

unsigned LCGStep(unsigned *z, unsigned A, unsigned C)
{
return *z=(A**z+C);
}

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

float ScaleRange(float u1, float low, float range)
{
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

__kernel void Gaussian2DIntegrator(
   __global float d_Results[512][512],
   __global unsigned seed1[512][512],
   __global unsigned seed2[512][512],
   __global unsigned seed3[512][512],
   __global unsigned seed4[512][512],
   __global const float* args)
{
	GaussVars_t xGauss, yGauss;
	ThreadDims_t ThreadDims;
	int localx = get_local_id(0);
	int localy = get_local_id(1);
	int globalx = get_global_id(0);
	int globaly = get_global_id(1);
	int groupx = get_group_id(0);
	int groupy = get_group_id(1);

	int iRand, iBin;					//for loop variables
   	int i;
	float rfx, rfy;						//random float (rf)
 	int l_NPERTHREAD = args[0];				//local, number of randoms produced by this thread

	//Gaussian Variables
	xGauss.mean = args[1];
	xGauss.sd = args[2];
	xGauss.low = args[3];
	xGauss.high = args[4];

	yGauss.mean = args[5];
	yGauss.sd = args[6];
	yGauss.low = args[7];
	yGauss.high = args[8];

	ThreadDims.xlength = (xGauss.high-xGauss.low)/get_global_size(0);		//width of the block
  	ThreadDims.ylength = (yGauss.high-yGauss.low)/get_global_size(1);
	ThreadDims.x = (get_global_id(0)*ThreadDims.xlength)+xGauss.low;
	ThreadDims.y = (get_global_id(1)*ThreadDims.ylength)+yGauss.low;

	unsigned z1,z2,z3,z4;					//local seeds for random number generator, unique to thread
	//initialise seeds from global memory
	z1 = seed1[globalx][globaly];
 	z2 = seed2[globalx][globaly];
        z3 = seed3[globalx][globaly];
        z4 = seed4[globalx][globaly];

	//loop over the random number generation
	__private float sum = 0.0f;
	__private int n = 0;

	for (iRand=0;iRand<l_NPERTHREAD;iRand+=1)
		{
		//Get uniform random number, and scale to range [l_LOW, l_HIGH]
		rfx = HybridTaus(&z1,&z2,&z3,&z4);
		rfx = ScaleRange(rfx, ThreadDims.x, ThreadDims.xlength);
		rfy = HybridTaus(&z1, &z2, &z3, &z4);
		rfy = ScaleRange(rfy, ThreadDims.y, ThreadDims.ylength);
		n++;
		sum += Gauss2D(rfx, rfy, xGauss, yGauss); //<---breaking it 
		}

	d_Results[globalx][globaly] = (sum/n)*ThreadDims.xlength*ThreadDims.ylength;		
//	d_Results[globalx][globaly] = sum;
}


