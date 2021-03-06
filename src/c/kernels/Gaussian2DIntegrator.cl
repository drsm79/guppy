//////////////////////////////////////////
// Gaussian 2D Integrator kernel	//
// Joe Jenkinson jj9854@bris.ac.uk	//
// August 2011				//
//////////////////////////////////////////

// Structs

typedef struct
	{
        float mean;
        float sd;
        float high;
        float low;
        } GaussVars_t;
 
//Shared Functions

unsigned TausStep(unsigned *z, int S1, int S2, int S3, unsigned M)
{
   unsigned b=(((*z << S1) ^ *z) >> S2);
   return *z = (((*z & M) << S3) ^ b);
 }

unsigned LCGStep(unsigned *z, unsigned A, unsigned C)
{
return *z=(A**z+C);
}

//Random number generator. Requires 4 unsigned seeds >180 
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

//Scale a [0,1] number to a number between [low, high]
float ScaleRange(float u1, float low, float high)
{
float range = high-low;
return (u1*range)+low;
}

//2D Gaussian function, Gaussian(x,y) 
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

// Kernel function

__kernel void Gaussian2DIntegrator(
   __global float (*d_Results)[3],
   __global unsigned (*seeds)[4],
   __global const float* args)
{
	GaussVars_t xGauss, yGauss;							// structs to hold the means, sd and range for each variable
	int gid = get_global_size(0)*get_global_id(1) + get_global_id(0);		//1D global index for 'flattened' 2D array
	int iRand, iBin_x, iBin_y;							//for loop variables
	float rfx, rfy;									//random x,y floats (rf)
 	int TrialsPerThread = 	args[0];						//number of random numbers produced by this thread
	xGauss.mean = 	 	args[1];						//Gaussian Variables
	xGauss.sd =		args[2];						// ...
	xGauss.low = 		args[3];						// ...
	xGauss.high = 		args[4];						// ...
	yGauss.mean = 		args[5];						// ...
	yGauss.sd = 		args[6];						// ...	
	yGauss.low = 		args[7];						// ...
	yGauss.high = 	 	args[8];						// ...
	int bincount_x = 	args[9];						//number of bins in x dimension
	int bincount_y = 	args[10];						//number of bins in y dimension
	float binwidth_x = (xGauss.high-xGauss.low)/bincount_x;				//width of one bin in x direction
  	float binwidth_y = (yGauss.high-yGauss.low)/bincount_y;				//width of one bin in the y direction
	unsigned z1,z2,z3,z4;								//local seeds for random number generator, unique to thread
	int binindex;									//1D index to each bin, from 'flattened' 2D array of bins	
	int index;									//1D index to memory from 'flattened' 2D arrays of threads/bins

	//initialise seeds from global memory
	z1 = seeds[gid][0];
 	z2 = seeds[gid][1];
        z3 = seeds[gid][2];
        z4 = seeds[gid][3];

	//Repeat for the required number of trials
	for (iRand=0;iRand<TrialsPerThread;iRand+=1)
		{
		//Get uniform random number, and scale to range [l_LOW, l_HIGH]
		rfx = HybridTaus(&z1,&z2,&z3,&z4);
		rfx = ScaleRange(rfx, xGauss.low, xGauss.high);
		rfy = HybridTaus(&z1, &z2, &z3, &z4);
		rfy = ScaleRange(rfy, yGauss.low, yGauss.high);
		//loop over the x direction
		for (iBin_x =0; iBin_x < bincount_x; iBin_x++)
			{
			//test for correct x bin
			if (rfx > iBin_x*binwidth_x+xGauss.low && rfx <= (iBin_x+1)*binwidth_x+xGauss.low)
				{
				//loop over the y direction
				for (iBin_y = 0; iBin_y < bincount_y; iBin_y++)
					{
					//test for correct y bin
					if (rfy > iBin_y*binwidth_y+yGauss.low && rfy <= (iBin_y+1)*binwidth_y+yGauss.low)
						{
						//calculate indexes and update global memory
   						binindex  = (iBin_y*bincount_x)+iBin_x;
                                		index = (gid*bincount_x*bincount_y) + binindex;
						d_Results[index][0] += 1;
                                		d_Results[index][1] += Gauss2D(rfx, rfy, xGauss, yGauss);
                                		d_Results[index][2] = d_Results[index][1]/d_Results[index][0];
						break;
						}
					}
				break;				
				}
			}
		}
}


