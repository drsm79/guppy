typedef struct
	{
        float mean;
        float sd;
        float high;
        float low;
        } GaussVars_t;
 
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
   __global float (*d_Results)[3],
   __global unsigned (*seeds)[4],
   __global const float* args)
{
	GaussVars_t xGauss, yGauss;
	//Get 1D global index from 2 dimensional workgroups
	int gid = get_global_size(0)*get_global_id(1) + get_global_id(0);
	int iRand, iBin_x, iBin_y;					//for loop variables
   	int i;
	float rfx, rfy;						//random float (rf)
 	int TrialsPerThread = args[0];				//local, number of randoms produced by this thread

	//Gaussian Variables
	xGauss.mean = args[1];
	xGauss.sd = args[2];
	xGauss.low = args[3];
	xGauss.high = args[4];

	yGauss.mean = args[5];
	yGauss.sd = args[6];
	yGauss.low = args[7];
	yGauss.high = args[8];

	int bincount_x = args[9];
	int bincount_y = args[10];

	float binwidth_x = (xGauss.high-xGauss.low)/bincount_x;		//width of the block
  	float binwidth_y = (yGauss.high-yGauss.low)/bincount_y;

	unsigned z1,z2,z3,z4;					//local seeds for random number generator, unique to thread
	//initialise seeds from global memory
	z1 = seeds[gid][0];
 	z2 = seeds[gid][1];
        z3 = seeds[gid][2];
        z4 = seeds[gid][3];

	int index;
	int binindex;
	int t1=0, t2=0,t3=0;

	for (iRand=0;iRand<TrialsPerThread;iRand+=1)
		{
		//Get uniform random number, and scale to range [l_LOW, l_HIGH]
		rfx = HybridTaus(&z1,&z2,&z3,&z4);
		rfx = ScaleRange(rfx, xGauss.low, xGauss.high);
		rfy = HybridTaus(&z1, &z2, &z3, &z4);
		rfy = ScaleRange(rfy, yGauss.low, yGauss.high);
		for (iBin_x =0; iBin_x < bincount_x; iBin_x++)
			{
			//if ((rfx > (iBin_x*binwidth_x)+xGauss.low) && (rfx <= ((iBin_x+1)*binwidth_x)+xGauss.low))
			//	{
				for (iBin_y = 0; iBin_y < bincount_y; iBin_y++)
					{
   					binindex  = (iBin_y*bincount_x)+iBin_x;
                                       	index = (gid*bincount_x*bincount_y) + binindex;
					if (index > t1) t1 = index;
					if (binindex > t2) t2 = binindex;
					if (gid > t3) t3 = gid;
					d_Results[index][0] = t1;
                                        d_Results[index][1] = t2;
                                        d_Results[index][2] = bincount_y;

					//if ((rfy > (iBin_y*binwidth_y)+yGauss.low) && (rfy <= ((iBin_y+1)*binwidth_y)+yGauss.low))
					//	{
					//	Update the bin
					//	binindex  = iBin_y*bincount_x +iBin_x; 
					//	index = gid*bincount_x*bincount_y + binindex;
					//	d_Results[index][0] += 1;
					//	d_Results[index][1] += Gauss2D(rfx, rfy, xGauss, yGauss);
					//	d_Results[index][2] = d_Results[index][1]/d_Results[index][0];						
					//	break;
					//	}
			//		}
				break;			
				}
			}
		}
}


