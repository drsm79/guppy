
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

float ScaleRange(float u1, float low, float high)
{
float range;
range = high-low;
return (u1*range)+low;
}

float Gauss(float x, const float mean, const float sd)
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

__kernel void Gaussian1DIntegrator( 
   __global float (*d_Results)[3],
   __global unsigned int (*d_Seed)[4],
   __global const float* args)
{
	int gid = get_global_id(0);			//global id for this thread
	int iRand, iBin;					//for loop variables
   	float rf;						//random float (rf)
   	int TrialsPerThread = args[0];				//local, number of randoms produced by this thread
	int bincount = args[1];
	int TargetTrials = args[2];
	float mean =args[3];					//local mean and standard deviation of the Gaussian
	float sd = args[4];	
 	float low = args[5];					//local lower and upper limits to integrate between
	float high = args[6];
	unsigned z1,z2,z3,z4;					//local seeds for random number generator, unique to thread
	float binwidth = (high-low)/bincount;
	int index = 0;
	
	//initialise seeds from global memory
	z1 = d_Seed[gid][0];
   	z2 = d_Seed[gid][1];
   	z3 = d_Seed[gid][2];
   	z4 = d_Seed[gid][3];
	// initialise memory to zero
 	for (iBin=0; iBin<bincount; iBin++)
              {
               index = gid*bincount+iBin;
               d_Results[index][0] = 0;
               d_Results[index][1] = 0;
               d_Results[index][2] = 0;
              }
	//loop over the random number generation
 	for (iRand=0;iRand<TrialsPerThread;iRand+=1)
		{
		//Get uniform random number, and scale to range [l_LOW, l_HIGH]
		rf = HybridTaus(&z1,&z2,&z3,&z4);
		rf = ScaleRange(rf, low, high);
		//Bin the number
		if ((rf>low) && (rf<=high))
			for (iBin=0; iBin<bincount; iBin++)
				{
				if ((rf > (iBin*binwidth)+low) && (rf <= ((iBin+1)*binwidth)+low))
					{
					index = gid*bincount+iBin; 
					d_Results[index][0] += 1;
					d_Results[index][1] += Gauss(rf, mean, sd);
					d_Results[index][2] = d_Results[index][1]/d_Results[index][0];
					break;
					}
				}
		}
}

