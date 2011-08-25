
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

pwr = pow(x-mean, 2)/(-2*pow(sd,2));
temp = exp(pwr);
temp /= sqrt(2*PI*pow(sd,2));
return temp;
}

__kernel void Gaussian1DIntegrator( 
   __global float * d_Results,
   __global unsigned int (*d_Seed)[4],
   __global const float* args,
   __local float (* d_Bins)[3])
{
	int globalID = get_global_id(0);			//global id for this thread
   	int localID = get_local_id(0);				//local id for within the workgroup
	int groupID = get_group_id(0);				//group id of the workgroup
	int iRand, iBin;					//for loop variables
   	float rf;						//random float (rf)
   	int l_NPERTHREAD = args[0];				//local, number of randoms produced by this thread
   	int   l_BLOCKS = args[1];                               //Number of concurrent workgroups
        int   l_THREADS = args[2];                              //Number of threads within the workgroup
	float l_MEAN =args[3];					//local mean and standard deviation of the Gaussian
	float l_SD = args[4];	
 	float l_LOW = args[5];					//local lower and upper limits to integrate between
	float   l_HIGH = args[6];
	float th_low, th_high;
	float l_BLOCKWIDTH = (l_HIGH-l_LOW)/l_BLOCKS;		//width of the block
   	float l_THREADWIDTH = l_BLOCKWIDTH/l_THREADS;		//width of a bin within the block
	unsigned z1,z2,z3,z4;					//local seeds for random number generator, unique to thread
	float temp;
	int stride;

	//initialise seeds from global memory
	z1 = d_Seed[globalID][0];
   	z1 = d_Seed[globalID][1];
   	z2 = d_Seed[globalID][2];
   	z3 = d_Seed[globalID][3];

	d_Bins[localID][0] = 0;
	d_Bins[localID][1] = 0;
	d_Bins[localID][2] = 0;

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
					d_Bins[localID][1] += Gauss(rf, l_MEAN, l_SD);
					if (d_Bins[localID][0] != 0)
					d_Bins[localID][2] = d_Bins[localID][1]/d_Bins[localID][0];
					}
		}

	//GPU reduction within the block
	for (stride = get_local_size(0)/2; stride > 0; stride /= 2)
		{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localID < stride)
			{
			d_Bins[localID][2] += d_Bins[localID+stride][2];
			}
		
		}	

	if (localID ==0)
		{
		d_Results[groupID] = d_Bins[0][2]*l_THREADWIDTH;
		barrier(CLK_LOCAL_MEM_FENCE);
		}
}

