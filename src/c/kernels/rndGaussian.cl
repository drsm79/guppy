//Kernel to create a Gaussian random number

// S1, S2, S3, and M are all constants, and z is part of the
// private per-thread generator state.
unsigned TausStep(unsigned *z, int S1, int S2, int S3, unsigned M)
{
   unsigned b=(((*z << S1) ^ *z) >> S2);
   return *z = (((*z & M) << S3) ^ b);
 }

 // A and C are constants
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


void BoxMullerTrans(float *u1, float *u2)
{
	const float PI = 3.14159265358979f;
    float   r = native_sqrt(-2.0f * log(*u1));
    float phi = 2 * PI * (*u2);
    *u1 = r * native_cos(phi);
    *u2 = r * native_sin(phi);
}

__kernel void rndGaussian(
   __global float* d_Rand,
   __global unsigned int (*d_Seed)[3],
   __global const float* args)
{
   int globalID = get_global_id(0);
   int i, index;
   float rf1, rf2;
   int l_nPerRNG, l_RNG_COUNT, l_TARGET;
   float l_mean, l_sd;

   unsigned z1,z2,z3,z4;
   z1 = d_Seed[globalID][0];
   z1 = d_Seed[globalID][1];
   z2 = d_Seed[globalID][2];
   z3 = d_Seed[globalID][3];

	l_nPerRNG = args[0];
	l_RNG_COUNT = args[1];
	l_TARGET = args[2];
	l_mean = args[3];
	l_sd = args[4];
 	for (i=0;i<l_nPerRNG;i+=2)
	{
	//Get 2 uniform random numbers
	rf1 = HybridTaus(&z1,&z2,&z3,&z4);
	rf2 = HybridTaus(&z1,&z2,&z3,&z4);
	
	//Do Box-Muller Transform
	BoxMullerTrans(&rf1, &rf2);
	
	//Write Random 1
	index  =globalID+i*l_RNG_COUNT;
	if (index < l_TARGET)
		d_Rand[index] = l_mean+l_sd*rf1;
	else break;
	//Write Random 2
	index  =globalID+(i+1)*l_RNG_COUNT;
	if (index < l_TARGET)
		d_Rand[index] = l_mean+l_sd*rf2;
	else break;
	}
}







