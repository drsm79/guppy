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


__kernel void GPUrand(
   __global float* d_Rand,
   __global unsigned int (*d_Seed)[3],
   __global const unsigned int* nPerRNG,
   __global const int* RNG_COUNT,
   __global const int* TARGET)
{
   int globalID = get_global_id(0);
   int i, index;
   int l_nPerRNG, l_RNG_COUNT, l_TARGET; 
   unsigned z1,z2,z3,z4;
   z1 = d_Seed[globalID][0];
   z1 = d_Seed[globalID][1];
   z2 = d_Seed[globalID][2];
   z3 = d_Seed[globalID][3];
   l_nPerRNG = nPerRNG[0];
   l_RNG_COUNT = RNG_COUNT[0];
   l_TARGET = TARGET[0];
 	for (i=0;i<l_nPerRNG;i++)
	{
	index  =globalID+i*l_RNG_COUNT;
	if (index < l_TARGET)
   		d_Rand[index] = HybridTaus(&z1,&z2,&z3,&z4);
	else break;
	}
}

