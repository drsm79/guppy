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

typedef struct{
  unsigned s1;
  unsigned s2;
  unsigned s3;
  unsigned s4;
} seed_t;

__kernel void GPUrand(
   __global float* d_Rand,
   __global seed_t* d_Seed, //change
   const unsigned int nPerRNG,
   const int RNG_COUNT)
{
   int globalID = get_global_id(0);
   float rfl;
   unsigned z1,z2,z3,z4;
   z1 = d_Seed[globalID].s1;
   z2 = d_Seed[globalID].s2;
   z3 = d_Seed[globalID].s3;
   z4 = d_Seed[globalID].s4;
   for (int i=0;i<nPerRNG;i++)
       {
       d_Rand[globalID+i*RNG_COUNT] = HybridTaus(&z1,&z2,&z3,&z4);
       }
}

