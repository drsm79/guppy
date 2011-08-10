// Header file for oclRandom
// Joe Jenkinson
// August 2011

#ifndef OCLRANDOM_H
#define OCLRANDOM_H

#define PI 3.14159265358979f

//structure to hold 4 seeds for each random number generator. 
typedef struct{
  unsigned s1;
  unsigned s2;
  unsigned s3;
  unsigned s4;
} seed_t;

//All available Distributions
typedef enum
{
dtFlat,
dtGaussian,
dtPoissonDiscrete,
dtPoissionCont,
dtGamma,
dtLogNormal
} dist_t;

// Histogram Bin structure. Each bin has a cumulative sum (cSum)
// and a frequency (n) in order to calculate the mean value in the bin
//typedef struct{
 //       float cSum;
  //      int n;
//} bin_t;

#endif
































































































































































