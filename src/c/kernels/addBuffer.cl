__kernel void sum(
   __global float* input_1,
   __global float* input_2,
   __global float* output
)
{
	unsigned int gid = get_global_id(0);
	output[gid] = input_1[gid] ;

}
