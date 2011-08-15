__kernel void sum(
   __global const float* input_1,
   __global const float* input_2,
   __global float* output
)
{
	int gid = get_global_id(0);
	output[gid] = input_1[gid]+input_2[gid] ;

}
