__kernel void Array2D(__global float (* input)[2],
		 __global float (* output)[2])
{
int id = get_global_id(0);
output[id][0] = input[id][0];
output[id][1] = input[id][1];
} 
