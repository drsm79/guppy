__kernel void sum(
   __global const float* input_1,
   __global float* output,
   __local float *loc
)
{
	int gid = get_global_id(0);
	int tid = get_local_id(0);
	uint stride;

	loc[tid] = input_1[gid];

	for (stride = get_local_size(0)/2; stride > 0; stride /= 2)
		{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (get_local_id(0) < stride)
			{
			loc[get_local_id(0)] += loc[get_local_id(0)+stride];
			}
		}

	if (get_local_id(0) == 0)
		{
		output[get_group_id(0)] = loc[0];
		}
	barrier(CLK_LOCAL_MEM_FENCE);
}

