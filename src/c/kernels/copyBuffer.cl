__kernel void sum(
   __global const float* input_1,
   __global float* output
)
{

__local float* temp;

	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int groupid = get_group_id(0);
	temp[lid] = groupid;
	output[gid] = temp[lid];
}
