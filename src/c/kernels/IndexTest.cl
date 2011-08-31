__kernel void IndexTest(
   __global float* output
)
{

//get sizes and ids
int lx = get_local_id(0);
int ly = get_local_id(1);
int lxsize = get_local_size(0);
int lysize = get_local_size(1);
int localIndex, blockIndex, globalIndex;
int bx = get_group_id(0);
int by = get_group_id(1);
int bxsize = get_num_groups(0);
int bysize = get_num_groups(1);
 
//Get unique local ID
localIndex = ly*lxsize + lx;
blockIndex = by*lxsize*lysize + localIndex;
globalIndex = bx*bysize*lxsize*lysize + blockIndex;
if (globalIndex < get_global_size(0)*get_global_size(1))
   output[globalIndex] = globalIndex;
}
