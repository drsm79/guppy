__kernel void thread_names(
    __global float *output
) {
    int gid = get_global_id(0);
    output[gid] = gid;
}
