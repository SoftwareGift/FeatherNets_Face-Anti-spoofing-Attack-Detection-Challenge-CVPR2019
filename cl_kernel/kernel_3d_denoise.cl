/*
 * function: kernel_3d_denoise
 *     3D Noise Reduction
 * gain:        The parameter determines the filtering strength for the reference block
 * threshold:   Noise variances of observed image
 * restoredPrev: The previous restored image, image2d_t as read only
 * output:      restored image, image2d_t as write only
 * input:       observed image, image2d_t as read only
 * inputPrev1:  reference image, image2d_t as read only
 * inputPrev2:  reference image, image2d_t as read only
 */

#ifndef REFERENCE_FRAME_COUNT
#define REFERENCE_FRAME_COUNT 2
#endif

#ifndef ENABLE_IIR_FILERING
#define ENABLE_IIR_FILERING 1
#endif

#define ENABLE_GRADIENT     1

#ifndef WORKGROUP_WIDTH
#define WORKGROUP_WIDTH    2
#endif

#ifndef WORKGROUP_HEIGHT
#define WORKGROUP_HEIGHT   32
#endif

#define REF_BLOCK_X_OFFSET  1
#define REF_BLOCK_Y_OFFSET  4

#define REF_BLOCK_WIDTH     (WORKGROUP_WIDTH + 2 * REF_BLOCK_X_OFFSET)
#define REF_BLOCK_HEIGHT    (WORKGROUP_HEIGHT + 2 * REF_BLOCK_Y_OFFSET)

inline int2 subgroup_pos(const int sg_id, const int sg_lid)
{
    int2 pos;
    pos.x = mad24(2, sg_id % 2, sg_lid % 2);
    pos.y = mad24(4, sg_id / 2, sg_lid / 2);

    return pos;
}

inline void average_slice(float8 ref,
                          float8 observe,
                          float8* restore,
                          float2* sum_weight,
                          float gain,
                          float threshold,
                          uint sg_id,
                          uint sg_lid)
{
    float8 grad = 0.0f;
    float8 gradient = 0.0f;
    float8 dist = 0.0f;
    float8 distance = 0.0f;
    float weight = 0.0f;

#if ENABLE_GRADIENT
    // calculate & cumulate gradient
    if (sg_lid % 2 == 0) {
        grad = intel_sub_group_shuffle(ref, 4);
    } else {
        grad = intel_sub_group_shuffle(ref, 5);
    }
    gradient = (float8)(grad.s1, grad.s1, grad.s1, grad.s1, grad.s5, grad.s5, grad.s5, grad.s5);

    // normalize gradient "1/(4*255.0f) = 0.00098039f"
    grad = fabs(gradient - ref) * 0.00098039f;
    //grad = mad(-2, gradient, (ref + grad)) * 0.0004902f;

    grad.s0 = (grad.s0 + grad.s1 + grad.s2 + grad.s3);
    grad.s4 = (grad.s4 + grad.s5 + grad.s6 + grad.s7);
#endif
    // calculate & normalize distance "1/255.0f = 0.00392157f"
    dist = (observe - ref) * 0.00392157f;
    dist = dist * dist;

    float8 dist_shuffle[8];
    dist_shuffle[0] = (intel_sub_group_shuffle(dist, 0));
    dist_shuffle[1] = (intel_sub_group_shuffle(dist, 1));
    dist_shuffle[2] = (intel_sub_group_shuffle(dist, 2));
    dist_shuffle[3] = (intel_sub_group_shuffle(dist, 3));
    dist_shuffle[4] = (intel_sub_group_shuffle(dist, 4));
    dist_shuffle[5] = (intel_sub_group_shuffle(dist, 5));
    dist_shuffle[6] = (intel_sub_group_shuffle(dist, 6));
    dist_shuffle[7] = (intel_sub_group_shuffle(dist, 7));

    if (sg_lid % 2 == 0) {
        distance = dist_shuffle[0];
        distance += dist_shuffle[2];
        distance += dist_shuffle[4];
        distance += dist_shuffle[6];
    }
    else {
        distance = dist_shuffle[1];
        distance += dist_shuffle[3];
        distance += dist_shuffle[5];
        distance += dist_shuffle[7];
    }

    // cumulate distance
    dist.s0 = (distance.s0 + distance.s1 + distance.s2 + distance.s3);
    dist.s4 = (distance.s4 + distance.s5 + distance.s6 + distance.s7);
    gain = (grad.s0 < threshold) ? gain : 2.0f * gain;
    weight = native_exp(-gain * dist.s0);
    (*restore).lo = mad(weight, ref.lo, (*restore).lo);
    (*sum_weight).lo = (*sum_weight).lo + weight;

    gain = (grad.s4 < threshold) ? gain : 2.0f * gain;
    weight = native_exp(-gain * dist.s4);
    (*restore).hi = mad(weight, ref.hi, (*restore).hi);
    (*sum_weight).hi = (*sum_weight).hi + weight;
}

inline void weighted_average (__read_only image2d_t input,
                              __local uchar8* ref_cache,
                              bool load_observe,
                              float8* observe,
                              float8* restore,
                              float2* sum_weight,
                              float gain,
                              float threshold,
                              uint sg_id,
                              uint sg_lid)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);
    const int group_id_x = get_group_id(0);
    const int group_id_y = get_group_id(1);

    int start_x = mad24(group_id_x, WORKGROUP_WIDTH, -REF_BLOCK_X_OFFSET);
    int start_y = mad24(group_id_y, WORKGROUP_HEIGHT, -REF_BLOCK_Y_OFFSET);

    int i = local_id_x + local_id_y * WORKGROUP_WIDTH;
    for ( int j = i; j < (REF_BLOCK_HEIGHT * REF_BLOCK_WIDTH);
            j += (WORKGROUP_HEIGHT * WORKGROUP_WIDTH) ) {
        int corrd_x = start_x + (j % REF_BLOCK_WIDTH);
        int corrd_y = start_y + (j / REF_BLOCK_WIDTH);

        ref_cache[j] = as_uchar8( convert_ushort4(read_imageui(input,
                                  sampler,
                                  (int2)(corrd_x, corrd_y))));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if WORKGROUP_WIDTH == 4
    int2 pos = subgroup_pos(sg_id, sg_lid);
    local_id_x = pos.x;
    local_id_y = pos.y;
#endif

    if (load_observe) {
        (*observe) = convert_float8(
                         ref_cache[mad24(local_id_y + REF_BLOCK_Y_OFFSET,
                                         REF_BLOCK_WIDTH,
                                         local_id_x + REF_BLOCK_X_OFFSET)]);
        (*restore) = (*observe);
        (*sum_weight) = 1.0f;
    }

    float8 ref[2] = {0.0f, 0.0f};
    __local uchar4* p_ref = (__local uchar4*)(ref_cache);

    // top-left
    ref[0] = convert_float8(*(__local uchar8*)(p_ref + mad24(local_id_y,
                            2 * REF_BLOCK_WIDTH,
                            mad24(2, local_id_x, 1))));
    average_slice(ref[0], *observe, restore, sum_weight, gain, threshold, sg_id, sg_lid);

    // top-right
    ref[1] = convert_float8(*(__local uchar8*)(p_ref + mad24(local_id_y,
                            2 * REF_BLOCK_WIDTH,
                            mad24(2, local_id_x, 3))));
    average_slice(ref[1], *observe, restore, sum_weight, gain, threshold, sg_id, sg_lid);

    // top-mid
    average_slice((float8)(ref[0].hi, ref[1].lo), *observe, restore, sum_weight, gain, threshold, sg_id, sg_lid);

    // mid-left
    ref[0] = convert_float8(*(__local uchar8*)(p_ref + mad24((local_id_y + 4),
                            2 * REF_BLOCK_WIDTH,
                            mad24(2, local_id_x, 1))));
    average_slice(ref[0], *observe, restore, sum_weight, gain, threshold, sg_id, sg_lid);

    // mid-right
    ref[1] = convert_float8(*(__local uchar8*)(p_ref + mad24((local_id_y + 4),
                            2 * REF_BLOCK_WIDTH,
                            mad24(2, local_id_x, 3))));
    average_slice(ref[1], *observe, restore, sum_weight, gain, threshold, sg_id, sg_lid);

    // mid-mid
    if (!load_observe) {
        average_slice((float8)(ref[0].hi, ref[1].lo), *observe, restore, sum_weight, gain, threshold, sg_id, sg_lid);
    }

    // bottom-left
    ref[0] = convert_float8(*(__local uchar8*)(p_ref + mad24((local_id_y + 8),
                            2 * REF_BLOCK_WIDTH,
                            mad24(2, local_id_x, 1))));
    average_slice(ref[0], *observe, restore, sum_weight, gain, threshold, sg_id, sg_lid);

    // bottom-right
    ref[1] = convert_float8(*(__local uchar8*)(p_ref + mad24((local_id_y + 8),
                            2 * REF_BLOCK_WIDTH,
                            mad24(2, local_id_x, 3))));
    average_slice(ref[1], *observe, restore, sum_weight, gain, threshold, sg_id, sg_lid);

    // bottom-mid
    average_slice((float8)(ref[0].hi, ref[1].lo), *observe, restore, sum_weight, gain, threshold, sg_id, sg_lid);
}

__kernel void kernel_3d_denoise ( float gain,
                                  float threshold,
                                  __write_only image2d_t restoredPrev,
                                  __write_only image2d_t output,
                                  __read_only image2d_t input,
                                  __read_only image2d_t inputPrev1,
                                  __read_only image2d_t inputPrev2)
{
    float8 restore = 0.0f;
    float8 observe = 0.0f;
    float2 sum_weight = 0.0f;

    const int sg_id = get_sub_group_id();
    const int sg_lid = (get_local_id(1) * WORKGROUP_WIDTH + get_local_id(0)) % 8;

    __local uchar8 ref_cache[REF_BLOCK_HEIGHT * REF_BLOCK_WIDTH];

    weighted_average (input, ref_cache, true, &observe, &restore, &sum_weight, gain, threshold, sg_id, sg_lid);

#if ENABLE_IIR_FILERING
    weighted_average (restoredPrev, ref_cache, false, &observe, &restore, &sum_weight, gain, threshold, sg_id, sg_lid);
#else
#if REFERENCE_FRAME_COUNT > 1
    weighted_average (inputPrev1, ref_cache, false, &observe, &restore, &sum_weight, gain, threshold, sg_id, sg_lid);
#endif

#if REFERENCE_FRAME_COUNT > 2
    weighted_average (inputPrev2, ref_cache, false, &observe, &restore, &sum_weight, gain, threshold, sg_id, sg_lid);
#endif
#endif

    restore.lo = restore.lo / sum_weight.lo;
    restore.hi = restore.hi / sum_weight.hi;

    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);
    const int group_id_x = get_group_id(0);
    const int group_id_y = get_group_id(1);

#if WORKGROUP_WIDTH == 4
    int2 pos = subgroup_pos(sg_id, sg_lid);
    local_id_x = pos.x;
    local_id_y = pos.y;
#endif

    int coor_x = mad24(group_id_x, WORKGROUP_WIDTH, local_id_x);
    int coor_y = mad24(group_id_y, WORKGROUP_HEIGHT, local_id_y);

    write_imageui(output,
                  (int2)(coor_x, coor_y),
                  convert_uint4(as_ushort4(convert_uchar8(restore))));
}

