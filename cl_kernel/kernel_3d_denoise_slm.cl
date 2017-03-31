/*
 * function: kernel_3d_denoise_slm
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

#define WORK_GROUP_WIDTH    8
#define WORK_GROUP_HEIGHT   1

#define WORK_BLOCK_WIDTH    8
#define WORK_BLOCK_HEIGHT   8

#define REF_BLOCK_X_OFFSET  1
#define REF_BLOCK_Y_OFFSET  4

#define REF_BLOCK_WIDTH  (WORK_BLOCK_WIDTH + 2 * REF_BLOCK_X_OFFSET)
#define REF_BLOCK_HEIGHT (WORK_BLOCK_HEIGHT + 2 * REF_BLOCK_Y_OFFSET)


inline void weighted_average (__read_only image2d_t input,
                              __local float4* ref_cache,
                              bool load_observe,
                              __local float4* observe_cache,
                              float4* restore,
                              float2* sum_weight,
                              float gain,
                              float threshold)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    const int local_id_x = get_local_id(0);
    const int local_id_y = get_local_id(1);
    const int group_id_x = get_group_id(0);
    const int group_id_y = get_group_id(1);

    int i = local_id_x + local_id_y * WORK_BLOCK_WIDTH;
    int start_x = mad24(group_id_x, WORK_BLOCK_WIDTH, -REF_BLOCK_X_OFFSET);
    int start_y = mad24(group_id_y, WORK_BLOCK_HEIGHT, -REF_BLOCK_Y_OFFSET);
    for (int j = i; j < REF_BLOCK_WIDTH * REF_BLOCK_HEIGHT; j += (WORK_GROUP_WIDTH * WORK_GROUP_HEIGHT)) {
        int corrd_x = start_x + (j % REF_BLOCK_WIDTH);
        int corrd_y = start_y + (j / REF_BLOCK_WIDTH);
        ref_cache[j] = read_imagef(input, sampler, (int2)(corrd_x, corrd_y));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (load_observe) {
        for (int i = 0; i < WORK_BLOCK_HEIGHT; i++) {
            observe_cache[i * WORK_BLOCK_WIDTH + local_id_x] =
                ref_cache[(i + REF_BLOCK_Y_OFFSET) * REF_BLOCK_WIDTH
                          + local_id_x + REF_BLOCK_X_OFFSET];
        }
    }

    float4 dist = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 gradient = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float weight = 0.0f;

#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            dist = (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, local_id_x + j)] -
                    observe_cache[local_id_x]) *
                   (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, local_id_x + j)] -
                    observe_cache[local_id_x]);
            dist = mad((ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[WORK_BLOCK_WIDTH + local_id_x]),
                       (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[WORK_BLOCK_WIDTH + local_id_x]),
                       dist);
            dist = mad((ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 2 * REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[2 * WORK_BLOCK_WIDTH + local_id_x]),
                       (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 2 * REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[2 * WORK_BLOCK_WIDTH + local_id_x]),
                       dist);
            dist = mad((ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 3 * REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[3 * WORK_BLOCK_WIDTH + local_id_x]),
                       (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 3 * REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[3 * WORK_BLOCK_WIDTH + local_id_x]),
                       dist);

            gradient = (float4)(ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)].s2,
                                ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)].s2,
                                ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)].s2,
                                ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)].s2);
            gradient = (gradient - ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, local_id_x + j)]) +
                       (gradient - ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)]) +
                       (gradient - ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 2 * REF_BLOCK_WIDTH + local_id_x + j)]) +
                       (gradient - ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 3 * REF_BLOCK_WIDTH + local_id_x + j)]);
            gradient.s0 = (gradient.s0 + gradient.s1 + gradient.s2 + gradient.s3) / 15.0f;
            gain = (gradient.s0 < threshold) ? gain : 2.0f * gain;

            weight = native_exp(-gain * (dist.s0 + dist.s1 + dist.s2 + dist.s3));
            weight = (weight < 0) ? 0 : weight;
            (*sum_weight).s0 = (*sum_weight).s0 + weight;

            restore[0] = mad(weight, ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, local_id_x + j)], restore[0]);
            restore[1] = mad(weight, ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)], restore[1]);
            restore[2] = mad(weight, ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 2 * REF_BLOCK_WIDTH + local_id_x + j)], restore[2]);
            restore[3] = mad(weight, ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 3 * REF_BLOCK_WIDTH + local_id_x + j)], restore[3]);
        }
    }

#pragma unroll
    for (int i = 1; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            dist = (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, local_id_x + j)] -
                    observe_cache[4 * WORK_BLOCK_WIDTH + local_id_x]) *
                   (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, local_id_x + j)] -
                    observe_cache[4 * WORK_BLOCK_WIDTH + local_id_x]);
            dist = mad((ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[5 * WORK_BLOCK_WIDTH + local_id_x]),
                       (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[5 * WORK_BLOCK_WIDTH + local_id_x]),
                       dist);
            dist = mad((ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 2 * REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[6 * WORK_BLOCK_WIDTH + local_id_x]),
                       (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 2 * REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[6 * WORK_BLOCK_WIDTH + local_id_x]),
                       dist);
            dist = mad((ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 3 * REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[7 * WORK_BLOCK_WIDTH + local_id_x]),
                       (ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 3 * REF_BLOCK_WIDTH + local_id_x + j)] -
                        observe_cache[7 * WORK_BLOCK_WIDTH + local_id_x]),
                       dist);

            gradient = (float4)(ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)].s2,
                                ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)].s2,
                                ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)].s2,
                                ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)].s2);
            gradient = (gradient - ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, local_id_x + j)]) +
                       (gradient - ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)]) +
                       (gradient - ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 2 * REF_BLOCK_WIDTH + local_id_x + j)]) +
                       (gradient - ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 3 * REF_BLOCK_WIDTH + local_id_x + j)]);
            gradient.s0 = (gradient.s0 + gradient.s1 + gradient.s2 + gradient.s3) / 15.0f;
            gain = (gradient.s0 < threshold) ? gain : 2.0f * gain;

            weight = native_exp(-gain * (dist.s0 + dist.s1 + dist.s2 + dist.s3));
            weight = (weight < 0) ? 0 : weight;
            (*sum_weight).s1 = (*sum_weight).s1 + weight;

            restore[4] = mad(weight, ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, local_id_x + j)], restore[4]);
            restore[5] = mad(weight, ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, REF_BLOCK_WIDTH + local_id_x + j)], restore[5]);
            restore[6] = mad(weight, ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 2 * REF_BLOCK_WIDTH + local_id_x + j)], restore[6]);
            restore[7] = mad(weight, ref_cache[mad24(i, 4 * REF_BLOCK_WIDTH, 3 * REF_BLOCK_WIDTH + local_id_x + j)], restore[7]);
        }
    }
}

__kernel void kernel_3d_denoise_slm( float gain,
                                     float threshold,
                                     __write_only image2d_t restoredPrev,
                                     __write_only image2d_t output,
                                     __read_only image2d_t input,
                                     __read_only image2d_t inputPrev1,
                                     __read_only image2d_t inputPrev2)
{
    float4 restore[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float2 sum_weight = {0.0f, 0.0f};

    __local float4 ref_cache[REF_BLOCK_HEIGHT * REF_BLOCK_WIDTH];
    __local float4 observe_cache[WORK_BLOCK_HEIGHT * WORK_BLOCK_WIDTH];

    weighted_average (input, ref_cache, true, observe_cache, restore, &sum_weight, gain, threshold);

#if 1

#if ENABLE_IIR_FILERING
    weighted_average (restoredPrev, ref_cache, false, observe_cache, restore, &sum_weight, gain, threshold);
#else
#if REFERENCE_FRAME_COUNT > 1
    weighted_average (inputPrev1, ref_cache, false, observe_cache, restore, &sum_weight, gain, threshold);
#endif

#if REFERENCE_FRAME_COUNT > 2
    weighted_average (inputPrev2, ref_cache, false, observe_cache, restore, &sum_weight, gain, threshold);
#endif
#endif

#endif

    restore[0] = restore[0] / sum_weight.s0;
    restore[1] = restore[1] / sum_weight.s0;
    restore[2] = restore[2] / sum_weight.s0;
    restore[3] = restore[3] / sum_weight.s0;

    restore[4] = restore[4] / sum_weight.s1;
    restore[5] = restore[5] / sum_weight.s1;
    restore[6] = restore[6] / sum_weight.s1;
    restore[7] = restore[7] / sum_weight.s1;

    const int global_id_x = get_global_id (0);
    const int global_id_y = get_global_id (1);

    write_imagef(output, (int2)(global_id_x, 8 * global_id_y), restore[0]);
    write_imagef(output, (int2)(global_id_x, mad24(8, global_id_y, 1)), restore[1]);
    write_imagef(output, (int2)(global_id_x, mad24(8, global_id_y, 2)), restore[2]);
    write_imagef(output, (int2)(global_id_x, mad24(8, global_id_y, 3)), restore[3]);
    write_imagef(output, (int2)(global_id_x, mad24(8, global_id_y, 4)), restore[4]);
    write_imagef(output, (int2)(global_id_x, mad24(8, global_id_y, 5)), restore[5]);
    write_imagef(output, (int2)(global_id_x, mad24(8, global_id_y, 6)), restore[6]);
    write_imagef(output, (int2)(global_id_x, mad24(8, global_id_y, 7)), restore[7]);
}

