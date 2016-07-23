/*
 * function: kernel_3d_denoise
 *     3D Noise Reduction
 * gain:        The parameter determines the filtering strength for the reference block
 * threshold:   Noise variances of observed image
 * output:      restored image, image2d_t as write only
 * input:       observed image, image2d_t as read only
 * inputPrev1:  reference image, image2d_t as read only
 * inputPrev2:  reference image, image2d_t as read only
 */

#define GROUP_WIDTH          8
#define GROUP_HEIGHT         1

#define BLOCK_WIDTH          3
#define BLOCK_HEIGHT         8

void __gen_ocl_force_simd8(void);

inline void weighted_average_subblock (
    __read_only image2d_t input, __write_only image2d_t output, float* sum_weight,
    bool load_abserved_block, float4* observed_block, float4* restored_block,
    float gain, float threshold, int row_batch, int col_batch)
{

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    const int group_id_x = get_group_id(0);
    const int group_id_y = get_group_id(1);
    const int local_id_x = get_local_id(0);
    const int local_id_y = get_local_id(1);

    int2 coord_read = (int2)( group_id_x * GROUP_WIDTH * BLOCK_WIDTH, group_id_y * GROUP_HEIGHT * BLOCK_HEIGHT);
    int2 coord_write = (int2)( group_id_x * GROUP_WIDTH * BLOCK_WIDTH, group_id_y * GROUP_HEIGHT * BLOCK_HEIGHT);

    uint subgroup_size = get_sub_group_size ();
    uint cur_id = get_sub_group_local_id ();
    int left_id = (cur_id == 0) ? 0 : cur_id - 1;
    int right_id = (cur_id == subgroup_size - 1) ? subgroup_size - 1 : cur_id + 1;

    uint8 ref_subgroup[2];
    uint8 middle_block[2];

    float4 ref_block[16][3];

    float4 dist = 0.0f;
    float weight[2] = {0.0f, 0.0f};

    // middle
    for (int i = 0; i < 2; i++) {
        middle_block[i] = intel_sub_group_block_read8 (input, coord_read);
        coord_read.y += BLOCK_HEIGHT;
    }
    ref_block[0][1] = convert_float4(as_uchar4(middle_block[0].s0));
    ref_block[1][1] = convert_float4(as_uchar4(middle_block[0].s1));
    ref_block[2][1] = convert_float4(as_uchar4(middle_block[0].s2));
    ref_block[3][1] = convert_float4(as_uchar4(middle_block[0].s3));
    ref_block[4][1] = convert_float4(as_uchar4(middle_block[0].s4));
    ref_block[5][1] = convert_float4(as_uchar4(middle_block[0].s5));
    ref_block[6][1] = convert_float4(as_uchar4(middle_block[0].s6));
    ref_block[7][1] = convert_float4(as_uchar4(middle_block[0].s7));

    ref_block[8][1] = convert_float4(as_uchar4(middle_block[1].s0));
    ref_block[9][1] = convert_float4(as_uchar4(middle_block[1].s1));
    ref_block[10][1] = convert_float4(as_uchar4(middle_block[1].s2));
    ref_block[11][1] = convert_float4(as_uchar4(middle_block[1].s3));
    ref_block[12][1] = convert_float4(as_uchar4(middle_block[1].s4));
    ref_block[13][1] = convert_float4(as_uchar4(middle_block[1].s5));
    ref_block[14][1] = convert_float4(as_uchar4(middle_block[1].s6));
    ref_block[15][1] = convert_float4(as_uchar4(middle_block[1].s7));

    // left
    coord_read.y -= 2 * BLOCK_HEIGHT;
    for (int i = 0; i < 2; i++) {
        ref_subgroup[i] = intel_sub_group_shuffle(middle_block[i], left_id);
        coord_read.y += BLOCK_HEIGHT;
    }
    ref_block[0][0] = convert_float4(as_uchar4(ref_subgroup[0].s0));
    ref_block[1][0] = convert_float4(as_uchar4(ref_subgroup[0].s1));
    ref_block[2][0] = convert_float4(as_uchar4(ref_subgroup[0].s2));
    ref_block[3][0] = convert_float4(as_uchar4(ref_subgroup[0].s3));
    ref_block[4][0] = convert_float4(as_uchar4(ref_subgroup[0].s4));
    ref_block[5][0] = convert_float4(as_uchar4(ref_subgroup[0].s5));
    ref_block[6][0] = convert_float4(as_uchar4(ref_subgroup[0].s6));
    ref_block[7][0] = convert_float4(as_uchar4(ref_subgroup[0].s7));

    ref_block[8][0] = convert_float4(as_uchar4(ref_subgroup[1].s0));
    ref_block[9][0] = convert_float4(as_uchar4(ref_subgroup[1].s1));
    ref_block[10][0] = convert_float4(as_uchar4(ref_subgroup[1].s2));
    ref_block[11][0] = convert_float4(as_uchar4(ref_subgroup[1].s3));
    ref_block[12][0] = convert_float4(as_uchar4(ref_subgroup[1].s4));
    ref_block[13][0] = convert_float4(as_uchar4(ref_subgroup[1].s5));
    ref_block[14][0] = convert_float4(as_uchar4(ref_subgroup[1].s6));
    ref_block[15][0] = convert_float4(as_uchar4(ref_subgroup[1].s7));


    // right
    coord_read.y -= 2 * BLOCK_HEIGHT;
    for (int i = 0; i < 2; i++) {
        ref_subgroup[i] = intel_sub_group_shuffle(middle_block[i], right_id);
        coord_read.y += BLOCK_HEIGHT;
    }
    ref_block[0][2] = convert_float4(as_uchar4(ref_subgroup[0].s0));
    ref_block[1][2] = convert_float4(as_uchar4(ref_subgroup[0].s1));
    ref_block[2][2] = convert_float4(as_uchar4(ref_subgroup[0].s2));
    ref_block[3][2] = convert_float4(as_uchar4(ref_subgroup[0].s3));
    ref_block[4][2] = convert_float4(as_uchar4(ref_subgroup[0].s4));
    ref_block[5][2] = convert_float4(as_uchar4(ref_subgroup[0].s5));
    ref_block[6][2] = convert_float4(as_uchar4(ref_subgroup[0].s6));
    ref_block[7][2] = convert_float4(as_uchar4(ref_subgroup[0].s7));

    ref_block[8][2] = convert_float4(as_uchar4(ref_subgroup[1].s0));
    ref_block[9][2] = convert_float4(as_uchar4(ref_subgroup[1].s1));
    ref_block[10][2] = convert_float4(as_uchar4(ref_subgroup[1].s2));
    ref_block[11][2] = convert_float4(as_uchar4(ref_subgroup[1].s3));
    ref_block[12][2] = convert_float4(as_uchar4(ref_subgroup[1].s4));
    ref_block[13][2] = convert_float4(as_uchar4(ref_subgroup[1].s5));
    ref_block[14][2] = convert_float4(as_uchar4(ref_subgroup[1].s6));
    ref_block[15][2] = convert_float4(as_uchar4(ref_subgroup[1].s7));

    if (load_abserved_block) {
        observed_block[0] = ref_block[4][1];
        observed_block[1] = ref_block[5][1];
        observed_block[2] = ref_block[6][1];
        observed_block[3] = ref_block[7][1];

        observed_block[4] = ref_block[8][1];
        observed_block[5] = ref_block[9][1];
        observed_block[6] = ref_block[10][1];
        observed_block[7] = ref_block[11][1];
    }

#if 0
#pragma unroll
    for (int j = 0;  j < 3; j++) {
#pragma unroll
        for (int i = 0; i < 3; i++) {
            dist = (ref_block[4 * i][j] - observed_block[0]) * (ref_block[4 * i][j] - observed_block[0]);
            dist = mad((ref_block[4 * i + 1][j] - observed_block[1]), (ref_block[4 * i + 1][j] - observed_block[1]), dist);
            dist = mad((ref_block[4 * i + 2][j] - observed_block[2]), (ref_block[4 * i + 2][j] - observed_block[2]), dist);
            dist = mad((ref_block[4 * i + 3][j] - observed_block[3]), (ref_block[4 * i + 3][j] - observed_block[3]), dist);
            weight[0] = exp(gain * (dist.s0 + dist.s1 + dist.s2 + dist.s3));
            sum_weight[0] = sum_weight[0] + weight[0];

            restored_block[0] = mad(weight[0], ref_block[4 * i][j], restored_block[0]);
            restored_block[1] = mad(weight[0], ref_block[4 * i + 1][j], restored_block[1]);
            restored_block[2] = mad(weight[0], ref_block[4 * i + 2][j], restored_block[2]);
            restored_block[3] = mad(weight[0], ref_block[4 * i + 3][j], restored_block[3]);
        }
    }
    restored_block[0] = restored_block[0] / sum_weight[0];
    restored_block[1] = restored_block[1] / sum_weight[0];
    restored_block[2] = restored_block[2] / sum_weight[0];
    restored_block[3] = restored_block[3] / sum_weight[0];

//#else

#pragma unroll
    for (int j = 0;  j < 3; j++) {
#pragma unroll
        for (int i = 1; i < 4; i++) {
            dist = (ref_block[4 * i][j] - observed_block[4]) * (ref_block[4 * i][j] - observed_block[4]);
            dist = mad((ref_block[4 * i + 1][j] - observed_block[5]), (ref_block[4 * i + 1][j] - observed_block[5]), dist);
            dist = mad((ref_block[4 * i + 2][j] - observed_block[6]), (ref_block[4 * i + 2][j] - observed_block[6]), dist);
            dist = mad((ref_block[4 * i + 3][j] - observed_block[7]), (ref_block[4 * i + 3][j] - observed_block[7]), dist);
            weight[1] = exp(gain * (dist.s0 + dist.s1 + dist.s2 + dist.s3));
            sum_weight[1] = sum_weight[1] + weight[1];

            restored_block[4] = mad(weight[1], ref_block[4 * i][j], restored_block[4]);
            restored_block[5] = mad(weight[1], ref_block[4 * i + 1][j], restored_block[5]);
            restored_block[6] = mad(weight[1], ref_block[4 * i + 2][j], restored_block[6]);
            restored_block[7] = mad(weight[1], ref_block[4 * i + 3][j], restored_block[7]);
        }
    }
    restored_block[4] = restored_block[4] / sum_weight[1];
    restored_block[5] = restored_block[5] / sum_weight[1];
    restored_block[6] = restored_block[6] / sum_weight[1];
    restored_block[7] = restored_block[7] / sum_weight[1];
#endif

    if (load_abserved_block) {
        for (int i = 0; i < 2; i++) {
            intel_sub_group_block_write8 (output, coord_write, middle_block[i]);
            coord_write.y += BLOCK_HEIGHT;
        }
    }
}


inline void weighted_average (__read_only image2d_t input,  __write_only image2d_t output, float* sum_weight,
                              bool load_abserved_block, float4* observed_block, float4* restored_block,
                              float gain, float threshold, int row_batch, int col_batch)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);

    float4 ref_block[12][4];

    ref_block[0][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y - 4));
    ref_block[0][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y - 4));
    ref_block[0][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y - 4));
    ref_block[0][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y - 4));

    ref_block[1][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y - 3));
    ref_block[1][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y - 3));
    ref_block[1][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y - 3));
    ref_block[1][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y - 3));

    ref_block[2][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y - 2));
    ref_block[2][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y - 2));
    ref_block[2][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y - 2));
    ref_block[2][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y - 2));

    ref_block[3][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y - 1));
    ref_block[3][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y - 1));
    ref_block[3][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y - 1));
    ref_block[3][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y - 1));

    ref_block[4][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y));
    ref_block[4][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y));
    ref_block[4][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y));
    ref_block[4][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y));

    ref_block[5][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y + 1));
    ref_block[5][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y + 1));
    ref_block[5][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 1));
    ref_block[5][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y + 1));

    ref_block[6][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y + 2));
    ref_block[6][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y + 2));
    ref_block[6][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 2));
    ref_block[6][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y + 2));

    ref_block[7][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y + 3));
    ref_block[7][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y + 3));
    ref_block[7][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 3));
    ref_block[7][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y + 3));

    ref_block[8][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y + 4));
    ref_block[8][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y + 4));
    ref_block[8][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 4));
    ref_block[8][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y + 4));

    ref_block[9][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y + 5));
    ref_block[9][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y + 5));
    ref_block[9][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 5));
    ref_block[9][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y + 5));

    ref_block[10][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y + 6));
    ref_block[10][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y + 6));
    ref_block[10][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 6));
    ref_block[10][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y + 6));

    ref_block[11][0] = read_imagef(input, sampler, (int2)(col_batch * g_id_x - 1, row_batch * g_id_y + 7));
    ref_block[11][1] = read_imagef(input, sampler, (int2)(col_batch * g_id_x, row_batch * g_id_y + 7));
    ref_block[11][2] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 7));
    ref_block[11][3] = read_imagef(input, sampler, (int2)(col_batch * g_id_x + 2, row_batch * g_id_y + 7));

    float4 dist = 0.0f;
    float4 gradient = {0.0f, 0.0f, 0.0f, 0.0f};
    float MAG = 0.0f;
    float2 weight = {0.0f, 0.0f};

    if (load_abserved_block) {
        observed_block[0] = ref_block[4][1];
        observed_block[2] = ref_block[5][1];
        observed_block[4] = ref_block[6][1];
        observed_block[6] = ref_block[7][1];

        observed_block[1] = ref_block[4][2];
        observed_block[3] = ref_block[5][2];
        observed_block[5] = ref_block[6][2];
        observed_block[7] = ref_block[7][2];
    }

#pragma unroll
    for (int i = 0;  i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            dist = (ref_block[4 * i][j] - observed_block[0]) * (ref_block[4 * i][j] - observed_block[0]);
            dist = mad((ref_block[4 * i + 1][j] - observed_block[2]), (ref_block[4 * i + 1][j] - observed_block[2]), dist);
            dist = mad((ref_block[4 * i + 2][j] - observed_block[4]), (ref_block[4 * i + 2][j] - observed_block[4]), dist);
            dist = mad((ref_block[4 * i + 3][j] - observed_block[6]), (ref_block[4 * i + 3][j] - observed_block[6]), dist);

            gradient = (float4)(
                           ref_block[4 * i + 1][j].s2, ref_block[4 * i + 1][j].s2,
                           ref_block[4 * i + 1][j].s2, ref_block[4 * i + 1][j].s2);
            gradient = fabs(gradient - ref_block[4 * i][j]) +
                       fabs(gradient - ref_block[4 * i + 1][j]) +
                       fabs(gradient - ref_block[4 * i + 2][j]) +
                       fabs(gradient - ref_block[4 * i + 3][j]);
            MAG = fabs(gradient.s0 + gradient.s1 + gradient.s2 + gradient.s3) / 15.0f;
            gain = (MAG < 0.4) ? gain : 2 * gain;

            weight.s0 = exp(gain * (dist.s0 + dist.s1 + dist.s2 + dist.s3));
            weight.s0 = (weight.s0 < 0) ? 0 : weight.s0;
            sum_weight[0] = sum_weight[0] + weight.s0;

            restored_block[0] = mad(weight.s0, ref_block[4 * i][j], restored_block[0]);
            restored_block[2] = mad(weight.s0, ref_block[4 * i + 1][j], restored_block[2]);
            restored_block[4] = mad(weight.s0, ref_block[4 * i + 2][j], restored_block[4]);
            restored_block[6] = mad(weight.s0, ref_block[4 * i + 3][j], restored_block[6]);
        }
    }

#pragma unroll
    for (int i = 0;  i < 3; i++) {
#pragma unroll
        for (int j = 1; j < 4; j++) {
            dist = (ref_block[4 * i][j] - observed_block[1]) * (ref_block[4 * i][j] - observed_block[1]);
            dist = mad((ref_block[4 * i + 1][j] - observed_block[3]), (ref_block[4 * i + 1][j] - observed_block[3]), dist);
            dist = mad((ref_block[4 * i + 2][j] - observed_block[5]), (ref_block[4 * i + 2][j] - observed_block[5]), dist);
            dist = mad((ref_block[4 * i + 3][j] - observed_block[7]), (ref_block[4 * i + 3][j] - observed_block[7]), dist);

            gradient = (float4)(
                           ref_block[4 * i + 1][j].s2, ref_block[4 * i + 1][j].s2,
                           ref_block[4 * i + 1][j].s2, ref_block[4 * i + 1][j].s2);
            gradient = fabs(gradient - ref_block[4 * i][j]) +
                       fabs(gradient - ref_block[4 * i + 1][j]) +
                       fabs(gradient - ref_block[4 * i + 2][j]) +
                       fabs(gradient - ref_block[4 * i + 3][j]);
            MAG = (gradient.s0 + gradient.s1 + gradient.s2 + gradient.s3) / 15.0f;
            gain = (MAG < 0.4) ? gain : 2 * gain;

            weight.s1 = exp(gain * (dist.s0 + dist.s1 + dist.s2 + dist.s3));
            weight.s1 = (weight.s1 < 0) ? 0 : weight.s1;
            sum_weight[1] = sum_weight[1] + weight.s1;

            restored_block[1] = mad(weight.s1, ref_block[4 * i][j], restored_block[1]);
            restored_block[3] = mad(weight.s1, ref_block[4 * i + 1][j], restored_block[3]);
            restored_block[5] = mad(weight.s1, ref_block[4 * i + 2][j], restored_block[5]);
            restored_block[7] = mad(weight.s1, ref_block[4 * i + 3][j], restored_block[7]);
        }
    }

}


__kernel void kernel_3d_denoise (
    float gain, float threshold,
    __write_only image2d_t output,
    __read_only image2d_t input, __read_only image2d_t inputPrev1, __read_only image2d_t inputPrev2)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);

    int row_batch = 4;
    int col_batch = 2;

    float4 observed_block[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float4 restored_block[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sum_weight[2] = {0.0f, 0.0f};

#if REFERENCE_FRAME_COUNT == 1
    //printf("reference frame count 1! \n");
    weighted_average (
        input, output, sum_weight, true, observed_block, restored_block, gain, threshold, row_batch, col_batch);
#endif

#if REFERENCE_FRAME_COUNT == 2
    //printf("reference frame count 2! \n");
    weighted_average (
        input, output, sum_weight, true, observed_block, restored_block, gain, threshold, row_batch, col_batch);
    weighted_average (
        inputPrev1, output, sum_weight, false, observed_block, restored_block,
        gain, threshold, row_batch, col_batch);
#endif

#if REFERENCE_FRAME_COUNT == 3
    //printf("reference frame count 3! \n");
    weighted_average (
        input, output, sum_weight, true, observed_block, restored_block, gain, threshold, row_batch, col_batch);
    weighted_average (
        inputPrev1, output, sum_weight, false, observed_block, restored_block, gain, threshold, row_batch, col_batch);
    weighted_average (
        inputPrev2, output, sum_weight, false, observed_block, restored_block, gain, threshold, row_batch, col_batch);
#endif

    restored_block[0] = restored_block[0] / sum_weight[0];
    restored_block[2] = restored_block[2] / sum_weight[0];
    restored_block[4] = restored_block[4] / sum_weight[0];
    restored_block[6] = restored_block[6] / sum_weight[0];

    restored_block[1] = restored_block[1] / sum_weight[1];
    restored_block[3] = restored_block[3] / sum_weight[1];
    restored_block[5] = restored_block[5] / sum_weight[1];
    restored_block[7] = restored_block[7] / sum_weight[1];

    write_imagef(output, (int2)(col_batch * g_id_x, row_batch * g_id_y), restored_block[0]);
    write_imagef(output, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y), restored_block[1]);
    write_imagef(output, (int2)(col_batch * g_id_x, row_batch * g_id_y + 1), restored_block[2]);
    write_imagef(output, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 1), restored_block[3]);
    write_imagef(output, (int2)(col_batch * g_id_x, row_batch * g_id_y + 2), restored_block[4]);
    write_imagef(output, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 2), restored_block[5]);
    write_imagef(output, (int2)(col_batch * g_id_x, row_batch * g_id_y + 3), restored_block[6]);
    write_imagef(output, (int2)(col_batch * g_id_x + 1, row_batch * g_id_y + 3), restored_block[7]);

}

