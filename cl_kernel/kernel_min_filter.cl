/*
 * function:    kernel_min_filter
 * input:      image2d_t as read only
 * output:    image2d_t as write only
 *
 * data_type CL_UNSIGNED_INT16
 * channel_order CL_RGBA
 */

//#define VERTICAL_MIN_KERNEL 1
#define PATCH_RADIUS 8

// offset X should be PATCH_RADIUS and aligned by 8
// offset Y should be PATCH_RADIUS aligned

#if VERTICAL_MIN_KERNEL  // vertical
#define OFFSET_X 0
#define OFFSET_Y PATCH_RADIUS
#define GROUP_X 128
#define GROUP_Y 8
#define LINES_OF_WI 2

#else  //horizontal
// offset X should be PATCH_RADIUS and aligned with 8
#define OFFSET_X 8
#define OFFSET_Y 0
#define GROUP_X 128
#define GROUP_Y 4
#define LINES_OF_WI 1
#endif

#define DOT_X_SIZE (GROUP_X + OFFSET_X * 2)
#define DOT_Y_SIZE (GROUP_Y + OFFSET_Y * 2)

//__constant const int slm_x_size = DOT_X_SIZE / 8;
//__constant const int slm_y_size = DOT_Y_SIZE;
#define slm_x_size  (DOT_X_SIZE / 8)
#define slm_y_size   DOT_Y_SIZE
__constant int uchar8_offset = OFFSET_X / 8;

void load_to_slm (__read_only image2d_t input, __local uchar8 *slm, int group_start_x, int group_start_y)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int local_x = get_local_id (0);
    int local_y = get_local_id (1);
    int local_index = local_y * get_local_size (0) + local_x;

    int group_offset_x = group_start_x - uchar8_offset;
    int group_offset_y = group_start_y - OFFSET_Y;

    for (; local_index < slm_x_size * slm_y_size; local_index += get_local_size(0) * get_local_size(1)) {
        int slm_x = local_index % slm_x_size;
        int slm_y = local_index / slm_x_size;
        int pos_x = group_offset_x + slm_x;
        int pos_y = group_offset_y + slm_y;
        uchar8 data = as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(pos_x, pos_y))));
        slm[local_index] = data;
    }

}

void finish_vertical_min (
    __local uchar8 *data_center, __write_only image2d_t output,
    int group_start_x, int group_start_y, int local_x, int local_y)
{
    int pos_x, pos_y;
    uchar8 min_val = data_center[0];
    int v;

    // process 2 line with each uchar8 pixels by each work-item
#pragma unroll
    for (v = 1; v < OFFSET_Y; ++v) {
        min_val = min (min_val, data_center[slm_x_size * v]);
        min_val = min (min_val, data_center[-slm_x_size * v]);
    }
    min_val = min (min_val, data_center[slm_x_size * OFFSET_Y]);

    uchar8 min_val_1 = min (min_val, data_center[-slm_x_size * OFFSET_Y]);
    uchar8 min_val_2 = min (min_val, data_center[slm_x_size * (OFFSET_Y + 1)]);

    pos_x = group_start_x + local_x;
    pos_y = group_start_y + local_y;

    write_imageui(output, (int2)(pos_x, pos_y), convert_uint4(as_ushort4(min_val_1)));
    write_imageui(output, (int2)(pos_x, pos_y + 1), convert_uint4(as_ushort4(min_val_2)));
}


void finish_horizontal_min (
    __local uchar8 *data_center, __write_only image2d_t output,
    int group_start_x, int group_start_y, int local_x, int local_y)
{

    uchar8 value = data_center[0];
    uchar8 v_left = ((__local uchar8 *)data_center)[-1];
    uchar8 v_right = ((__local uchar8 *)data_center)[1];
    /*
     * Order 1st uchar4
     * 1st 4 values's common min, value.lo
     * - - - 3 4 5 6 7  X X X X 4 5 6 7  0 - - - - - - -
     * 2nd 4 values's common min, value.hi
     * - - - - - - - 7  0 1 2 3 X X X X  0 1 2 3 4 - - -
     * 1st and 2nd 4 value's shared common
     * - - - - - - - 7  0 1 2 3 4 5 6 7  0 - - - - - - -
     */

    uchar4 tmp4;
    uchar2 tmp2;
    uchar tmp1_left, tmp1_right;

    uchar shared_common;
    uchar first_common_min, second_common_min;
    uchar8 out_data;

    tmp4 = min (value.lo, value.hi);
    tmp2 = min (tmp4.s01, tmp4.s23);
    shared_common = min (tmp2.s0, tmp2.s1);
    shared_common = min (shared_common, v_left.s7);
    shared_common = min (shared_common, v_right.s0);

    tmp2 = min (v_left.s34, v_left.s56);
    first_common_min = min (tmp2.s0, tmp2.s1);
    first_common_min = min (first_common_min, shared_common);

    tmp2 = min (v_right.s12, v_right.s34);
    second_common_min = min (tmp2.s0, tmp2.s1);
    second_common_min = min (second_common_min, shared_common);

    //final first 4 values
    tmp1_left = min (v_left.s1, v_left.s2);
    tmp1_right = min (v_right.s1, v_right.s2);
    out_data.s0 = min (tmp1_left, v_left.s0);
    out_data.s0 = min (out_data.s0, first_common_min);

    out_data.s1 = min (tmp1_left, first_common_min);
    out_data.s1 = min (out_data.s1, v_right.s1);

    out_data.s2 = min (v_left.s2, first_common_min);
    out_data.s2 = min (out_data.s2, tmp1_right);

    out_data.s3 = min (first_common_min, tmp1_right);
    out_data.s3 = min (out_data.s3, v_right.s3);

    //second 4 values
    tmp1_left = min (v_left.s5, v_left.s6);
    tmp1_right = min (v_right.s5, v_right.s6);
    out_data.s4 = min (tmp1_left, v_left.s4);
    out_data.s4 = min (out_data.s4, second_common_min);

    out_data.s5 = min (tmp1_left, second_common_min);
    out_data.s5 = min (out_data.s5, v_right.s5);

    out_data.s6 = min (v_left.s6, second_common_min);
    out_data.s6 = min (out_data.s6, tmp1_right);

    out_data.s7 = min (second_common_min, tmp1_right);
    out_data.s7 = min (out_data.s7, v_right.s7);

    int pos_x = group_start_x + local_x;
    int pos_y = group_start_y + local_y;

    write_imageui(output, (int2)(pos_x, pos_y), convert_uint4(as_ushort4(out_data)));
}

__kernel void kernel_min_filter (
    __read_only image2d_t input,
    __write_only image2d_t output)
{
    int group_start_x = get_group_id (0) * (GROUP_X / 8);
    int group_start_y = get_group_id (1) * GROUP_Y;

    __local uchar8 slm_cache[slm_x_size * slm_y_size];

    //load to slm
    load_to_slm (input, slm_cache, group_start_x, group_start_y);
    barrier (CLK_LOCAL_MEM_FENCE);

    int local_x = get_local_id (0) ;
    int local_y = get_local_id (1) * LINES_OF_WI;
    int slm_x = local_x + uchar8_offset;
    int slm_y = local_y + OFFSET_Y;
    int slm_index = slm_x + slm_y * slm_x_size;
    __local uchar8 *data_center = slm_cache + slm_index;

#if VERTICAL_MIN_KERNEL
    finish_vertical_min (data_center, output, group_start_x, group_start_y, local_x, local_y);
#else
    finish_horizontal_min (data_center, output, group_start_x, group_start_y, local_x, local_y);
#endif
}

