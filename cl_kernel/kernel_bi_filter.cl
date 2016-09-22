/*
 * function:     kernel_bi_filter
 *               bilateral filter
 * input_y:      Y channel image2d_t as read only
 * input_dark:   dark channel image2d_t as read only
 * output_dark:  dark channel image2d_t as write only
 *
 * data_type      CL_UNSIGNED_INT16
 * channel_order  CL_RGBA
 */

#define PATCH_RADIUS 7
#define PATCH_DIAMETER (2 * PATCH_RADIUS + 1)

#define CALC_SUM(y1,y2,dark1,dark2) \
    cur_y = (float8)(y1, y2); \
    cur_dark = (float8)(dark1, dark2); \
    calc_sum (cur_y, cur_dark, center_y, &weight_sum, &data_sum);

__inline void calc_sum (float8 cur_y, float8 cur_dark, float8 center_y, float8 *weight_sum, float8 *data_sum)
{
    float8 delta = (cur_y - center_y) / 28.0f;
    delta = -0.5f * delta * delta;

    float8 weight = native_exp(delta);
    float8 data = cur_dark * weight;
    (*weight_sum) += weight;
    (*data_sum) += data;
}

__kernel void kernel_bi_filter (
    __read_only image2d_t input_y,
    __read_only image2d_t input_dark,
    __write_only image2d_t output_dark)
{
    int pos_x = get_global_id (0);
    int pos_y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float8 y1, y2, dark1, dark2;
    float8 cur_y, cur_dark;

    float8 weight_sum = 0.0f;
    float8 data_sum = 0.0f;
    float8 center_y = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_y, sampler, (int2)(pos_x, pos_y)))));
    for (int i = 0; i < PATCH_DIAMETER; i++) {
        y1 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_y, sampler, (int2)(pos_x - 1, pos_y - PATCH_RADIUS + i)))));
        y2 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_y, sampler, (int2)(pos_x, pos_y - PATCH_RADIUS + i)))));
        dark1 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_dark, sampler, (int2)(pos_x - 1, pos_y - PATCH_RADIUS + i)))));
        dark2 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_dark, sampler, (int2)(pos_x, pos_y - PATCH_RADIUS + i)))));
        CALC_SUM (y1.s1234567, y2.s0, dark1.s1234567, dark2.s0);
        CALC_SUM (y1.s234567, y2.s01, dark1.s234567, dark2.s01);
        CALC_SUM (y1.s34567, y2.s012, dark1.s34567, dark2.s012);
        CALC_SUM (y1.s4567, y2.s0123, dark1.s4567, dark2.s0123);
        CALC_SUM (y1.s567, y2.s01234, dark1.s567, dark2.s01234);
        CALC_SUM (y1.s67, y2.s012345, dark1.s67, dark2.s012345);
        CALC_SUM (y1.s7, y2.s0123456, dark1.s7, dark2.s0123456);
        CALC_SUM (y2.s0123, y2.s4567, dark2.s0123, dark2.s4567);

        y1 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_y, sampler, (int2)(pos_x + 1, pos_y - PATCH_RADIUS + i)))));
        dark1 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_dark, sampler, (int2)(pos_x + 1, pos_y - PATCH_RADIUS + i)))));
        CALC_SUM (y2.s1234567, y1.s0, dark2.s1234567, dark1.s0);
        CALC_SUM (y2.s234567, y1.s01, dark2.s234567, dark1.s01);
        CALC_SUM (y2.s34567, y1.s012, dark2.s34567, dark1.s012);
        CALC_SUM (y2.s4567, y1.s0123, dark2.s4567, dark1.s0123);
        CALC_SUM (y2.s567, y1.s01234, dark2.s567, dark1.s01234);
        CALC_SUM (y2.s67, y1.s012345, dark2.s67, dark1.s012345);
        CALC_SUM (y2.s7, y1.s0123456, dark2.s7, dark1.s0123456);
    }

    float8 out_data = data_sum / weight_sum;
    write_imageui(output_dark, (int2)(pos_x, pos_y), convert_uint4(as_ushort4(convert_uchar8(out_data))));
}

