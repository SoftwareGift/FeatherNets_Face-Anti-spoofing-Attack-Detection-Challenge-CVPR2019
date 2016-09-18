/*
 * kernel_gauss_lap_pyramid.cl
 * input0
 * input1
 * output
 * window, pos_x, pos_y, width, height
 */

#ifndef PYRAMID_UV
#define PYRAMID_UV 0
#endif

#ifndef CL_PYRAMID_ENABLE_DUMP
#define CL_PYRAMID_ENABLE_DUMP 0
#endif

#define fixed_pixels 8
#define GAUSS_V_R 2
#define GAUSS_H_R 1
#define COEFF_MID 4

#define zero8 (float8)(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)

__constant const float coeffs[9] = {0.0f, 0.0f, 0.1f, 0.25f, 0.3f, 0.25f, 0.1f, 0.0f, 0.0f};

/*
 * input: RGBA-CL_UNSIGNED_INT16
 * output_gauss: RGBA-CL_UNSIGNED_INT8
 * output_lap:RGBA-CL_UNSIGNED_INT16
 * each work-item calc 2 lines
 */
__kernel void
kernel_gauss_scale_transform (
    __read_only image2d_t input, int in_offset_x,
    __write_only image2d_t output_gauss
#if CL_PYRAMID_ENABLE_DUMP
    , __write_only image2d_t dump_orig
#endif
)
{
    int g_x = get_global_id (0);
    int in_x = g_x + in_offset_x;
    int g_y = get_global_id (1) * 4;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int g_out_x = get_global_id (0);
    int g_out_y = get_global_id (1) * 2;
    float8 data[2];
    data[0] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(in_x, g_y)))));
    data[1] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(in_x, g_y + 2)))));

#if CL_PYRAMID_ENABLE_DUMP
    write_imageui (dump_orig, (int2)(g_x, g_y + 0), read_imageui(input, sampler, (int2)(in_x, g_y)));
    write_imageui (dump_orig, (int2)(g_x, g_y + 1), read_imageui(input, sampler, (int2)(in_x, g_y + 1)));
    write_imageui (dump_orig, (int2)(g_x, g_y + 2), read_imageui(input, sampler, (int2)(in_x, g_y + 2)));
    write_imageui (dump_orig, (int2)(g_x, g_y + 3), read_imageui(input, sampler, (int2)(in_x, g_y + 3)));
#endif

    float8 result_pre[2] = {zero8, zero8};
    float8 result_next[2] = {zero8, zero8};
    float8 result_cur[2];
    float4 final_g[2];
    result_cur[0] = data[0] * coeffs[COEFF_MID] + data[1] * coeffs[COEFF_MID + 2];
    result_cur[1] = data[1] * coeffs[COEFF_MID] + data[0] * coeffs[COEFF_MID + 2];

    float8 tmp_data;
    int i_ver;

#pragma unroll
    for (i_ver = -GAUSS_V_R; i_ver <= GAUSS_V_R + 2; i_ver++) {
        int cur_g_y = g_y + i_ver;
        float coeff0 = coeffs[i_ver + COEFF_MID];
        float coeff1 = coeffs[i_ver + COEFF_MID - 2];
        tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(in_x - 1, cur_g_y)))));
        result_pre[0] += tmp_data * coeff0;
        result_pre[1] += tmp_data * coeff1;

        if (i_ver != 0 && i_ver != 2) {
            tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(in_x, cur_g_y)))));
            result_cur[0] += tmp_data * coeff0;
            result_cur[1] += tmp_data * coeff1;
        }
        tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(in_x + 1, cur_g_y)))));
        result_next[1] += tmp_data * coeff1;
        result_next[0] += tmp_data * coeff0;
    }

    int i_line;
#pragma unroll
    for (i_line = 0; i_line < 2; ++i_line) {
#if !PYRAMID_UV
        final_g[i_line] = result_cur[i_line].even * coeffs[COEFF_MID] +
                          (float4)(result_pre[i_line].s7, result_cur[i_line].s135) * coeffs[COEFF_MID + 1] +
                          (float4)(result_pre[i_line].s6, result_cur[i_line].s024) * coeffs[COEFF_MID + 2] +
                          (float4)(result_cur[i_line].s1357) * coeffs[COEFF_MID + 1] +
                          (float4)(result_cur[i_line].s246, result_next[i_line].s0) * coeffs[COEFF_MID + 2];
#else
        final_g[i_line] = result_cur[i_line].s0145 * coeffs[COEFF_MID] +
                          (float4)(result_pre[i_line].s67, result_cur[i_line].s23) * coeffs[COEFF_MID + 1] +
                          (float4)(result_pre[i_line].s45, result_cur[i_line].s01) * coeffs[COEFF_MID + 2] +
                          (float4)(result_cur[i_line].s2367) * coeffs[COEFF_MID + 1] +
                          (float4)(result_cur[i_line].s45, result_next[i_line].s01) * coeffs[COEFF_MID + 2];
#endif
        final_g[i_line] = clamp (final_g[i_line] + 0.5f, 0.0f, 255.0f);
        write_imageui (output_gauss, (int2)(g_out_x, g_out_y + i_line), convert_uint4(final_g[i_line]));
    }

}

inline float8
read_scale_y (__read_only image2d_t input, const sampler_t sampler, float2 pos_start, float step_x)
{
    float8 data;
    data.s0 = read_imagef (input, sampler, pos_start).x;
    pos_start.x += step_x;
    data.s1 = read_imagef (input, sampler, pos_start).x;
    pos_start.x += step_x;
    data.s2 = read_imagef (input, sampler, pos_start).x;
    pos_start.x += step_x;
    data.s3 = read_imagef (input, sampler, pos_start).x;
    pos_start.x += step_x;
    data.s4 = read_imagef (input, sampler, pos_start).x;
    pos_start.x += step_x;
    data.s5 = read_imagef (input, sampler, pos_start).x;
    pos_start.x += step_x;
    data.s6 = read_imagef (input, sampler, pos_start).x;
    pos_start.x += step_x;
    data.s7 = read_imagef (input, sampler, pos_start).x;
    return data;
}

inline float8
read_scale_uv (__read_only image2d_t input, const sampler_t sampler, float2 pos_start, float step_x)
{
    float8 data;
    data.s01 = read_imagef (input, sampler, pos_start).xy;
    pos_start.x += step_x;
    data.s23 = read_imagef (input, sampler, pos_start).xy;
    pos_start.x += step_x;
    data.s45 = read_imagef (input, sampler, pos_start).xy;
    pos_start.x += step_x;
    data.s67 = read_imagef (input, sampler, pos_start).xy;
    return data;
}

/*
 * input_gauss: RGBA-CL_UNSIGNED_INT18
 * input_lap: RGBA-CL_UNSIGNED_INT16
 * output:     RGBA-CL_UNSIGNED_INT16
 * each work-item calc 2 lines
 */
__kernel void
kernel_gauss_lap_reconstruct (
    __read_only image2d_t input_gauss,
    float in_sampler_offset_x, float in_sampler_offset_y,
    __read_only image2d_t input_lap,
    __write_only image2d_t output, int out_offset_x, float out_width, float out_height
#if CL_PYRAMID_ENABLE_DUMP
    , __write_only image2d_t dump_resize, __write_only image2d_t dump_final
#endif
)
{
    int g_x = get_global_id (0);
    int g_y = get_global_id (1);
    const sampler_t lap_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const sampler_t gauss_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    //if (g_x > out_width + 0.9f || g_y > out_height + 0.5f)
    //    return;

    float8 lap = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_lap, lap_sampler, (int2)(g_x, g_y)))));
    lap = (lap - 128.0f) * 2.0f;

    float8 data_g;
    float2 input_gauss_pos;
    float step_x;
    input_gauss_pos.x = g_x / out_width + in_sampler_offset_x;
    input_gauss_pos.y = g_y / out_height + in_sampler_offset_y;
#if !PYRAMID_UV
    step_x = 0.125f / out_width;
    data_g = read_scale_y (input_gauss, gauss_sampler, input_gauss_pos, step_x) * 256.0f;
#else
    step_x = 0.25f / out_width;
    data_g = read_scale_uv (input_gauss, gauss_sampler, input_gauss_pos, step_x) * 256.0f;
#endif

#if CL_PYRAMID_ENABLE_DUMP
    write_imageui (dump_resize, (int2)(g_x, g_y), convert_uint4(as_ushort4(convert_uchar8(data_g))));
#endif

    data_g += lap + 0.5f;
    data_g = clamp (data_g, 0.0f, 255.0f);
    write_imageui (output, (int2)(g_x + out_offset_x, g_y), convert_uint4(as_ushort4(convert_uchar8(data_g))));
#if CL_PYRAMID_ENABLE_DUMP
    write_imageui (dump_final, (int2)(g_x, g_y), convert_uint4(as_ushort4(convert_uchar8(data_g))));
#endif
}

__kernel void
kernel_pyramid_blend (
    __read_only image2d_t input0, __read_only image2d_t input1,
#if !PYRAMID_UV
    __global const float8 *input0_mask,
#else
    __global const float4 *input0_mask,
#endif
    __write_only image2d_t output)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const int g_x = get_global_id (0);
    const int g_y = get_global_id (1);
    int2 pos = (int2) (g_x, g_y);

    float8 data0 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input0, sampler, pos))));
    float8 data1 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input1, sampler, pos))));
    float8 out_data;

#if !PYRAMID_UV
    out_data = (data0 - data1) * input0_mask[g_x] + data1;
#else
    float8 coeff;
    coeff.even = input0_mask[g_x];
    coeff.odd = coeff.even;
    out_data = (data0 - data1) * coeff + data1;
#endif

    out_data = clamp (out_data + 0.5f, 0.0f, 255.0f);

    write_imageui(output, pos, convert_uint4(as_ushort4(convert_uchar8(out_data))));
}

__kernel void
kernel_pyramid_copy (
    __read_only image2d_t input, int in_offset_x,
    __write_only image2d_t output, int out_offset_x,
    int max_g_x, int max_g_y)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const int g_x = get_global_id (0);
    const int g_y = get_global_id (1);

    if (g_x >= max_g_x || g_y >= max_g_y)
        return;

    uint4 data = read_imageui (input, sampler, (int2)(g_x + in_offset_x, g_y));
    write_imageui (output, (int2)(g_x + out_offset_x, g_y), data);
}

/*
 * input_gauss: RGBA-CL_UNSIGNED_INT18
 * input_lap: RGBA-CL_UNSIGNED_INT16
 * output:     RGBA-CL_UNSIGNED_INT16
 * each work-item calc 2 lines
 */
__kernel void
kernel_lap_transform (
    __read_only image2d_t input_gauss0, int gauss0_offset_x,
    __read_only image2d_t input_gauss1,
    float gauss1_sampler_offset_x, float gauss1_sampler_offset_y,
    __write_only image2d_t output, int lap_offset_x, float out_width, float out_height)
{
    int g_x = get_global_id (0);
    int g_y = get_global_id (1);
    const sampler_t gauss0_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const sampler_t gauss1_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    float8 orig = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_gauss0, gauss0_sampler, (int2)(g_x + gauss0_offset_x, g_y)))));
    float8 zoom_in;
    float2 gauss1_pos;
    float sampler_step;
    gauss1_pos.y = (g_y / out_height) + gauss1_sampler_offset_y;
    gauss1_pos.x = (g_x / out_width) + gauss1_sampler_offset_x;

#if !PYRAMID_UV
    sampler_step = 0.125f / out_width;
    zoom_in.s0 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).x;
    gauss1_pos.x += sampler_step;
    zoom_in.s1 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).x;
    gauss1_pos.x += sampler_step;
    zoom_in.s2 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).x;
    gauss1_pos.x += sampler_step;
    zoom_in.s3 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).x;
    gauss1_pos.x += sampler_step;
    zoom_in.s4 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).x;
    gauss1_pos.x += sampler_step;
    zoom_in.s5 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).x;
    gauss1_pos.x += sampler_step;
    zoom_in.s6 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).x;
    gauss1_pos.x += sampler_step;
    zoom_in.s7 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).x;
#else
    sampler_step = 0.25f / out_width;
    zoom_in.s01 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).xy;
    gauss1_pos.x += sampler_step;
    zoom_in.s23 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).xy;
    gauss1_pos.x += sampler_step;
    zoom_in.s45 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).xy;
    gauss1_pos.x += sampler_step;
    zoom_in.s67 = read_imagef (input_gauss1, gauss1_sampler, gauss1_pos).xy;
#endif
    float8 lap = (orig - zoom_in * 256.0f) * 0.5f + 128.0f + 0.5f;
    lap = clamp (lap, 0.0f, 255.0f);
    write_imageui (output, (int2)(g_x + lap_offset_x, g_y), convert_uint4(as_ushort4(convert_uchar8(lap))));
}

