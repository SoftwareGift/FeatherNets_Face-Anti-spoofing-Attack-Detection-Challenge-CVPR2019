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

#ifndef ENABLE_MASK_GAUSS_SCALE
#define ENABLE_MASK_GAUSS_SCALE 0
#endif

#define fixed_pixels 8
#define GAUSS_V_R 2
#define GAUSS_H_R 1
#define COEFF_MID 4

#define zero8 (float8)(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)

__constant const float coeffs[9] = {0.0f, 0.0f, 0.152f, 0.222f, 0.252f, 0.222f, 0.152f, 0.0f, 0.0f};

#define ARG_FORMAT4 "(%.1f,%.1f,%.1f,%.1f)"
#define ARGS4(a) a.s0, a.s1, a.s2, a.s3

#define ARG_FORMAT8 "(%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f)"
#define ARGS8(a) a.s0, a.s1, a.s2, a.s3, a.s4, a.s5, a.s6, a.s7

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

#if CL_PYRAMID_ENABLE_DUMP
    write_imageui (dump_orig, (int2)(g_x, g_y + 0), read_imageui(input, sampler, (int2)(in_x, g_y)));
    write_imageui (dump_orig, (int2)(g_x, g_y + 1), read_imageui(input, sampler, (int2)(in_x, g_y + 1)));
    write_imageui (dump_orig, (int2)(g_x, g_y + 2), read_imageui(input, sampler, (int2)(in_x, g_y + 2)));
    write_imageui (dump_orig, (int2)(g_x, g_y + 3), read_imageui(input, sampler, (int2)(in_x, g_y + 3)));
#endif

    float8 result_pre[2] = {zero8, zero8};
    float8 result_next[2] = {zero8, zero8};
    float8 result_cur[2] = {zero8, zero8};
    float4 final_g[2];

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
        tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(in_x, cur_g_y)))));
        result_cur[0] += tmp_data * coeff0;
        result_cur[1] += tmp_data * coeff1;
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
kernel_pyramid_scale (
    __read_only image2d_t input, __write_only image2d_t output,
    int out_offset_x, int output_width, int output_height)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    int g_x = get_global_id (0);
    int g_y = get_global_id (1);

    float2 normCoor = (float2)(g_x, g_y) / (float2)(output_width, output_height);
    float8 out_data;
    float step_x;

#if !PYRAMID_UV
    step_x = 0.125f / output_width;
    out_data = read_scale_y (input, sampler, normCoor, step_x) * 255.0f;
#else
    step_x = 0.25f / output_width;
    out_data = read_scale_uv (input, sampler, normCoor, step_x) * 255.0f;
#endif

    out_data = clamp (out_data + 0.5f, 0.0f, 255.0f);
    write_imageui (output, (int2)(g_x + out_offset_x, g_y), convert_uint4(as_ushort4(convert_uchar8(out_data))));
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


/*
 * input0: RGBA-CL_UNSIGNED_INT16
 * input1: RGBA-CL_UNSIGNED_INT16
 * out_diff:  RGBA-CL_UNSIGNED_INT16
 */
__kernel void
kernel_image_diff (
    __read_only image2d_t input0, int offset0,
    __read_only image2d_t input1, int offset1,
    __write_only image2d_t out_diff)
{
    int g_x = get_global_id (0);
    int g_y = get_global_id (1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int8 data0 = convert_int8(as_uchar8(convert_ushort4(read_imageui(input0, sampler, (int2)(g_x + offset0, g_y)))));
    int8 data1 = convert_int8(as_uchar8(convert_ushort4(read_imageui(input1, sampler, (int2)(g_x + offset1, g_y)))));
    uint8 diff = abs_diff (data0, data1);
    write_imageui (out_diff, (int2)(g_x, g_y), convert_uint4(as_ushort4(convert_uchar8(diff))));
}


/*
 * input0: RGBA-CL_UNSIGNED_INT16
 */
#define LEFT_POS (int)(-1)
#define MID_POS (int)(0)
#define RIGHT_POS (int)(1)

__inline int pos_buf_index (int x, int y, int stride)
{
    return mad24 (stride, y, x);
}

__kernel void
kernel_seam_dp (
    __read_only image2d_t image,
    __global short *pos_buf, __global float *sum_buf, int offset_x, int valid_width,
    int max_pos, int seam_height, int seam_stride)
{
    int l_x = get_local_id (0);
    int group_id = get_group_id (0);
    if (l_x >= valid_width)
        return;

    // group0 fill first half slice image curve y = [0, seam_height/2 - 1]
    // group1 fill send half slice image curve = [seam_height - 1, seam_height/2]
    int first_slice_h = seam_height / 2;
    int group_h = (group_id == 0 ? first_slice_h : seam_height - first_slice_h);

    __local float slm_sum[4096];
    float mid, left, right, cur;
    int slm_idx;
    int default_pos;

    int x = l_x + offset_x;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int y = (group_id == 0 ? 0 : seam_height - 1);
    float sum = convert_float(read_imageui(image, sampler, (int2)(x, y)).x);

    default_pos = x;
    slm_sum[l_x] = sum;
    barrier (CLK_LOCAL_MEM_FENCE);
    pos_buf[pos_buf_index(x, y, seam_stride)] = convert_short(default_pos);

    for (int i = 0; i < group_h; ++i) {
        y = (group_id == 0 ? i : seam_height - i - 1);
        slm_idx = l_x - 1;
        slm_idx = (slm_idx > 0 ? slm_idx : 0);
        left = slm_sum[slm_idx];
        slm_idx = l_x + 1;
        slm_idx = (slm_idx < valid_width ? slm_idx : valid_width - 1);
        right = slm_sum[slm_idx];

        cur = convert_float(read_imageui(image, sampler, (int2)(x, y)).x);

        left = left + cur;
        right = right + cur;
        mid = sum + cur;

        int pos;
        pos = (left < mid) ? LEFT_POS : MID_POS;
        sum = min (left, mid);
        pos = (sum < right) ? pos : RIGHT_POS;
        sum = min (sum, right);
        slm_sum[l_x] = sum;
        barrier (CLK_LOCAL_MEM_FENCE);

        pos += default_pos;
        pos = clamp (pos, offset_x, max_pos);
        //if (l_x == 3)
        //    printf ("s:%f, pos:%d, mid:%f, offset_x:%d\n", sum.s0, pos.s0, mid.s0, offset_x);
        pos_buf[pos_buf_index(x, y, seam_stride)] = convert_short(pos);
    }
    sum_buf[group_id * seam_stride + x] = sum;
    //printf ("sum(x):%f(x:%d)\n", sum_buf[x].s0, x);
}

__kernel void
kernel_seam_mask_blend (
    __read_only image2d_t input0, __read_only image2d_t input1,
    __read_only image2d_t seam_mask,
    __write_only image2d_t output)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const int g_x = get_global_id (0);
    const int g_y = get_global_id (1);
    int2 pos = (int2) (g_x, g_y);

    float8 data0 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input0, sampler, pos))));
    float8 data1 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input1, sampler, pos))));
    float8 coeff0 = convert_float8(as_uchar8(convert_ushort4(read_imageui(seam_mask, sampler, pos)))) / 255.0f;
    float8 out_data;

#if !PYRAMID_UV
    out_data = (data0 - data1) * coeff0 + data1;
#else
    coeff0.even = (coeff0.even + coeff0.odd) * 0.5f;
    coeff0.odd = coeff0.even;
    out_data = (data0 - data1) * coeff0 + data1;
#endif

    out_data = clamp (out_data + 0.5f, 0.0f, 255.0f);

    write_imageui(output, pos, convert_uint4(as_ushort4(convert_uchar8(out_data))));
}



#define MASK_GAUSS_R 4
#define MASK_COEFF_MID 7

__constant const float mask_coeffs[] = {0.0f, 0.0f, 0.0f, 0.082f, 0.102f, 0.119f, 0.130f, 0.134f, 0.130f, 0.119f, 0.102f, 0.082f, 0.0f, 0.0f, 0.0f};

/*
 * input: RGBA-CL_UNSIGNED_INT16
 * output_gauss: RGBA-CL_UNSIGNED_INT8 ?
 * output_lap:RGBA-CL_UNSIGNED_INT16
 * each work-item calc 2 lines
 */
__kernel void
kernel_mask_gauss_scale_slm (
    __read_only image2d_t input,
    __write_only image2d_t output_gauss,
    int image_width
#if ENABLE_MASK_GAUSS_SCALE
    , __write_only image2d_t output_scale
#endif
)
{
#define WI_LINES 2
// input image width MUST < MASK_GAUSS_SLM_WIDTH*4
#define MASK_GAUSS_SLM_WIDTH  256
#define CONV_COEFF 128.0f

    int g_x = get_global_id (0);
    int g_y = get_global_id (1) * WI_LINES;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    __local ushort4 slm_gauss_y[WI_LINES][MASK_GAUSS_SLM_WIDTH];

    float8 result_cur[WI_LINES] = {zero8, zero8};
    float8 tmp_data;
    int i_line;
    int cur_g_y;

#pragma unroll
    for (i_line = -MASK_GAUSS_R; i_line <= MASK_GAUSS_R + 1; i_line++) {
        cur_g_y = g_y + i_line;
        tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(g_x, cur_g_y)))));
        result_cur[0] += tmp_data * mask_coeffs[i_line + MASK_COEFF_MID];
        result_cur[1] += tmp_data * mask_coeffs[i_line + MASK_COEFF_MID - 1];
    }
    ((__local ushort8*)(slm_gauss_y[0]))[g_x] = convert_ushort8(result_cur[0] * CONV_COEFF);
    ((__local ushort8*)(slm_gauss_y[1]))[g_x] = convert_ushort8(result_cur[1] * CONV_COEFF);
    barrier (CLK_LOCAL_MEM_FENCE);

    float8 final_g[WI_LINES];
    float4 result_pre;
    float4 result_next;

#pragma unroll
    for (i_line = 0; i_line < WI_LINES; ++i_line) {
        result_pre = convert_float4(slm_gauss_y[i_line][clamp (g_x * 2 - 1, 0, image_width * 2)]) / CONV_COEFF;
        result_next = convert_float4(slm_gauss_y[i_line][clamp (g_x * 2 + 2, 0, image_width * 2)]) / CONV_COEFF;
        final_g[i_line] = result_cur[i_line] * mask_coeffs[MASK_COEFF_MID] +
                          (float8)(result_pre.s3, result_cur[i_line].s0123456) * mask_coeffs[MASK_COEFF_MID + 1] +
                          (float8)(result_cur[i_line].s1234567, result_next.s0) * mask_coeffs[MASK_COEFF_MID + 1] +
                          (float8)(result_pre.s23, result_cur[i_line].s012345) * mask_coeffs[MASK_COEFF_MID + 2] +
                          (float8)(result_cur[i_line].s234567, result_next.s01) * mask_coeffs[MASK_COEFF_MID + 2] +
                          (float8)(result_pre.s123, result_cur[i_line].s01234) * mask_coeffs[MASK_COEFF_MID + 3] +
                          (float8)(result_cur[i_line].s34567, result_next.s012) * mask_coeffs[MASK_COEFF_MID + 3] +
                          (float8)(result_pre.s0123, result_cur[i_line].s0123) * mask_coeffs[MASK_COEFF_MID + 4] +
                          (float8)(result_cur[i_line].s4567, result_next.s0123) * mask_coeffs[MASK_COEFF_MID + 4];
        final_g[i_line] = clamp (final_g[i_line] + 0.5f, 0.0f, 255.0f);
        //if ((g_x == 9 || g_x == 8) && g_y == 0) {
        //    printf ("(x:%d, y:0), pre:" ARG_FORMAT4 "cur" ARG_FORMAT8 "next" ARG_FORMAT4 "final:" ARG_FORMAT8 "\n",
        //        g_x, ARGS4(result_pre), ARGS8(result_cur[i_line]), ARGS4(result_next), ARGS8(final_g[i_line]));
        //}
        write_imageui (output_gauss, (int2)(g_x, g_y + i_line), convert_uint4(as_ushort4(convert_uchar8(final_g[i_line]))));
    }

#if ENABLE_MASK_GAUSS_SCALE
    write_imageui (output_scale, (int2)(g_x, get_global_id (1)), convert_uint4(final_g[0].even));
#endif
}

__kernel void
kernel_mask_gauss_scale (
    __read_only image2d_t input,
    __write_only image2d_t output_gauss
#if ENABLE_MASK_GAUSS_SCALE
    , __write_only image2d_t output_scale
#endif
)
{
    int g_x = get_global_id (0);
    int in_x = g_x;
    int g_y = get_global_id (1) * 2;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float8 result_pre[2] = {zero8, zero8};
    float8 result_next[2] = {zero8, zero8};
    float8 result_cur[2] = {zero8, zero8};
    float8 final_g[2];

    float8 tmp_data;
    int i_line;
    int cur_g_y;
    float coeff0, coeff1;

#pragma unroll
    for (i_line = -MASK_GAUSS_R; i_line <= MASK_GAUSS_R + 1; i_line++) {
        cur_g_y = g_y + i_line;
        coeff0 = mask_coeffs[i_line + MASK_COEFF_MID];
        coeff1 = mask_coeffs[i_line + MASK_COEFF_MID - 1];
        tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(in_x - 1, cur_g_y)))));
        result_pre[0] += tmp_data * coeff0;
        result_pre[1] += tmp_data * coeff1;

        tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(in_x, cur_g_y)))));
        result_cur[0] += tmp_data * coeff0;
        result_cur[1] += tmp_data * coeff1;
        tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(in_x + 1, cur_g_y)))));
        result_next[1] += tmp_data * coeff1;
        result_next[0] += tmp_data * coeff0;
    }

#pragma unroll
    for (i_line = 0; i_line < 2; ++i_line) {
        final_g[i_line] = result_cur[i_line] * mask_coeffs[MASK_COEFF_MID] +
                          (float8)(result_pre[i_line].s7, result_cur[i_line].s0123456) * mask_coeffs[MASK_COEFF_MID + 1] +
                          (float8)(result_cur[i_line].s1234567, result_next[i_line].s0) * mask_coeffs[MASK_COEFF_MID + 1] +
                          (float8)(result_pre[i_line].s67, result_cur[i_line].s012345) * mask_coeffs[MASK_COEFF_MID + 2] +
                          (float8)(result_cur[i_line].s234567, result_next[i_line].s01) * mask_coeffs[MASK_COEFF_MID + 2] +
                          (float8)(result_pre[i_line].s567, result_cur[i_line].s01234) * mask_coeffs[MASK_COEFF_MID + 3] +
                          (float8)(result_cur[i_line].s34567, result_next[i_line].s012) * mask_coeffs[MASK_COEFF_MID + 3] +
                          (float8)(result_pre[i_line].s4567, result_cur[i_line].s0123) * mask_coeffs[MASK_COEFF_MID + 4] +
                          (float8)(result_cur[i_line].s4567, result_next[i_line].s0123) * mask_coeffs[MASK_COEFF_MID + 4];
        final_g[i_line] = clamp (final_g[i_line] + 0.5f, 0.0f, 255.0f);
        write_imageui (output_gauss, (int2)(g_x, g_y + i_line), convert_uint4(as_ushort4(convert_uchar8(final_g[i_line]))));
    }

#if ENABLE_MASK_GAUSS_SCALE
    write_imageui (output_scale, (int2)(g_x, get_global_id (1)), convert_uint4(final_g[0].even));
#endif

}

