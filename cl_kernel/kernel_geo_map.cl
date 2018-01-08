/*
 * kernel_geo_map
 * input_y,      input image, CL_R + CL_UNORM_INT8
 * input_uv, CL_RG + CL_UNORM_INT8
 * geo_table, CL_RGBA + CL_FLOAT
 * output_y,  CL_RGBA + CL_UNSIGNED_INT16
 * output_uv,  CL_RGBA + CL_UNSIGNED_INT16
 *
 * description:
 * the center of geo_table and output positons are both mapped to (0, 0)
 */

#ifndef ENABLE_LSC
#define ENABLE_LSC 0
#endif

#define CONST_DATA_Y 0.0f
#define CONST_DATA_UV (float2)(0.5f, 0.5f)

// 8 bytes for each pixel
#define PIXEL_RES_STEP_X 8

void get_geo_mapped_y (
    __read_only image2d_t input,
    __read_only image2d_t geo_table, float2 table_pos, float step_x,
    bool *out_of_bound, float2 *input_pos, float8 *out_y)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    float *output_data = (float*)(out_y);
    int i = 0;

    for (i = 0; i < PIXEL_RES_STEP_X; ++i) {
        out_of_bound[i] =
            (min (table_pos.x, table_pos.y) < 0.0f) ||
            (max (table_pos.x, table_pos.y) > 1.0f);
        input_pos[i] = read_imagef (geo_table, sampler, table_pos).xy;
        out_of_bound[i] =
            out_of_bound[i] ||
            (min (input_pos[i].x, input_pos[i].y) < 0.0f) ||
            (max (input_pos[i].x, input_pos[i].y) > 1.0f);
        //need convert input_pos to (0.0 ~ 1.0)????
        output_data[i] = out_of_bound[i] ? CONST_DATA_Y : read_imagef (input, sampler, input_pos[i]).x;
        table_pos.x += step_x;
    }
}

void get_lsc_data (
    image2d_t lsc_table, int2 g_pos, float step_x,
    float2 gray_threshold, float8 output, float8 *lsc_data)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    float *lsc_ptr = (float *)(lsc_data);

    float2 pos = convert_float2((int2)(g_pos.x * PIXEL_RES_STEP_X, g_pos.y)) * step_x;
    for (int i = 0; i < PIXEL_RES_STEP_X; ++i) {
        lsc_ptr[i] = read_imagef (lsc_table, sampler, pos).x;
        pos.x += step_x;
    }

    float8 diff_ratio = (gray_threshold.y - output * 255.0f) / (gray_threshold.y - gray_threshold.x);
    diff_ratio = clamp (diff_ratio, 0.0f, 1.0f);
    (*lsc_data) = diff_ratio * diff_ratio * ((*lsc_data) - 1.0f) + 1.0f;
}

__kernel void
kernel_geo_map (
    __read_only image2d_t input_y, __read_only image2d_t input_uv,
    __read_only image2d_t geo_table, float2 table_scale_size,
    float2 left_scale_factor, float2 right_scale_factor,
#if ENABLE_LSC
    __read_only image2d_t lsc_table, float2 gray_threshold,
#endif
    __write_only image2d_t output_y, __write_only image2d_t output_uv, float2 out_size)
{
    const int g_x = get_global_id (0);
    const int g_y_uv = get_global_id (1);
    const int g_y = get_global_id (1) * 2;
    float8 output_data;
    float2 from_pos;
    bool out_of_bound[8];
    float2 input_pos[8];
    // map to [-0.5, 0.5)
    table_scale_size *= (g_x * PIXEL_RES_STEP_X < out_size.x / 2.0f) ? left_scale_factor : right_scale_factor;

    float2 table_scale_step = 1.0f / table_scale_size;
    float2 out_map_pos;
    sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    out_map_pos = (convert_float2((int2)(g_x * PIXEL_RES_STEP_X, g_y)) - out_size / 2.0f) * table_scale_step + 0.5f;

    get_geo_mapped_y (input_y, geo_table, out_map_pos, table_scale_step.x, out_of_bound, input_pos, &output_data);

#if ENABLE_LSC
    float8 lsc_data;
    get_lsc_data (lsc_table, (int2)(g_x, g_y), table_scale_step.x, gray_threshold, output_data, &lsc_data);
    output_data = clamp (output_data * lsc_data, 0.0f, 1.0f);
#endif
    write_imageui (output_y, (int2)(g_x, g_y), convert_uint4(as_ushort4(convert_uchar8(output_data * 255.0f))));

    output_data.s01 = out_of_bound[0] ? CONST_DATA_UV : read_imagef (input_uv, sampler, input_pos[0]).xy;
    output_data.s23 = out_of_bound[2] ? CONST_DATA_UV : read_imagef (input_uv, sampler, input_pos[2]).xy;
    output_data.s45 = out_of_bound[4] ? CONST_DATA_UV : read_imagef (input_uv, sampler, input_pos[4]).xy;
    output_data.s67 = out_of_bound[6] ? CONST_DATA_UV : read_imagef (input_uv, sampler, input_pos[6]).xy;
    write_imageui (output_uv, (int2)(g_x, g_y_uv), convert_uint4(as_ushort4(convert_uchar8(output_data * 255.0f))));

    out_map_pos.y += table_scale_step.y;
    get_geo_mapped_y (input_y, geo_table, out_map_pos, table_scale_step.x, out_of_bound, input_pos, &output_data);

#if ENABLE_LSC
    get_lsc_data (lsc_table, (int2)(g_x, g_y + 1), table_scale_step.x, gray_threshold, output_data, &lsc_data);
    output_data = clamp (output_data * lsc_data, 0.0f, 1.0f);
#endif
    write_imageui (output_y, (int2)(g_x, g_y + 1), convert_uint4(as_ushort4(convert_uchar8(output_data * 255.0f))));
}
