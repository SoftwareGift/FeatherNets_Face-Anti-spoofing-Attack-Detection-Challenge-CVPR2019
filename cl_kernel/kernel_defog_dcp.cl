/*
 * function:    kernel_dark_channel
 * input_y:     Y channel image2d_t as read only
 * input_uv:   UV channel image2d_t as read only
 * out_dark_channel: dark channel image2d_t as write only
 * output_r:    R channel image2d_t as write only
 * output_g:   G channel image2d_t as write only
 * output_b:   B channel image2d_t as write only
 *
 * data_type CL_UNSIGNED_INT16
 * channel_order CL_RGBA
 */

__kernel void kernel_dark_channel (
    __read_only image2d_t input_y, __read_only image2d_t input_uv,
    __write_only image2d_t out_dark_channel,
    __write_only image2d_t output_r, __write_only image2d_t output_g, __write_only image2d_t output_b)
{
    int pos_x = get_global_id (0);
    int pos_y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float8 y[2];
    float8 r, g, b;
    float8 uv_r, uv_g, uv_b;
    uint4 ret;
    int2 pos;

    y[0] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_y, sampler, (int2)(pos_x, pos_y * 2)))));
    y[1] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_y, sampler, (int2)(pos_x, pos_y * 2 + 1)))));
    float8 uv = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_uv, sampler, (int2)(pos_x, pos_y))))) - 128.0f;

    uv_r.even = -0.001f * uv.even + 1.402f * uv.odd;
    uv_r.odd = uv_r.even;
    uv_g.even = -0.344f * uv.even - 0.714f * uv.odd;
    uv_g.odd = uv_g.even;
    uv_b.even = 1.772f * uv.even + 0.001f * uv.odd;
    uv_b.odd = uv_b.even;

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        r = y[i] + uv_r;
        g = y[i] + uv_g;
        b = y[i] + uv_b;
        r = clamp (r, 0.0f, 255.0f);
        g = clamp (g, 0.0f, 255.0f);
        b = clamp (b, 0.0f, 255.0f);

        pos = (int2)(pos_x, 2 * pos_y + i);

        ret = convert_uint4(as_ushort4(convert_uchar8(r)));
        write_imageui(output_r, pos, ret);
        ret = convert_uint4(as_ushort4(convert_uchar8(g)));
        write_imageui(output_g, pos, ret);
        ret = convert_uint4(as_ushort4(convert_uchar8(b)));
        write_imageui(output_b, pos, ret);

        r = min (r, g);
        r = min (r, b);
        ret = convert_uint4(as_ushort4(convert_uchar8(r)));
        write_imageui(out_dark_channel, pos, ret);
    }

}


/*
 * function:    kernel_defog_recover
 * input_dark: dark channel image2d_t as read only
 * max_v:       atmospheric light
 * input_r:      R channel image2d_t as read only
 * input_g:     G channel image2d_t as read only
 * input_b:     B channel image2d_t as read only
 * output_y:   Y channel image2d_t as write only
 * output_uv: uv channel image2d_t as write only
 *
 * data_type        CL_UNSIGNED_INT16
 * channel_order  CL_RGBA
 */

#define transmit_map_coeff 0.95f

__kernel void kernel_defog_recover (
    __read_only image2d_t input_dark, float max_v, float max_r, float max_g, float max_b,
    __read_only image2d_t input_r, __read_only image2d_t input_g, __read_only image2d_t input_b,
    __write_only image2d_t out_y, __write_only image2d_t output_uv)
{
    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);
    int pos_x = g_id_x;
    int pos_y = g_id_y * 2;
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float8 in_r[2], in_g[2], in_b[2];
    float8 transmit_map[2];
    float8 out_data;

    in_r[0] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_r, sampler, (int2)(pos_x, pos_y)))));
    in_r[1] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_r, sampler, (int2)(pos_x, pos_y + 1)))));
    in_g[0] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_g, sampler, (int2)(pos_x, pos_y)))));
    in_g[1] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_g, sampler, (int2)(pos_x, pos_y + 1)))));
    in_b[0] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_b, sampler, (int2)(pos_x, pos_y)))));
    in_b[1] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_b, sampler, (int2)(pos_x, pos_y + 1)))));
    transmit_map[0] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_dark, sampler, (int2)(pos_x, pos_y)))));
    transmit_map[1] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_dark, sampler, (int2)(pos_x, pos_y + 1)))));

    transmit_map[0] = 1.0f - transmit_map_coeff * transmit_map[0] / max_v;
    transmit_map[1] = 1.0f - transmit_map_coeff * transmit_map[1] / max_v;

    transmit_map[0] = max (transmit_map[0], 0.1f);
    transmit_map[1] = max (transmit_map[1], 0.1f);

    float8 gain = 2.0f;  // adjust the brightness temporarily
    in_r[0] = (max_r + (in_r[0] - max_r) / transmit_map[0]) * gain;
    in_r[1] = (max_r + (in_r[1] - max_r) / transmit_map[1]) * gain;
    in_g[0] = (max_g + (in_g[0] - max_g) / transmit_map[0]) * gain;
    in_g[1] = (max_g + (in_g[1] - max_g) / transmit_map[1]) * gain;
    in_b[0] = (max_b + (in_b[0] - max_b) / transmit_map[0]) * gain;
    in_b[1] = (max_b + (in_b[1] - max_b) / transmit_map[1]) * gain;

    out_data = 0.299f * in_r[0] + 0.587f * in_g[0] + 0.114f * in_b[0];
    out_data = clamp (out_data, 0.0f, 255.0f);
    write_imageui(out_y, (int2)(pos_x, pos_y), convert_uint4(as_ushort4(convert_uchar8(out_data))));
    out_data = 0.299f * in_r[1] + 0.587f * in_g[1] + 0.114f * in_b[1];
    out_data = clamp (out_data, 0.0f, 255.0f);
    write_imageui(out_y, (int2)(pos_x, pos_y + 1), convert_uint4(as_ushort4(convert_uchar8(out_data))));

    float4 r, g, b;
    r = (in_r[0].even + in_r[0].odd + in_r[1].even + in_r[1].odd) * 0.25f;
    g = (in_g[0].even + in_g[0].odd + in_g[1].even + in_g[1].odd) * 0.25f;
    b = (in_b[0].even + in_b[0].odd + in_b[1].even + in_b[1].odd) * 0.25f;
    out_data.even = (-0.169f * r - 0.331f * g + 0.5f * b) + 128.0f;
    out_data.odd = (0.5f * r - 0.419f * g - 0.081f * b) + 128.0f;
    out_data = clamp (out_data, 0.0f, 255.0f);
    write_imageui(output_uv, (int2)(g_id_x, g_id_y), convert_uint4(as_ushort4(convert_uchar8(out_data))));
}

