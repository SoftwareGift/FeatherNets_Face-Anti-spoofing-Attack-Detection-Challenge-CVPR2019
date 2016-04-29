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

