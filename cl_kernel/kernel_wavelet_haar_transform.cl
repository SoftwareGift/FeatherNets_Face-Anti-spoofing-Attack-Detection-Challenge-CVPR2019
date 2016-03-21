/*
 * function: kernel_wavelet_haar_transform
 *     wavelet haar transform kernel
 * input:        input image data as read only
 * ll/hl/lh/hh:  wavelet decomposition image
 * layer:        wavelet decomposition layer
 * decomLevels:  wavelet decomposition levels
 */
__kernel void kernel_wavelet_haar_transform (__read_only image2d_t input,
        __write_only image2d_t ll, __write_only image2d_t hl,
        __write_only image2d_t lh, __write_only image2d_t hh,
        uint vertical_offset_in, uint vertical_offset_out,
        int layer, int decomLevels,
        float hardThresh, float softThresh)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float8 line[2];
    line[0].lo = read_imagef(input, sampler, (int2)(2 * x, 2 * y));
    line[0].hi = read_imagef(input, sampler, (int2)(2 * x, 2 * y + 1));
    line[1].lo = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y));
    line[1].hi = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y + 1));

    // row transform
    float8 row_l;
    float8 row_h;
    row_l = ((float8)(line[0].lo, line[0].hi) + (float8)(line[1].lo, line[1].hi)) / 2.0f;
    row_h = ((float8)(line[0].lo, line[0].hi) - (float8)(line[1].lo, line[1].hi)) / 2.0f;

    // column transform
    write_imagef(ll, (int2)(x, y), (row_l.odd + row_l.even) / 2.0f);
    write_imagef(hl, (int2)(x, y), (row_l.odd - row_l.even) / 2.0f + 0.5f);
    write_imagef(lh, (int2)(x, y), (row_h.odd + row_h.even) / 2.0f + 0.5f);
    write_imagef(hh, (int2)(x, y), (row_h.odd - row_h.even) / 2.0f + 0.5f);
}

