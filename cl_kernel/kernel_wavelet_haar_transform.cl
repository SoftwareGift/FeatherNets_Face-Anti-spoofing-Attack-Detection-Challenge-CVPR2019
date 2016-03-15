/*
 * function: kernel_wavelet_haar_transform
 *     wavelet haar transform kernel
 * input:        input image data as read only
 * ll/hl/lh/hh:  wavelet decomposation image
 * layer:        wavelet decomposation layer
 * decomLevels:  wavelet decomposation levels
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

    float4 pixel[2][2];
    pixel[0][0] = read_imagef(input, sampler, (int2)(2 * x, 2 * y));
    pixel[0][1] = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y));
    pixel[1][0] = read_imagef(input, sampler, (int2)(2 * x, 2 * y + 1));
    pixel[1][1] = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y + 1));

    // column transform
    float4 col_l[2];
    float4 col_h[2];
    col_l[0] = (pixel[0][0] + pixel[0][1]) / 2;
    col_h[0] = (pixel[0][0] - pixel[0][1]) / 2;
    col_l[1] = (pixel[1][0] + pixel[1][1]) / 2;
    col_h[1] = (pixel[1][0] - pixel[1][1]) / 2;

    // row transform
    write_imagef(ll, (int2)(x, y), (col_l[0] + col_l[1]) / 2);
    write_imagef(hl, (int2)(x, y), (col_l[0] - col_l[1]) / 2 + 0.5f);
    write_imagef(lh, (int2)(x, y), (col_h[0] + col_h[1]) / 2 + 0.5f);
    write_imagef(hh, (int2)(x, y), (col_h[0] - col_h[1]) / 2 + 0.5f);
}

