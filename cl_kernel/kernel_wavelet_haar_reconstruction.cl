/*
 * function: kernel_wavelet_haar_reconstruction
 *     wavelet haar reconstruction kernel
 * ll/hl/lh/hh: input wavelet transform data as read only
 * output:      output wavelet reconstruction image
 * threshold:   noise threshold
 * layer:       wavelet reconstruction layer
 * decomLevels: wavelet decomposation levels
 */
__kernel void kernel_wavelet_haar_reconstruction (__write_only image2d_t output,
        __read_only image2d_t ll, __read_only image2d_t hl,
        __read_only image2d_t lh, __read_only image2d_t hh,
        uint vertical_offset_in, uint vertical_offset_out,
        int layer, int decomLevels,
        float hardThresh, float softThresh)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 pixel_ll;
    float4 pixel_hl;
    float4 pixel_lh;
    float4 pixel_hh;

    pixel_ll = read_imagef(ll, sampler, (int2)(x, y));
    pixel_hl = read_imagef(hl, sampler, (int2)(x, y)) - 0.5f;
    pixel_lh = read_imagef(lh, sampler, (int2)(x, y)) - 0.5f;
    pixel_hh = read_imagef(hh, sampler, (int2)(x, y)) - 0.5f;

    // row reconstruction
    float4 row_l[2];
    float4 row_h[2];
    row_l[0] = pixel_ll + pixel_lh;
    row_l[1] = pixel_ll - pixel_lh;
    row_h[0] = pixel_hl + pixel_hh;
    row_h[1] = pixel_hl - pixel_hh;

    // column reconstruction
    write_imagef(output, (int2)(2 * x, 2 * y), row_l[0] + row_h[0]);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), row_l[0] - row_h[0]);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), row_l[1] + row_h[1]);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), row_l[1] - row_h[1]);
}

