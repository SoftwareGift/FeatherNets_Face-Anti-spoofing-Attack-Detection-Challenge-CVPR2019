/*
 * function: kernel_wavelet_haar_reconstruction
 *     wavelet haar reconstruction kernel
 * output:      output wavelet reconstruction image
 * ll/hl/lh/hh: input wavelet transform data as read only
 * layer:       wavelet reconstruction layer
 * decomLevels: wavelet decomposition levels
 * threshold:   hard/soft denoise thresholding
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

    float4 line_ll;
    float4 line_hl;
    float4 line_lh;
    float4 line_hh;

    line_ll = read_imagef(ll, sampler, (int2)(x, y));
    line_hl = read_imagef(hl, sampler, (int2)(x, y)) - 0.5f;
    line_lh = read_imagef(lh, sampler, (int2)(x, y)) - 0.5f;
    line_hh = read_imagef(hh, sampler, (int2)(x, y)) - 0.5f;

    // row reconstruction
    float4 row_l[2];
    float4 row_h[2];
    row_l[0] = line_ll + line_lh;
    row_l[1] = line_hl + line_hh;
    row_h[0] = line_ll - line_lh;
    row_h[1] = line_hl - line_hh;

    // column reconstruction
    float8 line[2];
    line[0].odd = row_l[0] + row_l[1];
    line[0].even = row_l[0] - row_l[1];
    line[1].odd = row_h[0] + row_h[1];
    line[1].even = row_h[0] - row_h[1];

    write_imagef(output, (int2)(2 * x, 2 * y), line[0].lo);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), line[0].hi);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), line[1].lo);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), line[1].hi);

}

