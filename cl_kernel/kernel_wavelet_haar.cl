/*
 * function: kernel_wavelet_haar_decomposition
 *     wavelet haar decomposition kernel
 * input:        input image data as read only
 * ll/hl/lh/hh:  wavelet decomposition image
 * layer:        wavelet decomposition layer
 * decomLevels:  wavelet decomposition levels
 */
#ifndef WAVELET_DENOISE_Y
#define WAVELET_DENOISE_Y 1
#endif

#ifndef WAVELET_DENOISE_UV
#define WAVELET_DENOISE_UV 0
#endif

#ifndef WAVELET_BAYES_SHRINK
#define WAVELET_BAYES_SHRINK 1
#endif

__kernel void kernel_wavelet_haar_decomposition (
    __read_only image2d_t input,
    __write_only image2d_t ll, __write_only image2d_t hl,
    __write_only image2d_t lh, __write_only image2d_t hh,
    int layer, int decomLevels,
    float hardThresh, float softThresh)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float8 line[2];
    line[0].lo = read_imagef(input, sampler, (int2)(2 * x, 2 * y));
    line[0].hi = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y));
    line[1].lo = read_imagef(input, sampler, (int2)(2 * x, 2 * y + 1));
    line[1].hi = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y + 1));

    // row transform
    float8 row_l;
    float8 row_h;
    row_l = (float8)(line[0].lo + line[1].lo, line[0].hi + line[1].hi) / 2.0f;
    row_h = (float8)(line[0].lo - line[1].lo, line[0].hi - line[1].hi) / 2.0f;

    float4 line_ll;
    float4 line_hl;
    float4 line_lh;
    float4 line_hh;

#if WAVELET_DENOISE_Y
    // column transform
    line_ll = (row_l.odd + row_l.even) / 2.0f;
    line_hl = (row_l.odd - row_l.even) / 2.0f;
    line_lh = (row_h.odd + row_h.even) / 2.0f;
    line_hh = (row_h.odd - row_h.even) / 2.0f;
#endif

#if WAVELET_DENOISE_UV
    // U column transform
    line_ll.odd = (row_l.odd.odd + row_l.odd.even) / 2.0f;
    line_hl.odd = (row_l.odd.odd - row_l.odd.even) / 2.0f;
    line_lh.odd = (row_h.odd.odd + row_h.odd.even) / 2.0f;
    line_hh.odd = (row_h.odd.odd - row_h.odd.even) / 2.0f;

    // V column transform
    line_ll.even = (row_l.even.odd + row_l.even.even) / 2.0f;
    line_hl.even = (row_l.even.odd - row_l.even.even) / 2.0f;
    line_lh.even = (row_h.even.odd + row_h.even.even) / 2.0f;
    line_hh.even = (row_h.even.odd - row_h.even.even) / 2.0f;
#endif

    write_imagef(ll, (int2)(x, y), line_ll);
    write_imagef(hl, (int2)(x, y), line_hl + 0.5f);
    write_imagef(lh, (int2)(x, y), line_lh + 0.5f);
    write_imagef(hh, (int2)(x, y), line_hh + 0.5f);
}

/*
 * function: kernel_wavelet_haar_reconstruction
 *     wavelet haar reconstruction kernel
 * output:      output wavelet reconstruction image
 * ll/hl/lh/hh: input wavelet transform data as read only
 * layer:       wavelet reconstruction layer
 * decomLevels: wavelet decomposition levels
 * threshold:   hard/soft denoise thresholding
 */

__constant float uv_threshConst[5] = { 0.1659f, 0.06719f, 0.03343f, 0.01713f, 0.01043f };
__constant float y_threshConst[5] = { 0.06129f, 0.027319f, 0.012643f, 0.006513f, 0.003443f };

__kernel void kernel_wavelet_haar_reconstruction (
    __write_only image2d_t output,
    __read_only image2d_t ll, __read_only image2d_t hl,
    __read_only image2d_t lh, __read_only image2d_t hh,
    int layer, int decomLevels,
    float hardThresh, float softThresh)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float thresh = 0.0f;

    float4 line_ll;
    float4 line_hl;
    float4 line_lh;
    float4 line_hh;

    line_ll = read_imagef(ll, sampler, (int2)(x, y));
    line_hl = read_imagef(hl, sampler, (int2)(x, y)) - 0.5f;
    line_lh = read_imagef(lh, sampler, (int2)(x, y)) - 0.5f;
    line_hh = read_imagef(hh, sampler, (int2)(x, y)) - 0.5f;

#if WAVELET_DENOISE_Y
    thresh = hardThresh * y_threshConst[layer - 1];
#endif

#if WAVELET_DENOISE_UV
    thresh = hardThresh * uv_threshConst[layer - 1];
#endif

#if !WAVELET_BAYES_SHRINK
    // thresholding
    line_hl = (line_hl < -thresh) ? line_hl + (thresh - thresh * softThresh) : line_hl;
    line_hl = (line_hl > thresh) ? line_hl - (thresh - thresh * softThresh) : line_hl;
    line_hl = (line_hl > -thresh && line_hl < thresh) ? line_hl * softThresh : line_hl;

    line_lh = (line_lh < -thresh) ? line_lh + (thresh - thresh * softThresh) : line_lh;
    line_lh = (line_lh > thresh) ? line_lh - (thresh - thresh * softThresh) : line_lh;
    line_lh = (line_lh > -thresh && line_lh < thresh) ? line_lh * softThresh : line_lh;

    line_hh = (line_hh < -thresh) ? line_hh + (thresh - thresh * softThresh) : line_hh;
    line_hh = (line_hh > thresh) ? line_hh - (thresh - thresh * softThresh) : line_hh;
    line_hh = (line_hh > -thresh && line_hh < thresh) ? line_hh * softThresh : line_hh;
#endif

#if WAVELET_DENOISE_Y
    // row reconstruction
    float8 row_l;
    float8 row_h;
    row_l = (float8)(line_ll + line_lh, line_hl + line_hh);
    row_h = (float8)(line_ll - line_lh, line_hl - line_hh);

    // column reconstruction
    float8 line[2];
    line[0].odd = row_l.lo + row_l.hi;
    line[0].even = row_l.lo - row_l.hi;
    line[1].odd = row_h.lo + row_h.hi;
    line[1].even = row_h.lo - row_h.hi;

    write_imagef(output, (int2)(2 * x, 2 * y), line[0].lo);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), line[0].hi);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), line[1].lo);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), line[1].hi);
#endif

#if WAVELET_DENOISE_UV
    // row reconstruction
    float8 row_l;
    float8 row_h;
    row_l = (float8)(line_ll + line_lh, line_hl + line_hh);
    row_h = (float8)(line_ll - line_lh, line_hl - line_hh);

    float8 line[2];

    // U column reconstruction
    line[0].odd.odd = row_l.lo.odd + row_l.hi.odd;
    line[0].odd.even = row_l.lo.odd - row_l.hi.odd;
    line[1].odd.odd = row_h.lo.odd + row_h.hi.odd;
    line[1].odd.even = row_h.lo.odd - row_h.hi.odd;

    // V column reconstruction
    line[0].even.odd = row_l.lo.even + row_l.hi.even;
    line[0].even.even = row_l.lo.even - row_l.hi.even;
    line[1].even.odd = row_h.lo.even + row_h.hi.even;
    line[1].even.even = row_h.lo.even - row_h.hi.even;

    write_imagef(output, (int2)(2 * x, 2 * y), line[0].lo);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), line[0].hi);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), line[1].lo);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), line[1].hi);
#endif
}
