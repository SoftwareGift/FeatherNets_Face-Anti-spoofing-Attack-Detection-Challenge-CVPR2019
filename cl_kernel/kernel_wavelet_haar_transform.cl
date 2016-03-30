/*
 * function: kernel_wavelet_haar_transform
 *     wavelet haar transform kernel
 * input:        input image data as read only
 * ll/hl/lh/hh:  wavelet decomposition image
 * layer:        wavelet decomposition layer
 * decomLevels:  wavelet decomposition levels
 */

__constant float uv_threshConst[5] = { 0.3129, 0.13319, 0.06643, 0.03513, 0.02143 };
__constant float y_threshConst[5] = { 0.06129, 0.027319, 0.012643, 0.006513, 0.003443 };

__kernel void kernel_wavelet_haar_transform (__read_only image2d_t input,
        __write_only image2d_t ll, __write_only image2d_t hl,
        __write_only image2d_t lh, __write_only image2d_t hh,
        int layer, int decomLevels,
        float hardThresh, float softThresh)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float thold = 0.0;

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

    thold = hardThresh * y_threshConst[layer - 1];
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

    thold = hardThresh * uv_threshConst[layer - 1];
#endif

    // thresholding
    line_hl = (line_hl < -thold) ? line_hl + (thold - thold * softThresh) : line_hl;
    line_hl = (line_hl > thold) ? line_hl - (thold - thold * softThresh) : line_hl;
    line_hl = (line_hl > -thold && line_hl < thold) ? line_hl * softThresh : line_hl;

    line_lh = (line_lh < -thold) ? line_lh + (thold - thold * softThresh) : line_lh;
    line_lh = (line_lh > thold) ? line_lh - (thold - thold * softThresh) : line_lh;
    line_lh = (line_lh > -thold && line_lh < thold) ? line_lh * softThresh : line_lh;

    line_hh = (line_hh < -thold) ? line_hh + (thold - thold * softThresh) : line_hh;
    line_hh = (line_hh > thold) ? line_hh - (thold - thold * softThresh) : line_hh;
    line_hh = (line_hh > -thold && line_hh < thold) ? line_hh * softThresh : line_hh;

    write_imagef(ll, (int2)(x, y), line_ll);
    write_imagef(hl, (int2)(x, y), line_hl + 0.5f);
    write_imagef(lh, (int2)(x, y), line_lh + 0.5f);
    write_imagef(hh, (int2)(x, y), line_hh + 0.5f);
}

