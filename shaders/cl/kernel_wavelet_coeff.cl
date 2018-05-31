/*
 * function: kernel_wavelet_coeff_variance
 *     Calculate wavelet coefficients variance
 * input:  Wavelet coefficients as read only
 * output: Wavelet coefficients variance
 */

#ifndef WAVELET_DENOISE_Y
#define WAVELET_DENOISE_Y 1
#endif

#ifndef WAVELET_DENOISE_UV
#define WAVELET_DENOISE_UV 0
#endif

#define WG_CELL_X_SIZE 8
#define WG_CELL_Y_SIZE 8

#define SLM_CELL_X_OFFSET 1
#define SLM_CELL_Y_OFFSET 2

// 10x12
#define SLM_CELL_X_SIZE (WG_CELL_X_SIZE + SLM_CELL_X_OFFSET * 2)
#define SLM_CELL_Y_SIZE (WG_CELL_Y_SIZE + SLM_CELL_Y_OFFSET * 2)

__kernel void kernel_wavelet_coeff_variance (__read_only image2d_t input, __write_only image2d_t output, int layer)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);

    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int g_size_x = get_global_size (0);
    int g_size_y = get_global_size (1);

    int l_size_x = get_local_size (0);
    int l_size_y = get_local_size (1);

    int local_index = local_id_y * WG_CELL_X_SIZE + local_id_x;

    float offset = 0.5f;
    float4 line_sum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float4 line_var = 0.0f;

    __local float4 local_src_data[SLM_CELL_X_SIZE * SLM_CELL_Y_SIZE];

    int i = local_id_x + local_id_y * WG_CELL_X_SIZE;
    int start_x = mad24(group_id_x, WG_CELL_X_SIZE, -SLM_CELL_X_OFFSET);
    int start_y = mad24(group_id_y, WG_CELL_Y_SIZE, -SLM_CELL_Y_OFFSET);

    for (int j = i;  j < SLM_CELL_X_SIZE * SLM_CELL_Y_SIZE; j += WG_CELL_X_SIZE * WG_CELL_Y_SIZE)
    {
        int x = start_x + (j % SLM_CELL_X_SIZE);
        int y = start_y + (j / SLM_CELL_X_SIZE);
        local_src_data[j] = read_imagef (input, sampler, (int2)(x, y)) - offset;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float16 line0 = *((__local float16 *)(local_src_data + local_id_y * SLM_CELL_X_SIZE + local_id_x));
    float16 line1 = *((__local float16 *)(local_src_data + (local_id_y + 1) * SLM_CELL_X_SIZE + local_id_x));
    float16 line2 = *((__local float16 *)(local_src_data + (local_id_y + 2) * SLM_CELL_X_SIZE + local_id_x));
    float16 line3 = *((__local float16 *)(local_src_data + (local_id_y + 3) * SLM_CELL_X_SIZE + local_id_x));
    float16 line4 = *((__local float16 *)(local_src_data + (local_id_y + 4) * SLM_CELL_X_SIZE + local_id_x));

#if WAVELET_DENOISE_Y
    line_sum[0] = mad(line0.s0123, line0.s0123, line_sum[0]);
    line_sum[0] = mad(line0.s1234, line0.s1234, line_sum[0]);
    line_sum[0] = mad(line0.s2345, line0.s2345, line_sum[0]);
    line_sum[0] = mad(line0.s3456, line0.s3456, line_sum[0]);
    line_sum[0] = mad(line0.s4567, line0.s4567, line_sum[0]);
    line_sum[0] = mad(line0.s5678, line0.s5678, line_sum[0]);
    line_sum[0] = mad(line0.s6789, line0.s6789, line_sum[0]);
    line_sum[0] = mad(line0.s789a, line0.s789a, line_sum[0]);
    line_sum[0] = mad(line0.s89ab, line0.s89ab, line_sum[0]);

    line_sum[1] = mad(line1.s0123, line1.s0123, line_sum[1]);
    line_sum[1] = mad(line1.s1234, line1.s1234, line_sum[1]);
    line_sum[1] = mad(line1.s2345, line1.s2345, line_sum[1]);
    line_sum[1] = mad(line1.s3456, line1.s3456, line_sum[1]);
    line_sum[1] = mad(line1.s4567, line1.s4567, line_sum[1]);
    line_sum[1] = mad(line1.s5678, line1.s5678, line_sum[1]);
    line_sum[1] = mad(line1.s6789, line1.s6789, line_sum[1]);
    line_sum[1] = mad(line1.s789a, line1.s789a, line_sum[1]);
    line_sum[1] = mad(line1.s89ab, line1.s89ab, line_sum[1]);

    line_sum[2] = mad(line2.s0123, line2.s0123, line_sum[2]);
    line_sum[2] = mad(line2.s1234, line2.s1234, line_sum[2]);
    line_sum[2] = mad(line2.s2345, line2.s2345, line_sum[2]);
    line_sum[2] = mad(line2.s3456, line2.s3456, line_sum[2]);
    line_sum[2] = mad(line2.s4567, line2.s4567, line_sum[2]);
    line_sum[2] = mad(line2.s5678, line2.s5678, line_sum[2]);
    line_sum[2] = mad(line2.s6789, line2.s6789, line_sum[2]);
    line_sum[2] = mad(line2.s789a, line2.s789a, line_sum[2]);
    line_sum[2] = mad(line2.s89ab, line2.s89ab, line_sum[2]);

    line_sum[3] = mad(line3.s0123, line3.s0123, line_sum[3]);
    line_sum[3] = mad(line3.s1234, line3.s1234, line_sum[3]);
    line_sum[3] = mad(line3.s2345, line3.s2345, line_sum[3]);
    line_sum[3] = mad(line3.s3456, line3.s3456, line_sum[3]);
    line_sum[3] = mad(line3.s4567, line3.s4567, line_sum[3]);
    line_sum[3] = mad(line3.s5678, line3.s5678, line_sum[3]);
    line_sum[3] = mad(line3.s6789, line3.s6789, line_sum[3]);
    line_sum[3] = mad(line3.s789a, line3.s789a, line_sum[3]);
    line_sum[3] = mad(line3.s89ab, line3.s89ab, line_sum[3]);

    line_sum[4] = mad(line4.s0123, line4.s0123, line_sum[4]);
    line_sum[4] = mad(line4.s1234, line4.s1234, line_sum[4]);
    line_sum[4] = mad(line4.s2345, line4.s2345, line_sum[4]);
    line_sum[4] = mad(line4.s3456, line4.s3456, line_sum[4]);
    line_sum[4] = mad(line4.s4567, line4.s4567, line_sum[4]);
    line_sum[4] = mad(line4.s5678, line4.s5678, line_sum[4]);
    line_sum[4] = mad(line4.s6789, line4.s6789, line_sum[4]);
    line_sum[4] = mad(line4.s789a, line4.s789a, line_sum[4]);
    line_sum[4] = mad(line4.s89ab, line4.s89ab, line_sum[4]);

    line_var = (line_sum[0] + line_sum[1] + line_sum[2] + line_sum[3] + line_sum[4]) / 45;
#endif

#if WAVELET_DENOISE_UV
    line_sum[0] = mad(line0.s0123, line0.s0123, line_sum[0]);
    line_sum[0] = mad(line0.s2345, line0.s2345, line_sum[0]);
    line_sum[0] = mad(line0.s4567, line0.s4567, line_sum[0]);
    line_sum[0] = mad(line0.s6789, line0.s6789, line_sum[0]);
    line_sum[0] = mad(line0.s89ab, line0.s89ab, line_sum[0]);
    line_sum[0] = mad(line0.sabcd, line0.sabcd, line_sum[0]);
    line_sum[0] = mad(line0.scdef, line0.scdef, line_sum[0]);

    line_sum[1] = mad(line1.s0123, line1.s0123, line_sum[1]);
    line_sum[1] = mad(line1.s2345, line1.s2345, line_sum[1]);
    line_sum[1] = mad(line1.s4567, line1.s4567, line_sum[1]);
    line_sum[1] = mad(line1.s6789, line1.s6789, line_sum[1]);
    line_sum[1] = mad(line1.s89ab, line1.s89ab, line_sum[1]);
    line_sum[1] = mad(line1.sabcd, line1.sabcd, line_sum[1]);
    line_sum[1] = mad(line1.scdef, line1.scdef, line_sum[1]);

    line_sum[2] = mad(line2.s0123, line2.s0123, line_sum[2]);
    line_sum[2] = mad(line2.s2345, line2.s2345, line_sum[2]);
    line_sum[2] = mad(line2.s4567, line2.s4567, line_sum[2]);
    line_sum[2] = mad(line2.s6789, line2.s6789, line_sum[2]);
    line_sum[2] = mad(line2.s89ab, line2.s89ab, line_sum[2]);
    line_sum[2] = mad(line2.sabcd, line2.sabcd, line_sum[2]);
    line_sum[2] = mad(line2.scdef, line2.scdef, line_sum[2]);

    line_sum[3] = mad(line3.s0123, line3.s0123, line_sum[3]);
    line_sum[3] = mad(line3.s2345, line3.s2345, line_sum[3]);
    line_sum[3] = mad(line3.s4567, line3.s4567, line_sum[3]);
    line_sum[3] = mad(line3.s6789, line3.s6789, line_sum[3]);
    line_sum[3] = mad(line3.s89ab, line3.s89ab, line_sum[3]);
    line_sum[3] = mad(line3.sabcd, line3.sabcd, line_sum[3]);
    line_sum[3] = mad(line3.scdef, line3.scdef, line_sum[3]);

    line_sum[4] = mad(line4.s0123, line4.s0123, line_sum[4]);
    line_sum[4] = mad(line4.s2345, line4.s2345, line_sum[4]);
    line_sum[4] = mad(line4.s4567, line4.s4567, line_sum[4]);
    line_sum[4] = mad(line4.s6789, line4.s6789, line_sum[4]);
    line_sum[4] = mad(line4.s89ab, line4.s89ab, line_sum[4]);
    line_sum[4] = mad(line4.sabcd, line4.sabcd, line_sum[4]);
    line_sum[4] = mad(line4.scdef, line4.scdef, line_sum[4]);

    line_var = ((line_sum[0] + line_sum[1] + line_sum[2] + line_sum[3] + line_sum[4]) / 35);
#endif

    write_imagef(output, (int2)(g_id_x, g_id_y), line_var);
}

/*
 * function: kernel_wavelet_coeff_thresholding
 *     wavelet coefficient thresholding kernel
 * hl/lh/hh:  wavelet coefficients
 * layer:        wavelet decomposition layer
 * decomLevels:  wavelet decomposition levels
 */

__kernel void kernel_wavelet_coeff_thresholding (float noise_var1, float noise_var2,
        __read_only image2d_t in_hl, __read_only image2d_t var_hl, __write_only image2d_t out_hl,
        __read_only image2d_t in_lh, __read_only image2d_t var_lh, __write_only image2d_t out_lh,
        __read_only image2d_t in_hh, __read_only image2d_t var_hh, __write_only image2d_t out_hh,
        int layer, int decomLevels,
        float hardThresh, float softThresh, float ag_weight)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 input_hl;
    float4 input_lh;
    float4 input_hh;

    float4 output_hl;
    float4 output_lh;
    float4 output_hh;

    float4 coeff_var_hl;
    float4 coeff_var_lh;
    float4 coeff_var_hh;

    float4 stddev_hl;
    float4 stddev_lh;
    float4 stddev_hh;

    float4 thresh_hl;
    float4 thresh_lh;
    float4 thresh_hh;

    float4 noise_var = (float4) (noise_var1, noise_var2, noise_var1, noise_var2);

    input_hl = read_imagef(in_hl, sampler, (int2)(x, y)) - 0.5f;
    input_lh = read_imagef(in_lh, sampler, (int2)(x, y)) - 0.5f;
    input_hh = read_imagef(in_hh, sampler, (int2)(x, y)) - 0.5f;

    coeff_var_hl = 65025 * (1 << 2 * layer) * read_imagef(var_hl, sampler, (int2)(x, y));
    coeff_var_lh = 65025 * (1 << 2 * layer) * read_imagef(var_lh, sampler, (int2)(x, y));
    coeff_var_hh = 65025 * (1 << 2 * layer) * read_imagef(var_hh, sampler, (int2)(x, y));

    stddev_hl = coeff_var_hl - noise_var;
    stddev_hl = (stddev_hl > 0) ? sqrt(stddev_hl) : 0.000001f;

    stddev_lh = coeff_var_lh - noise_var;
    stddev_lh = (stddev_lh > 0) ? sqrt(stddev_lh) : 0.000001f;

    stddev_hh = coeff_var_hh - noise_var;
    stddev_hh = (stddev_hh > 0) ? sqrt(stddev_hh) : 0.000001f;

    thresh_hl = (ag_weight * noise_var / stddev_hl) / (255 * (1 << layer));
    thresh_lh = (ag_weight * noise_var / stddev_lh) / (255 * (1 << layer));
    thresh_hh = (ag_weight * noise_var / stddev_hh) / (255 * (1 << layer));

    // Soft thresholding
    output_hl = (fabs(input_hl) < thresh_hl) ? 0 : ((input_hl > 0) ? fabs(input_hl) - thresh_hl : thresh_hl - fabs(input_hl));
    output_lh = (fabs(input_lh) < thresh_lh) ? 0 : ((input_lh > 0) ? fabs(input_lh) - thresh_lh : thresh_lh - fabs(input_lh));
    output_hh = (fabs(input_hh) < thresh_hh) ? 0 : ((input_hh > 0) ? fabs(input_hh) - thresh_hh : thresh_hh - fabs(input_hh));

    write_imagef(out_hl, (int2)(x, y), output_hl + 0.5f);
    write_imagef(out_lh, (int2)(x, y), output_lh + 0.5f);
    write_imagef(out_hh, (int2)(x, y), output_hh + 0.5f);
}

