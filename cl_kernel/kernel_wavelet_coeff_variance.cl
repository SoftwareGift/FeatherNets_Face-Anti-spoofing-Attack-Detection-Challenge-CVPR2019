/*
 * function: kernel_wavelet_coeff_variance
 *     Calculate wavelet coefficients variance
 * input:  Wavelet coefficients as read only
 * output: Wavelet coefficients variance
 */

__kernel void kernel_wavelet_coeff_variance (__read_only image2d_t input, __write_only image2d_t output, int layer)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 line0[3];
    float4 line1[3];
    float4 line2[3];
    float4 line3[3];
    float4 line4[3];
    float4 line5[3];
    float4 line6[3];
    float4 line7[3];

    float4 line_sum[8];
    float4 line_var[4];

    float offset = 0.5;

    line0[0] = (read_imagef(input, sampler, (int2)(x - 1, 4 * y - 2)) - offset);
    line0[1] = (read_imagef(input, sampler, (int2)(x, 4 * y - 2)) - offset);
    line0[2] = (read_imagef(input, sampler, (int2)(x + 1, 4 * y - 2)) - offset);

    line1[0] = (read_imagef(input, sampler, (int2)(x - 1, 4 * y - 1)) - offset);
    line1[1] = (read_imagef(input, sampler, (int2)(x, 4 * y - 1)) - offset);
    line1[2] = (read_imagef(input, sampler, (int2)(x + 1, 4 * y - 1)) - offset);

    line2[0] = (read_imagef(input, sampler, (int2)(x - 1, 4 * y)) - offset);
    line2[1] = (read_imagef(input, sampler, (int2)(x, 4 * y)) - offset);
    line2[2] = (read_imagef(input, sampler, (int2)(x + 1, 4 * y)) - offset);

    line3[0] = (read_imagef(input, sampler, (int2)(x - 1, 4 * y + 1)) - offset);
    line3[1] = (read_imagef(input, sampler, (int2)(x, 2 * 4 * y + 1)) - offset);
    line3[2] = (read_imagef(input, sampler, (int2)(x + 1, 4 * y + 1)) - offset);

    line4[0] = (read_imagef(input, sampler, (int2)(x - 1, 4 * y + 2)) - offset);
    line4[1] = (read_imagef(input, sampler, (int2)(x, 4 * y + 2)) - offset);
    line4[2] = (read_imagef(input, sampler, (int2)(x + 1, 4 * y + 2)) - offset);

    line5[0] = (read_imagef(input, sampler, (int2)(x - 1, 4 * y + 3)) - offset);
    line5[1] = (read_imagef(input, sampler, (int2)(x, 4 * y + 3)) - offset);
    line5[2] = (read_imagef(input, sampler, (int2)(x + 1, 4 * y + 3)) - offset);

    line6[0] = (read_imagef(input, sampler, (int2)(x - 1, 4 * y + 4)) - offset);
    line6[1] = (read_imagef(input, sampler, (int2)(x, 4 * y + 4)) - offset);
    line6[2] = (read_imagef(input, sampler, (int2)(x + 1, 4 * y + 4)) - offset);

    line7[0] = (read_imagef(input, sampler, (int2)(x - 1, 4 * y + 5)) - offset);
    line7[1] = (read_imagef(input, sampler, (int2)(x, 4 * y + 5)) - offset);
    line7[2] = (read_imagef(input, sampler, (int2)(x + 1, 4 * y + 5)) - offset);

#if WAVELET_DENOISE_Y

    line_sum[0] = line0[0] * line0[0]
                  + (float4)(line0[0].s123, line0[1].s0) * (float4)(line0[0].s123, line0[1].s0)
                  + (float4)(line0[0].s23, line0[1].s01) * (float4)(line0[0].s23, line0[1].s01)
                  + (float4)(line0[0].s3, line0[1].s012) * (float4)(line0[0].s3, line0[1].s012)
                  + line0[1] * line0[1]
                  + (float4)(line0[1].s123, line0[2].s0) * (float4)(line0[1].s123, line0[2].s0)
                  + (float4)(line0[1].s23, line0[2].s01) * (float4)(line0[1].s23, line0[2].s01)
                  + (float4)(line0[1].s3, line0[2].s012) * (float4)(line0[1].s3, line0[2].s012)
                  + line0[2] * line0[2];

    line_sum[1] = line1[0] * line1[0]
                  + (float4)(line1[0].s123, line1[1].s0) * (float4)(line1[0].s123, line1[1].s0)
                  + (float4)(line1[0].s23, line1[1].s01) * (float4)(line1[0].s23, line1[1].s01)
                  + (float4)(line1[0].s3, line1[1].s012) * (float4)(line1[0].s3, line1[1].s012)
                  + line1[1] * line1[1]
                  + (float4)(line1[1].s123, line1[2].s0) * (float4)(line1[1].s123, line1[2].s0)
                  + (float4)(line1[1].s23, line1[2].s01) * (float4)(line1[1].s23, line1[2].s01)
                  + (float4)(line1[1].s3, line1[2].s012) * (float4)(line1[1].s3, line1[2].s012)
                  + line1[2] * line1[2];

    line_sum[2] = line2[0] * line2[0]
                  + (float4)(line2[0].s123, line2[1].s0) * (float4)(line2[0].s123, line2[1].s0)
                  + (float4)(line2[0].s23, line2[1].s01) * (float4)(line2[0].s23, line2[1].s01)
                  + (float4)(line2[0].s3, line2[1].s012) * (float4)(line2[0].s3, line2[1].s012)
                  + line2[1] * line2[1]
                  + (float4)(line2[1].s123, line2[2].s0) * (float4)(line2[1].s123, line2[2].s0)
                  + (float4)(line2[1].s23, line2[2].s01) * (float4)(line2[1].s23, line2[2].s01)
                  + (float4)(line2[1].s3, line2[2].s012) * (float4)(line2[1].s3, line2[2].s012)
                  + line2[2] * line2[2];

    line_sum[3] = line3[0] * line3[0]
                  + (float4)(line3[0].s123, line3[1].s0) * (float4)(line3[0].s123, line3[1].s0)
                  + (float4)(line3[0].s23, line3[1].s01) * (float4)(line3[0].s23, line3[1].s01)
                  + (float4)(line3[0].s3, line3[1].s012) * (float4)(line3[0].s3, line3[1].s012)
                  + line3[1] * line3[1]
                  + (float4)(line3[1].s123, line3[2].s0) * (float4)(line3[1].s123, line3[2].s0)
                  + (float4)(line3[1].s23, line3[2].s01) * (float4)(line3[1].s23, line3[2].s01)
                  + (float4)(line3[1].s3, line3[2].s012) * (float4)(line3[1].s3, line3[2].s012)
                  + line3[2] * line3[2];

    line_sum[4] = line4[0] * line4[0]
                  + (float4)(line4[0].s123, line4[1].s0) * (float4)(line4[0].s123, line4[1].s0)
                  + (float4)(line4[0].s23, line4[1].s01) * (float4)(line4[0].s23, line4[1].s01)
                  + (float4)(line4[0].s3, line4[1].s012) * (float4)(line4[0].s3, line4[1].s012)
                  + line4[1] * line4[1]
                  + (float4)(line4[1].s123, line4[2].s0) * (float4)(line4[1].s123, line4[2].s0)
                  + (float4)(line4[1].s23, line4[2].s01) * (float4)(line4[1].s23, line4[2].s01)
                  + (float4)(line4[1].s3, line4[2].s012) * (float4)(line4[1].s3, line4[2].s012)
                  + line4[2] * line4[2];

    line_sum[5] = line5[0] * line5[0]
                  + (float4)(line5[0].s123, line5[1].s0) * (float4)(line5[0].s123, line5[1].s0)
                  + (float4)(line5[0].s23, line5[1].s01) * (float4)(line5[0].s23, line5[1].s01)
                  + (float4)(line5[0].s3, line5[1].s012) * (float4)(line5[0].s3, line5[1].s012)
                  + line5[1] * line5[1]
                  + (float4)(line5[1].s123, line5[2].s0) * (float4)(line5[1].s123, line5[2].s0)
                  + (float4)(line5[1].s23, line5[2].s01) * (float4)(line5[1].s23, line5[2].s01)
                  + (float4)(line5[1].s3, line5[2].s012) * (float4)(line5[1].s3, line5[2].s012)
                  + line5[2] * line5[2];

    line_sum[6] = line6[0] * line6[0]
                  + (float4)(line6[0].s123, line6[1].s0) * (float4)(line6[0].s123, line6[1].s0)
                  + (float4)(line6[0].s23, line6[1].s01) * (float4)(line6[0].s23, line6[1].s01)
                  + (float4)(line6[0].s3, line6[1].s012) * (float4)(line6[0].s3, line6[1].s012)
                  + line6[1] * line6[1]
                  + (float4)(line6[1].s123, line6[2].s0) * (float4)(line6[1].s123, line6[2].s0)
                  + (float4)(line6[1].s23, line6[2].s01) * (float4)(line6[1].s23, line6[2].s01)
                  + (float4)(line6[1].s3, line6[2].s012) * (float4)(line6[1].s3, line6[2].s012)
                  + line6[2] * line6[2];

    line_sum[7] = line7[0] * line7[0]
                  + (float4)(line7[0].s123, line7[1].s0) * (float4)(line7[0].s123, line7[1].s0)
                  + (float4)(line7[0].s23, line7[1].s01) * (float4)(line7[0].s23, line7[1].s01)
                  + (float4)(line7[0].s3, line7[1].s012) * (float4)(line7[0].s3, line7[1].s012)
                  + line7[1] * line7[1]
                  + (float4)(line7[1].s123, line7[2].s0) * (float4)(line7[1].s123, line7[2].s0)
                  + (float4)(line7[1].s23, line7[2].s01) * (float4)(line7[1].s23, line7[2].s01)
                  + (float4)(line7[1].s3, line7[2].s012) * (float4)(line7[1].s3, line7[2].s012)
                  + line7[2] * line7[2];

    line_var[0] = (line_sum[0] + line_sum[1] + line_sum[2] + line_sum[3] + line_sum[4]) / 45;
    line_var[1] = (line_sum[1] + line_sum[2] + line_sum[3] + line_sum[4] + line_sum[5]) / 45;
    line_var[2] = (line_sum[2] + line_sum[3] + line_sum[4] + line_sum[5] + line_sum[6]) / 45;
    line_var[3] = (line_sum[3] + line_sum[4] + line_sum[5] + line_sum[6] + line_sum[7]) / 45;
#endif

#if WAVELET_DENOISE_UV
    line_sum[0] = line0[0] * line0[0]
                  + (float4)(line0[0].s23, line0[1].s01) * (float4)(line0[0].s23, line0[1].s01)
                  + line0[1] * line0[1]
                  + (float4)(line0[1].s23, line0[2].s01) * (float4)(line0[1].s23, line0[2].s01)
                  + line0[2] * line0[2];

    line_sum[1] = line1[0] * line1[0]
                  + (float4)(line1[0].s23, line1[1].s01) * (float4)(line1[0].s23, line1[1].s01)
                  + line1[1] * line1[1]
                  + (float4)(line1[1].s23, line1[2].s01) * (float4)(line1[1].s23, line1[2].s01)
                  + line1[2] * line1[2];

    line_sum[2] = line2[0] * line2[0]
                  + (float4)(line2[0].s23, line2[1].s01) * (float4)(line2[0].s23, line2[1].s01)
                  + line2[1] * line2[1]
                  + (float4)(line2[1].s23, line2[2].s01) * (float4)(line2[1].s23, line2[2].s01)
                  + line2[2] * line2[2];

    line_sum[3] = line3[0] * line3[0]
                  + (float4)(line3[0].s23, line3[1].s01) * (float4)(line3[0].s23, line3[1].s01)
                  + line3[1] * line3[1]
                  + (float4)(line3[1].s23, line3[2].s01) * (float4)(line3[1].s23, line3[2].s01)
                  + line3[2] * line3[2];

    line_sum[4] = line4[0] * line4[0]
                  + (float4)(line4[0].s23, line4[1].s01) * (float4)(line4[0].s23, line4[1].s01)
                  + line4[1] * line4[1]
                  + (float4)(line4[1].s23, line4[2].s01) * (float4)(line4[1].s23, line4[2].s01)
                  + line4[2] * line4[2];

    line_sum[5] = line5[0] * line5[0]
                  + (float4)(line5[0].s23, line5[1].s01) * (float4)(line5[0].s23, line5[1].s01)
                  + line5[1] * line5[1]
                  + (float4)(line5[1].s23, line5[2].s01) * (float4)(line5[1].s23, line5[2].s01)
                  + line5[2] * line5[2];

    line_sum[6] = line6[0] * line6[0]
                  + (float4)(line6[0].s23, line6[1].s01) * (float4)(line6[0].s23, line6[1].s01)
                  + line6[1] * line6[1]
                  + (float4)(line6[1].s23, line6[2].s01) * (float4)(line6[1].s23, line6[2].s01)
                  + line6[2] * line6[2];

    line_sum[7] = line7[0] * line7[0]
                  + (float4)(line7[0].s23, line7[1].s01) * (float4)(line7[0].s23, line7[1].s01)
                  + line7[1] * line7[1]
                  + (float4)(line7[1].s23, line7[2].s01) * (float4)(line7[1].s23, line7[2].s01)
                  + line7[2] * line7[2];
    line_var[0] = (line_sum[0] + line_sum[1] + line_sum[2] + line_sum[3] + line_sum[4]) / 25;
    line_var[1] = (line_sum[1] + line_sum[2] + line_sum[3] + line_sum[4] + line_sum[5]) / 25;
    line_var[2] = (line_sum[2] + line_sum[3] + line_sum[4] + line_sum[5] + line_sum[6]) / 25;
    line_var[3] = (line_sum[3] + line_sum[4] + line_sum[5] + line_sum[6] + line_sum[7]) / 25;
#endif

    write_imagef(output, (int2)(x, 4 * y), line_var[0]);
    write_imagef(output, (int2)(x, 4 * y + 1), line_var[1]);
    write_imagef(output, (int2)(x, 4 * y + 2), line_var[2]);
    write_imagef(output, (int2)(x, 4 * y + 3), line_var[3]);
}
