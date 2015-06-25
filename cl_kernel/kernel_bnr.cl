/*
 * function: kernel_bnr
 *     implementation of bayer noise reduction
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * bnr_gain: strength of noise reduction
 * direction: sensitivity of edge
 * todo: add the upstream algorithm for BNR
 */

__kernel void kernel_bnr (__read_only image2d_t input, __write_only image2d_t output,
                          float bnr_gain, float direction)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 p;
    p = read_imagef(input, sampler, (int2)(x, y));

    write_imagef(output, (int2)(x, y), p);
}

