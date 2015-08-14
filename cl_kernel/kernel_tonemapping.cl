/*
 * function: kernel_tonemapping
 *     implementation of tone mapping
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

__kernel void kernel_tonemapping (__read_only image2d_t input, __write_only image2d_t output, uint color_bits)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    //printf("In tonemapping kernel input color_bits = %d\n", color_bits);

    float4 p;
    p = read_imagef(input, sampler, (int2)(x , y));
    write_imagef(output, (int2)(x, y), p);
}
