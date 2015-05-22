/*
 * function: kernel_demo
 *     sample code of default kernel arguments
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

__kernel void kernel_demo (__read_only image2d_t input, __write_only image2d_t output)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    int2 pos = (int2)(x, y);
    uint4 pixel = read_imageui(input, sampler, pos);
    write_imageui(output, pos, pixel);
}

