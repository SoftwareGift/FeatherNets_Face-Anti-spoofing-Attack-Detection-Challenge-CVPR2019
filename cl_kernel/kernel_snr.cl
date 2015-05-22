/*
 * function: kernel_snr
 *     implementation of simple noise reduction
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

__kernel void kernel_snr (__read_only image2d_t input, __write_only image2d_t output)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 p[9];
    p[0] = read_imagef(input, sampler, (int2)(x - 1, y - 1));
    p[1] = read_imagef(input, sampler, (int2)(x, y - 1));
    p[2] = read_imagef(input, sampler, (int2)(x + 1, y - 1));
    p[3] = read_imagef(input, sampler, (int2)(x - 1, y));
    p[4] = read_imagef(input, sampler, (int2)(x, y));
    p[5] = read_imagef(input, sampler, (int2)(x + 1, y));
    p[6] = read_imagef(input, sampler, (int2)(x - 1, y + 1));
    p[7] = read_imagef(input, sampler, (int2)(x, y + 1));
    p[8] = read_imagef(input, sampler, (int2)(x + 1, y + 1));

    float4 pixel_out;
    pixel_out.x = (p[0].x + p[1].x + p[2].x + p[3].x + p[4].x + p[5].x + p[6].x + p[7].x + p[8].x) / 9.0f;
    pixel_out.y = (p[0].y + p[1].y + p[2].y + p[3].y + p[4].y + p[5].y + p[6].y + p[7].y + p[8].y) / 9.0f;
    pixel_out.z = (p[0].z + p[1].z + p[2].z + p[3].z + p[4].z + p[5].z + p[6].z + p[7].z + p[8].z) / 9.0f;
    pixel_out.w = p[4].w;
    write_imagef(output, (int2)(x, y), pixel_out);
}
