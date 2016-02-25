/*
 * function: kernel_gauss
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

__kernel void kernel_gauss (__read_only image2d_t input, __write_only image2d_t output, uint vertical_offset_in, uint vertical_offset_out, __global float *table)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 y_in[36];
    float4 out1, out2, out3, out4;
    int i, j;

    for(i = 0; i < 36; i++)
        y_in[i] = read_imagef(input, sampler, (int2)(2 * x - 2 + i % 6, 2 * y - 2 + i / 6));

    for(i = 0; i < 5; i++)
        for(j = 0; j < 5; j++) {
            out1.x += y_in[i * 6 + j].x *  table[i * 5 + j];
            out2.x += y_in[i * 6 + j + 1].x *  table[i * 5 + j];
            out3.x += y_in[(i + 1) * 6 + j].x *  table[i * 5 + j];
            out4.x += y_in[(i + 1) * 6 + j + 1].x *  table[i * 5 + j];
        }

    write_imagef(output, (int2)(2 * x, 2 * y), out1);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), out2);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), out3);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), out4);

}

