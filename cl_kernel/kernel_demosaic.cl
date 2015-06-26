/*
 * function: kernel_demosaic
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

__kernel void kernel_demosaic (__read_only image2d_t input, __write_only image2d_t output)
{
    int x = 2 * get_global_id (0);
    int y = 2 * get_global_id (1);
//    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int x0 = x - 1;
    int y0 = y - 1;
    float4 p[16];
#if 0
    p[0] = read_imagef (input, sampler, (int2)(x0, y0));
    p[1] = read_imagef (input, sampler, (int2)(x0 + 1, y0));
    p[2] = read_imagef (input, sampler, (int2)(x0 + 2, y0));
    p[3] = read_imagef (input, sampler, (int2)(x0 + 3, y0));
    p[4] = read_imagef (input, sampler, (int2)(x0, y0 + 1));
    p[5] = read_imagef (input, sampler, (int2)(x0 + 1, y0 + 1));
    p[6] = read_imagef (input, sampler, (int2)(x0 + 2, y0 + 1));
    p[7] = read_imagef (input, sampler, (int2)(x0 + 3, y0 + 1));
    p[8] = read_imagef (input, sampler, (int2)(x0, y0 + 2));
    p[9] = read_imagef (input, sampler, (int2)(x0 + 1, y0 + 2));
    p[10] = read_imagef (input, sampler, (int2)(x0 + 2, y0 + 2));
    p[11] = read_imagef (input, sampler, (int2)(x0 + 3, y0 + 2));
    p[12] = read_imagef (input, sampler, (int2)(x0, y0 + 3));
    p[13] = read_imagef (input, sampler, (int2)(x0 + 1, y0 + 3));
    p[14] = read_imagef (input, sampler, (int2)(x0 + 2, y0 + 3));
    p[15] = read_imagef (input, sampler, (int2)(x0 + 3, y0 + 3));
#endif

#pragma unroll

    for (int i = 0; i < 16; ++i) {
        p[i] = read_imagef (input, sampler, (int2)(x0 + i % 4, y0 + i / 4));
    }

    float4 p00, p01, p10, p11;
    p00.x = (p[4].x + p[6].x) / 2.0;
    p00.y = (p[5].x * 4 + p[0].x + p[2].x + p[8].x + p[10].x) / 8.0;
    p00.z = (p[1].x + p[9].x) / 2.0;
    p01.x = p[6].x;
    p01.y = (p[2].x + p[5].x + p[7].x + p[10].x) / 4.0;
    p01.z = (p[1].x + p[3].x + p[9].x + p[11].x) / 4.0;
    p10.x = (p[4].x + p[6].x + p[12].x + p[14].x) / 4.0;
    p10.y = (p[5].x + p[8].x + p[10].x + p[13].x) / 4.0;
    p10.z = p[9].x;
    p11.x = (p[6].x + p[14].x) / 2.0;
    p11.y = (p[10].x * 4 + p[5].x + p[7].x + p[13].x + p[15].x) / 8.0;
    p11.z = (p[9].x + p[11].x) / 2.0;

    write_imagef (output, (int2)(x, y), p00);
    write_imagef (output, (int2)(x + 1, y), p01);
    write_imagef (output, (int2)(x, y + 1), p10);
    write_imagef (output, (int2)(x + 1, y + 1), p11);
}

