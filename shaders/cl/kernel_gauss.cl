/*
 * function: kernel_gauss
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * workitem = 4x2 pixel ouptut
 * GAUSS_RADIUS must be defined in build options.
 */

#ifndef GAUSS_RADIUS
#define GAUSS_RADIUS 2
#endif

#define GAUSS_SCALE (2 * GAUSS_RADIUS + 1)

__kernel void kernel_gauss (__read_only image2d_t input, __write_only image2d_t output, __global float *table)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 in1;
    int i, j;
    int index;
    float4 out1 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 out2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for(i = 0; i < GAUSS_SCALE + 1; i++)
        for(j = 0; j < GAUSS_SCALE + 3; j++) {
            in1 = read_imagef (input, sampler, (int2)(4 * x - GAUSS_RADIUS + j, 2 * y - GAUSS_RADIUS + i));
            //first line
            if (i < GAUSS_SCALE) {
                index = i * GAUSS_SCALE + j;
                out1.x +=  (j < GAUSS_SCALE ? table[index] * in1.x : 0.0f);
                index -= 1;
                out1.y += ((j < GAUSS_SCALE + 1) && j > 0 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out1.z += ((j < GAUSS_SCALE + 2) && j > 1 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out1.w += (j > 2 ? table[index] * in1.x : 0.0f);
            }
            //second line
            if (i > 0) {
                index = (i - 1) * GAUSS_SCALE + j;
                out2.x +=  (j < GAUSS_SCALE ? table[index] * in1.x : 0.0f);
                index -= 1;
                out2.y += ((j < GAUSS_SCALE + 1) && j > 0 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out2.z += ((j < GAUSS_SCALE + 2) && j > 1 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out2.w += (j > 2 ? table[index] * in1.x : 0.0f);
            }
        }

    write_imagef(output, (int2)(x, 2 * y), out1);
    write_imagef(output, (int2)(x,  2 * y + 1), out2);

}

