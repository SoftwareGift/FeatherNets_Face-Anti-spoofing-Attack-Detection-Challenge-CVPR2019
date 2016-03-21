/*
 * function: kernel_gauss
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

__kernel void kernel_gauss (__read_only image2d_t input, __write_only image2d_t output, __global float *table)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 y_in[36];
    float4 in1;
    int i, j;
    int index;
    float4 out1 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 out2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
    for(i = 0; i < 5 + 1; i++)
        for(j = 0; j < 5 + 3; j++) {
            in1 = read_imagef (input, sampler, (int2)(4 * x - 2 + j, 2 * y - 2 + i));
            //first line
            if (i < 5) {
                index = i * 5 + j;
                out1.x +=  (j < 5 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out1.y += (j < 6 && j > 0 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out1.z += (j < 7 && j > 1 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out1.w += (j > 2 ? table[index] * in1.x : 0.0f);
            }
            //second line
            if (i > 0) {
                index = (i - 1) * 5 + j;
                out2.x +=  (j < 5 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out2.y += (j < 6 && j > 0 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out2.z += (j < 7 && j > 1 ? table[index] * in1.x : 0.0f);
                index -= 1;
                out2.w += (j > 2 ? table[index] * in1.x : 0.0f);
            }
        }

    write_imagef(output, (int2)(x, 2 * y), out1);
    write_imagef(output, (int2)(x,  2 * y + 1), out2);

}

