/*
 * function: kernel_macc
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * table: macc table.
 */
unsigned int get_sector_id (float u, float v)
{
    u = fabs(u) > 0.00001f ? u : 0.00001f;
    float tg = v / u;
    unsigned int se = tg > 1 ? (tg > 2 ? 3 : 2) : (tg > 0.5 ? 1 : 0);
    unsigned int so = tg > -1 ? (tg > -0.5 ? 3 : 2) : (tg > -2 ? 1 : 0);
    return tg > 0 ? (u > 0 ? se : (se + 8)) : (u > 0 ? (so + 12) : (so + 4));
}
__kernel void kernel_macc (__read_only image2d_t input, __write_only image2d_t output, __global float *table)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_in[8], pixel_out[8];
    float Y[8], ui[8], vi[8], uo[8], vo[8];
    unsigned int table_id[8];
    int i = 0, j = 0;

#pragma unroll
    for(j = 0; j < 2; j++) {
#pragma unroll
        for(i = 0; i < 4; i++) {
            pixel_in[j * 4 + i] = read_imagef(input, sampler, (int2)(4 * x + i, 2 * y + j));
            Y[j * 4 + i] = 0.3 * pixel_in[j * 4 + i].x + 0.59 * pixel_in[j * 4 + i].y + 0.11 * pixel_in[j * 4 + i].z;
            ui[j * 4 + i] = 0.493 * (pixel_in[j * 4 + i].z - Y[j * 4 + i]);
            vi[j * 4 + i] = 0.877 * (pixel_in[j * 4 + i].x - Y[j * 4 + i]);
            table_id[j * 4 + i] = get_sector_id(ui[j * 4 + i], vi[j * 4 + i]);
            uo[j * 4 + i] = ui[j * 4 + i] * table[4 * table_id[j * 4 + i]] + vi[j * 4 + i] * table[4 * table_id[j * 4 + i] + 1];
            vo[j * 4 + i] = ui[j * 4 + i] * table[4 * table_id[j * 4 + i] + 2] + vi[j * 4 + i] * table[4 * table_id[j * 4 + i] + 3];
            pixel_out[j * 4 + i].x = Y[j * 4 + i] + 1.14 * vo[j * 4 + i];
            pixel_out[j * 4 + i].y = Y[j * 4 + i] - 0.39 * uo[j * 4 + i] - 0.58 * vo[j * 4 + i];
            pixel_out[j * 4 + i].z = Y[j * 4 + i] + 2.03 * uo[j * 4 + i];
            pixel_out[j * 4 + i].w = 0.0;
            write_imagef(output, (int2)(4 * x + i, 2 * y + j), pixel_out[j * 4 + i]);
        }
    }
}


