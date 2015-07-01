/*
 * function: kernel_gamma
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * table: gamma table.
 */
__kernel void kernel_gamma (__read_only image2d_t input, __write_only image2d_t output, __global float *table)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_in[8], pixel_out[8];
    int i=0,j=0;

#pragma unroll
    for(j=0;j<2;j++) {
#pragma unroll
	for(i=0;i<4;i++) {
	     pixel_in[j*4 + i] = read_imagef(input, sampler,(int2)(4*x + i, 2*y + j));
	     pixel_out[j*4 + i].x = table[convert_int(pixel_in[j*4 + i].x * 255.0)] / 255.0;
	     pixel_out[j*4 + i].y = table[convert_int(pixel_in[j*4 + i].y * 255.0)] / 255.0;
	     pixel_out[j*4 + i].z = table[convert_int(pixel_in[j*4 + i].z * 255.0)] / 255.0;
	     pixel_out[j*4 + i].w = 0.0;
	     write_imagef(output, (int2)(4*x + i, 2*y + j), pixel_out[j*4 + i]);
	 }
    }
}
