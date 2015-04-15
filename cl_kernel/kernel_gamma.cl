/*
 * function: kernel_gamma
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * table: gamma table.
 */
"__kernel void kernel_gamma (__read_only image2d_t input, __write_only image2d_t output, __global float *table)        "
"{                                                                                             "
"    int x = get_global_id (0);                                                                "
"    int y = get_global_id (1);                                                                "
"    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;  "
"    int2 pos = (int2)(x, y);                                                                  "
"    float4 pixel_in,pixel_out;									"
"    pixel_in = read_imagef(input, sampler, pos);                             		   "
"    pixel_out.x = table[convert_int(pixel_in.x*255.0)]/255.0;                            "
"    pixel_out.y = table[convert_int(pixel_in.y*255.0)]/255.0;                            "
"    pixel_out.z = table[convert_int(pixel_in.z*255.0)]/255.0;                            "
"    pixel_out.w = 0.0;                            "
"    write_imagef(output, pos, pixel_out);                                                        "
"}                                                                                             "

