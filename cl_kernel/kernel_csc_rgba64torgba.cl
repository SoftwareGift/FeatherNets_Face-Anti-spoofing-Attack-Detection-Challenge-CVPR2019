/*
 * function: kernel_csc_rgba64torgba
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */
"__kernel void kernel_csc_rgba64torgba (__read_only image2d_t input, __write_only image2d_t output)        "
"{                                                                                             "
"    int x = get_global_id (0);                                                            "
"    int y = get_global_id (1);                                                             "
"    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;  "
"    float4 pixel_in = read_imagef(input, sampler, (int2)(x,y));                                          "
"    write_imagef(output, (int2)(x,y), pixel_in);                                                        "
"}                                                                                             "

