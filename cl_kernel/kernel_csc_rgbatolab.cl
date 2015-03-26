/*
 * function: kernel_csc_rgbatolab
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

"static float fun(float a)    "
"{                            "
"if (a > 0.008856)            "
"    return pow(a,1.0/3);     "
"else                         "
"    return (float)(7.787*a + 16.0/116);    "
"}                            "
"__kernel void kernel_csc_rgbatolab (__read_only image2d_t input, __write_only image2d_t output)        "
"{                                                                                             "
"    int x = get_global_id (0);                                                            "
"    int y = get_global_id (1);                                                             "
"    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;  "
"    float4 pixel_in = read_imagef(input, sampler, (int2)(x,y));                                          "
"    float X,Y,Z,L,a,b;                                                                                          "
"    X = 0.433910*pixel_in.x + 0.376220*pixel_in.y + 0.189860*pixel_in.z;    "
"    Y = 0.212649*pixel_in.x + 0.715169*pixel_in.y + 0.072182*pixel_in.z;    "
"    Z = 0.017756*pixel_in.x + 0.109478*pixel_in.y + 0.872915*pixel_in.z;    "
"    if(Y > 0.008856)             "
"    L = 116*(pow(Y,1.0/3));    "
"    else                       "
"    L = 903.3*Y;               "
"    a = 500*(fun(X) - fun(Y));    "
"    b = 200*(fun(Y) - fun(Z));    "
"    write_imagef(output, (int2)(3*x,y), L);                                                        "
"    write_imagef(output, (int2)(3*x+1,y), a);                                                        "
"    write_imagef(output, (int2)(3*x+2,y), b);                                                        "
"}                                                                                             "

