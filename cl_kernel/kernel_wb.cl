/*
 * function: kernel_wb
 *     black level correction for sensor data input
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * wb_config: white balance configuration
 */

typedef struct
{
    float r_gain;
    float gr_gain;
    float gb_gain;
    float b_gain;
} CLWBConfig;

__kernel void kernel_wb (__read_only image2d_t input,
                         __write_only image2d_t output,
                         CLWBConfig wb_config)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 Gr_in, R_in, B_in, Gb_in;
    float4 Gr_out, R_out, B_out, Gb_out;
    Gr_in = read_imagef(input, sampler, (int2)(2 * x, 2 * y));
    R_in = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y));
    B_in = read_imagef(input, sampler, (int2)(2 * x, 2 * y + 1));
    Gb_in = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y + 1));
    Gr_out.x = Gr_in.x * wb_config.gr_gain;
    Gr_out.y = 0.0f;
    Gr_out.z = 0.0f;
    Gr_out.w = 1.0f;
    R_out.x = R_in.x * wb_config.r_gain;
    R_out.y = 0.0f;
    R_out.z = 0.0f;
    R_out.w = 1.0f;
    B_out.x = B_in.x * wb_config.b_gain;
    B_out.y = 0.0f;
    B_out.z = 0.0f;
    B_out.w = 1.0f;
    Gb_out.x = Gb_in.x * wb_config.gb_gain;
    Gb_out.y = 0.0f;
    Gb_out.z = 0.0f;
    Gb_out.w = 1.0f;
    write_imagef(output, (int2)(2 * x, 2 * y), Gr_out);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), R_out);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), B_out);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), Gb_out);
}
