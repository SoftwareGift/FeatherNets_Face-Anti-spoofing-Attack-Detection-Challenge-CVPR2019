/*
 * function: kernel_retinex
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */
typedef struct {
    float    gain;
    float    threshold;
    float    log_min;
    float    log_max;
    float    width;
    float    height;
} CLRetinexConfig;

__kernel void kernel_retinex (__read_only image2d_t input, __read_only image2d_t ga_input, __write_only image2d_t output, uint vertical_offset_in, uint vertical_offset_out, CLRetinexConfig re_config)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    sampler_t sampler1 = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 y_out, uv_in;
    float4 y_in, y_ga;
    float y_lg;
    int i;
    // cpy UV
    if(y % 2 == 0) {
        uv_in = read_imagef(input, sampler, (int2)(x, y / 2 + vertical_offset_in));
        write_imagef(output, (int2)(x, y / 2 + vertical_offset_out), uv_in);
    }

    y_in = read_imagef(input, sampler, (int2)(x, y)) * 255.0;
    y_ga = read_imagef(ga_input, sampler1, (float2)(x / re_config.width, y / (re_config.height / 2 * 3))) * 255.0;

    y_lg = log(y_in.x) - log(y_ga.x);

    y_out.x = re_config.gain * y_in.x / 128.0 * (y_lg - re_config.log_min) / 255.0;
    write_imagef(output, (int2)(x, y), y_out);
}
