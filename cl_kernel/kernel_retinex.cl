/*
 * function: kernel_retinex
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */
typedef struct {
    float  gain;
    float  threshold;
    float           log_min;
    float           log_max;
} CLRetinexConfig;

__kernel void kernel_retinex (__read_only image2d_t input, __write_only image2d_t output, uint vertical_offset_in, uint vertical_offset_out, CLRetinexConfig re_config, __global float *table)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 y_out, uv_in;
    float4 y_in[25];
    float y_ga, y_lg;
    int i;
    // cpy UV
    if(y % 2 == 0) {
        uv_in = read_imagef(input, sampler, (int2)(x, y / 2 + vertical_offset_in));
        write_imagef(output, (int2)(x, y / 2 + vertical_offset_out), uv_in);
    }

    for(i = 0; i < 25; i++)
        y_in[i] = read_imagef(input, sampler, (int2)(x - 2 + i % 5, y - 2 + i / 5)) * 255.0;

    for(i = 0; i < 25; i++)
        y_ga += y_in[i].x * table[i];
    y_lg = log(y_in[12].x) - log(y_ga);

    if(y_lg < re_config.log_min)
        y_out.x = 0.0f;
    else if(y_lg > re_config.log_max)
        y_out.x = 1.0;
    else
        y_out.x = re_config.gain * (y_lg - re_config.log_min) / 255.0;
    write_imagef(output, (int2)(x, y), y_out);
}

