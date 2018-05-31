/*
 * function: kernel_retinex
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

#ifndef RETINEX_SCALE_SIZE
#define RETINEX_SCALE_SIZE 2
#endif

typedef struct {
    float    gain;
    float    threshold;
    float    log_min;
    float    log_max;
    float    width;
    float    height;
} CLRetinexConfig;

__constant float log_table[256] = {
    0.000000f, 0.693147f, 1.098612f, 1.386294f, 1.609438f, 1.791759f, 1.945910f, 2.079442f,
    2.197225f, 2.302585f, 2.397895f, 2.484907f, 2.564949f, 2.639057f, 2.708050f, 2.772589f,
    2.833213f, 2.890372f, 2.944439f, 2.995732f, 3.044522f, 3.091042f, 3.135494f, 3.178054f,
    3.218876f, 3.258097f, 3.295837f, 3.332205f, 3.367296f, 3.401197f, 3.433987f, 3.465736f,
    3.496508f, 3.526361f, 3.555348f, 3.583519f, 3.610918f, 3.637586f, 3.663562f, 3.688879f,
    3.713572f, 3.737670f, 3.761200f, 3.784190f, 3.806662f, 3.828641f, 3.850148f, 3.871201f,
    3.891820f, 3.912023f, 3.931826f, 3.951244f, 3.970292f, 3.988984f, 4.007333f, 4.025352f,
    4.043051f, 4.060443f, 4.077537f, 4.094345f, 4.110874f, 4.127134f, 4.143135f, 4.158883f,
    4.174387f, 4.189655f, 4.204693f, 4.219508f, 4.234107f, 4.248495f, 4.262680f, 4.276666f,
    4.290459f, 4.304065f, 4.317488f, 4.330733f, 4.343805f, 4.356709f, 4.369448f, 4.382027f,
    4.394449f, 4.406719f, 4.418841f, 4.430817f, 4.442651f, 4.454347f, 4.465908f, 4.477337f,
    4.488636f, 4.499810f, 4.510860f, 4.521789f, 4.532599f, 4.543295f, 4.553877f, 4.564348f,
    4.574711f, 4.584967f, 4.595120f, 4.605170f, 4.615121f, 4.624973f, 4.634729f, 4.644391f,
    4.653960f, 4.663439f, 4.672829f, 4.682131f, 4.691348f, 4.700480f, 4.709530f, 4.718499f,
    4.727388f, 4.736198f, 4.744932f, 4.753590f, 4.762174f, 4.770685f, 4.779123f, 4.787492f,
    4.795791f, 4.804021f, 4.812184f, 4.820282f, 4.828314f, 4.836282f, 4.844187f, 4.852030f,
    4.859812f, 4.867534f, 4.875197f, 4.882802f, 4.890349f, 4.897840f, 4.905275f, 4.912655f,
    4.919981f, 4.927254f, 4.934474f, 4.941642f, 4.948760f, 4.955827f, 4.962845f, 4.969813f,
    4.976734f, 4.983607f, 4.990433f, 4.997212f, 5.003946f, 5.010635f, 5.017280f, 5.023881f,
    5.030438f, 5.036953f, 5.043425f, 5.049856f, 5.056246f, 5.062595f, 5.068904f, 5.075174f,
    5.081404f, 5.087596f, 5.093750f, 5.099866f, 5.105945f, 5.111988f, 5.117994f, 5.123964f,
    5.129899f, 5.135798f, 5.141664f, 5.147494f, 5.153292f, 5.159055f, 5.164786f, 5.170484f,
    5.176150f, 5.181784f, 5.187386f, 5.192957f, 5.198497f, 5.204007f, 5.209486f, 5.214936f,
    5.220356f, 5.225747f, 5.231109f, 5.236442f, 5.241747f, 5.247024f, 5.252273f, 5.257495f,
    5.262690f, 5.267858f, 5.273000f, 5.278115f, 5.283204f, 5.288267f, 5.293305f, 5.298317f,
    5.303305f, 5.308268f, 5.313206f, 5.318120f, 5.323010f, 5.327876f, 5.332719f, 5.337538f,
    5.342334f, 5.347108f, 5.351858f, 5.356586f, 5.361292f, 5.365976f, 5.370638f, 5.375278f,
    5.379897f, 5.384495f, 5.389072f, 5.393628f, 5.398163f, 5.402677f, 5.407172f, 5.411646f,
    5.416100f, 5.420535f, 5.424950f, 5.429346f, 5.433722f, 5.438079f, 5.442418f, 5.446737f,
    5.451038f, 5.455321f, 5.459586f, 5.463832f, 5.468060f, 5.472271f, 5.476464f, 5.480639f,
    5.484797f, 5.488938f, 5.493061f, 5.497168f, 5.501258f, 5.505332f, 5.509388f, 5.513429f,
    5.517453f, 5.521461f, 5.525453f, 5.529429f, 5.533389f, 5.537334f, 5.541264f, 5.545177f
};

__kernel void kernel_retinex (
    __read_only image2d_t input_y, __read_only image2d_t input_uv,
    __read_only image2d_t ga_input0,
#if RETINEX_SCALE_SIZE > 1
    __read_only image2d_t ga_input1,
#endif
#if RETINEX_SCALE_SIZE > 2
    __read_only image2d_t ga_input2,
#endif
    __write_only image2d_t output_y, __write_only image2d_t output_uv,
    CLRetinexConfig re_config)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler_orig = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    sampler_t sampler_ga = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    float4 y_out, uv_in;
    float4 y_in, y_ga[RETINEX_SCALE_SIZE];
    float4 y_in_lg, y_lg;
    int i;

    y_in = read_imagef(input_y, sampler_orig, (int2)(x, y)) * 255.0f;
    y_in_lg.x = log_table[convert_int(y_in.x)];
    y_in_lg.y = log_table[convert_int(y_in.y)];
    y_in_lg.z = log_table[convert_int(y_in.z)];
    y_in_lg.w = log_table[convert_int(y_in.w)];

    float ga_x_step = 1.0f / re_config.width;
    float2 pos_ga = (float2)(x * 4.0f * ga_x_step, y / re_config.height);
    y_ga[0].x = read_imagef(ga_input0, sampler_ga, pos_ga).x * 255.0f;
    pos_ga.x += ga_x_step;
    y_ga[0].y = read_imagef(ga_input0, sampler_ga, pos_ga).x * 255.0f;
    pos_ga.x += ga_x_step;
    y_ga[0].z = read_imagef(ga_input0, sampler_ga, pos_ga).x * 255.0f;
    pos_ga.x += ga_x_step;
    y_ga[0].w = read_imagef(ga_input0, sampler_ga, pos_ga).x * 255.0f;

#if RETINEX_SCALE_SIZE > 1
    y_ga[1].x = read_imagef(ga_input1, sampler_ga, pos_ga).x * 255.0f;
    pos_ga.x += ga_x_step;
    y_ga[1].y = read_imagef(ga_input1, sampler_ga, pos_ga).x * 255.0f;
    pos_ga.x += ga_x_step;
    y_ga[1].z = read_imagef(ga_input1, sampler_ga, pos_ga).x * 255.0f;
    pos_ga.x += ga_x_step;
    y_ga[1].w = read_imagef(ga_input1, sampler_ga, pos_ga).x * 255.0f;
#endif

#if RETINEX_SCALE_SIZE > 2
    y_ga[2].x = read_imagef(ga_input2, sampler_ga, pos_ga).x * 255.0f;
    pos_ga.x += ga_x_step;
    y_ga[2].y = read_imagef(ga_input2, sampler_ga, pos_ga).x * 255.0f;
    pos_ga.x += ga_x_step;
    y_ga[2].z = read_imagef(ga_input2, sampler_ga, pos_ga).x * 255.0f;
    pos_ga.x += ga_x_step;
    y_ga[2].w = read_imagef(ga_input2, sampler_ga, pos_ga).x * 255.0f;
#endif


    y_lg = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < RETINEX_SCALE_SIZE; ++i) {
        y_lg.x += y_in_lg.x - log_table[convert_int(y_ga[i].x)];
        y_lg.y += y_in_lg.y - log_table[convert_int(y_ga[i].y)];
        y_lg.z += y_in_lg.z - log_table[convert_int(y_ga[i].z)];
        y_lg.w += y_in_lg.w - log_table[convert_int(y_ga[i].w)];
    }
    y_lg = y_lg / (float)(RETINEX_SCALE_SIZE);

    //y_out = re_config.gain * (y_in + 20.0f) / 128.0f * (y_lg - re_config.log_min);
    y_out = re_config.gain * (y_ga[0] + 20.0f) / 128.0f * (y_lg - re_config.log_min);
    write_imagef(output_y, (int2)(x, y), y_out);

    // copy UV
    if(y % 2 == 0) {
        float2 avg_y_out, avg_y_in, gain_y;
        float4 uv_out, gain_uv;
        y_in = y_in / 255.0f;
        avg_y_in = (float2)((y_in.x + y_in.y) * 0.5f, (y_in.z + y_in.w) * 0.5f);
        avg_y_out = (float2)((y_out.x + y_out.y) * 0.5f, (y_out.z + y_out.w) * 0.5f);
        avg_y_out = clamp (avg_y_out, 0.0f, 1.0f);
        avg_y_in = (avg_y_in > 0.5f) ? (1.0f - avg_y_in) : avg_y_in;
        avg_y_out = (avg_y_out > 0.5f) ? (1.0f - avg_y_out) : avg_y_out;
        gain_y = (avg_y_out + 0.1f) / (avg_y_in + 0.05f);
        gain_y = gain_y * (avg_y_in * 2.0f + 1.0f);

        uv_in = read_imagef(input_uv, sampler_orig, (int2)(x, y / 2)) - 0.5f;
        float2 v_coef = 1.01f / (1.13f * uv_in.xz + 0.01f);
        float2 v_gain_1 = v_coef - avg_y_in * v_coef;
        float2 v_gain_2 = -v_coef;
        float2 v_gain_min = (v_gain_1 < v_gain_2) ? v_gain_1 : v_gain_2;
        float2 v_gain_max = (v_gain_1 < v_gain_2) ? v_gain_2 : v_gain_1;
        v_gain_min = max (v_gain_min, 0.1f);
        v_gain_max = max (v_gain_max, 0.1f);
        gain_y = clamp (gain_y, v_gain_min, v_gain_max);

        float2 u_coef = 1.01f / (2.03f * uv_in.yw + 0.01f);
        float2 u_gain_1 = u_coef - avg_y_in * u_coef;
        float2 u_gain_2 = -u_coef;
        float2 u_gain_min = (u_gain_1 < u_gain_2) ? u_gain_1 : u_gain_2;
        float2 u_gain_max = (u_gain_1 < u_gain_2) ? u_gain_2 : u_gain_1;
        u_gain_min = max (u_gain_min, 0.1f);
        u_gain_max = max (u_gain_max, 0.1f);
        gain_y = clamp (gain_y, u_gain_min, u_gain_max);
        gain_uv = (float4) (gain_y, gain_y);
        //printf (" (%.2f) ", gain_uv.x);
        uv_out = uv_in * gain_uv + 0.5f;
        write_imagef(output_uv, (int2)(x, y / 2), uv_out);
    }
}
