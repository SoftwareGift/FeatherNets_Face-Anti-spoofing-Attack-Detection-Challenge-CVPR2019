/*
 * function: kernel_bayer_pipe
 * params:
 *   input:    image2d_t as read only
 *   output:   image2d_t as write only
 *   blc_config: black level correction configuration
 *   wb_config: whitebalance configuration
 *   gamma_table: RGGB table
 *   stats_output: 3a stats output
 */


#define WORKGROUP_CELL_WIDTH 64
#define WORKGROUP_CELL_HEIGHT 4

#define DEMOSAIC_X_CELL_PER_WORKITEM 2

#define PIXEL_PER_CELL 2

#define SLM_CELL_X_OFFSET 4
#define SLM_CELL_Y_OFFSET 1

// 8x8
#define SLM_CELL_X_VALID_SIZE WORKGROUP_CELL_WIDTH
#define SLM_CELL_Y_VALID_SIZE WORKGROUP_CELL_HEIGHT

// 10x10
#define SLM_CELL_X_SIZE (SLM_CELL_X_VALID_SIZE + SLM_CELL_X_OFFSET * 2)
#define SLM_CELL_Y_SIZE (SLM_CELL_Y_VALID_SIZE + SLM_CELL_Y_OFFSET * 2)

#define GUASS_DELTA_S_1      1.031739f
#define GUASS_DELTA_S_1_5    1.072799f
#define GUASS_DELTA_S_2      1.133173f
#define GUASS_DELTA_S_2_5    1.215717f

typedef struct
{
    float           ee_gain;
    float           ee_threshold;
    float           nr_gain;
} CLEeConfig;

inline int get_shared_pos_x (int i)
{
    return i % SLM_CELL_X_SIZE;
}

inline int get_shared_pos_y (int i)
{
    return i / SLM_CELL_X_SIZE;
}

inline int shared_pos (int x, int y)
{
    return mad24(y, SLM_CELL_X_SIZE, x);
}

/* BA10=> GRBG  */
inline void grbg_slm_load (
    __local float *px, __local float *py, __local float *pz, __local float *pw,
    int index, __read_only image2d_t input, uint input_height, int x_start, int y_start
)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float4 data1, data2, line1, line2;
    int x0 = (get_shared_pos_x (index) + x_start) / 4;
    int y0 = get_shared_pos_y (index) + y_start;
    int2 pos = (int2)(x0, y0);
    float4 gr, r, b, gb;

    y0 = y0 > 0 ? y0 : 0;

    gr = read_imagef (input, sampler, (int2)(x0, y0));
    r = read_imagef (input, sampler, (int2)(x0, y0 + input_height));
    b = read_imagef (input, sampler, (int2)(x0, y0 + input_height * 2));
    gb = read_imagef (input, sampler, (int2)(x0, y0 + input_height * 3));

    (*(__local float4 *)(px + index)) = gr;
    (*(__local float4 *)(py + index)) = r;
    (*(__local float4 *)(pz + index)) = b;
    (*(__local float4 *)(pw + index)) = gb;
}

#define MAX_DELTA_COFF 5.0f
#define MIN_DELTA_COFF 1.0f
#define DEFAULT_DELTA_COFF 4.0f

inline float2 delta_coff (float2 in, __local float *table)
{
    float2 out;
    out.x = table[(int)(fabs(in.x * 64.0f))];
    out.y = table[(int)(fabs(in.y * 64.0f))];

    return out;
}

inline float2 dot_denoise (float2 value, float2 in1, float2 in2, float2 in3, float2 in4, __local float *table, float coff0)
{
    float2 coff1, coff2, coff3, coff4, coff5;
    coff1 = delta_coff (in1 - value, table);
    coff2 = delta_coff (in2 - value, table);
    coff3 = delta_coff (in3 - value, table);
    coff4 = delta_coff (in4 - value, table);
    //(in1 * coff1 + in2 * coff2 + in3 * coff3 + in4 * coff4 + value * coff0)
    float2 sum1 = (mad (in1, coff1,
                        mad (in2, coff2,
                             mad (in3, coff3,
                                  mad (in4, coff4, value * coff0)))));
    return  sum1 / (coff0 + coff1 + coff2 + coff3 + coff4);
}

inline float2 dot_ee (float2 value, float2 in1, float2 in2, float2 in3, float2 in4, float2 out, CLEeConfig ee_config, float2 *egain)
{

    float2 ee = mad(in1 + in2 + in3 + in4, -0.25f, value);
    ee =  fabs(ee) > ee_config.ee_threshold ? ee : 0.0f;

    egain[0] = mad(ee, ee_config.ee_gain, out + 0.03f) / (out + 0.03f);

    return out * egain[0];
}

inline float2 dot_denoise_ee (float2 value, float2 in1, float2 in2, float2 in3, float2 in4, __local float *table, float coff0, float2 *egain, CLEeConfig ee_config)
{
    float2 out = dot_denoise(value, in1, in2, in3, in4, table, coff0);
    return dot_ee(value, in1, in2, in3, in4, out, ee_config, egain);
}

void demosaic_2_cell (
    __local float *x_data_in, __local float *y_data_in, __local float *z_data_in, __local float *w_data_in,
    int in_x, int in_y,
    __write_only image2d_t out, uint out_height, int out_x, int out_y)
{
    float4 out_data;
    float2 value;
    int index;
    {
        float3 R_y[2];
        index = shared_pos (in_x - 1, in_y);
        R_y[0] = *(__local float3*)(y_data_in + index);
        index = shared_pos (in_x - 1, in_y + 1);
        R_y[1] = *(__local float3*)(y_data_in + index);

        out_data.s02 = (R_y[0].s01 + R_y[0].s12) * 0.5f;
        out_data.s13 = R_y[0].s12;
        write_imagef (out, (int2)(out_x, out_y), out_data);

        out_data.s02 = (R_y[0].s01 + R_y[0].s12 + R_y[1].s01 + R_y[1].s12) * 0.25f;
        out_data.s13 = (R_y[0].s12 + R_y[1].s12) * 0.5f;
        write_imagef (out, (int2)(out_x, out_y + 1), out_data);
    }

    {
        float3 B_z[2];
        index = shared_pos (in_x, in_y - 1);
        B_z[0] = *(__local float3*)(z_data_in + index);
        index = shared_pos (in_x, in_y);
        B_z[1] = *(__local float3*)(z_data_in + index);

        out_data.s02 = (B_z[0].s01 + B_z[1].s01) * 0.5f;
        out_data.s13 = (B_z[0].s01 + B_z[0].s12 + B_z[1].s01 + B_z[1].s12) * 0.25f;
        write_imagef (out, (int2)(out_x, out_y + out_height * 2), out_data);

        out_data.s02 = B_z[1].s01;
        out_data.s13 = (B_z[1].s01 + B_z[1].s12) * 0.5f;
        write_imagef (out, (int2)(out_x, out_y + 1 + out_height * 2), out_data);
    }

    {
        float3 Gr_x[2], Gb_w[2];
        index = shared_pos (in_x, in_y);
        Gr_x[0] = *(__local float3*)(x_data_in + index);
        index = shared_pos (in_x, in_y + 1);
        Gr_x[1] = *(__local float3*)(x_data_in + index);

        index = shared_pos (in_x - 1, in_y - 1);
        Gb_w[0] = *(__local float3*)(w_data_in + index);
        index = shared_pos (in_x - 1, in_y);
        Gb_w[1] = *(__local float3*)(w_data_in + index);

        out_data.s02 = (Gr_x[0].s01 * 4.0f + Gb_w[0].s01 +
                        Gb_w[0].s12 + Gb_w[1].s01 + Gb_w[1].s12) * 0.125f;
        out_data.s13 = (Gr_x[0].s01 + Gr_x[0].s12 + Gb_w[0].s12 + Gb_w[1].s12) * 0.25f;
        write_imagef (out, (int2)(out_x, out_y + out_height), out_data);

        out_data.s02 = (Gr_x[0].s01 + Gr_x[1].s01 + Gb_w[1].s01 + Gb_w[1].s12) * 0.25f;

        out_data.s13 = (Gb_w[1].s12 * 4.0f + Gr_x[0].s01 +
                        Gr_x[0].s12 + Gr_x[1].s01 + Gr_x[1].s12) * 0.125f;
        write_imagef (out, (int2)(out_x, out_y + 1 + out_height), out_data);
    }
}

void demosaic_denoise_2_cell (
    __local float *x_data_in, __local float *y_data_in, __local float *z_data_in, __local float *w_data_in,
    int in_x, int in_y,
    __write_only image2d_t out, uint out_height, int out_x, int out_y, __local float *table, CLEeConfig ee_config)
{
    float4 out_data_r[2];
    float4 out_data_g[2];
    float4 out_data_b[2];
    float2 value;
    int index;
    float2 egain[4];
    float2 de;
    float gain_coff0 = table[0];

    float4 R_y[3], B_z[3];;
    float2 Gr_x0, Gb_w2;
    float4 Gr_x1, Gb_w1;
    float3 Gr_x2, Gb_w0;

    // R egain
    {
        index = shared_pos (in_x - 1, in_y - 1);
        R_y[0] = *(__local float4*)(y_data_in + index);
        index = shared_pos (in_x - 1, in_y);
        R_y[1] = *(__local float4*)(y_data_in + index);
        index = shared_pos (in_x - 1, in_y + 1);
        R_y[2] = *(__local float4*)(y_data_in + index);

        out_data_r[0].s13 = dot_denoise_ee (R_y[1].s12, R_y[0].s12, R_y[1].s01, R_y[1].s23, R_y[2].s12,
                                            table, gain_coff0 * GUASS_DELTA_S_2, &egain[1], ee_config);
    }

    // Gr, Gb egain
    {
        index = shared_pos (in_x, in_y - 1);
        Gr_x0 = *(__local float2*)(x_data_in + index);
        index = shared_pos (in_x - 1, in_y);
        Gr_x1 = *(__local float4*)(x_data_in + index);
        index = shared_pos (in_x, in_y + 1);
        Gr_x2 = *(__local float3*)(x_data_in + index);

        index = shared_pos (in_x - 1, in_y - 1);
        Gb_w0 = *(__local float3*)(w_data_in + index);
        index = shared_pos (in_x - 1, in_y);
        Gb_w1 = *(__local float4*)(w_data_in + index);
        index = shared_pos (in_x, in_y + 1);
        Gb_w2 = *(__local float2*)(w_data_in + index);

        value = mad (Gr_x1.s12, 4.0f, (Gb_w0.s01 + Gb_w0.s12 + Gb_w1.s01 + Gb_w1.s12)) * 0.125f;
        de = dot_denoise (value, Gb_w0.s01, Gb_w0.s12, Gb_w1.s01, Gb_w1.s12, table, gain_coff0 * GUASS_DELTA_S_1_5);
        out_data_g[0].s02 = dot_ee(Gr_x1.s12, Gr_x0, Gr_x1.s01, Gr_x1.s23, Gr_x2.s01, de, ee_config, &egain[0]);

        value = mad (Gb_w1.s12, 4.0f, (Gr_x1.s12 + Gr_x1.s23 + Gr_x2.s01 + Gr_x2.s12)) * 0.125f;
        de = dot_denoise (value, Gr_x1.s12, Gr_x1.s23, Gr_x2.s01, Gr_x2.s12, table, gain_coff0 * GUASS_DELTA_S_1_5);
        out_data_g[1].s13 = dot_ee(Gb_w1.s12, Gb_w0.s12, Gb_w1.s01, Gb_w1.s23, Gb_w2, de, ee_config, &egain[3]);
    }

    // B egain
    {
        index = shared_pos (in_x - 1, in_y - 1);
        B_z[0] = *(__local float4*)(z_data_in + index);
        index = shared_pos (in_x - 1, in_y);
        B_z[1] = *(__local float4*)(z_data_in + index);
        index = shared_pos (in_x - 1, in_y + 1);
        B_z[2] = *(__local float4*)(z_data_in + index);

        out_data_b[1].s02 = dot_denoise_ee (B_z[1].s12, B_z[0].s12, B_z[1].s01, B_z[1].s23, B_z[2].s12,
                                            table, gain_coff0 * GUASS_DELTA_S_2, &egain[2], ee_config);
    }

    ////////////////////////////////R//////////////////////////////////////////
    {
        value = (R_y[1].s01 + R_y[1].s12) * 0.5f;
        de =  dot_denoise (value, R_y[0].s01, R_y[0].s12, R_y[2].s01, R_y[2].s12, table, gain_coff0 * GUASS_DELTA_S_2_5);
        out_data_r[0].s02 = de * egain[0];

        value = (R_y[1].s01 + R_y[1].s12 + R_y[2].s01 + R_y[2].s12) * 0.25f;
        de = dot_denoise (value, R_y[1].s01, R_y[1].s12, R_y[2].s01, R_y[2].s12, table, gain_coff0 * GUASS_DELTA_S_1_5);
        out_data_r[1].s02 = de * egain[2];

        value = (R_y[1].s12 + R_y[2].s12) * 0.5f;
        de = dot_denoise (value, R_y[1].s01, R_y[1].s23, R_y[2].s01, R_y[2].s23, table, gain_coff0 * GUASS_DELTA_S_2_5);
        out_data_r[1].s13 = de * egain[3];

        write_imagef (out, (int2)(out_x, out_y), out_data_r[0]);
        write_imagef (out, (int2)(out_x, out_y + 1), out_data_r[1]);
    }

    ////////////////////////////////G//////////////////////////////////////////
    {
        value = (Gr_x1.s12 + Gr_x1.s23 + Gb_w0.s12 + Gb_w1.s12) * 0.25f;
        de = dot_denoise(value, Gr_x1.s12, Gr_x1.s23, Gb_w0.s12, Gb_w1.s12, table, gain_coff0 * GUASS_DELTA_S_1);
        out_data_g[0].s13 = de * egain[1];

        value = (Gr_x1.s12 + Gr_x2.s01 + Gb_w1.s01 + Gb_w1.s12) * 0.25f;
        de = dot_denoise (value, Gr_x1.s12, Gr_x2.s01, Gb_w1.s01, Gb_w1.s12, table, gain_coff0 * GUASS_DELTA_S_1);
        out_data_g[1].s02 = de * egain[2];

        write_imagef (out, (int2)(out_x, out_y + out_height), out_data_g[0]);
        write_imagef (out, (int2)(out_x, out_y + 1 + out_height), out_data_g[1]);
    }

    ////////////////////////////////B//////////////////////////////////////////
    {
        value = (B_z[0].s12 + B_z[1].s12) * 0.5f;
        de = dot_denoise (value, B_z[0].s01, B_z[0].s23, B_z[1].s01, B_z[1].s23, table, gain_coff0 * GUASS_DELTA_S_2_5);
        out_data_b[0].s02 = de * egain[0];

        value = (B_z[0].s12 + B_z[0].s23 +
                 B_z[1].s12 + B_z[1].s23) * 0.25f;
        de = dot_denoise (value, B_z[0].s12, B_z[0].s23, B_z[1].s12, B_z[1].s23, table, gain_coff0 * GUASS_DELTA_S_1_5);
        out_data_b[0].s13 = de * egain[1];

        value = (B_z[1].s12 + B_z[1].s23) * 0.5f;
        de = dot_denoise (value, B_z[0].s12, B_z[0].s23, B_z[2].s12, B_z[2].s23, table, gain_coff0 * GUASS_DELTA_S_2_5);
        out_data_b[1].s13 = de * egain[3];

        write_imagef (out, (int2)(out_x, out_y + out_height * 2), out_data_b[0]);
        write_imagef (out, (int2)(out_x, out_y + 1 + out_height * 2), out_data_b[1]);
    }
}

void shared_demosaic (
    __local float *x_data_in, __local float *y_data_in, __local float *z_data_in, __local float *w_data_in,
    int in_x, int in_y,
    __write_only image2d_t out, uint output_height, int out_x, int out_y,
    uint has_denoise, __local float *table, CLEeConfig ee_config)
{
    if (has_denoise) {
        demosaic_denoise_2_cell (
            x_data_in, y_data_in, z_data_in, w_data_in, in_x, in_y,
            out, output_height, out_x, out_y, table, ee_config);
    } else {
        demosaic_2_cell (
            x_data_in, y_data_in, z_data_in, w_data_in, in_x, in_y,
            out, output_height, out_x, out_y);
    }
}

__kernel void kernel_bayer_pipe (__read_only image2d_t input,
                                 uint input_height,
                                 __write_only image2d_t output,
                                 uint output_height,
                                 __global float * bnr_table,
                                 uint has_denoise,
                                 CLEeConfig ee_config
                                )
{
    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);
    int g_size_x = get_global_size (0);
    int g_size_y = get_global_size (1);

    int l_id_x = get_local_id(0);
    int l_id_y = get_local_id(1);
    int l_size_x = get_local_size (0);
    int l_size_y = get_local_size (1);

    __local float p1_x[SLM_CELL_X_SIZE * SLM_CELL_Y_SIZE], p1_y[SLM_CELL_X_SIZE * SLM_CELL_Y_SIZE], p1_z[SLM_CELL_X_SIZE * SLM_CELL_Y_SIZE], p1_w[SLM_CELL_X_SIZE * SLM_CELL_Y_SIZE];
    __local float SLM_delta_coef_table[64];

    int out_x_start, out_y_start;
    int x_start = get_group_id (0) * WORKGROUP_CELL_WIDTH;
    int y_start = get_group_id (1) * WORKGROUP_CELL_HEIGHT;
    int i = mad24 (l_id_y, l_size_x, l_id_x);
    int j = i;

    i *= 4;
    if(i < SLM_CELL_X_SIZE * SLM_CELL_Y_SIZE)
    {
        grbg_slm_load (p1_x, p1_y, p1_z, p1_w, i,
                       input, input_height,
                       x_start - SLM_CELL_X_OFFSET, y_start - SLM_CELL_Y_OFFSET);
    }
    if(j < 64)
        SLM_delta_coef_table[j] = bnr_table[j];

    barrier(CLK_LOCAL_MEM_FENCE);

    i = mad24 (l_id_y, l_size_x, l_id_x);
    int workitem_x_size = (SLM_CELL_X_VALID_SIZE / DEMOSAIC_X_CELL_PER_WORKITEM);
    int input_x = (i % workitem_x_size) * DEMOSAIC_X_CELL_PER_WORKITEM;
    int input_y = i / workitem_x_size;

    shared_demosaic (
        p1_x, p1_y, p1_z, p1_w,
        input_x + SLM_CELL_X_OFFSET, input_y + SLM_CELL_Y_OFFSET,
        output, output_height,
        (input_x + x_start) * PIXEL_PER_CELL / 4, (input_y + y_start) * PIXEL_PER_CELL, has_denoise, SLM_delta_coef_table, ee_config);
}

