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


#define WORKGROUP_PIXEL_WIDTH 16
#define WORKGROUP_PIXEL_HEIGHT 16

#define DEMOSAIC_X_CELL_PER_WORKITEM 2

#define PIXEL_PER_CELL 2

#define SLM_CELL_X_OFFSET 2
#define SLM_CELL_Y_OFFSET 1

// 8x8
#define SLM_CELL_X_VALID_SIZE (WORKGROUP_PIXEL_WIDTH/PIXEL_PER_CELL)
#define SLM_CELL_Y_VALID_SIZE (WORKGROUP_PIXEL_HEIGHT/PIXEL_PER_CELL)

// 10x10
#define SLM_CELL_X_SIZE (SLM_CELL_X_VALID_SIZE + SLM_CELL_X_OFFSET * 2)
#define SLM_CELL_Y_SIZE (SLM_CELL_Y_VALID_SIZE + SLM_CELL_Y_OFFSET * 2)

#define SLM_PIXEL_X_OFFSET (SLM_CELL_X_OFFSET * PIXEL_PER_CELL)
#define SLM_PIXEL_Y_OFFSET (SLM_CELL_Y_OFFSET * PIXEL_PER_CELL)

#define STATS_3A_GRID_SIZE (16/PIXEL_PER_CELL)

typedef struct  {
    float  level_gr;  /* Black level for GR pixels */
    float  level_r;   /* Black level for R pixels */
    float  level_b;   /* Black level for B pixels */
    float  level_gb;  /* Black level for GB pixels */
    uint   color_bits;
} CLBLCConfig;


typedef struct
{
    float r_gain;
    float gr_gain;
    float gb_gain;
    float b_gain;
} CLWBConfig;

typedef struct
{
    unsigned int avg_y;

    unsigned int avg_r;
    unsigned int avg_gr;
    unsigned int avg_gb;
    unsigned int avg_b;
    unsigned int valid_wb_count;

    unsigned int f_value1;
    unsigned int f_value2;
} XCamGridStat;

/* BA10=> GRBG  */
inline void blc (float4 *in_out, CLBLCConfig *blc_config, float blc_multiplier)
{
    in_out->x = mad (in_out->x, blc_multiplier, - blc_config->level_gr);
    in_out->y = mad (in_out->y, blc_multiplier, - blc_config->level_r);
    in_out->z = mad (in_out->z, blc_multiplier, - blc_config->level_b);
    in_out->w = mad (in_out->w, blc_multiplier, - blc_config->level_gb);
}

inline void wb (float4 *in_out, CLWBConfig *wbconfig)
{
    in_out->x *= wbconfig->gr_gain;
    in_out->y *= wbconfig->r_gain;
    in_out->z *= wbconfig->b_gain;
    in_out->w *= wbconfig->gb_gain;
}

inline void gamma_correct(float4 *in_out, __global float *table)
{
    in_out->x = table[clamp(convert_int(in_out->x * 255.0f), 0, 255)];
    in_out->y = table[clamp(convert_int(in_out->y * 255.0f), 0, 255)];
    in_out->z = table[clamp(convert_int(in_out->z * 255.0f), 0, 255)];
    in_out->w = table[clamp(convert_int(in_out->w * 255.0f), 0, 255)];
}

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
inline void simple_calculate (
    __local float *px, __local float *py, __local float *pz, __local float *pw,
    int index, __read_only image2d_t input, int x_start, int y_start,
    __local float4 *stats_cache,
    CLBLCConfig *blc_config,
    float blc_multiplier,
    CLWBConfig *wb_config,
    uint enable_gamma,
    __global float *gamma_table)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float4 data1, data2, line1, line2;
    int x0 = (get_shared_pos_x (index) * PIXEL_PER_CELL + x_start) / 4;
    int y0 = get_shared_pos_y (index) * PIXEL_PER_CELL + y_start;

    line1 = read_imagef (input, sampler, (int2)(x0, y0));
    line2 = read_imagef (input, sampler, (int2)(x0, y0 + 1));

    data1 = (float4)(line1.s01, line2.s01);
    data2 = (float4)(line1.s23, line2.s23);

    blc (&data1, blc_config, blc_multiplier);
    blc (&data2, blc_config, blc_multiplier);

    /* write back for 3a stats calculation R, G, B, Y */
    (*(__local float8 *)(stats_cache + index)) = (float8)(data1, data2);

    wb (&data1, wb_config);
    wb (&data2, wb_config);
    if (enable_gamma) {
        gamma_correct (&data1, gamma_table);
        gamma_correct (&data2, gamma_table);
    }

    (*(__local float2 *)(px + index)) = (float2)(data1.x, data2.x);
    (*(__local float2 *)(py + index)) = (float2)(data1.y, data2.y);
    (*(__local float2 *)(pz + index)) = (float2)(data1.z, data2.z);
    (*(__local float2 *)(pw + index)) = (float2)(data1.w, data2.w);
}

#define MAX_DELTA_COFF 5.0f
#define MIN_DELTA_COFF 1.0f
#define DEFAULT_DELTA_COFF 4.0f

inline float2 delta_coff (float2 delta)
{
    float2 coff = mad (fabs(delta), - 20.0f, MAX_DELTA_COFF);
    return fmax (1.0f, coff);
}

inline float2 dot_denoise (float2 value, float2 in1, float2 in2, float2 in3, float2 in4)
{
    float2 coff0, coff1, coff2, coff3, coff4, coff5;
    coff0 = DEFAULT_DELTA_COFF;
    coff1 = delta_coff (in1 - value);
    coff2 = delta_coff (in2 - value);
    coff3 = delta_coff (in3 - value);
    coff4 = delta_coff (in4 - value);
    //(in1 * coff1 + in2 * coff2 + in3 * coff3 + in4 * coff4 + value * coff0)
    float2 sum1 = (mad (in1, coff1,
                        mad (in2, coff2,
                             mad (in3, coff3,
                                  mad (in4, coff4, value * coff0)))));
    return  sum1 / (coff0 + coff1 + coff2 + coff3 + coff4);
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
    __write_only image2d_t out, uint out_height, int out_x, int out_y)
{
    float4 out_data[2];
    float2 value;
    int index;

    ////////////////////////////////R//////////////////////////////////////////
    {
        float4 R_y[3];
        index = shared_pos (in_x - 1, in_y - 1);
        R_y[0] = *(__local float4*)(y_data_in + index);
        index = shared_pos (in_x - 1, in_y);
        R_y[1] = *(__local float4*)(y_data_in + index);
        index = shared_pos (in_x - 1, in_y + 1);
        R_y[2] = *(__local float4*)(y_data_in + index);

        value = (R_y[1].s01 + R_y[1].s12) * 0.5f;
        out_data[0].s02 = dot_denoise (value, R_y[0].s01, R_y[0].s12, R_y[2].s01, R_y[2].s12);

        value = R_y[1].s12;
        out_data[0].s13 = dot_denoise (value, R_y[0].s12, R_y[1].s01, R_y[1].s23, R_y[2].s12);

        value = (R_y[1].s01 + R_y[1].s12 +
                 R_y[2].s01 + R_y[2].s12) * 0.25f;
        out_data[1].s02 = dot_denoise (value, R_y[1].s01, R_y[1].s12, R_y[2].s01, R_y[2].s12);

        value = (R_y[1].s12 + R_y[2].s12) * 0.5f;
        out_data[1].s13 = dot_denoise (value, R_y[1].s01, R_y[1].s23, R_y[2].s01, R_y[2].s23);

        write_imagef (out, (int2)(out_x, out_y), out_data[0]);
        write_imagef (out, (int2)(out_x, out_y + 1), out_data[1]);

    }

    ////////////////////////////////B//////////////////////////////////////////
    {
        float4 B_z[3];
        index = shared_pos (in_x - 1, in_y - 1);
        B_z[0] = *(__local float4*)(z_data_in + index);
        index = shared_pos (in_x - 1, in_y);
        B_z[1] = *(__local float4*)(z_data_in + index);
        index = shared_pos (in_x - 1, in_y + 1);
        B_z[2] = *(__local float4*)(z_data_in + index);

        value = (B_z[0].s12 + B_z[1].s12) * 0.5f;
        out_data[0].s02 = dot_denoise (value, B_z[0].s01, B_z[0].s23, B_z[1].s01, B_z[1].s23);

        value = (B_z[0].s12 + B_z[0].s23 +
                 B_z[1].s12 + B_z[1].s23) * 0.25f;
        out_data[0].s13 = dot_denoise (value, B_z[0].s12, B_z[0].s23, B_z[1].s12, B_z[1].s23);

        value = B_z[1].s12;
        out_data[1].s02 = dot_denoise (value, B_z[0].s12, B_z[1].s01, B_z[1].s23, B_z[2].s12);

        value = (B_z[1].s12 + B_z[1].s23) * 0.5f;
        out_data[1].s13 = dot_denoise (value, B_z[0].s12, B_z[0].s23, B_z[2].s12, B_z[2].s23);

        write_imagef (out, (int2)(out_x, out_y + out_height * 2), out_data[0]);
        write_imagef (out, (int2)(out_x, out_y + 1 + out_height * 2), out_data[1]);
    }

    ///////////////////////////////////////G///////////////////////////////////
    {
        float3 Gr_x[2], Gb_w[2];
        index = shared_pos (in_x - 1, in_y - 1);
        Gb_w[0] = *(__local float3*)(w_data_in + index);
        index = shared_pos (in_x - 1, in_y);
        Gb_w[1] = *(__local float3*)(w_data_in + index);

        index = shared_pos (in_x, in_y);
        Gr_x[0] = *(__local float3*)(x_data_in + index);
        index = shared_pos (in_x, in_y + 1);
        Gr_x[1] = *(__local float3*)(x_data_in + index);

        value = mad (Gr_x[0].s01, 4.0f,  (Gb_w[0].s01 +
                                          Gb_w[0].s12 + Gb_w[1].s01 + Gb_w[1].s12)) * 0.125f;
        out_data[0].s02 = dot_denoise (value, Gb_w[0].s01, Gb_w[0].s12, Gb_w[1].s01, Gb_w[1].s12);
        value = (Gr_x[0].s01 + Gr_x[0].s12 +
                 Gb_w[0].s12 + Gb_w[1].s12) * 0.25f;
        out_data[0].s13 = dot_denoise(value, Gr_x[0].s01, Gr_x[0].s12, Gb_w[0].s12, Gb_w[1].s12);

        value = (Gr_x[0].s01 + Gr_x[1].s01 +
                 Gb_w[1].s01 + Gb_w[1].s12) * 0.25f;
        out_data[1].s02 = dot_denoise (value, Gr_x[0].s01, Gr_x[1].s01, Gb_w[1].s01, Gb_w[1].s12);

        value = mad (Gb_w[1].s12, 4.0f, (Gr_x[0].s01 +
                                         Gr_x[0].s12 + Gr_x[1].s01 + Gr_x[1].s12)) * 0.125f;
        out_data[1].s13 = dot_denoise (value, Gr_x[0].s01, Gr_x[0].s12, Gr_x[1].s01, Gr_x[1].s12);

        write_imagef (out, (int2)(out_x, out_y + out_height), out_data[0]);
        write_imagef (out, (int2)(out_x, out_y + 1 + out_height), out_data[1]);
    }
}

void shared_demosaic (
    __local float *x_data_in, __local float *y_data_in, __local float *z_data_in, __local float *w_data_in,
    int in_x, int in_y,
    __write_only image2d_t out, uint output_height, int out_x, int out_y,
    uint has_denoise)
{
    if (has_denoise) {
        demosaic_denoise_2_cell (
            x_data_in, y_data_in, z_data_in, w_data_in, in_x, in_y,
            out, output_height, out_x, out_y);
    } else {
        demosaic_2_cell (
            x_data_in, y_data_in, z_data_in, w_data_in, in_x, in_y,
            out, output_height, out_x, out_y);
    }
}

inline void stats_3a_calculate (
    __local float4 * input,
    __global XCamGridStat * stats_output,
    CLWBConfig *wb_config)
{
    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);
    int g_size_x = get_global_size (0);
    int g_size_y = get_global_size (1);

    int l_id_x = get_local_id(0);
    int l_id_y = get_local_id(1);
    int count = STATS_3A_GRID_SIZE * STATS_3A_GRID_SIZE / 4;

    for (; count > 0; count /= 4) {
        if ((l_id_x % STATS_3A_GRID_SIZE) + (l_id_y % STATS_3A_GRID_SIZE)* STATS_3A_GRID_SIZE < count) {
            int index1 = shared_pos (l_id_x + SLM_CELL_X_OFFSET, l_id_y + SLM_CELL_Y_OFFSET);
            int index2 = shared_pos (SLM_CELL_X_OFFSET + ((l_id_x + count) % STATS_3A_GRID_SIZE),
                                     SLM_CELL_Y_OFFSET + l_id_y + (l_id_x + count) / STATS_3A_GRID_SIZE);
            int index3 = shared_pos (SLM_CELL_X_OFFSET + ((l_id_x + count * 2) % STATS_3A_GRID_SIZE),
                                     SLM_CELL_Y_OFFSET + l_id_y + (l_id_x + count * 2) / STATS_3A_GRID_SIZE);
            int index4 = shared_pos (SLM_CELL_X_OFFSET + ((l_id_x + count * 3) % STATS_3A_GRID_SIZE),
                                     SLM_CELL_Y_OFFSET + l_id_y + (l_id_x + count * 3) / STATS_3A_GRID_SIZE);
            input[index1] = (input[index1] + input[index2] + input[index3] + input[index4]) / 4.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (l_id_x % STATS_3A_GRID_SIZE == 0 && l_id_y % STATS_3A_GRID_SIZE == 0) {
        float4 tmp_data;
        int out_index = mad24(g_id_y / STATS_3A_GRID_SIZE,  g_size_x / STATS_3A_GRID_SIZE, g_id_x / STATS_3A_GRID_SIZE);
        tmp_data = input[shared_pos (l_id_x + SLM_CELL_X_OFFSET, l_id_y + SLM_CELL_Y_OFFSET)];
        stats_output[out_index].avg_gr = convert_uchar_sat(tmp_data.x * 255.0f);
        stats_output[out_index].avg_r = convert_uchar_sat(tmp_data.y * 255.0f);
        stats_output[out_index].avg_b = convert_uchar_sat(tmp_data.z * 255.0f);
        stats_output[out_index].avg_gb = convert_uchar_sat(tmp_data.w * 255.0f);
        stats_output[out_index].valid_wb_count = STATS_3A_GRID_SIZE * STATS_3A_GRID_SIZE;
        stats_output[out_index].avg_y =
            convert_uchar_sat(
                mad ((tmp_data.x * wb_config->gr_gain + tmp_data.w * wb_config->gb_gain), 74.843f,
                     mad (tmp_data.y * wb_config->r_gain, 76.245f, (tmp_data.z * wb_config->b_gain * 29.070f))));
        stats_output[out_index].f_value1 = 0;
        stats_output[out_index].f_value2 = 0;
    }
}

__kernel void kernel_bayer_pipe (__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 uint output_height,
                                 CLBLCConfig blc_config,
                                 CLWBConfig wb_config,
                                 uint has_denoise,
                                 uint enable_gamma,
                                 __global float * gamma_table,
                                 __global XCamGridStat * stats_output)
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
    __local float4 p2[SLM_CELL_X_SIZE * SLM_CELL_Y_SIZE];
    __local float4 *stats_cache = p2;

    float blc_multiplier = (float)(1 << (16 - blc_config.color_bits));

    int out_x_start, out_y_start;
    int x_start = get_group_id (0) * WORKGROUP_PIXEL_WIDTH;
    int y_start = get_group_id (1) * WORKGROUP_PIXEL_HEIGHT;
    int i = mad24 (l_id_y, l_size_x, l_id_x);

    i *= 2;
    for (; i < SLM_CELL_X_SIZE * SLM_CELL_Y_SIZE; i += (l_size_x * l_size_y) * 2) {
        simple_calculate (p1_x, p1_y, p1_z, p1_w, i,
                          input,
                          x_start - SLM_PIXEL_X_OFFSET, y_start - SLM_PIXEL_Y_OFFSET,
                          stats_cache,
                          &blc_config,
                          blc_multiplier,
                          &wb_config,
                          enable_gamma,
                          gamma_table);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    stats_3a_calculate (stats_cache, stats_output, &wb_config);

    i = mad24 (l_id_y, l_size_x, l_id_x);
    int workitem_x_size = (SLM_CELL_X_VALID_SIZE / DEMOSAIC_X_CELL_PER_WORKITEM);
    int input_x = (i % workitem_x_size) * DEMOSAIC_X_CELL_PER_WORKITEM;
    int input_y = i / workitem_x_size;

    shared_demosaic (
        p1_x, p1_y, p1_z, p1_w,
        input_x + SLM_CELL_X_OFFSET, input_y + SLM_CELL_Y_OFFSET,
        output, output_height,
        mad24 (input_x, PIXEL_PER_CELL, x_start) / 4, mad24 (input_y, PIXEL_PER_CELL, y_start), has_denoise);
}

