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

#define GRID_X_SIZE 2
#define GRID_Y_SIZE 2

#define SHARED_PIXEL_WIDTH 16
#define SHARED_PIXEL_HEIGHT 16
#define SHARED_PIXEL_X_OFFSET 2
#define SHARED_PIXEL_Y_OFFSET 2
#define SHARED_PIXEL_X_SIZE  (SHARED_PIXEL_WIDTH + SHARED_PIXEL_X_OFFSET * 2)
#define SHARED_PIXEL_Y_SIZE  (SHARED_PIXEL_HEIGHT + SHARED_PIXEL_Y_OFFSET * 2)

#define SHARED_GRID_WIDTH (SHARED_PIXEL_WIDTH/GRID_X_SIZE)
#define SHARED_GRID_HEIGHT (SHARED_PIXEL_HEIGHT/GRID_Y_SIZE)
#define SHARED_GRID_X_OFFSET (SHARED_PIXEL_X_OFFSET/GRID_X_SIZE)
#define SHARED_GRID_Y_OFFSET (SHARED_PIXEL_Y_OFFSET/GRID_Y_SIZE)
#define SHARED_GRID_X_SIZE (SHARED_PIXEL_X_SIZE/GRID_X_SIZE)
#define SHARED_GRID_Y_SIZE (SHARED_PIXEL_Y_SIZE/GRID_Y_SIZE)

#define WORK_ITEM_X_SIZE GRID_X_SIZE
#define WORK_ITEM_Y_SIZE GRID_Y_SIZE

#define STATS_3A_GRID_SIZE (16/GRID_X_SIZE)

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
inline void blc (float4 *in_out, CLBLCConfig *blc_config)
{
    float multiplier = (float)(1 << (16 - blc_config->color_bits));
    in_out->x = in_out->x * multiplier - blc_config->level_gr;
    in_out->y = in_out->y * multiplier - blc_config->level_r;
    in_out->z = in_out->z * multiplier - blc_config->level_b;
    in_out->w = in_out->w * multiplier - blc_config->level_gb;
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
    return i % SHARED_GRID_X_SIZE;
}

inline int get_shared_pos_y (int i)
{
    return i / SHARED_GRID_X_SIZE;
}

inline int shared_pos (int x, int y)
{
    return mad24(y, SHARED_GRID_X_SIZE, x);
}

/* BA10=> GRBG  */
inline float4 simple_calculate (
    __local float4 *p, int index, __read_only image2d_t input, sampler_t sampler, int x_start, int y_start,
    __local float4 *stats_cache,
    CLBLCConfig *blc_config,
    CLWBConfig *wb_config,
    __global float *gamma_table)
{
    float4 data;
    int x0 = get_shared_pos_x (index) * WORK_ITEM_X_SIZE + x_start;
    int y0 = get_shared_pos_y (index) * WORK_ITEM_Y_SIZE + y_start;
    //Gr
    data.x = read_imagef (input, sampler, (int2)(x0, y0)).x;
    //R
    data.y = read_imagef (input, sampler, (int2)(x0 + 1, y0)).x;
    //B
    data.z = read_imagef (input, sampler, (int2)(x0, y0 + 1)).x;
    //Gb
    data.w = read_imagef (input, sampler, (int2)(x0 + 1, y0 + 1)).x;

    blc (&data, blc_config);

    /* write back for 3a stats calculation R, G, B, Y */
    stats_cache[index] = data;

    wb (&data, wb_config);
    gamma_correct (&data, gamma_table);

    p[index] = data;
}

inline void shared_demosaic (
    __local float4 *in, int in_x, int in_y,
    __write_only image2d_t out, int out_x, int out_y)
{
    float4 out_data;

    out_data.x = (in[shared_pos(in_x - 1, in_y)].y + in[shared_pos(in_x, in_y)].y) / 2.0f;
    out_data.y = (in[shared_pos(in_x, in_y)].x * 4.0f + in[shared_pos(in_x - 1, in_y - 1)].w +
                  in[shared_pos(in_x, in_y - 1)].w + in[shared_pos(in_x - 1, in_y)].w + in[shared_pos(in_x, in_y)].w) / 8.0f;
    out_data.z = (in[shared_pos(in_x, in_y - 1)].z + in[shared_pos(in_x, in_y)].z) / 2.0f;
    out_data.w = 0.0f;
    write_imagef(out, (int2)(out_x, out_y), out_data);


    out_data.x = in[shared_pos(in_x, in_y)].y;
    out_data.y = (in[shared_pos(in_x, in_y)].x + in[shared_pos(in_x, in_y)].w +
                  in[shared_pos(in_x + 1, in_y)].x + in[shared_pos(in_x, in_y - 1)].w) / 4.0f;
    out_data.z = (in[shared_pos(in_x, in_y - 1)].z + in[shared_pos(in_x + 1, in_y - 1)].z +
                  in[shared_pos(in_x, in_y)].z + in[shared_pos(in_x + 1, in_y)].z) / 4.0f;
    out_data.w = 0.0f;
    write_imagef(out, (int2)(out_x + 1, out_y), out_data);

    out_data.x = (in[shared_pos(in_x - 1, in_y)].y + in[shared_pos(in_x, in_y)].y +
                  in[shared_pos(in_x - 1, in_y + 1)].y + in[shared_pos(in_x, in_y + 1)].y) / 4.0f;
    out_data.y = (in[shared_pos(in_x, in_y)].x + in[shared_pos(in_x, in_y)].w +
                  in[shared_pos(in_x - 1, in_y)].w + in[shared_pos(in_x, in_y + 1)].x) / 4.0f;
    out_data.z = in[shared_pos(in_x, in_y)].z;
    out_data.w = 0.0f;
    write_imagef(out, (int2)(out_x, out_y + 1), out_data);

    out_data.x = (in[shared_pos(in_x, in_y)].y + in[shared_pos(in_x, in_y + 1)].y) / 2.0f;
    out_data.y = (in[shared_pos(in_x, in_y)].w * 4.0f + in[shared_pos(in_x, in_y)].x +
                  in[shared_pos(in_x + 1, in_y)].x + in[shared_pos(in_x, in_y + 1)].x + in[shared_pos(in_x + 1, in_y + 1)].x) / 8.0f;
    out_data.z = (in[shared_pos(in_x, in_y)].z + in[shared_pos(in_x + 1, in_y)].x) / 2.0f;
    out_data.w = 0.0f;
    write_imagef(out, (int2)(out_x + 1, out_y + 1), out_data);

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
    int count = STATS_3A_GRID_SIZE * STATS_3A_GRID_SIZE / 2;

    for (; count > 0; count /= 2) {
        if ((l_id_x % STATS_3A_GRID_SIZE) + (l_id_y % STATS_3A_GRID_SIZE)* STATS_3A_GRID_SIZE < count) {
            int index1 = shared_pos (l_id_x + SHARED_GRID_X_OFFSET, l_id_y + SHARED_GRID_Y_OFFSET);
            int index2 = shared_pos (l_id_x + SHARED_GRID_X_OFFSET + count % STATS_3A_GRID_SIZE,
                                     l_id_y + SHARED_GRID_Y_OFFSET + count / STATS_3A_GRID_SIZE);
            input[index1].x = (input[index1].x + input[index2].x) / 2.0f;
            input[index1].y = (input[index1].y + input[index2].y) / 2.0f;
            input[index1].z = (input[index1].z + input[index2].z) / 2.0f;
            input[index1].w = (input[index1].w + input[index2].w) / 2.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (l_id_x % STATS_3A_GRID_SIZE == 0 && l_id_y % STATS_3A_GRID_SIZE == 0) {
        float4 tmp_data;
        int out_index = mad24(g_id_y / STATS_3A_GRID_SIZE,  g_size_x / STATS_3A_GRID_SIZE, g_id_x / STATS_3A_GRID_SIZE);
        tmp_data = input[shared_pos (l_id_x + SHARED_GRID_X_OFFSET, l_id_y + SHARED_GRID_Y_OFFSET)];
        stats_output[out_index].avg_gr = convert_uchar_sat(tmp_data.x * 255.0f);
        stats_output[out_index].avg_r = convert_uchar_sat(tmp_data.y * 255.0f);
        stats_output[out_index].avg_b = convert_uchar_sat(tmp_data.z * 255.0f);
        stats_output[out_index].avg_gb = convert_uchar_sat(tmp_data.w * 255.0f);
        stats_output[out_index].valid_wb_count = STATS_3A_GRID_SIZE * STATS_3A_GRID_SIZE;
        stats_output[out_index].avg_y =
            convert_uchar_sat(((tmp_data.x * wb_config->gr_gain + tmp_data.w * wb_config->gb_gain) * 0.2935f +
                               tmp_data.y * wb_config->r_gain * 0.299f + tmp_data.z * wb_config->b_gain * 0.114f) * 255.0f);
        stats_output[out_index].f_value1 = 0;
        stats_output[out_index].f_value2 = 0;
    }
}

__kernel void kernel_bayer_pipe (__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 CLBLCConfig blc_config,
                                 CLWBConfig wb_config,
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

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    __local float4 p1[SHARED_GRID_X_SIZE * SHARED_GRID_Y_SIZE];
    __local float4 p2[SHARED_GRID_X_SIZE * SHARED_GRID_Y_SIZE];
    __local float4 *stats_cache = p2;

    int out_x_start, out_y_start;
    int x_start = (g_id_x - l_id_x) * WORK_ITEM_X_SIZE - SHARED_PIXEL_X_OFFSET;
    int y_start = (g_id_y - l_id_y) * WORK_ITEM_Y_SIZE - SHARED_PIXEL_Y_OFFSET;
    int i = l_id_x + l_id_y * l_size_x;

    for (; i < SHARED_GRID_X_SIZE * SHARED_GRID_Y_SIZE; i += l_size_x * l_size_y) {
        simple_calculate (p1, i,
                          input, sampler, x_start, y_start,
                          stats_cache,
                          &blc_config,
                          &wb_config,
                          gamma_table);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    stats_3a_calculate (stats_cache, stats_output, &wb_config);

    shared_demosaic (
        p1, l_id_x + SHARED_GRID_X_OFFSET, l_id_y + SHARED_GRID_Y_OFFSET,
        output, g_id_x * WORK_ITEM_X_SIZE, g_id_y * WORK_ITEM_Y_SIZE);
}

