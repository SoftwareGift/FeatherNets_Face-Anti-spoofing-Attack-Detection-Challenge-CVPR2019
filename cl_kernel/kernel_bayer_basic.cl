/*
 * function: kernel_bayer_copy
 *     sample code of default kernel arguments
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

//#define ENABLE_IMAGE_2D_INPUT 0

/*
 * GROUP_PIXEL_X_SIZE = 2 * GROUP_CELL_X_SIZE
 * GROUP_PIXEL_Y_SIZE = 2 * GROUP_CELL_Y_SIZE
*/

#define GROUP_CELL_X_SIZE 64
#define GROUP_CELL_Y_SIZE 4

//float4; 16
#define SLM_X_SIZE (GROUP_CELL_X_SIZE / 4)
#define SLM_Y_SIZE GROUP_CELL_Y_SIZE

#define STATS_3A_CELL_X_SIZE 8
#define STATS_3A_CELL_Y_SIZE GROUP_CELL_Y_SIZE

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

inline int slm_pos (const int x, const int y)
{
    return mad24 (y, SLM_X_SIZE, x);
}

inline void gamma_correct(float8 *in_out, __global float *table)
{
    in_out->s0 = table[clamp(convert_int(in_out->s0 * 255.0f), 0, 255)];
    in_out->s1 = table[clamp(convert_int(in_out->s1 * 255.0f), 0, 255)];
    in_out->s2 = table[clamp(convert_int(in_out->s2 * 255.0f), 0, 255)];
    in_out->s3 = table[clamp(convert_int(in_out->s3 * 255.0f), 0, 255)];
    in_out->s4 = table[clamp(convert_int(in_out->s4 * 255.0f), 0, 255)];
    in_out->s5 = table[clamp(convert_int(in_out->s5 * 255.0f), 0, 255)];
    in_out->s6 = table[clamp(convert_int(in_out->s6 * 255.0f), 0, 255)];
    in_out->s7 = table[clamp(convert_int(in_out->s7 * 255.0f), 0, 255)];
}

inline float avg_float8 (float8 data)
{
    return (data.s0 + data.s1 + data.s2 + data.s3 + data.s4 + data.s5 + data.s6 + data.s7) * 0.125f;
}

inline void stats_3a_calculate (
    __local float4 * slm_gr,
    __local float4 * slm_r,
    __local float4 * slm_b,
    __local float4 * slm_gb,
    __global ushort8 * stats_output,
    CLWBConfig *wb_config)
{
    const int group_x_size = get_num_groups (0);
    const int group_id_x = get_group_id (0);
    const int group_id_y = get_group_id (1);

    const int l_id_x = get_local_id (0);
    const int l_id_y = get_local_id (1);
    const int l_size_x = get_local_size (0);
    const int stats_float4_x_count = STATS_3A_CELL_X_SIZE / 4;
    int count =  stats_float4_x_count * STATS_3A_CELL_Y_SIZE / 4;

    int index = mad24 (l_id_y, l_size_x, l_id_x);
    int index_x = index % SLM_X_SIZE;
    int index_y = index / SLM_X_SIZE;

    if (mad24 (index_y,  stats_float4_x_count, index_x % stats_float4_x_count) < count) {
        int pitch_count = count / stats_float4_x_count * SLM_X_SIZE;
        int index1 = index + pitch_count;
        int index2 = index1 + pitch_count;
        int index3 = index2 + pitch_count;
        slm_gr[index] = (slm_gr[index] + slm_gr[index1] + slm_gr[index2] + slm_gr[index3]) * 0.25f;
        slm_r[index] = (slm_r[index] + slm_r[index1] + slm_r[index2] + slm_r[index3]) * 0.25f;
        slm_b[index] = (slm_b[index] + slm_b[index1] + slm_b[index2] + slm_b[index3]) * 0.25f;
        slm_gb[index] = (slm_gb[index] + slm_gb[index1] + slm_gb[index2] + slm_gb[index3]) * 0.25f;
    }
    barrier (CLK_LOCAL_MEM_FENCE);

    if (index < SLM_X_SIZE / 2) {
        float result_gr, result_r, result_b, result_gb, avg_y;
        float8 tmp;
        tmp = ((__local float8*)slm_gr)[index];
        result_gr = avg_float8 (tmp);

        tmp = ((__local float8*)slm_r)[index];
        result_r = avg_float8 (tmp);

        tmp = ((__local float8*)slm_b)[index];
        result_b = avg_float8 (tmp);

        tmp = ((__local float8*)slm_gb)[index];
        result_gb = avg_float8 (tmp);

        int out_index = mad24 (mad24 (group_id_y, group_x_size, group_id_x),
                               (GROUP_CELL_X_SIZE / STATS_3A_CELL_X_SIZE) * (GROUP_CELL_Y_SIZE / STATS_3A_CELL_Y_SIZE),
                               index);

#if STATS_BITS==8
        avg_y = mad ((result_gr * wb_config->gr_gain + result_gb * wb_config->gb_gain), 74.843f,
                     mad (result_r * wb_config->r_gain, 76.245f, result_b * 29.070f));

        //ushort avg_y; avg_r; avg_gr; avg_gb; avg_b; valid_wb_count; f_value1; f_value2;
        stats_output[out_index] = (ushort8) (
                                      convert_ushort (convert_uchar_sat (avg_y)),
                                      convert_ushort (convert_uchar_sat (result_r * 255.0f)),
                                      convert_ushort (convert_uchar_sat (result_gr * 255.0f)),
                                      convert_ushort (convert_uchar_sat (result_gb * 255.0f)),
                                      convert_ushort (convert_uchar_sat (result_b * 255.0f)),
                                      STATS_3A_CELL_X_SIZE * STATS_3A_CELL_Y_SIZE,
                                      0,
                                      0);
#elif STATS_BITS==12
        avg_y = mad ((result_gr * wb_config->gr_gain + result_gb * wb_config->gb_gain), 1201.883f,
                     mad (result_r * wb_config->r_gain, 1224.405f, result_b * 466.830f));

        stats_output[out_index] = (ushort8) (
                                      convert_ushort (clamp (avg_y, 0.0f, 4095.0f)),
                                      convert_ushort (clamp (result_r * 4096.0f, 0.0f, 4095.0f)),
                                      convert_ushort (clamp (result_gr * 4096.0f, 0.0f, 4095.0f)),
                                      convert_ushort (clamp (result_gb * 4096.0f, 0.0f, 4095.0f)),
                                      convert_ushort (clamp (result_b * 4096.0f, 0.0f, 4095.0f)),
                                      STATS_3A_CELL_X_SIZE * STATS_3A_CELL_Y_SIZE,
                                      0,
                                      0);
#else
        printf ("kernel 3a-stats error, wrong bit depth:%d\n", STATS_BITS);
#endif
    }
}


__kernel void kernel_bayer_basic (
#if ENABLE_IMAGE_2D_INPUT
    __read_only image2d_t input,
#else
    __global const ushort8 *input,
#endif
    uint input_aligned_width,
    __write_only image2d_t output,
    uint out_height,
    CLBLCConfig blc_config,
    CLWBConfig wb_config,
    __global float *gamma_table,
    __global ushort8 *stats_output
)
{
    int g_x = get_global_id (0);
    int g_y = get_global_id (1);

    const int l_x = get_local_id (0);
    const int l_y = get_local_id (1);
    const int l_x_size = get_local_size (0);
    const int l_y_size = get_local_size (1);
    const int group_id_x = get_group_id (0);
    const int group_id_y = get_group_id (1);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    int index = mad24 (l_y, l_x_size, l_x);
    int x_cell_start = (GROUP_CELL_X_SIZE / 4) * group_id_x;
    int y_cell_start = GROUP_CELL_Y_SIZE * group_id_y;
    int x, y;

    float blc_multiplier = (float)(1 << (16 - blc_config.color_bits));

    __local float4 slm_gr[SLM_X_SIZE * SLM_Y_SIZE], slm_r[SLM_X_SIZE * SLM_Y_SIZE], slm_b[SLM_X_SIZE * SLM_Y_SIZE], slm_gb[SLM_X_SIZE * SLM_Y_SIZE];

    for (; index < SLM_X_SIZE * SLM_Y_SIZE; index += l_x_size * l_y_size) {
        float8 line1;
        float8 line2;

        x = index % SLM_X_SIZE + x_cell_start;
        y = index / SLM_X_SIZE + y_cell_start;

#if ENABLE_IMAGE_2D_INPUT
        line1 = convert_float8 (as_ushort8 (read_imageui(input, sampler, (int2)(x, y * 2)))) / 65536.0f;
        line2 = convert_float8 (as_ushort8 (read_imageui(input, sampler, (int2)(x, y * 2 + 1)))) / 65536.0f;
#else
        line1 = convert_float8 (input [y * 2 * input_aligned_width + x]) / 65536.0f;
        line2 = convert_float8 (input [(y * 2 + 1) * input_aligned_width + x]) / 65536.0f;
#endif

        float4 gr = mad (line1.even, blc_multiplier, - blc_config.level_gr);
        float4 r = mad (line1.odd, blc_multiplier, - blc_config.level_r);
        float4 b = mad (line2.even, blc_multiplier, - blc_config.level_b);
        float4 gb = mad (line2.odd, blc_multiplier, - blc_config.level_gb);

        slm_gr[index] = gr;
        slm_r[index] =  r;
        slm_b[index] =  b;
        slm_gb[index] = gb;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float8 data_gr, data_r, data_b, data_gb;
    index = mad24 (l_y, l_x_size, l_x);
    x = mad24 (GROUP_CELL_X_SIZE / 8, group_id_x,  index % (SLM_X_SIZE / 2));
    y = mad24 (GROUP_CELL_Y_SIZE, group_id_y,  index / (SLM_X_SIZE / 2));

    data_gr = ((__local float8*)slm_gr)[index];
    data_gr = data_gr * wb_config.gr_gain;

    data_r = ((__local float8*)slm_r)[index];
    data_r  = data_r * wb_config.r_gain;

    data_b = ((__local float8*)slm_b)[index];
    data_b = data_b * wb_config.b_gain;

    data_gb = ((__local float8*)slm_gb)[index];
    data_gb = data_gb * wb_config.gb_gain;

#if ENABLE_GAMMA
    gamma_correct (&data_gr, gamma_table);
    gamma_correct (&data_r, gamma_table);
    gamma_correct (&data_b, gamma_table);
    gamma_correct (&data_gb, gamma_table);
#endif

#if 0
    if (x % 16 == 0 && y % 16 == 0) {
        uint8 value = convert_uint8(convert_uchar8_sat(data_gr * 255.0f));
        printf ("(x:%d, y:%d) (blc.bit:%d, level:%d) (wb.gr:%f)=> (%d, %d, %d, %d, %d, %d, %d, %d)\n",
                x * 8, y,
                blc_config.color_bits, convert_uint(blc_config.level_gr * 255.0f),
                wb_config.gr_gain,
                value.s0, value.s1, value.s2, value.s3, value.s4, value.s5, value.s6, value.s7);
    }
#endif

    write_imageui (output, (int2)(x, y), as_uint4 (convert_ushort8 (data_gr * 65536.0f)));
    write_imageui (output, (int2)(x, y + out_height), as_uint4 (convert_ushort8 (data_r * 65536.0f)));
    write_imageui (output, (int2)(x, y + out_height * 2), as_uint4 (convert_ushort8 (data_b * 65536.0f)));
    write_imageui (output, (int2)(x, y + out_height * 3), as_uint4 (convert_ushort8 (data_gb * 65536.0f)));

    stats_3a_calculate (slm_gr, slm_r, slm_b, slm_gb, stats_output, &wb_config);
}

