/*
 * function: kernel_tonemapping
 *     implementation of tone mapping
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

#define STATS_BLOCK_FACTOR 7

typedef struct
{
    float r_gain;
    float gr_gain;
    float gb_gain;
    float b_gain;
} CLWBConfig;

typedef struct _XCamGridStat {
    unsigned int avg_y;

    unsigned int avg_r;
    unsigned int avg_gr;
    unsigned int avg_gb;
    unsigned int avg_b;
    unsigned int valid_wb_count;

    unsigned int f_value1;
    unsigned int f_value2;
} XCamGridStat;

__kernel void kernel_tonemapping (
    __read_only image2d_t input, __write_only image2d_t output,
    __global XCamGridStat *stats_input,
    CLWBConfig wb_config, float tm_gamma)
{
    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);

    int g_size_x = get_global_size (0);
    int g_size_y = get_global_size (1);

    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

    int group_size_x = get_num_groups(0);
    int group_size_y = get_num_groups(1);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 data = read_imagef (input, sampler, (int2)(g_id_x, g_id_y));

    float4 local_avg = 0.0f;

    int index;
#if 1
    int x_start = g_id_x - 8;
    int x_end = g_id_x + 8;
    int y_start = g_id_y - 8;
    int y_end = g_id_y + 8;
    int x, y, index_x, index_y;

    for(y = y_start; y <= y_end; y++)
    {
        for(x = x_start; x <= x_end; x++)
        {
            index_y = y > 0 ? y : 0;
            index_y = index_y < g_size_y ? index_y : (g_size_y - 1);
            index_x = x > 0 ? x : 0;
            index_x = index_x < g_size_x ? index_x : (g_size_x - 1);

            local_avg = local_avg + read_imagef (input, sampler, (int2)(index_x, index_y));
        }
    }

    local_avg = local_avg / (17 * 17);
#else
    int x_start = group_id_x - STATS_BLOCK_FACTOR / 2;
    int x_end = group_id_x + STATS_BLOCK_FACTOR / 2;
    int y_start = group_id_y - STATS_BLOCK_FACTOR / 2;
    int y_end = group_id_y + STATS_BLOCK_FACTOR / 2;
    int x, y, index_x, index_y;
    for(y = y_start; y <= y_end; y++)
    {
        for(x = x_start; x <= x_end; x++)
        {
            index_y = y > 0 ? y : 0;
            index_y = index_y < group_size_y ? index_y : (group_size_y - 1);
            index_x = x > 0 ? x : 0;
            index_x = index_x < group_size_x ? index_x : (group_size_x - 1);
            index = index_y * group_size_x + index_x;
            local_avg.x = local_avg.x + stats_input[index].avg_r * wb_config.r_gain;
            local_avg.y = local_avg.y +
                          (stats_input[index].avg_gr * wb_config.gr_gain +
                           stats_input[index].avg_gb * wb_config.gb_gain) * 0.5f;
            local_avg.z = local_avg.z + stats_input[index].avg_b * wb_config.b_gain;
            local_avg.w = 0.0f;
        }
    }

    local_avg = local_avg / (STATS_BLOCK_FACTOR * STATS_BLOCK_FACTOR);
    local_avg = local_avg / 255.0f;
#endif

    float4 gain = 2.0f;
    float4 delta = data - local_avg;

    local_avg = pow(local_avg, 1 / tm_gamma);
    data = local_avg + gain * delta;

    write_imagef(output, (int2)(g_id_x, g_id_y), data);
}
