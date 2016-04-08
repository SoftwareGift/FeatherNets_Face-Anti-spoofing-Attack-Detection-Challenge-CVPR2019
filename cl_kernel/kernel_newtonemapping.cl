/*
 * function: kernel_newtonemapping
 *     implementation of tone mapping
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

#define WORK_ITEM_X_SIZE 8
#define WORK_ITEM_Y_SIZE 8
#define BLOCK_FACTOR 4

__kernel void kernel_newtonemapping (
    __read_only image2d_t input, __write_only image2d_t output,
    __global float *hist_leq, int image_width, int image_height)
{
    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);

    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int g_size_x = get_global_size (0);
    int g_size_y = get_global_size (1);

    int local_index = local_id_y * WORK_ITEM_X_SIZE + local_id_x;
    int row_per_block = image_height / BLOCK_FACTOR;
    int col_per_block = image_width / BLOCK_FACTOR;
    int row_block_id = g_id_y / row_per_block;
    int col_block_id = g_id_x / col_per_block;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 src_data_Gr = read_imagef (input, sampler, (int2)(g_id_x, g_id_y));
    float4 src_data_R = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height));
    float4 src_data_B = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height * 2));
    float4 src_data_Gb = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height * 3));

    float4 src_data_G = (src_data_Gr + src_data_Gb) / 2;

    float4 src_y_data = 0.0f;
    src_y_data = mad(src_data_R, 0.299f, src_y_data);
    src_y_data = mad(src_data_G, 0.587f, src_y_data);
    src_y_data = mad(src_data_B, 0.114f, src_y_data);

    float4 dst_y_data;
    float4 d, wd, haleq;
    float4 total_wd = 0.0f;
    float4 total_haleq = 0.0f;

    for(int i = 0; i < BLOCK_FACTOR; i++)
    {
        for(int j = 0; j < BLOCK_FACTOR; j++)
        {
            int center_x = col_per_block * j + col_per_block / 2;
            int center_y = row_per_block * i + row_per_block / 2;
            int start_index = (i * BLOCK_FACTOR + j) * 4096;

            d.x = (g_id_x * 4 - center_x) * (g_id_x * 4 - center_x) + (g_id_y - center_y) * (g_id_y - center_y);
            d.y = (g_id_x * 4 + 1 - center_x) * (g_id_x * 4 + 1 - center_x) + (g_id_y - center_y) * (g_id_y - center_y);
            d.z = (g_id_x * 4 + 2 - center_x) * (g_id_x * 4 + 2 - center_x) + (g_id_y - center_y) * (g_id_y - center_y);
            d.w = (g_id_x * 4 + 3 - center_x) * (g_id_x * 4 + 3 - center_x) + (g_id_y - center_y) * (g_id_y - center_y);

            d = sqrt(d);

            wd = 100.0f / (d + 100.0f);

            haleq.x = hist_leq[start_index + (int)(src_y_data.x * 65535.0f + 0.5f) / 16];
            haleq.y = hist_leq[start_index + (int)(src_y_data.y * 65535.0f + 0.5f) / 16];
            haleq.z = hist_leq[start_index + (int)(src_y_data.z * 65535.0f + 0.5f) / 16];
            haleq.w = hist_leq[start_index + (int)(src_y_data.w * 65535.0f + 0.5f) / 16];

            total_wd += wd;
            total_haleq += haleq * wd;
        }
    }

    dst_y_data = total_haleq / total_wd;

    float4 gain = (dst_y_data  + 0.0001f) / (src_y_data + 0.0001f);
    src_data_Gr = src_data_Gr * gain;
    src_data_R = src_data_R * gain;
    src_data_B = src_data_B * gain;
    src_data_Gb = src_data_Gb * gain;

    write_imagef(output, (int2)(g_id_x, g_id_y), src_data_Gr);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height), src_data_R);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height * 2), src_data_B);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height * 3), src_data_Gb);
}
