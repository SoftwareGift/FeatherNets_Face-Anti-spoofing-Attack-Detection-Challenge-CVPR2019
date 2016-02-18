/*
 * function: kernel_newtonemapping
 *     implementation of tone mapping
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

#define WORK_ITEM_X_SIZE 8
#define WORK_ITEM_Y_SIZE 8

__kernel void kernel_newtonemapping (__read_only image2d_t input, __write_only image2d_t output, __global float *hist_leq, int image_height, int y_max, int y_min)
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

    float t = 0.01f;
    float4 log_y_data = (log(src_y_data * 65535.0f / y_max + t) - log((float4)(y_min / y_max + t))) / (log((float4)(1.0f + t)) - log((float4)(y_min / y_max + t)));
    float4 dst_y_data = log_y_data;
    for(int i = 0; i < 256; i++)
    {
        dst_y_data = log_y_data <= hist_leq[i] ? dst_y_data : i;
    }
    float4 gain = dst_y_data / (src_y_data * 255.0f);
    src_data_Gr = src_data_Gr * gain;
    src_data_R = src_data_R * gain;
    src_data_B = src_data_B * gain;
    src_data_Gb = src_data_Gb * gain;

    write_imagef(output, (int2)(g_id_x, g_id_y), src_data_Gr);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height), src_data_R);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height * 2), src_data_B);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height * 3), src_data_Gb);
}
