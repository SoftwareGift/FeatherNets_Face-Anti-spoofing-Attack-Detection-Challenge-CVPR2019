/*
 * function: kernel_tonemapping
 *     implementation of tone mapping
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

#define WORK_ITEM_X_SIZE 8
#define WORK_ITEM_Y_SIZE 8

#define SHARED_PIXEL_X_SIZE 10
#define SHARED_PIXEL_Y_SIZE 10

__kernel void kernel_tonemapping (__read_only image2d_t input, __write_only image2d_t output, float y_max, float y_target, int image_height)
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

    __local float4 local_src_data[SHARED_PIXEL_X_SIZE * SHARED_PIXEL_Y_SIZE];

    float4 src_data_Gr = read_imagef (input, sampler, (int2)(g_id_x, g_id_y));
    float4 src_data_R = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height));
    float4 src_data_B = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height * 2));
    float4 src_data_Gb = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height * 3));

    float4 src_data_G = (src_data_Gr + src_data_Gb) / 2;

    float4 src_y_data = 0.0f;
    src_y_data = mad(src_data_R, 255.f * 0.299f, src_y_data);
    src_y_data = mad(src_data_G, 255.f * 0.587f, src_y_data);
    src_y_data = mad(src_data_B, 255.f * 0.114f, src_y_data);

    local_src_data[(local_id_y + 1) * SHARED_PIXEL_X_SIZE + local_id_x + 1] = src_y_data;

    if(local_index < SHARED_PIXEL_X_SIZE * SHARED_PIXEL_Y_SIZE - WORK_ITEM_X_SIZE * WORK_ITEM_Y_SIZE)
    {
        int target_index = local_index <= SHARED_PIXEL_X_SIZE ? local_index : (local_index <= (SHARED_PIXEL_X_SIZE * SHARED_PIXEL_Y_SIZE - WORK_ITEM_X_SIZE * WORK_ITEM_Y_SIZE - SHARED_PIXEL_X_SIZE) ? (local_index + WORK_ITEM_X_SIZE + (local_index - (SHARED_PIXEL_X_SIZE + 1)) / 2 * WORK_ITEM_X_SIZE) : (local_index + WORK_ITEM_X_SIZE * WORK_ITEM_Y_SIZE));
        int start_x = mad24(group_id_x, WORK_ITEM_X_SIZE, -1);
        int start_y = mad24(group_id_y, WORK_ITEM_Y_SIZE, -1);
        int offset_x = target_index % SHARED_PIXEL_X_SIZE;
        int offset_y = target_index / SHARED_PIXEL_X_SIZE;

        float4 data_Gr = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y));
        float4 data_R = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y + image_height));
        float4 data_B = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y + image_height * 2));
        float4 data_Gb = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y + image_height * 3));

        float4 data_G = (data_Gr + data_Gb) / 2;

        float4 y_data = 0.0f;
        y_data = mad(data_R, 255.f * 0.299f, y_data);
        y_data = mad(data_G, 255.f * 0.587f, y_data);
        y_data = mad(data_B, 255.f * 0.114f, y_data);
        local_src_data[target_index] = y_data;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float gaussian_table[9] = {0.075f, 0.124f, 0.075f,
                               0.124f, 0.204f, 0.124f,
                               0.075f, 0.124f, 0.075f
                              };
    float4 src_ym_data = 0.0f;

    float16 integrate_data = *((__local float16 *)(local_src_data + local_id_y * SHARED_PIXEL_X_SIZE + local_id_x));

    src_ym_data = mad(integrate_data.s3456, (float4)gaussian_table[0], src_ym_data);
    src_ym_data = mad(integrate_data.s4567, (float4)gaussian_table[1], src_ym_data);
    src_ym_data = mad(integrate_data.s5678, (float4)gaussian_table[2], src_ym_data);

    integrate_data = *((__local float16 *)(local_src_data + (local_id_y + 1) * SHARED_PIXEL_X_SIZE + local_id_x));

    src_ym_data = mad(integrate_data.s3456, (float4)gaussian_table[3], src_ym_data);
    src_ym_data = mad(src_y_data, (float4)gaussian_table[4], src_ym_data);
    src_ym_data = mad(integrate_data.s5678, (float4)gaussian_table[5], src_ym_data);

    integrate_data = *((__local float16 *)(local_src_data + (local_id_y + 2) * SHARED_PIXEL_X_SIZE + local_id_x));

    src_ym_data = mad(integrate_data.s3456, (float4)gaussian_table[6], src_ym_data);
    src_ym_data = mad(integrate_data.s4567, (float4)gaussian_table[7], src_ym_data);
    src_ym_data = mad(integrate_data.s5678, (float4)gaussian_table[8], src_ym_data);

    float4 gain = ((float4)(y_max + y_target) + src_ym_data) / (src_y_data + src_ym_data + (float4)y_target);
    src_data_Gr = src_data_Gr * gain;
    src_data_R = src_data_R * gain;
    src_data_B = src_data_B * gain;
    src_data_Gb = src_data_Gb * gain;

    write_imagef(output, (int2)(g_id_x, g_id_y), src_data_Gr);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height), src_data_R);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height * 2), src_data_B);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height * 3), src_data_Gb);
}
