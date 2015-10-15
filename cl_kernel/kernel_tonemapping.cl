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

__kernel void kernel_tonemapping (__read_only image2d_t input, __write_only image2d_t output, float y_max, float y_target)
{
    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);

    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int g_size_x = get_global_size (0);
    int g_size_y = get_global_size (1);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#if 1 // use SLM
    __local float local_src_data[SHARED_PIXEL_X_SIZE * SHARED_PIXEL_Y_SIZE];

    int start_x = WORK_ITEM_X_SIZE * group_id_x - 1;
    int start_y = WORK_ITEM_Y_SIZE * group_id_y - 1;
    int local_index = local_id_y * WORK_ITEM_X_SIZE + local_id_x;
    int offset_x = local_index % SHARED_PIXEL_X_SIZE;
    int offset_y = local_index / SHARED_PIXEL_X_SIZE;

    float4 data = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y));
    local_src_data[offset_y * SHARED_PIXEL_X_SIZE + offset_x] = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;

    if(local_index < SHARED_PIXEL_X_SIZE * SHARED_PIXEL_Y_SIZE - WORK_ITEM_X_SIZE * WORK_ITEM_Y_SIZE)
    {
        offset_x = (local_index + WORK_ITEM_X_SIZE * WORK_ITEM_Y_SIZE) % SHARED_PIXEL_X_SIZE;
        offset_y = (local_index + WORK_ITEM_X_SIZE * WORK_ITEM_Y_SIZE) / SHARED_PIXEL_X_SIZE;

        data = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y));
        local_src_data[offset_y * SHARED_PIXEL_X_SIZE + offset_x] = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    }

    float4 src_data = read_imagef (input, sampler, (int2)(g_id_x, g_id_y));
    float src_y_data = src_data.x * 255 * 0.299 + src_data.y * 255 * 0.587 + src_data.z * 255 * 0.114;

    barrier(CLK_LOCAL_MEM_FENCE);

    float gaussian_table[9] = {0.075f, 0.124f, 0.075f,
                               0.124f, 0.204f, 0.124f,
                               0.075f, 0.124f, 0.075f
                              };
    float src_ym_data = 0.0f;

    src_ym_data += local_src_data[local_id_y * SHARED_PIXEL_X_SIZE + local_id_x] * gaussian_table[0];
    src_ym_data += local_src_data[local_id_y * SHARED_PIXEL_X_SIZE + local_id_x + 1] * gaussian_table[1];
    src_ym_data += local_src_data[local_id_y * SHARED_PIXEL_X_SIZE + local_id_x + 2] * gaussian_table[2];
    src_ym_data += local_src_data[(local_id_y + 1) * SHARED_PIXEL_X_SIZE + local_id_x] * gaussian_table[3];
    src_ym_data += local_src_data[(local_id_y + 1) * SHARED_PIXEL_X_SIZE + local_id_x + 1] * gaussian_table[4];
    src_ym_data += local_src_data[(local_id_y + 1) * SHARED_PIXEL_X_SIZE + local_id_x + 2] * gaussian_table[5];
    src_ym_data += local_src_data[(local_id_y + 2) * SHARED_PIXEL_X_SIZE + local_id_x] * gaussian_table[6];
    src_ym_data += local_src_data[(local_id_y + 2) * SHARED_PIXEL_X_SIZE + local_id_x + 1] * gaussian_table[7];
    src_ym_data += local_src_data[(local_id_y + 2) * SHARED_PIXEL_X_SIZE + local_id_x + 2] * gaussian_table[8];

#else // use global memory
    float4 src_data = read_imagef (input, sampler, (int2)(g_id_x, g_id_y));
    float src_y_data = src_data.x * 255 * 0.299 + src_data.y * 255 * 0.587 + src_data.z * 255 * 0.114;

    float gaussian_table[9] = {0.075f, 0.124f, 0.075f,
                               0.124f, 0.204f, 0.124f,
                               0.075f, 0.124f, 0.075f
                              };
    float src_ym_data = 0.0f;

    float4 data = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y - 1));
    float y_data = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    src_ym_data += y_data * gaussian_table[0];

    data = read_imagef (input, sampler, (int2)(g_id_x, g_id_y - 1));
    y_data = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    src_ym_data += y_data * gaussian_table[1];

    data = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y - 1));
    y_data = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    src_ym_data += y_data * gaussian_table[2];

    data = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y));
    y_data = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    src_ym_data += y_data * gaussian_table[3];

    data = read_imagef (input, sampler, (int2)(g_id_x, g_id_y));
    y_data = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    src_ym_data += y_data * gaussian_table[4];

    data = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y));
    y_data = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    src_ym_data += y_data * gaussian_table[5];

    data = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y + 1));
    y_data = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    src_ym_data += y_data * gaussian_table[6];

    data = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + 1));
    y_data = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    src_ym_data += y_data * gaussian_table[7];

    data = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y + 1));
    y_data = data.x * 255 * 0.299 + data.y * 255 * 0.587 + data.z * 255 * 0.114;
    src_ym_data += y_data * gaussian_table[8];

#endif

    float gain = (y_max + src_ym_data + y_target) / (src_y_data + src_ym_data + y_target);
    src_data = src_data * gain;

    write_imagef(output, (int2)(g_id_x, g_id_y), src_data);
}
