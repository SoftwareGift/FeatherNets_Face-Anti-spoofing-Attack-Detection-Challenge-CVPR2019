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

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 src_data_R = read_imagef (input, sampler, (int2)(g_id_x, g_id_y));
    float4 src_data_G = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height));
    float4 src_data_B = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height * 2));

    float4 src_y_data = src_data_R * 255 * 0.299 + src_data_G * 255 * 0.587 + src_data_B * 255 * 0.114;

    float gaussian_table[9] = {0.075f, 0.124f, 0.075f,
                               0.124f, 0.204f, 0.124f,
                               0.075f, 0.124f, 0.075f
                              };
    float4 src_ym_data = 0.0f;

#if 1 // use SLM
    __local float4 local_src_data[SHARED_PIXEL_X_SIZE * SHARED_PIXEL_Y_SIZE];

    int start_x = WORK_ITEM_X_SIZE * group_id_x - 1;
    int start_y = WORK_ITEM_Y_SIZE * group_id_y - 1;
    int local_index = local_id_y * WORK_ITEM_X_SIZE + local_id_x;
    int offset_x = local_index % SHARED_PIXEL_X_SIZE;
    int offset_y = local_index / SHARED_PIXEL_X_SIZE;

    float4 data_R = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y));
    float4 data_G = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y + image_height));
    float4 data_B = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y + image_height * 2));
    local_src_data[offset_y * SHARED_PIXEL_X_SIZE + offset_x] = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;

    if(local_index < SHARED_PIXEL_X_SIZE * SHARED_PIXEL_Y_SIZE - WORK_ITEM_X_SIZE * WORK_ITEM_Y_SIZE)
    {
        offset_x = (local_index + WORK_ITEM_X_SIZE * WORK_ITEM_Y_SIZE) % SHARED_PIXEL_X_SIZE;
        offset_y = (local_index + WORK_ITEM_X_SIZE * WORK_ITEM_Y_SIZE) / SHARED_PIXEL_X_SIZE;

        data_R = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y));
        data_G = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y + image_height));
        data_B = read_imagef (input, sampler, (int2)(start_x + offset_x, start_y + offset_y + image_height * 2));
        local_src_data[offset_y * SHARED_PIXEL_X_SIZE + offset_x] = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 y_data;
    float4 top_Y, bottom_Y;

    top_Y = local_src_data[local_id_y * SHARED_PIXEL_X_SIZE + local_id_x + 1];
    bottom_Y = local_src_data[(local_id_y + 2) * SHARED_PIXEL_X_SIZE + local_id_x + 1];

    y_data.x = local_src_data[local_id_y * SHARED_PIXEL_X_SIZE + local_id_x].w;
    y_data.yzw = top_Y.xyz;
    src_ym_data += y_data * (float4)gaussian_table[0];

    src_ym_data += top_Y * (float4)gaussian_table[1];

    y_data.xyz = top_Y.yzw;
    y_data.w = local_src_data[local_id_y * SHARED_PIXEL_X_SIZE + local_id_x + 2].x;
    src_ym_data += y_data * (float4)gaussian_table[2];

    y_data.x = local_src_data[(local_id_y + 1) * SHARED_PIXEL_X_SIZE + local_id_x].w;
    y_data.yzw = src_y_data.xyz;
    src_ym_data += y_data * (float4)gaussian_table[3];

    src_ym_data += src_y_data * (float4)gaussian_table[4];

    y_data.xyz = src_y_data.yzw;
    y_data.w = local_src_data[(local_id_y + 1) * SHARED_PIXEL_X_SIZE + local_id_x + 2].x;
    src_ym_data += y_data * (float4)gaussian_table[5];

    y_data.x = local_src_data[(local_id_y + 2) * SHARED_PIXEL_X_SIZE + local_id_x].w;
    y_data.yzw = bottom_Y.xyz;
    src_ym_data += y_data * (float4)gaussian_table[6];

    src_ym_data += bottom_Y * (float4)gaussian_table[7];

    y_data.xyz = bottom_Y.yzw;
    y_data.w = local_src_data[(local_id_y + 2) * SHARED_PIXEL_X_SIZE + local_id_x + 2].x;
    src_ym_data += y_data * (float4)gaussian_table[8];

#else // use global memory
    float4 data_R, data_G, data_B;
    float4 top_R, bottom_R, top_G, bottom_G, top_B, bottom_B;
    top_R = read_imagef (input, sampler, (int2)(g_id_x, g_id_y - 1));
    bottom_R = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + 1));
    top_G = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height - 1));
    bottom_G = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height + 1));
    top_B = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height * 2 - 1));
    bottom_B = read_imagef (input, sampler, (int2)(g_id_x, g_id_y + image_height * 2 + 1));

    data_R.x = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y - 1)).w;
    data_R.yzw = top_R.xyz;
    data_G.x = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y + image_height - 1)).w;
    data_G.yzw = top_G.xyz;
    data_B.x = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y + image_height * 2 - 1)).w;
    data_B.yzw = top_B.xyz;
    float4 y_data = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;
    src_ym_data += y_data * (float4)gaussian_table[0];

    data_R = top_R;
    data_G = top_G;
    data_B = top_B;
    y_data = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;
    src_ym_data += y_data * (float4)gaussian_table[1];

    data_R.xyz = top_R.yzw;
    data_R.w = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y - 1)).x;
    data_G.xyz = top_G.yzw;
    data_G.w = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y + image_height - 1)).x;
    data_B.xyz = top_B.yzw;
    data_B.w = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y + image_height * 2 - 1)).x;
    y_data = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;
    src_ym_data += y_data * (float4)gaussian_table[2];

    data_R.x = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y)).w;
    data_R.yzw = src_data_R.xyz;
    data_G.x = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y + image_height)).w;
    data_G.yzw = src_data_G.xyz;
    data_B.x = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y + image_height * 2)).w;
    data_B.yzw = src_data_B.xyz;
    y_data = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;
    src_ym_data += y_data * (float4)gaussian_table[3];

    y_data = src_data_R * 255 * 0.299 + src_data_G * 255 * 0.587 + src_data_B * 255 * 0.114;
    src_ym_data += y_data * (float4)gaussian_table[4];

    data_R.xyz = src_data_R.yzw;
    data_R.w = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y)).x;
    data_G.xyz = src_data_G.yzw;
    data_G.w = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y + image_height)).x;
    data_B.xyz = src_data_B.yzw;
    data_B.w = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y + image_height * 2)).x;
    y_data = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;
    src_ym_data += y_data * (float4)gaussian_table[5];

    data_R.x = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y + 1)).w;
    data_R.yzw = bottom_R.xyz;
    data_G.x = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y + image_height + 1)).w;
    data_G.yzw = bottom_G.xyz;
    data_B.x = read_imagef (input, sampler, (int2)(g_id_x - 1, g_id_y + image_height * 2 + 1)).w;
    data_B.yzw = bottom_B.xyz;
    y_data = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;
    src_ym_data += y_data * (float4)gaussian_table[6];

    data_R = bottom_R;
    data_G = bottom_G;
    data_B = bottom_B;
    y_data = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;
    src_ym_data += y_data * (float4)gaussian_table[7];

    data_R.xyz = bottom_R.yzw;
    data_R.w = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y + 1)).x;
    data_G.xyz = bottom_G.yzw;
    data_G.w = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y + image_height + 1)).x;
    data_B.xyz = bottom_B.yzw;
    data_B.w = read_imagef (input, sampler, (int2)(g_id_x + 1, g_id_y + image_height * 2 + 1)).x;
    y_data = data_R * 255 * 0.299 + data_G * 255 * 0.587 + data_B * 255 * 0.114;
    src_ym_data += y_data * (float4)gaussian_table[8];
#endif

    float4 gain = ((float4)y_max + src_ym_data + (float4)y_target) / (src_y_data + src_ym_data + (float4)y_target);
    src_data_R = src_data_R * gain;
    src_data_G = src_data_G * gain;
    src_data_B = src_data_B * gain;

    write_imagef(output, (int2)(g_id_x, g_id_y), src_data_R);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height), src_data_G);
    write_imagef(output, (int2)(g_id_x, g_id_y + image_height * 2), src_data_B);
#endif
}
