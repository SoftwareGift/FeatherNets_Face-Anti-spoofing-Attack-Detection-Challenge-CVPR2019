/*
 * function: kernel_rgb_pipe
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

#define WORK_ITEM_X_SIZE 1
#define WORK_ITEM_Y_SIZE 1

#define SHARED_PIXEL_X_OFFSET 1
#define SHARED_PIXEL_Y_OFFSET 1

#define SHARED_PIXEL_WIDTH 8
#define SHARED_PIXEL_HEIGHT 4

#define SHARED_PIXEL_X_SIZE  (SHARED_PIXEL_WIDTH * WORK_ITEM_X_SIZE + SHARED_PIXEL_X_OFFSET * 2)
#define SHARED_PIXEL_Y_SIZE  (SHARED_PIXEL_HEIGHT * WORK_ITEM_Y_SIZE + SHARED_PIXEL_Y_OFFSET * 2)

typedef struct {
    float           thr_r;
    float           thr_g;
    float           thr_b;
    float           gain;
} CLRgbTnrConfig;

__inline void cl_snr (__local float4 *in, float4 *out, int lx, int ly)
{
    int tmp_id = (SHARED_PIXEL_Y_OFFSET + ly * WORK_ITEM_Y_SIZE) * SHARED_PIXEL_X_SIZE + SHARED_PIXEL_X_OFFSET + lx * WORK_ITEM_X_SIZE;
    (*(out)).x = ((*(in + tmp_id)).x + (*(in + tmp_id - SHARED_PIXEL_X_SIZE - 1)).x + (*(in + tmp_id - SHARED_PIXEL_X_SIZE)).x + (*(in + tmp_id - SHARED_PIXEL_Y_OFFSET + 1)).x + (*(in + tmp_id - 1)).x + (*(in + tmp_id + 1)).x + (*(in + tmp_id + SHARED_PIXEL_X_SIZE - 1)).x + (*(in + tmp_id + SHARED_PIXEL_X_SIZE)).x + (*(in + tmp_id + SHARED_PIXEL_X_SIZE + 1)).x) / 9.0f;

    (*(out)).y = ((*(in + tmp_id)).y + (*(in + tmp_id - SHARED_PIXEL_X_SIZE - 1)).y + (*(in + tmp_id - SHARED_PIXEL_X_SIZE)).y + (*(in + tmp_id - SHARED_PIXEL_Y_OFFSET + 1)).y + (*(in + tmp_id - 1)).y + (*(in + tmp_id + 1)).y + (*(in + tmp_id + SHARED_PIXEL_X_SIZE - 1)).y + (*(in + tmp_id + SHARED_PIXEL_X_SIZE)).y + (*(in + tmp_id + SHARED_PIXEL_X_SIZE + 1)).y) / 9.0f;

    (*(out)).z = ((*(in + tmp_id)).z + (*(in + tmp_id - SHARED_PIXEL_X_SIZE - 1)).z + (*(in + tmp_id - SHARED_PIXEL_X_SIZE)).z + (*(in + tmp_id - SHARED_PIXEL_Y_OFFSET + 1)).z + (*(in + tmp_id - 1)).z + (*(in + tmp_id + 1)).z + (*(in + tmp_id + SHARED_PIXEL_X_SIZE - 1)).z + (*(in + tmp_id + SHARED_PIXEL_X_SIZE)).z + (*(in + tmp_id + SHARED_PIXEL_X_SIZE + 1)).z) / 9.0f;

}

__inline void cl_tnr (float4 *out, int gx, int gy, __read_only image2d_t inputFrame1, __read_only image2d_t inputFrame2, __read_only image2d_t inputFrame3, CLRgbTnrConfig tnr_config)
{
    float4 var;
    float gain;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float4 in1, in2, in3;

    in1 = read_imagef(inputFrame1, sampler, (int2)(gx, gy));
    in2 = read_imagef(inputFrame2, sampler, (int2)(gx, gy));
    in3 = read_imagef(inputFrame3, sampler, (int2)(gx, gy));

    var.x = (fabs((*(out)).x - in1.x) + fabs(in1.x - in2.x) + fabs(in2.x - in3.x)) / 3.0f;
    var.y = (fabs((*(out)).y - in1.y) + fabs(in1.y - in2.y) + fabs(in2.y - in3.y)) / 3.0f;
    var.z = (fabs((*(out)).z - in1.z) + fabs(in1.z - in2.z) + fabs(in2.z - in3.z)) / 3.0f;

    int cond = (var.x + var.y + var.z) < (tnr_config.thr_r + tnr_config.thr_g + tnr_config.thr_b);
    gain = cond ? 1.0f : 0.0f;
    (*(out)).x = (gain * (*(out)).x + gain * in1.x + gain * in2.x +  in3.x) / (1.0f + 3 * gain);
    (*(out)).y = (gain * (*(out)).y + gain * in1.y + gain * in2.y +  in3.y) / (1.0f + 3 * gain);
    (*(out)).z = (gain * (*(out)).z + gain * in1.z + gain * in2.z +  in3.z) / (1.0f + 3 * gain);
}

__kernel void kernel_rgb_pipe (__write_only image2d_t output, CLRgbTnrConfig tnr_config, __read_only image2d_t inputFrame0, __read_only image2d_t inputFrame1, __read_only image2d_t inputFrame2, __read_only image2d_t inputFrame3)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int g_id_x = get_global_id (0);
    int g_id_y = get_global_id (1);
    int g_size_x = get_global_size (0);
    int g_size_y = get_global_size (1);

    int l_id_x = get_local_id (0);
    int l_id_y = get_local_id (1);
    int l_size_x = get_local_size (0);
    int l_size_y = get_local_size (1);

    __local float4 p[SHARED_PIXEL_X_SIZE * SHARED_PIXEL_Y_SIZE];

    float4 out;
    int i = l_id_x + l_id_y * l_size_x;
    int xstart = (g_id_x - l_id_x) - SHARED_PIXEL_X_OFFSET;
    int ystart = (g_id_y - l_id_y) - SHARED_PIXEL_Y_OFFSET;

    for(; i < SHARED_PIXEL_X_SIZE * SHARED_PIXEL_Y_SIZE; i += l_size_x * l_size_y) {

        int x0 = i % SHARED_PIXEL_X_SIZE + xstart;
        int y0 = i / SHARED_PIXEL_X_SIZE + ystart;

        p[i] = read_imagef(inputFrame0, sampler, (int2)(x0, y0));
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    cl_snr(&p[0], &out, l_id_x, l_id_y);
    cl_tnr(&out, g_id_x, g_id_y, inputFrame1, inputFrame2, inputFrame3, tnr_config);

    write_imagef(output, (int2)(g_id_x, g_id_y), out);
}

