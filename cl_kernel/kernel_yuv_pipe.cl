/*
 * function: kernel_yuv_pipe
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */
unsigned int get_sector_id (float u, float v)
{
    u = fabs(u) > 0.00001f ? u : 0.00001f;
    float tg = v / u;
    unsigned int se = tg > 1 ? (tg > 2 ? 3 : 2) : (tg > 0.5 ? 1 : 0);
    unsigned int so = tg > -1 ? (tg > -0.5 ? 3 : 2) : (tg > -2 ? 1 : 0);
    return tg > 0 ? (u > 0 ? se : (se + 8)) : (u > 0 ? (so + 12) : (so + 4));
}

__inline void cl_csc_rgbatonv12(float4 *in, float *out, __global float *matrix)
{
    out[0] = matrix[0] * in[0].x + matrix[1] * in[0].y + matrix[2] * in[0].z;
    out[1] = matrix[0] * in[1].x + matrix[1] * in[1].y + matrix[2] * in[1].z;
    out[2] = matrix[0] * in[2].x + matrix[1] * in[2].y + matrix[2] * in[2].z;
    out[3] = matrix[0] * in[3].x + matrix[1] * in[3].y + matrix[2] * in[3].z;
    out[4] = matrix[3] * in[0].x + matrix[4] * in[0].y + matrix[5] * in[0].z;
    out[5] = matrix[6] * in[0].x + matrix[7] * in[0].y + matrix[8] * in[0].z;
}

__inline void cl_macc(float *in, __global float *table)
{
    unsigned int table_id;
    float ui, vi, uo, vo;
    ui = (*in);
    vi = (*(in + 1));
    table_id = get_sector_id(ui, vi);

    uo = ui * table[4 * table_id] + vi * table[4 * table_id + 1];
    vo = ui * table[4 * table_id + 2] + vi * table[4 * table_id + 3];

    (*in) = uo + 0.5;
    (*(in + 1)) = vo + 0.5;
}

__inline void cl_tnr_rgb(float4 *in, __read_only image2d_t inputFrame1, __read_only image2d_t inputFrame2, __read_only image2d_t inputFrame3, uint count, int x, int y, float rgb_gain, float thr_r, float thr_g, float thr_b)
{

    float4 in1[4], in2[4], in3[4];
    float4 var[4];
    float4 out[4];
    float gain[4];
    int cond[4];

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    in1[0] = read_imagef(inputFrame1, sampler, (int2)(2 * x, 2 * y));
    in1[1] = read_imagef(inputFrame1, sampler, (int2)(2 * x + 1, 2 * y));
    in1[2] = read_imagef(inputFrame1, sampler, (int2)(2 * x, 2 * y + 1));
    in1[3] = read_imagef(inputFrame1, sampler, (int2)(2 * x + 1, 2 * y + 1));

    if (count == 2)
    {
        // write follow code separate intead of using a 'for' circle to improve kernel performace.
        var[0] = fabs(in[0] - in1[0]);
        var[1] = fabs(in[1] - in1[1]);
        var[2] = fabs(in[2] - in1[2]);
        var[3] = fabs(in[3] - in1[3]);

        cond[0] = (var[0].x + var[0].y + var[0].z) < (thr_r + thr_g + thr_b);
        cond[1] = (var[1].x + var[1].y + var[1].z) < (thr_r + thr_g + thr_b);
        cond[2] = (var[2].x + var[2].y + var[2].z) < (thr_r + thr_g + thr_b);
        cond[3] = (var[3].x + var[3].y + var[3].z) < (thr_r + thr_g + thr_b);

        gain[0] = cond[0] ? 1.0f : 0.0f;
        gain[1] = cond[1] ? 1.0f : 0.0f;
        gain[2] = cond[2] ? 1.0f : 0.0f;
        gain[3] = cond[3] ? 1.0f : 0.0f;

        out[0] = (gain[0] *  in[0] + in1[0]) / (1.0f + gain[0]);
        out[1] = (gain[1] *  in[1] + in1[1]) / (1.0f + gain[1]);
        out[2] = (gain[2] *  in[2] + in1[2]) / (1.0f + gain[2]);
        out[3] = (gain[3] *  in[3] + in1[3]) / (1.0f + gain[3]);
    }
    else if(count == 3)
    {
        in2[0] = read_imagef(inputFrame2, sampler, (int2)(2 * x, 2 * y));
        in2[1] = read_imagef(inputFrame2, sampler, (int2)(2 * x + 1, 2 * y));
        in2[2] = read_imagef(inputFrame2, sampler, (int2)(2 * x, 2 * y + 1));
        in2[3] = read_imagef(inputFrame2, sampler, (int2)(2 * x + 1, 2 * y + 1));

        var[0] = (fabs(in[0] - in1[0]) + fabs(in1[0] - in2[0])) / 2.0;
        var[1] = (fabs(in[1] - in1[1]) + fabs(in1[1] - in2[1])) / 2.0 ;
        var[2] = (fabs(in[2] - in1[2]) + fabs(in1[2] - in2[2])) / 2.0;
        var[3] = (fabs(in[3] - in1[3]) + fabs(in1[0] - in2[3])) / 2.0;

        cond[0] = (var[0].x + var[0].y + var[0].z) < (thr_r + thr_g + thr_b);
        cond[1] = (var[1].x + var[1].y + var[1].z) < (thr_r + thr_g + thr_b);
        cond[2] = (var[2].x + var[2].y + var[2].z) < (thr_r + thr_g + thr_b);
        cond[3] = (var[3].x + var[3].y + var[3].z) < (thr_r + thr_g + thr_b);

        gain[0] = cond[0] ? 1.0f : 0.0f;
        gain[1] = cond[1] ? 1.0f : 0.0f;
        gain[2] = cond[2] ? 1.0f : 0.0f;
        gain[3] = cond[3] ? 1.0f : 0.0f;

        out[0] = (gain[0] *  in[0] + gain[0] * in1[0] + in2[0]) / (1.0f + 2 * gain[0]);
        out[1] = (gain[1] *  in[1] + gain[1] * in1[1] + in2[1]) / (1.0f + 2 * gain[1]);
        out[2] = (gain[2] *  in[2] + gain[2] * in1[2] + in2[2]) / (1.0f + 2 * gain[2]);
        out[3] = (gain[3] *  in[3] + gain[3] * in1[3] + in2[3]) / (1.0f + 2 * gain[3]);
    }
    else if(count == 4)
    {
        in2[0] = read_imagef(inputFrame2, sampler, (int2)(2 * x, 2 * y));
        in2[1] = read_imagef(inputFrame2, sampler, (int2)(2 * x + 1, 2 * y));
        in2[2] = read_imagef(inputFrame2, sampler, (int2)(2 * x, 2 * y + 1));
        in2[3] = read_imagef(inputFrame2, sampler, (int2)(2 * x + 1, 2 * y + 1));

        in3[0] = read_imagef(inputFrame3, sampler, (int2)(2 * x, 2 * y));
        in3[1] = read_imagef(inputFrame3, sampler, (int2)(2 * x + 1, 2 * y));
        in3[2] = read_imagef(inputFrame3, sampler, (int2)(2 * x, 2 * y + 1));
        in3[3] = read_imagef(inputFrame3, sampler, (int2)(2 * x + 1, 2 * y + 1));

        var[0] = (fabs(in[0] - in1[0]) + fabs(in1[0] - in2[0]) + fabs(in2[0] - in3[0])) / 3.0;
        var[1] = (fabs(in[1] - in1[1]) + fabs(in1[1] - in2[1]) + fabs(in2[1] - in3[1])) / 3.0 ;
        var[2] = (fabs(in[2] - in1[2]) + fabs(in1[2] - in2[2]) + fabs(in2[2] - in3[2])) / 3.0;
        var[3] = (fabs(in[3] - in1[3]) + fabs(in1[0] - in2[3]) + fabs(in2[3] - in3[3])) / 3.0;

        cond[0] = (var[0].x + var[0].y + var[0].z) < (thr_r + thr_g + thr_b);
        cond[1] = (var[1].x + var[1].y + var[1].z) < (thr_r + thr_g + thr_b);
        cond[2] = (var[2].x + var[2].y + var[2].z) < (thr_r + thr_g + thr_b);
        cond[3] = (var[3].x + var[3].y + var[3].z) < (thr_r + thr_g + thr_b);

        gain[0] = cond[0] ? 1.0f : 0.0f;
        gain[1] = cond[1] ? 1.0f : 0.0f;
        gain[2] = cond[2] ? 1.0f : 0.0f;
        gain[3] = cond[3] ? 1.0f : 0.0f;

        out[0] = (gain[0] *  in[0] + gain[0] * in1[0] + gain[0] * in2[0] + in3[0]) / (1.0f + 3 * gain[0]);
        out[1] = (gain[1] *  in[1] + gain[1] * in1[1] + gain[1] * in2[1] + in3[1]) / (1.0f + 3 * gain[1]);
        out[2] = (gain[2] *  in[2] + gain[2] * in1[2] + gain[2] * in2[2] + in3[2]) / (1.0f + 3 * gain[2]);
        out[3] = (gain[3] *  in[3] + gain[3] * in1[3] + gain[3] * in2[3] + in3[3]) / (1.0f + 3 * gain[3]);
    }

    in[0] = out[0];
    in[1] = out[1];
    in[2] = out[2];
    in[3] = out[3];
}

__inline void cl_tnr_yuv(float *in, __read_only image2d_t inputFramePre, int x, int y, float gain_yuv, float thr_y, float thr_uv, uint vertical_offset)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 in_prev[6];
    in_prev[0] = read_imagef(inputFramePre, sampler, (int2)(2 * x, 2 * y));
    in_prev[1] = read_imagef(inputFramePre, sampler, (int2)(2 * x + 1, 2 * y));
    in_prev[2] = read_imagef(inputFramePre, sampler, (int2)(2 * x, 2 * y + 1));
    in_prev[3] = read_imagef(inputFramePre, sampler, (int2)(2 * x + 1, 2 * y + 1));

    in_prev[4] = read_imagef(inputFramePre, sampler, (int2)(2 * x, y + vertical_offset));
    in_prev[5] = read_imagef(inputFramePre, sampler, (int2)(2 * x + 1, y + vertical_offset));

    float diff_max = 0.8;

    float diff_Y = 0.25 * (fabs(in[0] - in_prev[0].x) + fabs(in[1] - in_prev[1].x) + fabs(in[2] - in_prev[2].x) + fabs(in[3] - in_prev[3].x));

    float coeff_Y = (diff_Y < thr_y) ? gain_yuv : (diff_Y * (1 - gain_yuv) + diff_max * gain_yuv - thr_y) / (diff_max - thr_y);
    coeff_Y = (coeff_Y < 1.0) ? coeff_Y : 1.0;

    float out[6];
    in[0] =  in_prev[0].x + (in[0] - in_prev[0].x) * coeff_Y;
    in[1] =  in_prev[1].x + (in[1] - in_prev[1].x) * coeff_Y;
    in[2] =  in_prev[2].x + (in[2] - in_prev[2].x) * coeff_Y;
    in[3] =  in_prev[3].x + (in[3] - in_prev[3].x) * coeff_Y;

    float diff_U = fabs(in[4] -  in_prev[4].x);
    float diff_V = fabs(in[5] -  in_prev[5].x);

    float coeff_U = (diff_U < thr_uv) ? gain_yuv : (diff_U * (1 - gain_yuv) + diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv);
    float coeff_V = (diff_V < thr_uv) ? gain_yuv : (diff_V * (1 - gain_yuv) + diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv);
    coeff_U = (coeff_U < 1.0) ? coeff_U : 1.0;
    coeff_V = (coeff_V < 1.0) ? coeff_V : 1.0;

    in[4] =  in_prev[4].x + (in[4] - in_prev[4].x) * coeff_U;
    in[5] =  in_prev[5].x + (in[5] - in_prev[5].x) * coeff_V;
}

__kernel void kernel_yuv_pipe (__write_only image2d_t output, __read_only image2d_t inputFramePre, uint vertical_offset, __global float *matrix, __global float *table, uint count, float rgb_gain, float thr_r, float thr_g, float thr_b, float yuv_gain, float thr_y, float thr_uv, uint tnr_rgb_enable, uint tnr_yuv_enable, __read_only image2d_t inputFrame0, __read_only image2d_t inputFrame1, __read_only image2d_t inputFrame2, __read_only image2d_t inputFrame3)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    float4 in[4];
    float out[6];

    in[0] = read_imagef(inputFrame0, sampler, (int2)(2 * x, 2 * y));
    in[1] = read_imagef(inputFrame0, sampler, (int2)(2 * x + 1, 2 * y));
    in[2] = read_imagef(inputFrame0, sampler, (int2)(2 * x, 2 * y + 1));
    in[3] = read_imagef(inputFrame0, sampler, (int2)(2 * x + 1, 2 * y + 1));

    if (tnr_rgb_enable) {
        cl_tnr_rgb (&in[0], inputFrame1, inputFrame2, inputFrame3, count, x, y, rgb_gain, thr_r, thr_g, thr_b);
    }

    cl_csc_rgbatonv12(&in[0], &out[0], matrix);
    cl_macc(&out[4], table);

    if (tnr_yuv_enable) {
        cl_tnr_yuv (&out[0], inputFramePre, x, y, yuv_gain, thr_y, thr_uv, vertical_offset);
    }


    write_imagef(output, (int2)(2 * x, 2 * y), (float4)out[0]);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), (float4)out[1]);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), (float4)out[2]);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), (float4)out[3]);
    write_imagef(output, (int2)(2 * x, y + vertical_offset), (float4)out[4]);
    write_imagef(output, (int2)(2 * x + 1, y + vertical_offset), (float4)out[5]);
}

