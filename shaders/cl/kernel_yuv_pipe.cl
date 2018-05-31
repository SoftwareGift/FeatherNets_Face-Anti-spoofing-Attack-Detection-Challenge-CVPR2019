/*
 * function: kernel_yuv_pipe
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

#pragma OPENCL FP_CONTRACT OFF

//#define USE_BUFFER_OBJECT 0

unsigned int get_sector_id (float u, float v)
{
    u = fabs(u) > 0.00001f ? u : 0.00001f;
    float tg = v / u;
    unsigned int se = tg > 1.f ? (tg > 2.f ? 3 : 2) : (tg > 0.5f ? 1 : 0);
    unsigned int so = tg > -1.f ? (tg > -0.5f ? 3 : 2) : (tg > -2.f ? 1 : 0);
    return tg > 0 ? (u > 0 ? se : (se + 8)) : (u > 0 ? (so + 12) : (so + 4));
}

__inline void cl_csc_rgbatonv12(float8 *R, float8 *G, float8 *B, float8 *out, __global float *matrix)
{
    out[0] = mad(matrix[0], R[0], mad(matrix[1], G[0], matrix[2] * B[0]));
    out[1] = mad(matrix[0], R[1], mad(matrix[1], G[1], matrix[2] * B[1]));

    out[2].s0 = mad(matrix[3], R[0].s0, mad(matrix[4], G[0].s0, matrix[5] * B[0].s0));
    out[2].s1 = mad(matrix[6], R[0].s0, mad(matrix[7], G[0].s0, matrix[8] * B[0].s0));
    out[2].s2 = mad(matrix[3], R[0].s2, mad(matrix[4], G[0].s2, matrix[5] * B[0].s2));
    out[2].s3 = mad(matrix[6], R[0].s2, mad(matrix[7], G[0].s2, matrix[8] * B[0].s2));
    out[2].s4 = mad(matrix[3], R[0].s4, mad(matrix[4], G[0].s4, matrix[5] * B[0].s4));
    out[2].s5 = mad(matrix[6], R[0].s4, mad(matrix[7], G[0].s4, matrix[8] * B[0].s4));
    out[2].s6 = mad(matrix[3], R[0].s6, mad(matrix[4], G[0].s6, matrix[5] * B[0].s6));
    out[2].s7 = mad(matrix[6], R[0].s6, mad(matrix[7], G[0].s6, matrix[8] * B[0].s6));

}

__inline void cl_macc(float8 *in, __global float *table)
{
    unsigned int table_id[4];
    float8 out;

    table_id[0] = get_sector_id(in[0].s0, in[0].s1);
    table_id[1] = get_sector_id(in[0].s2, in[0].s3);
    table_id[2] = get_sector_id(in[0].s4, in[0].s5);
    table_id[3] = get_sector_id(in[0].s6, in[0].s7);

    out.s0 = mad(in[0].s0, table[4 * table_id[0]], in[0].s1 * table[4 * table_id[0] + 1]) + 0.5f;
    out.s1 = mad(in[0].s0, table[4 * table_id[0] + 2], in[0].s1 * table[4 * table_id[0] + 3]) + 0.5f;
    out.s2 = mad(in[0].s2, table[4 * table_id[1]], in[0].s3 * table[4 * table_id[1] + 1]) + 0.5f;
    out.s3 = mad(in[0].s2, table[4 * table_id[1] + 2], in[0].s3 * table[4 * table_id[1] + 3]) + 0.5f;
    out.s4 = mad(in[0].s4, table[4 * table_id[0]], in[0].s5 * table[4 * table_id[0] + 1]) + 0.5f;
    out.s5 = mad(in[0].s4, table[4 * table_id[0] + 2], in[0].s5 * table[4 * table_id[0] + 3]) + 0.5f;
    out.s6 = mad(in[0].s6, table[4 * table_id[1]], in[0].s7 * table[4 * table_id[1] + 1]) + 0.5f;
    out.s7 = mad(in[0].s6, table[4 * table_id[1] + 2], in[0].s7 * table[4 * table_id[1] + 3]) + 0.5f;

    in[0] = out;
}

#if USE_BUFFER_OBJECT
__inline void cl_tnr_yuv(
    float8 *in, __global uchar8 *inputFramePre,
    int x, int y,
    float gain_yuv, float thr_y, float thr_uv,
    uint vertical_offset, uint x_offset)
#else
__inline void cl_tnr_yuv(
    float8 *in,
    __read_only image2d_t inputFramePre, __read_only image2d_t inputFramePreUV,
    int x, int y,
    float gain_yuv, float thr_y, float thr_uv, uint x_offset)
#endif
{
    float8 in_prev[3];
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#if USE_BUFFER_OBJECT
    in_prev[0] = convert_float8(inputFramePre[2 * y * x_offset + x]) / 256.0f;
    in_prev[1] = convert_float8(inputFramePre[(2 * y + 1) * x_offset + x]) / 256.0f;
    in_prev[2] = convert_float8(inputFramePre[(y + vertical_offset) * x_offset + x]) / 256.0f;
#else
    in_prev[0] = convert_float8(as_uchar8(convert_ushort4(read_imageui(inputFramePre, sampler, (int2)(x, 2 * y))))) / 256.0f;
    in_prev[1] = convert_float8(as_uchar8(convert_ushort4(read_imageui(inputFramePre, sampler, (int2)(x, 2 * y + 1))))) / 256.0f;
    in_prev[2] = convert_float8(as_uchar8(convert_ushort4(read_imageui(inputFramePreUV, sampler, (int2)(x, y))))) / 256.0f;
#endif

    float diff_max = 0.8f;
    float diff_Y[4], coeff_Y[4];

    diff_Y[0] = 0.25f * (fabs(in[0].s0 - in_prev[0].s0) + fabs(in[0].s1 - in_prev[0].s1) + fabs(in[1].s0 - in_prev[1].s0) + fabs(in[1].s1 - in_prev[1].s1));
    diff_Y[1] = 0.25f * (fabs(in[0].s2 - in_prev[0].s2) + fabs(in[0].s3 - in_prev[0].s3) + fabs(in[1].s2 - in_prev[1].s2) + fabs(in[1].s3 - in_prev[1].s3));
    diff_Y[2] = 0.25f * (fabs(in[0].s4 - in_prev[0].s4) + fabs(in[0].s5 - in_prev[0].s5) + fabs(in[1].s4 - in_prev[1].s4) + fabs(in[1].s5 - in_prev[1].s5));
    diff_Y[3] = 0.25f * (fabs(in[0].s6 - in_prev[0].s6) + fabs(in[0].s7 - in_prev[0].s7) + fabs(in[1].s6 - in_prev[1].s6) + fabs(in[1].s7 - in_prev[1].s7));

    coeff_Y[0] = (diff_Y[0] < thr_y) ? gain_yuv : (mad(diff_Y[0], 1 - gain_yuv, diff_max * gain_yuv - thr_y) / (diff_max - thr_y));
    coeff_Y[1] = (diff_Y[1] < thr_y) ? gain_yuv : (mad(diff_Y[1], 1 - gain_yuv, diff_max * gain_yuv - thr_y) / (diff_max - thr_y));
    coeff_Y[2] = (diff_Y[2] < thr_y) ? gain_yuv : (mad(diff_Y[2], 1 - gain_yuv, diff_max * gain_yuv - thr_y) / (diff_max - thr_y));
    coeff_Y[3] = (diff_Y[3] < thr_y) ? gain_yuv : (mad(diff_Y[3], 1 - gain_yuv, diff_max * gain_yuv - thr_y) / (diff_max - thr_y));

    coeff_Y[0] = (coeff_Y[0] < 1.0f) ? coeff_Y[0] : 1.0f;
    coeff_Y[1] = (coeff_Y[1] < 1.0f) ? coeff_Y[1] : 1.0f;
    coeff_Y[2] = (coeff_Y[2] < 1.0f) ? coeff_Y[2] : 1.0f;
    coeff_Y[3] = (coeff_Y[3] < 1.0f) ? coeff_Y[3] : 1.0f;

    in[0].s01 = mad(in[0].s01 - in_prev[0].s01, coeff_Y[0], in_prev[0].s01);
    in[1].s01 = mad(in[1].s01 - in_prev[1].s01, coeff_Y[0], in_prev[1].s01);
    in[0].s23 = mad(in[0].s23 - in_prev[0].s23, coeff_Y[1], in_prev[0].s23);
    in[1].s23 = mad(in[1].s23 - in_prev[1].s23, coeff_Y[1], in_prev[1].s23);
    in[0].s45 = mad(in[0].s45 - in_prev[0].s45, coeff_Y[2], in_prev[0].s45);
    in[1].s45 = mad(in[1].s45 - in_prev[1].s45, coeff_Y[2], in_prev[1].s45);
    in[0].s67 = mad(in[0].s67 - in_prev[0].s67, coeff_Y[3], in_prev[0].s67);
    in[1].s67 = mad(in[1].s67 - in_prev[1].s67, coeff_Y[3], in_prev[1].s67);

    float diff_U[4], diff_V[4], coeff_U[4], coeff_V[4];

    diff_U[0] = fabs(in[3].s0 - in_prev[3].s0);
    diff_U[1] = fabs(in[3].s2 - in_prev[3].s2);
    diff_U[2] = fabs(in[3].s4 - in_prev[3].s4);
    diff_U[3] = fabs(in[3].s6 - in_prev[3].s6);

    diff_V[0] = fabs(in[3].s1 - in_prev[3].s1);
    diff_V[1] = fabs(in[3].s3 - in_prev[3].s3);
    diff_V[2] = fabs(in[3].s5 - in_prev[3].s5);
    diff_V[3] = fabs(in[3].s7 - in_prev[3].s7);

    coeff_U[0] = (diff_U[0] < thr_uv) ? gain_yuv : (mad(diff_U[0], 1 - gain_yuv, diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv));
    coeff_U[1] = (diff_U[1] < thr_uv) ? gain_yuv : (mad(diff_U[1], 1 - gain_yuv, diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv));
    coeff_U[2] = (diff_U[2] < thr_uv) ? gain_yuv : (mad(diff_U[2], 1 - gain_yuv, diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv));
    coeff_U[3] = (diff_U[3] < thr_uv) ? gain_yuv : (mad(diff_U[3], 1 - gain_yuv, diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv));

    coeff_V[0] = (diff_V[0] < thr_uv) ? gain_yuv : (mad(diff_V[0], 1 - gain_yuv, diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv));
    coeff_V[1] = (diff_V[1] < thr_uv) ? gain_yuv : (mad(diff_V[1], 1 - gain_yuv, diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv));
    coeff_V[2] = (diff_V[2] < thr_uv) ? gain_yuv : (mad(diff_V[2], 1 - gain_yuv, diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv));
    coeff_V[3] = (diff_V[3] < thr_uv) ? gain_yuv : (mad(diff_V[3], 1 - gain_yuv, diff_max * gain_yuv - thr_uv) / (diff_max - thr_uv));

    coeff_U[0] = (coeff_U[0] < 1.0f) ? coeff_U[0] : 1.0f;
    coeff_U[1] = (coeff_U[1] < 1.0f) ? coeff_U[1] : 1.0f;
    coeff_U[2] = (coeff_U[2] < 1.0f) ? coeff_U[2] : 1.0f;
    coeff_U[3] = (coeff_U[3] < 1.0f) ? coeff_U[3] : 1.0f;

    coeff_V[0] = (coeff_V[0] < 1.0f) ? coeff_V[0] : 1.0f;
    coeff_V[1] = (coeff_V[1] < 1.0f) ? coeff_V[1] : 1.0f;
    coeff_V[2] = (coeff_V[2] < 1.0f) ? coeff_V[2] : 1.0f;
    coeff_V[3] = (coeff_V[3] < 1.0f) ? coeff_V[3] : 1.0f;

    in[2].s0 = mad(in[2].s0 - in_prev[2].s0, coeff_U[0], in_prev[2].s0);
    in[2].s1 = mad(in[2].s1 - in_prev[2].s1, coeff_V[0], in_prev[2].s1);
    in[2].s2 = mad(in[2].s2 - in_prev[2].s2, coeff_U[1], in_prev[2].s2);
    in[2].s3 = mad(in[2].s3 - in_prev[2].s3, coeff_V[1], in_prev[2].s3);
    in[2].s4 = mad(in[2].s4 - in_prev[2].s4, coeff_U[2], in_prev[2].s4);
    in[2].s5 = mad(in[2].s5 - in_prev[2].s5, coeff_V[2], in_prev[2].s5);
    in[2].s6 = mad(in[2].s6 - in_prev[2].s6, coeff_U[3], in_prev[2].s6);
    in[2].s7 = mad(in[2].s7 - in_prev[2].s7, coeff_V[3], in_prev[2].s7);

}

#if USE_BUFFER_OBJECT
__kernel void kernel_yuv_pipe (
    __global uchar8 *output,
    __global uchar8 *inputFramePre, uint vertical_offset,
    uint plannar_offset,
    __global float *matrix, __global float *table,
    float yuv_gain, float thr_y, float thr_uv, uint tnr_yuv_enable,
    __global ushort8 *inputFrame0)

#else

__kernel void kernel_yuv_pipe (
    __write_only image2d_t output, __write_only image2d_t output_uv,
    __read_only image2d_t inputFramePre, __read_only image2d_t inputFramePreUV,
    uint plannar_offset,
    __global float *matrix, __global float *table,
    float yuv_gain, float thr_y, float thr_uv, uint tnr_yuv_enable,
    __read_only image2d_t inputFrame0)

#endif
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    int offsetX = get_global_size(0);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float8 inR[2], inG[2], inB[2];
    float8 out[3];

#if USE_BUFFER_OBJECT
    // x [0, 240]
    // y [0, 540]
    uint offsetE = 2 * y * offsetX + x;
    uint offsetO = (2 * y + 1) * offsetX + x;
    uint offsetUV = (y + vertical_offset) * offsetX + x;
    uint offsetG = offsetX * plannar_offset;
    uint offsetB = offsetX * plannar_offset * 2;

    inR[0] = convert_float8(inputFrame0[offsetE]) / 65536.0f;
    inR[1] = convert_float8(inputFrame0[offsetO]) / 65536.0f;
    inG[0] = convert_float8(inputFrame0[offsetE + offsetG]) / 65536.0f;
    inG[1] = convert_float8(inputFrame0[offsetO + offsetG]) / 65536.0f;
    inB[0] = convert_float8(inputFrame0[offsetE + offsetB]) / 65536.0f;
    inB[1] = convert_float8(inputFrame0[offsetO + offsetB]) / 65536.0f;
#else
    inR[0] = convert_float8(as_ushort8(read_imageui(inputFrame0, sampler, (int2)(x, 2 * y)))) / 65536.0f;
    inR[1] = convert_float8(as_ushort8(read_imageui(inputFrame0, sampler, (int2)(x, 2 * y + 1)))) / 65536.0f;
    inG[0] = convert_float8(as_ushort8(read_imageui(inputFrame0, sampler, (int2)(x, 2 * y + plannar_offset)))) / 65536.0f;
    inG[1] = convert_float8(as_ushort8(read_imageui(inputFrame0, sampler, (int2)(x, 2 * y + 1 + plannar_offset)))) / 65536.0f;
    inB[0] = convert_float8(as_ushort8(read_imageui(inputFrame0, sampler, (int2)(x, 2 * y + plannar_offset * 2)))) / 65536.0f;
    inB[1] = convert_float8(as_ushort8(read_imageui(inputFrame0, sampler, (int2)(x, 2 * y + 1 + plannar_offset * 2)))) / 65536.0f;
#endif

    cl_csc_rgbatonv12(&inR[0], &inG[0], &inB[0], &out[0], matrix);
    cl_macc(&out[2], table);

    if (tnr_yuv_enable) {
#if USE_BUFFER_OBJECT
        cl_tnr_yuv (&out[0], inputFramePre, x, y, yuv_gain, thr_y, thr_uv, vertical_offset, offsetX);
#else
        cl_tnr_yuv (&out[0], inputFramePre, inputFramePreUV, x, y, yuv_gain, thr_y, thr_uv, offsetX);
#endif

    }

#if USE_BUFFER_OBJECT
    output[offsetE] = convert_uchar8(out[0] * 255.0f);
    output[offsetO] = convert_uchar8(out[1] * 255.0f);
    output[offsetUV] = convert_uchar8(out[2] * 255.0f);
#else
    write_imageui(output, (int2)(x, 2 * y), convert_uint4(as_ushort4(convert_uchar8_sat(out[0] * 255.0f))));
    write_imageui(output, (int2)(x, 2 * y + 1), convert_uint4(as_ushort4(convert_uchar8_sat(out[1] * 255.0f))));
    write_imageui(output_uv, (int2)(x, y), convert_uint4(as_ushort4(convert_uchar8_sat(out[2] * 255.0f))));
#endif

}

