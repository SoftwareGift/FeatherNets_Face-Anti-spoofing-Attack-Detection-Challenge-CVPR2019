/**
* \brief Image warping kernel function.
* \param[in] input Input image object.
* \param[out] output scaled output image object.
* \param[in] warp_config: image warping parameters
*/

#ifndef WARP_Y
#define WARP_Y 1
#endif

// 8 bytes for each Y pixel
#define PIXEL_X_STEP   8

typedef struct {
    int frame_id;
    int width;
    int height;
    float trim_ratio;
    float proj_mat[9];
} CLWarpConfig;

__kernel void
kernel_image_warp_8_pixel (
    __read_only image2d_t input,
    __write_only image2d_t output,
    CLWarpConfig warp_config)
{
    // dest coordinate
    int d_x = get_global_id(0);
    int d_y = get_global_id(1);

    int out_width = get_image_width (output);
    int out_height = get_image_height (output);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    // source coordinate
    float s_x = 0.0f;
    float s_y = 0.0f;
    float warp_x = 0.0f;
    float warp_y = 0.0f;
    float w = 0.0f;

    float t_x = 0.0f;
    float t_y = 0.0f;

    float16 pixel = 0.0f;
    float* output_pixel = (float*)(&pixel);
    int i = 0;

    t_y = d_y;
#pragma unroll
    for (i = 0; i < PIXEL_X_STEP; i++) {
        t_x = (float)(PIXEL_X_STEP * d_x + i);

        s_x = warp_config.proj_mat[0] * t_x +
              warp_config.proj_mat[1] * t_y +
              warp_config.proj_mat[2];
        s_y = warp_config.proj_mat[3] * t_x +
              warp_config.proj_mat[4] * t_y +
              warp_config.proj_mat[5];
        w = warp_config.proj_mat[6] * t_x +
            warp_config.proj_mat[7] * t_y +
            warp_config.proj_mat[8];
        w = w != 0.0f ? 1.0f / w : 0.0f;

        warp_x = (s_x * w) / (float)(PIXEL_X_STEP * out_width);
        warp_y = (s_y * w) / (float)out_height;

#if WARP_Y
        output_pixel[i] = read_imagef(input, sampler, (float2)(warp_x, warp_y)).x;
#else
        float2 temp = read_imagef(input, sampler, (float2)(warp_x, warp_y)).xy;
        output_pixel[2 * i] = temp.x;
        output_pixel[2 * i + 1] = temp.y;
#endif
    }

#if WARP_Y
    write_imageui(output, (int2)(d_x, d_y), convert_uint4(as_ushort4(convert_uchar8(pixel.lo * 255.0f))));
#else
    write_imageui(output, (int2)(d_x, d_y), as_uint4(convert_uchar16(pixel * 255.0f)));
#endif

}

__kernel void
kernel_image_warp_1_pixel (
    __read_only image2d_t input,
    __write_only image2d_t output,
    CLWarpConfig warp_config)
{
    // dest coordinate
    int d_x = get_global_id(0);
    int d_y = get_global_id(1);

    int out_width = get_image_width (output);
    int out_height = get_image_height (output);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    // source coordinate
    float s_x = warp_config.proj_mat[0] * d_x +
                warp_config.proj_mat[1] * d_y +
                warp_config.proj_mat[2];
    float s_y = warp_config.proj_mat[3] * d_x +
                warp_config.proj_mat[4] * d_y +
                warp_config.proj_mat[5];
    float w = warp_config.proj_mat[6] * d_x +
              warp_config.proj_mat[7] * d_y +
              warp_config.proj_mat[8];
    w = w != 0.0f ? 1.0f / w : 0.0f;

    float warp_x = (s_x * w) / (float)out_width;
    float warp_y = (s_y * w) / (float)out_height;

    float4 pixel = read_imagef(input, sampler, (float2)(warp_x, warp_y));

    write_imagef(output, (int2)(d_x, d_y), pixel);
}

