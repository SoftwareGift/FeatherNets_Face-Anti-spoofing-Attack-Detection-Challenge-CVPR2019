/**
* \brief Image warping kernel function.
* \param[in] input Input image object.
* \param[out] output scaled output image object.
* \param[in] warp_config: image warping parameters
*/

typedef struct {
    int frame_id;
    int valid;
    int width;
    int height;
    float trim_ratio;
    float proj_mat[9];
} CLWarpConfig;

__kernel void kernel_image_warp (__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 CLWarpConfig warp_config)
{
#if WARP_Y

#endif

#if WARP_UV

#endif

    int dx = get_global_id(0);
    int dy = get_global_id(1);

    size_t width = get_global_size(0);
    size_t height = get_global_size(1);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    float sx = warp_config.proj_mat[0] * (float)dx + warp_config.proj_mat[1] * (float)dy + warp_config.proj_mat[2];
    float sy = warp_config.proj_mat[3] * (float)dx + warp_config.proj_mat[4] * (float)dy + warp_config.proj_mat[5];
    float w = warp_config.proj_mat[6] * (float)dx + warp_config.proj_mat[7] * (float)dy + warp_config.proj_mat[8];
    w = w != 0.0f ? 1.f / w : 0.0f;
    float warp_x = (sx * w) / (float)width;
    float warp_y = (sy * w) / (float)height;

    float4 pixel = read_imagef(input, sampler, (float2)(warp_x, warp_y));
    write_imagef(output, (int2)(dx, dy), pixel);
}


__kernel void kernel_image_trim (__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 float trim_ratio)
{
#if WARP_Y

#endif

#if WARP_UV

#endif

    int dx = get_global_id(0);
    int dy = get_global_id(1);

    size_t width = get_global_size(0);
    size_t height = get_global_size(1);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    float sx = (1.0f - 2.0f * trim_ratio) * ((float)dx / (float)width) + trim_ratio;
    float sy = (1.0f - 2.0f * trim_ratio) * ((float)dy / (float)height) + trim_ratio;

    float4 pixel = read_imagef(input, sampler, (float2)(sx, sy));
    write_imagef(output, (int2)(dx, dy), pixel);
}


