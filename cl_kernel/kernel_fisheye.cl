/*
 * kernel_fisheye_2_gps
 * input_y,      input image, CL_R + CL_UNORM_INT8 //sampler
 * input_uv, CL_RG + CL_UNORM_INT8  //sampler
 * output_y,  CL_RGBA + CL_UNSIGNED_INT8, // 4-pixel
 * output_uv,  CL_RGBA + CL_UNSIGNED_INT8, // 4-pixel
 *
 * all angles are in radian
 */

#define PI 3.1415926f
#define PIXEL_PER_WI 4

typedef struct {
    float    center_x;
    float    center_y;
    float    wide_angle;
    float    radius;
    float    rotate_angle;
} FisheyeInfo;

__inline float2 calculate_fisheye_pos (float2 gps_pos, FisheyeInfo *fisheye)
{
    float z = cos (gps_pos.y);
    float x = sin (gps_pos.y) * cos (gps_pos.x);
    float y = sin (gps_pos.y) * sin (gps_pos.x);
    float r_angle = acos (y);
    float r = r_angle * (fisheye->radius * 2.0f) / fisheye->wide_angle;
    float xz_size = sqrt(x * x + z * z);

    float2 dst;
    dst.x = -r * x / xz_size;
    dst.y = -r * z / xz_size;

    float2 ret;
    ret.x = cos(fisheye->rotate_angle) * dst.x - sin(fisheye->rotate_angle) * dst.y;
    ret.y = sin(fisheye->rotate_angle) * dst.x + cos (fisheye->rotate_angle) * dst.y;

    return ret + (float2)(fisheye->center_x, fisheye->center_y);
}

__kernel void
kernel_fisheye_2_gps (
    __read_only image2d_t input_y, __read_only image2d_t input_uv, float2 input_y_size, FisheyeInfo fisheye,
    __write_only image2d_t output_y, __write_only image2d_t output_uv, float2 dst_center, float2 radian_per_pixel)
{
    const int g_x = get_global_id (0);
    const int g_y_uv = get_global_id (1);
    const int g_y = get_global_id (1) * 2;
    float2 src_pos[4];
    float4 src_data;
    float *src_ptr = (float*)(&src_data);
    sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    float2 gps_start_pos =
        (convert_float2((int2)(g_x * PIXEL_PER_WI, g_y)) - dst_center) * radian_per_pixel + PI / 2.0f;
    float2 gps_pos = gps_start_pos;

#pragma unroll
    for (int i = 0; i < PIXEL_PER_WI; ++i) {
        float2 pos = calculate_fisheye_pos (gps_pos, &fisheye);
        src_pos[i] = pos / input_y_size;
        src_ptr[i] = read_imagef (input_y, sampler, src_pos[i]).x;
        gps_pos.x += radian_per_pixel.x;
    }
    write_imageui (output_y, (int2)(g_x, g_y), convert_uint4(convert_uchar4(src_data * 255.0f)));

    src_data.s01 = read_imagef (input_uv, sampler, src_pos[0]).xy;
    src_data.s23 = read_imagef (input_uv, sampler, src_pos[2]).xy;
    write_imageui (output_uv, (int2)(g_x, g_y_uv), convert_uint4(convert_uchar4(src_data * 255.0f)));

    gps_pos = gps_start_pos;
    gps_pos.y += radian_per_pixel.y;
#pragma unroll
    for (int i = 0; i < PIXEL_PER_WI; ++i) {
        float2 pos = calculate_fisheye_pos (gps_pos, &fisheye);
        pos /= input_y_size;
        src_ptr[i] = read_imagef (input_y, sampler, pos).x;
        gps_pos.x += radian_per_pixel.x;
    }
    write_imageui (output_y, (int2)(g_x, g_y + 1), convert_uint4(convert_uchar4(src_data * 255.0f)));

}
