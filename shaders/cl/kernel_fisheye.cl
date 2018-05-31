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

__inline float2 calculate_fisheye_pos (float2 gps_pos, const FisheyeInfo *info)
{
    float z = cos (gps_pos.y);
    float x = sin (gps_pos.y) * cos (gps_pos.x);
    float y = sin (gps_pos.y) * sin (gps_pos.x);
    float r_angle = acos (y);
    float r = r_angle * (info->radius * 2.0f) / info->wide_angle;
    float xz_size = sqrt(x * x + z * z);

    float2 dst;
    dst.x = -r * x / xz_size;
    dst.y = -r * z / xz_size;

    float2 ret;
    ret.x = cos(info->rotate_angle) * dst.x - sin(info->rotate_angle) * dst.y;
    ret.y = sin(info->rotate_angle) * dst.x + cos (info->rotate_angle) * dst.y;

    return ret + (float2)(info->center_x, info->center_y);
}

__kernel void
kernel_fisheye_table (
    const FisheyeInfo info, const float2 fisheye_image_size,
    __write_only image2d_t table, const float2 radian_per_pixel, const float2 table_center)
{
    int2 out_pos = (int2)(get_global_id (0), get_global_id (1));
    float2 gps_pos = (convert_float2 (out_pos) - table_center) * radian_per_pixel + PI / 2.0f;
    float2 pos = calculate_fisheye_pos (gps_pos, &info);
    float2 min_pos = (float2)(info.center_x - info.radius, info.center_y - info.radius);
    float2 max_pos = (float2)(info.center_x + info.radius, info.center_y + info.radius);
    pos = clamp (pos, min_pos, max_pos);
    pos /= fisheye_image_size;
    write_imagef (table, out_pos, (float4)(pos, 0.0f, 0.0f));
}

__kernel void
kernel_lsc_table (
    __read_only image2d_t geo_table, __write_only image2d_t lsc_table,
    __global float *lsc_array, int array_size, const FisheyeInfo info, const float2 fisheye_image_size)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2) (get_global_id (0), get_global_id (1));

    float2 geo_data = read_imagef (geo_table, sampler, pos).xy * fisheye_image_size;
    float2 dist = geo_data - (float2)(info.center_x, info.center_y);
    float r = sqrt (dist.x * dist.x + dist.y * dist.y);
    r /= (1.0f * info.radius / array_size);

    int min_idx = r;
    int max_idx = r + 1.0f;
    float lsc_data = max_idx > (array_size - 1) ? lsc_array[array_size - 1] :
                     (r - min_idx) * (lsc_array[max_idx] - lsc_array[min_idx]) + lsc_array[min_idx];

    write_imagef (lsc_table, pos, (float4)(lsc_data, 0.0f, 0.0f, 1.0f));
}

__kernel void
kernel_fisheye_2_gps (
    __read_only image2d_t input_y, __read_only image2d_t input_uv,
    const float2 input_y_size, const FisheyeInfo info,
    __write_only image2d_t output_y, __write_only image2d_t output_uv,
    const float2 dst_center, const float2 radian_per_pixel)
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
        float2 pos = calculate_fisheye_pos (gps_pos, &info);
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
        float2 pos = calculate_fisheye_pos (gps_pos, &info);
        pos /= input_y_size;
        src_ptr[i] = read_imagef (input_y, sampler, pos).x;
        gps_pos.x += radian_per_pixel.x;
    }
    write_imageui (output_y, (int2)(g_x, g_y + 1), convert_uint4(convert_uchar4(src_data * 255.0f)));

}
