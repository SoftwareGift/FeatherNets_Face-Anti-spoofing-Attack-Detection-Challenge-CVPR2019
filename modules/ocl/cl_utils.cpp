/*
 * cl_utils.cpp - CL Utilities
 *
 *  Copyright (c) 2016 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "cl_utils.h"
#include "image_file_handle.h"
#if HAVE_LIBDRM
#include "intel/cl_intel_context.h"
#include "intel/cl_va_memory.h"
#endif

namespace XCam {

struct NV12Pixel {
    float x_pos;
    float y_pos;

    float y;
    float u;
    float v;

    NV12Pixel ()
        : x_pos (0.0f), y_pos (0.0f)
        , y (0.0f), u (0.0f), v (0.0f)
    {}
};

static inline void
clamp (float &value, float min, float max)
{
    value = (value < min) ? min : ((value > max) ? max : value);
}

bool
dump_image (SmartPtr<CLImage> image, const char *file_name)
{
    XCAM_ASSERT (file_name);

    const CLImageDesc &desc = image->get_image_desc ();
    void *ptr = NULL;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {desc.width, desc.height, 1};
    size_t row_pitch;
    size_t slice_pitch;

    XCamReturn ret = image->enqueue_map (ptr, origin, region, &row_pitch, &slice_pitch, CL_MAP_READ);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), false, "dump image failed in enqueue_map");
    XCAM_ASSERT (ptr);
    XCAM_ASSERT (row_pitch == desc.row_pitch);
    uint8_t *buf_start = (uint8_t *)ptr;
    uint32_t width = image->get_pixel_bytes () * desc.width;

    FILE *fp = fopen (file_name, "wb");
    XCAM_FAIL_RETURN (ERROR, fp, false, "open file(%s) failed", file_name);

    for (uint32_t i = 0; i < desc.height; ++i) {
        uint8_t *buf_line = buf_start + row_pitch * i;
        fwrite (buf_line, width, 1, fp);
    }
    image->enqueue_unmap (ptr);
    fclose (fp);
    XCAM_LOG_INFO ("write image:%s\n", file_name);
    return true;
}

SmartPtr<CLBuffer>
convert_to_clbuffer (
    const SmartPtr<CLContext> &context,
    const SmartPtr<VideoBuffer> &buf)
{
    SmartPtr<CLBuffer> cl_buf;

    SmartPtr<CLVideoBuffer> cl_video_buf = buf.dynamic_cast_ptr<CLVideoBuffer> ();
    if (cl_video_buf.ptr ()) {
        cl_buf = cl_video_buf;
    }
#if HAVE_LIBDRM
    else {
        SmartPtr<DrmBoBuffer> bo_buf = buf.dynamic_cast_ptr<DrmBoBuffer> ();
        SmartPtr<CLIntelContext> ctx = context.dynamic_cast_ptr<CLIntelContext> ();
        XCAM_ASSERT (bo_buf.ptr () && ctx.ptr ());

        cl_buf = new CLVaBuffer (ctx, bo_buf);
    }
#else
    XCAM_UNUSED (context);
#endif

    XCAM_FAIL_RETURN (WARNING, cl_buf.ptr (), NULL, "convert to clbuffer failed");
    return cl_buf;
}

SmartPtr<CLImage>
convert_to_climage (
    const SmartPtr<CLContext> &context,
    SmartPtr<VideoBuffer> &buf,
    const CLImageDesc &desc,
    uint32_t offset,
    cl_mem_flags flags)
{
    SmartPtr<CLImage> cl_image;

    SmartPtr<CLVideoBuffer> cl_video_buf = buf.dynamic_cast_ptr<CLVideoBuffer> ();
    if (cl_video_buf.ptr ()) {
        SmartPtr<CLBuffer> cl_buf;

        if (offset == 0) {
            cl_buf = cl_video_buf;
        } else {
            uint32_t row_pitch = CLImage::calculate_pixel_bytes (desc.format) *
                                 XCAM_ALIGN_UP (desc.width, XCAM_CL_IMAGE_ALIGNMENT_X);
            uint32_t size = row_pitch * desc.height;

            cl_buf = new CLSubBuffer (context, cl_video_buf, flags, offset, size);
        }

        cl_image = new CLImage2D (context, desc, flags, cl_buf);
    }
#if HAVE_LIBDRM
    else {
        SmartPtr<DrmBoBuffer> bo_buf = buf.dynamic_cast_ptr<DrmBoBuffer> ();
        SmartPtr<CLIntelContext> ctx = context.dynamic_cast_ptr<CLIntelContext> ();
        XCAM_ASSERT (bo_buf.ptr () && ctx.ptr ());

        cl_image = new CLVaImage (ctx, bo_buf, desc, offset);
    }
#endif

    XCAM_FAIL_RETURN (WARNING, cl_image.ptr (), NULL, "convert to climage failed");
    return cl_image;
}

XCamReturn
convert_nv12_mem_to_video_buffer (
    void *nv12_mem, uint32_t width, uint32_t height, uint32_t row_pitch, uint32_t offset_uv,
    SmartPtr<VideoBuffer> &buf)
{
    XCAM_ASSERT (nv12_mem);
    XCAM_ASSERT (row_pitch >= width);

    VideoBufferPlanarInfo planar;
    const VideoBufferInfo info = buf->get_video_info ();
    XCAM_FAIL_RETURN (
        DEBUG, (width == info.width) && (height == info.height), XCAM_RETURN_ERROR_PARAM,
        "convert mem to video buffer failed since image sizes are not matched.");

    uint8_t *out_mem = buf->map ();
    XCAM_FAIL_RETURN (ERROR, out_mem, XCAM_RETURN_ERROR_MEM, "map buffer failed");

    uint8_t *src = (uint8_t *)nv12_mem;
    uint8_t *dest = NULL;
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);

        dest = out_mem + info.offsets[index];
        for (uint32_t i = 0; i < planar.height; i++) {
            memcpy (dest, src, width);
            src += row_pitch;
            dest += info.strides[index];
        }

        src = (uint8_t *)nv12_mem + offset_uv;
    }

    buf->unmap ();
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
interpolate_pixel_value (
    uint8_t* stitch_mem,
    float image_coord_x, float image_coord_y,
    float &y, float &u, float &v,
    const VideoBufferInfo& stitch_info)
{
    XCAM_ASSERT (image_coord_y < stitch_info.height && image_coord_x < stitch_info.width);

    uint8_t y00, y01, y10, y11;
    uint8_t u00, u01, u10, u11;
    uint8_t v00, v01, v10, v11;

    uint32_t x0 = (uint32_t) image_coord_x;
    uint32_t x1 = (x0 < stitch_info.width - 1) ? (x0 + 1) : x0;
    uint32_t y0 = (uint32_t) image_coord_y;
    uint32_t y1 = (y0 < stitch_info.height - 1) ? (y0 + 1) : y0;

    float rate00 = (x0 + 1 - image_coord_x) * (y0 + 1 - image_coord_y);
    float rate01 = (x0 + 1 - image_coord_x) * (image_coord_y - y0);
    float rate10 = (image_coord_x - x0) * (y0 + 1 - image_coord_y);
    float rate11 = (image_coord_x - x0) * (image_coord_y - y0);

    y00 = stitch_mem[y0 * stitch_info.strides[0] + x0];
    y01 = stitch_mem[y1 * stitch_info.strides[0] + x0];
    y10 = stitch_mem[y0 * stitch_info.strides[0] + x1];
    y11 = stitch_mem[y1 * stitch_info.strides[0] + x1];

    u00 = stitch_mem[stitch_info.offsets[1] + y0 / 2 * stitch_info.strides[1] + XCAM_ALIGN_DOWN (x0, 2)];
    u01 = stitch_mem[stitch_info.offsets[1] + y1 / 2 * stitch_info.strides[1] + XCAM_ALIGN_DOWN (x0, 2)];
    u10 = stitch_mem[stitch_info.offsets[1] + y0 / 2 * stitch_info.strides[1] + XCAM_ALIGN_DOWN (x1, 2)];
    u11 = stitch_mem[stitch_info.offsets[1] + y1 / 2 * stitch_info.strides[1] + XCAM_ALIGN_DOWN (x1, 2)];

    v00 = stitch_mem[stitch_info.offsets[1] + y0 / 2 * stitch_info.strides[1] + XCAM_ALIGN_DOWN (x0, 2) + 1];
    v01 = stitch_mem[stitch_info.offsets[1] + y1 / 2 * stitch_info.strides[1] + XCAM_ALIGN_DOWN (x0, 2) + 1];
    v10 = stitch_mem[stitch_info.offsets[1] + y0 / 2 * stitch_info.strides[1] + XCAM_ALIGN_DOWN (x1, 2) + 1];
    v11 = stitch_mem[stitch_info.offsets[1] + y1 / 2 * stitch_info.strides[1] + XCAM_ALIGN_DOWN (x1, 2) + 1];

    y = y00 * rate00 + y01 * rate01 + y10 * rate10 + y11 * rate11;
    u = u00 * rate00 + u01 * rate01 + u10 * rate10 + u11 * rate11;
    v = v00 * rate00 + v01 * rate01 + v10 * rate10 + v11 * rate11;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
map_to_specific_view (
    uint8_t *specific_view_mem, uint8_t* stitch_mem,
    uint32_t row, uint32_t col,
    float image_coord_x, float image_coord_y,
    const VideoBufferInfo& specific_view_info, const VideoBufferInfo& stitch_info)
{
    XCAM_ASSERT (row < specific_view_info.height && col < specific_view_info.width);

    float y, u, v;

    interpolate_pixel_value (stitch_mem, image_coord_x, image_coord_y, y, u, v, stitch_info);

    uint32_t y_index = row * specific_view_info.strides[0] + col;
    uint32_t u_index = specific_view_info.offsets[1] + row / 2 * specific_view_info.strides[1] + XCAM_ALIGN_DOWN (col, 2);

    specific_view_mem[y_index] = (uint8_t)y;
    specific_view_mem[u_index] = (uint8_t)u;
    specific_view_mem[u_index + 1] = (uint8_t)v;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
generate_topview_map_table (
    const VideoBufferInfo &stitch_info,
    const BowlDataConfig &config,
    std::vector<PointFloat2> &map_table,
    int width, int height)
{
    int center_x = width / 2;
    int center_y = height / 2;

    float show_width_mm = 5000.0f;
    float length_per_pixel = show_width_mm / height;

    map_table.resize (height * width);

    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            PointFloat3 world;
            world.x = (col - center_x) * length_per_pixel;
            world.y = (center_y - row) * length_per_pixel;
            world.z = 0.0f;

            PointFloat2 image_pos =
                bowl_view_coords_to_image (config, world, stitch_info.width, stitch_info.height);

            map_table[row * width + col] = image_pos;
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
generate_rectifiedview_map_table (
    const VideoBufferInfo &stitch_info,
    const BowlDataConfig &config,
    std::vector<PointFloat2> &map_table,
    float angle_start, float angle_end,
    int width, int height)
{
    float center_x = width / 2;

    float focal_plane_dist = 6000.0f;

    float angle_center = (angle_start + angle_end) / 2.0f;
    float theta = degree2radian((angle_end - angle_start)) / 2.0f;
    float length_per_pixel_x = 2 * focal_plane_dist * tan (theta) / width;

    float fov_up = degree2radian (20.0f);
    float fov_down = degree2radian (35.0f);

    float length_per_pixel_y = (focal_plane_dist * tan (fov_up) + focal_plane_dist * tan (fov_down)) / height;

    float center_y = tan (fov_up) / (tan (fov_up) + tan (fov_down)) * height;

    PointFloat3 world_pos;
    float plane_center_coords[3];

    plane_center_coords[0] = focal_plane_dist * cos (degree2radian (angle_center));
    plane_center_coords[1] = -focal_plane_dist * sin (degree2radian (angle_center));
    plane_center_coords[2] = 0.0f;

    map_table.resize (width * height);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float plane_point_coords[3];
            plane_point_coords[0] = (center_x - col) * length_per_pixel_x * cos (PI / 2 - degree2radian (angle_center)) + plane_center_coords[0];
            plane_point_coords[1] = (center_x - col) * length_per_pixel_x * sin (PI / 2 - degree2radian (angle_center)) + plane_center_coords[1];
            plane_point_coords[2] = (center_y - row) * length_per_pixel_y + plane_center_coords[2];

            float rate_xz, rate_yz;
            if (XCAM_DOUBLE_EQUAL_AROUND (plane_point_coords[2], 0.0f) && XCAM_DOUBLE_EQUAL_AROUND (plane_point_coords[1], 0.0f)) {
                world_pos.x = config.a;
                world_pos.y = 0;
                world_pos.z = 0;
            } else if (XCAM_DOUBLE_EQUAL_AROUND (plane_point_coords[2], 0.0f)) {
                world_pos.z = 0.0f;

                float rate_xy = plane_point_coords[0] / plane_point_coords[1];
                float square_y = 1 / (rate_xy * rate_xy / (config.a * config.a) + 1 / (config.b * config.b));
                world_pos.y = (plane_point_coords[1] > 0) ? sqrt (square_y) : -sqrt (square_y);
                world_pos.x = rate_xy * world_pos.y;
            } else {
                rate_xz = plane_point_coords[0] / plane_point_coords[2];
                rate_yz = plane_point_coords[1] / plane_point_coords[2];

                float square_z = 1 / (rate_xz * rate_xz / (config.a * config.a) + rate_yz * rate_yz / (config.b * config.b) + 1 / (config.c * config.c));
                world_pos.z = (plane_point_coords[2] > 0) ? sqrt (square_z) : -sqrt (square_z);
                world_pos.z = (world_pos.z <= -config.center_z) ? -config.center_z : world_pos.z;
                world_pos.x = rate_xz * world_pos.z;
                world_pos.y = rate_yz * world_pos.z;
            }

            world_pos.z += config.center_z;

            PointFloat2 image_coord =
                bowl_view_coords_to_image (config, world_pos, stitch_info.width, stitch_info.height);

            map_table[row * width + col] = image_coord;
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
sample_generate_top_view (
    SmartPtr<VideoBuffer> &stitch_buf,
    SmartPtr<VideoBuffer> top_view_buf,
    const BowlDataConfig &config,
    std::vector<PointFloat2> &map_table)
{
    const VideoBufferInfo top_view_info = top_view_buf->get_video_info ();
    const VideoBufferInfo stitch_info = stitch_buf->get_video_info ();

    int top_view_resolution_w = top_view_buf->get_video_info ().width;
    int top_view_resolution_h = top_view_buf->get_video_info ().height;

    if((int)map_table.size () != top_view_resolution_w * top_view_resolution_h) {
        map_table.clear ();
        generate_topview_map_table (stitch_info, config, map_table, top_view_resolution_w, top_view_resolution_h);
    }

    uint8_t *top_view_mem = NULL;
    uint8_t *stitch_mem = NULL;
    top_view_mem = top_view_buf->map ();
    stitch_mem = stitch_buf->map ();

    for(int row = 0; row < top_view_resolution_h; row++) {
        for(int col = 0; col < top_view_resolution_w; col++) {
            PointFloat2 image_coord = map_table[row * top_view_resolution_w + col];

            map_to_specific_view (top_view_mem, stitch_mem, row, col, image_coord.x, image_coord.y, top_view_info, stitch_info);
        }
    }

    top_view_buf->unmap();
    stitch_buf->unmap();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
sample_generate_rectified_view (
    SmartPtr<VideoBuffer> &stitch_buf,
    SmartPtr<VideoBuffer> rectified_view_buf,
    const BowlDataConfig &config,
    float angle_start, float angle_end,
    std::vector<PointFloat2> &map_table)
{
    const VideoBufferInfo rectified_view_info = rectified_view_buf->get_video_info ();
    const VideoBufferInfo stitch_info = stitch_buf->get_video_info ();

    int rectified_view_resolution_w = rectified_view_buf->get_video_info ().width;
    int rectified_view_resolution_h = rectified_view_buf->get_video_info ().height;

    if((int)map_table.size () != rectified_view_resolution_w * rectified_view_resolution_h) {
        map_table.clear ();
        generate_rectifiedview_map_table (stitch_info, config, map_table, angle_start, angle_end, rectified_view_resolution_w, rectified_view_resolution_h);
    }

    uint8_t *rectified_view_mem = NULL;
    uint8_t *stitch_mem = NULL;
    rectified_view_mem = rectified_view_buf->map ();
    stitch_mem = stitch_buf->map ();

    for(int row = 0; row < rectified_view_resolution_h; row++) {
        for(int col = 0; col < rectified_view_resolution_w; col++) {
            PointFloat2 image_coord = map_table[row * rectified_view_resolution_w + col];

            map_to_specific_view (rectified_view_mem, stitch_mem, row, col, image_coord.x, image_coord.y, rectified_view_info, stitch_info);
        }
    }

    rectified_view_buf->unmap();
    stitch_buf->unmap();

    return XCAM_RETURN_NO_ERROR;
}

}
