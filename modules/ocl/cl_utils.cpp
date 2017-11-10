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
    XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);
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

bool
dump_buffer (SmartPtr<VideoBuffer> buffer, char *file_name)
{
    ImageFileHandle file;

    XCamReturn ret = file.open (file_name, "wb");
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("open %s failed", file_name);
        return false;
    }

    ret = file.write_buf (buffer);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("write buffer to %s failed", file_name);
        file.close ();
        return false;
    }

    file.close ();
    return true;
}

SmartPtr<CLBuffer>
convert_to_clbuffer (
    const SmartPtr<CLContext> &context,
    SmartPtr<VideoBuffer> &buf)
{
    SmartPtr<CLBuffer> cl_buf;

    SmartPtr<CLVideoBuffer> cl_video_buf = buf.dynamic_cast_ptr<CLVideoBuffer> ();
    if (cl_video_buf.ptr ()) {
        cl_buf = cl_video_buf;
    }
#if HAVE_LIBDRM
    else {
        SmartPtr<DrmBoBuffer> bo_buf = buf.dynamic_cast_ptr<DrmBoBuffer> ();
        cl_buf = new CLVaBuffer (context, bo_buf);
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
        cl_image = new CLVaImage (context, bo_buf, desc, offset);
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
    XCAM_ASSERT ((width == info.width) && (height == info.height));

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

/*
 *  P00 ------------- P01
 *  |         |         |
 *  | -- Pinterpoint -- |
 *  |         |         |
 *  P10 ------------- P11
 */
static void
weighting_interpolate (
    const NV12Pixel &p00, const NV12Pixel &p01, const NV12Pixel &p10, const NV12Pixel &p11,
    NV12Pixel &inter_point)
{
    float weight00 = 0.0f, weight01 = 0.0f, weight10 = 0.0f, weight11 = 0.0f;
    if (p00.x_pos != p01.x_pos && p00.y_pos != p10.y_pos) {
        weight00 = (p11.x_pos - inter_point.x_pos) * (p11.y_pos - inter_point.y_pos);
        weight01 = (inter_point.x_pos - p10.x_pos) * (p10.y_pos - inter_point.y_pos);
        weight10 = (p01.x_pos - inter_point.x_pos) * (inter_point.y_pos - p00.y_pos);
        weight11 = (inter_point.x_pos - p00.x_pos) * (inter_point.y_pos - p00.y_pos);

        inter_point.y = p00.y * weight00 +  p01.y * weight01 +  p10.y * weight10 +  p11.y * weight11;
        inter_point.u = p00.u * weight00 +  p01.u * weight01 +  p10.u * weight10 +  p11.u * weight11;
        inter_point.v = p00.v * weight00 +  p01.v * weight01 +  p10.v * weight10 +  p11.v * weight11;
    } else if (p00.x_pos == p01.x_pos && p00.y_pos != p10.y_pos) {
        weight00 = p10.y_pos - inter_point.y_pos;
        weight10 = inter_point.y_pos - p00.y_pos;

        inter_point.y = p00.y * weight00 +  p10.y * weight10;
        inter_point.u = p00.u * weight00 +  p10.u * weight10;
        inter_point.v = p00.v * weight00 +  p10.v * weight10;
    } else if (p00.y_pos == p10.y_pos && p00.x_pos != p01.x_pos ) {
        weight00 = p01.x_pos - inter_point.x_pos;
        weight01 = inter_point.x_pos - p00.x_pos;

        inter_point.y = p00.y * weight00 +  p01.y * weight01;
        inter_point.u = p00.u * weight00 +  p01.u * weight01;
        inter_point.v = p00.v * weight00 +  p01.v * weight01;
    } else {
        inter_point.y = p00.y;
        inter_point.u = p00.u;
        inter_point.v = p00.v;
    }

    clamp (inter_point.y, 0.0f, 255.0f);
    clamp (inter_point.u, 0.0f, 255.0f);
    clamp (inter_point.v, 0.0f, 255.0f);
}

static void
transform_x_coordinate (
    const VideoBufferInfo &info,
    float x_pos, float y_pos,
    float &x_trans)
{
    float step = info.width / (2.0f * PI);
    float offset_radian = (x_pos < 0.0f) ? PI : ((y_pos >= 0.0f) ? 0.0f : 2.0f * PI);
    float arctan_radian = (x_pos != 0.0f) ? atan (y_pos / x_pos) : ((y_pos >= 0.0f) ? PI / 2.0f : -PI / 2.0f);

    x_trans = arctan_radian + offset_radian;
    x_trans *= step;
    clamp (x_trans, 0.0f, info.width - 1.0f);
}

static void
transform_y_coordinate (
    const VideoBufferInfo &info,
    const BowlDataConfig &config,
    float x_pos, float y_pos, float z_pos,
    float &y_trans)
{
    uint32_t wall_image_height = config.wall_height / (config.wall_height + config.ground_length) * info.height;
    uint32_t ground_image_height = info.height - wall_image_height;

    if (z_pos > 0.0f) {
        y_trans = (config.wall_height - z_pos) * wall_image_height / config.wall_height;
        clamp (y_trans, 0.0f, wall_image_height - 1.0f);
    } else {
        float max_semimajor = config.a *
                              sqrt (1 - config.center_z * config.center_z / (config.c * config.c));
        float min_semimajor = max_semimajor - config.ground_length;
        XCAM_ASSERT (max_semimajor > min_semimajor);
        float step = ground_image_height / (max_semimajor - min_semimajor);

        float axis_ratio = config.a / config.b;
        float cur_semimajor = sqrt (x_pos * x_pos + y_pos * y_pos * axis_ratio * axis_ratio);
        clamp (cur_semimajor, min_semimajor, max_semimajor);

        y_trans = (max_semimajor - cur_semimajor) * step + wall_image_height;
        clamp (y_trans, wall_image_height, info.height - 1.0f);
    }
}

static void
transform_coordinates (
    const VideoBufferInfo &info,
    const BowlDataConfig &config,
    float x_pos, float y_pos, float z_pos,
    NV12Pixel &p00, NV12Pixel &p01, NV12Pixel &p10, NV12Pixel &p11, NV12Pixel &inter_point)
{
    float x_trans, y_trans;
    transform_x_coordinate (info, x_pos, y_pos, x_trans);
    transform_y_coordinate (info, config, x_pos, y_pos, z_pos, y_trans);

    inter_point.x_pos = x_trans;
    inter_point.y_pos = y_trans;

    p00.x_pos = (uint32_t)inter_point.x_pos;
    p00.y_pos = (uint32_t)inter_point.y_pos;
    p01.x_pos = (p00.x_pos == (info.width - 1.0f)) ? p00.x_pos : (p00.x_pos + 1.0f);
    p01.y_pos = p00.y_pos;

    p10.x_pos = p00.x_pos;
    p10.y_pos = (p00.y_pos == (info.height - 1.0f)) ? p00.y_pos : (p00.y_pos + 1.0f);
    p11.x_pos = (p00.x_pos == (info.width - 1.0f)) ? p00.x_pos : (p00.x_pos + 1.0f);
    p11.y_pos = (p00.y_pos == (info.height - 1.0f)) ? p00.y_pos : (p00.y_pos + 1.0f);
}

static void
fill_pixel_value (uint8_t *mem, const VideoBufferInfo &info, NV12Pixel &p)
{
    uint32_t y_pos = (uint32_t)p.y_pos;
    uint32_t x_pos = (uint32_t)p.x_pos;

    uint32_t pos = y_pos * info.strides[0] + x_pos;
    p.y = mem[pos];

    pos = info.offsets[1] + y_pos / 2 * info.strides[1] + XCAM_ALIGN_DOWN (x_pos, 2);
    p.u = mem[pos];
    p.v = mem[pos + 1];
}

XCamReturn
get_bowl_view_data (
    SmartPtr<VideoBuffer> &buf, const BowlDataConfig &config,
    float x_pos, float y_pos, float z_pos,
    float &y, float &u, float &v)
{
    const VideoBufferInfo info = buf->get_video_info ();

    NV12Pixel p00, p01, p10, p11, inter_point;
    transform_coordinates (info, config, x_pos, y_pos, z_pos, p00, p01, p10, p11, inter_point);

    uint8_t *mem = buf->map ();
    XCAM_FAIL_RETURN (ERROR, mem, XCAM_RETURN_ERROR_MEM, "map buffer failed");

    fill_pixel_value (mem, info, p00);
    fill_pixel_value (mem, info, p01);
    fill_pixel_value (mem, info, p10);
    fill_pixel_value (mem, info, p11);
    buf->unmap ();

    weighting_interpolate (p00, p01, p10, p11, inter_point);
    y = inter_point.y;
    u = inter_point.u;
    v = inter_point.v;

    return XCAM_RETURN_NO_ERROR;
}

}
