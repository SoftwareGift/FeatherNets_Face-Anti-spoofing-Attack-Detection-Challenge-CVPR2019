/*
 * xcam_buffer.cpp - video buffer standard version
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

#include <base/xcam_buffer.h>

XCamReturn
xcam_video_buffer_info_reset (
    XCamVideoBufferInfo *info,
    uint32_t format,
    uint32_t width, uint32_t height,
    uint32_t aligned_width, uint32_t aligned_height, uint32_t size)
{
    uint32_t image_size = 0;
    uint32_t i = 0;

    XCAM_ASSERT (info && format);
    XCAM_ASSERT (!aligned_width  || aligned_width >= width);
    XCAM_ASSERT (!aligned_height  || aligned_height >= height);

    if (!aligned_width)
        aligned_width = XCAM_ALIGN_UP (width, 4);
    if (!aligned_height)
        aligned_height = XCAM_ALIGN_UP (height, 2);

    info->format = format;
    info->width = width;
    info->height = height;
    info->aligned_width = aligned_width;
    info->aligned_height = aligned_height;

    switch (format) {
    case V4L2_PIX_FMT_GREY:
        info->color_bits = 8;
        info->components = 1;
        info->strides [0] = aligned_width;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;
    case V4L2_PIX_FMT_NV12:
        info->color_bits = 8;
        info->components = 2;
        info->strides [0] = aligned_width;
        info->strides [1] = info->strides [0];
        info->offsets [0] = 0;
        info->offsets [1] = info->offsets [0] + info->strides [0] * aligned_height;
        image_size = info->strides [0] * aligned_height + info->strides [1] * aligned_height / 2;
        break;
    case V4L2_PIX_FMT_YUYV:
        info->color_bits = 8;
        info->components = 1;
        info->strides [0] = aligned_width * 2;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;
    case V4L2_PIX_FMT_RGB565:
        info->color_bits = 16;
        info->components = 1;
        info->strides [0] = aligned_width * 2;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;
    case V4L2_PIX_FMT_RGB24:
        info->color_bits = 8;
        info->components = 1;
        info->strides [0] = aligned_width * 3;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;
        // memory order RGBA 8-8-8-8
    case V4L2_PIX_FMT_RGBA32:
        // memory order: BGRA 8-8-8-8
    case V4L2_PIX_FMT_XBGR32:
    case V4L2_PIX_FMT_ABGR32:
    case V4L2_PIX_FMT_BGR32:
        // memory order: ARGB 8-8-8-8
    case V4L2_PIX_FMT_RGB32:
    case V4L2_PIX_FMT_ARGB32:
    case V4L2_PIX_FMT_XRGB32:
        info->color_bits = 8;
        info->components = 1;
        info->strides [0] = aligned_width * 4;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;
    case XCAM_PIX_FMT_RGB48:
        info->color_bits = 16;
        info->components = 1;
        info->strides [0] = aligned_width * 3 * 2;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;
    case XCAM_PIX_FMT_RGBA64:
        info->color_bits = 16;
        info->components = 1;
        info->strides [0] = aligned_width * 4 * 2;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;

    case V4L2_PIX_FMT_SBGGR8:
    case V4L2_PIX_FMT_SGBRG8:
    case V4L2_PIX_FMT_SGRBG8:
    case V4L2_PIX_FMT_SRGGB8:
        info->color_bits = 8;
        info->components = 1;
        info->strides [0] = aligned_width;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;

    case V4L2_PIX_FMT_SBGGR10:
    case V4L2_PIX_FMT_SGBRG10:
    case V4L2_PIX_FMT_SGRBG10:
    case V4L2_PIX_FMT_SRGGB10:
        info->color_bits = 10;
        info->components = 1;
        info->strides [0] = aligned_width * 2;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;

    case V4L2_PIX_FMT_SBGGR12:
    case V4L2_PIX_FMT_SGBRG12:
    case V4L2_PIX_FMT_SGRBG12:
    case V4L2_PIX_FMT_SRGGB12:
        info->color_bits = 12;
        info->components = 1;
        info->strides [0] = aligned_width * 2;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;

    case V4L2_PIX_FMT_SBGGR16:
    case XCAM_PIX_FMT_SGRBG16:
        info->color_bits = 16;
        info->components = 1;
        info->strides [0] = aligned_width * 2;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;

    case XCAM_PIX_FMT_LAB:
        info->color_bits = 8;
        info->components = 1;
        info->strides [0] = aligned_width * 3;
        info->offsets [0] = 0;
        image_size = info->strides [0] * aligned_height;
        break;

    case XCAM_PIX_FMT_RGB48_planar:
    case XCAM_PIX_FMT_RGB24_planar:
        if (XCAM_PIX_FMT_RGB48_planar == format)
            info->color_bits = 16;
        else
            info->color_bits = 8;
        info->components = 3;
        info->strides [0] = info->strides [1] = info->strides [2] = aligned_width * (info->color_bits / 8);
        info->offsets [0] = 0;
        info->offsets [1] = info->offsets [0] + info->strides [0] * aligned_height;
        info->offsets [2] = info->offsets [1] + info->strides [1] * aligned_height;
        image_size = info->offsets [2] + info->strides [2] * aligned_height;
        break;

    case XCAM_PIX_FMT_SGRBG16_planar:
    case XCAM_PIX_FMT_SGRBG8_planar:
        if (XCAM_PIX_FMT_SGRBG16_planar == format)
            info->color_bits = 16;
        else
            info->color_bits = 8;
        info->components = 4;
        for (i = 0; i < info->components; ++i) {
            info->strides [i] = aligned_width * (info->color_bits / 8);
        }
        info->offsets [0] = 0;
        for (i = 1; i < info->components; ++i) {
            info->offsets [i] = info->offsets [i - 1] + info->strides [i - 1] * aligned_height;
        }
        image_size = info->offsets [info->components - 1] + info->strides [info->components - 1] * aligned_height;
        break;

    default:
        XCAM_LOG_WARNING ("XCamVideoBufferInfo reset failed, unsupported format:%s", xcam_fourcc_to_string (format));
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (!size)
        info->size = image_size;
    else {
        XCAM_ASSERT (size >= image_size);
        info->size = size;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
xcam_video_buffer_get_planar_info (
    const XCamVideoBufferInfo *buf_info,  XCamVideoBufferPlanarInfo *planar_info, const uint32_t index)
{
    XCAM_ASSERT (buf_info);
    XCAM_ASSERT (planar_info);

    planar_info->width = buf_info->width;
    planar_info->height = buf_info->height;
    planar_info->pixel_bytes = XCAM_ALIGN_UP (buf_info->color_bits, 8) / 8;

    switch (buf_info->format) {
    case V4L2_PIX_FMT_NV12:
        XCAM_ASSERT (index <= 1);
        if (index == 1) {
            planar_info->height = buf_info->height / 2;
        }
        break;

    case V4L2_PIX_FMT_GREY:
    case V4L2_PIX_FMT_YUYV:
    case V4L2_PIX_FMT_RGB565:
    case V4L2_PIX_FMT_SBGGR8:
    case V4L2_PIX_FMT_SGBRG8:
    case V4L2_PIX_FMT_SGRBG8:
    case V4L2_PIX_FMT_SRGGB8:
    case V4L2_PIX_FMT_SBGGR10:
    case V4L2_PIX_FMT_SGBRG10:
    case V4L2_PIX_FMT_SGRBG10:
    case V4L2_PIX_FMT_SRGGB10:
    case V4L2_PIX_FMT_SBGGR12:
    case V4L2_PIX_FMT_SGBRG12:
    case V4L2_PIX_FMT_SGRBG12:
    case V4L2_PIX_FMT_SRGGB12:
    case V4L2_PIX_FMT_SBGGR16:
    case XCAM_PIX_FMT_SGRBG16:
        XCAM_ASSERT (index <= 0);
        break;

    case V4L2_PIX_FMT_RGB24:
        XCAM_ASSERT (index <= 0);
        planar_info->pixel_bytes = 3;
        break;

    case V4L2_PIX_FMT_RGBA32:
    case V4L2_PIX_FMT_XBGR32:
    case V4L2_PIX_FMT_ABGR32:
    case V4L2_PIX_FMT_BGR32:
    case V4L2_PIX_FMT_RGB32:
    case V4L2_PIX_FMT_ARGB32:
    case V4L2_PIX_FMT_XRGB32:
        XCAM_ASSERT (index <= 0);
        planar_info->pixel_bytes = 4;
        break;

    case XCAM_PIX_FMT_RGB48:
        XCAM_ASSERT (index <= 0);
        planar_info->pixel_bytes = 3 * 2;
        break;

    case XCAM_PIX_FMT_RGBA64:
        planar_info->pixel_bytes = 4 * 2;
        break;

    case XCAM_PIX_FMT_LAB:
        planar_info->pixel_bytes = 3;
        break;

    case XCAM_PIX_FMT_RGB48_planar:
    case XCAM_PIX_FMT_RGB24_planar:
        XCAM_ASSERT (index <= 2);
        break;

    case XCAM_PIX_FMT_SGRBG16_planar:
    case XCAM_PIX_FMT_SGRBG8_planar:
        XCAM_ASSERT (index <= 3);
        break;

    default:
        XCAM_LOG_WARNING ("VideoBufferInfo get_planar_info failed, unsupported format:%s", xcam_fourcc_to_string (buf_info->format));
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}
