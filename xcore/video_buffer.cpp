/*
 * video_buffer.cpp - video buffer base
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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

#include "video_buffer.h"
#include <linux/videodev2.h>

namespace XCam {

VideoBufferPlanarInfo::VideoBufferPlanarInfo ()
    : width (0)
    , height (0)
    , pixel_bytes (0)
{ }

VideoBufferInfo::VideoBufferInfo ()
    : format (0)
    , color_bits (8)
    , width (0)
    , height (0)
    , aligned_width (0)
    , aligned_height (0)
    , size (0)
    , components (0)
{
    xcam_mem_clear (strides);
    xcam_mem_clear (offsets);
}

bool
VideoBufferInfo::init (
    uint32_t format,
    uint32_t width, uint32_t height,
    uint32_t aligned_width, uint32_t aligned_height,
    uint32_t size)
{
    uint32_t image_size = 0;
    uint32_t i = 0;

    XCAM_ASSERT (!aligned_width  || aligned_width >= width);
    XCAM_ASSERT (!aligned_height  || aligned_height >= height);

    if (!aligned_width)
        aligned_width = XCAM_ALIGN_UP (width, 4);
    if (!aligned_height)
        aligned_height = XCAM_ALIGN_UP (height, 2);

    this->format = format;
    this->width = width;
    this->height = height;
    this->aligned_width = aligned_width;
    this->aligned_height = aligned_height;

    switch (format) {
    case V4L2_PIX_FMT_NV12:
        this->color_bits = 8;
        this->components = 2;
        this->strides [0] = aligned_width;
        this->strides [1] = this->strides [0];
        this->offsets [0] = 0;
        this->offsets [1] = this->offsets [0] + this->strides [0] * aligned_height;
        image_size = this->strides [0] * aligned_height + this->strides [1] * aligned_height / 2;
        break;
    case V4L2_PIX_FMT_YUYV:
        this->color_bits = 8;
        this->components = 1;
        this->strides [0] = aligned_width * 2;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;
    case V4L2_PIX_FMT_RGB565:
        this->color_bits = 16;
        this->components = 1;
        this->strides [0] = aligned_width * 2;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;
    case V4L2_PIX_FMT_RGB24:
        this->color_bits = 8;
        this->components = 1;
        this->strides [0] = aligned_width * 3;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
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
        this->color_bits = 8;
        this->components = 1;
        this->strides [0] = aligned_width * 4;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;
    case XCAM_PIX_FMT_RGB48:
        this->color_bits = 16;
        this->components = 1;
        this->strides [0] = aligned_width * 3 * 2;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;
    case XCAM_PIX_FMT_RGBA64:
        this->color_bits = 16;
        this->components = 1;
        this->strides [0] = aligned_width * 4 * 2;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;

    case V4L2_PIX_FMT_SBGGR8:
    case V4L2_PIX_FMT_SGBRG8:
    case V4L2_PIX_FMT_SGRBG8:
    case V4L2_PIX_FMT_SRGGB8:
        this->color_bits = 8;
        this->components = 1;
        this->strides [0] = aligned_width;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;

    case V4L2_PIX_FMT_SBGGR10:
    case V4L2_PIX_FMT_SGBRG10:
    case V4L2_PIX_FMT_SGRBG10:
    case V4L2_PIX_FMT_SRGGB10:
        this->color_bits = 10;
        this->components = 1;
        this->strides [0] = aligned_width * 2;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;

    case V4L2_PIX_FMT_SBGGR12:
    case V4L2_PIX_FMT_SGBRG12:
    case V4L2_PIX_FMT_SGRBG12:
    case V4L2_PIX_FMT_SRGGB12:
        this->color_bits = 12;
        this->components = 1;
        this->strides [0] = aligned_width * 2;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;

    case V4L2_PIX_FMT_SBGGR16:
    case XCAM_PIX_FMT_SGRBG16:
        this->color_bits = 16;
        this->components = 1;
        this->strides [0] = aligned_width * 2;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;

    case XCAM_PIX_FMT_LAB:
        this->color_bits = 32;
        this->components = 1;
        this->strides [0] = aligned_width * 3 * 4;
        this->offsets [0] = 0;
        image_size = this->strides [0] * aligned_height;
        break;

    case XCAM_PIX_FMT_RGB48_planar:
    case XCAM_PIX_FMT_RGB24_planar:
        if (XCAM_PIX_FMT_RGB48_planar == format)
            this->color_bits = 16;
        else
            this->color_bits = 8;
        this->components = 3;
        this->strides [0] = this->strides [1] = this->strides [2] = aligned_width * (this->color_bits / 8);
        this->offsets [0] = 0;
        this->offsets [1] = this->offsets [0] + this->strides [0] * aligned_height;
        this->offsets [2] = this->offsets [1] + this->strides [1] * aligned_height;
        image_size = this->offsets [2] + this->strides [2] * aligned_height;
        break;

    case XCAM_PIX_FMT_SGRBG16_planar:
    case XCAM_PIX_FMT_SGRBG8_planar:
        if (XCAM_PIX_FMT_SGRBG16_planar == format)
            this->color_bits = 16;
        else
            this->color_bits = 8;
        this->components = 4;
        for (i = 0; i < this->components; ++i) {
            this->strides [i] = aligned_width * (this->color_bits / 8);
        }
        this->offsets [0] = 0;
        for (i = 1; i < this->components; ++i) {
            this->offsets [i] = this->offsets [i - 1] + this->strides [i - 1] * aligned_height;
        }
        image_size = this->offsets [this->components - 1] + this->strides [this->components - 1] * aligned_height;
        break;

    default:
        XCAM_LOG_WARNING ("VideoBufferInfo init failed, unsupported format:%s", xcam_fourcc_to_string (format));
        return false;
    }

    if (!size)
        this->size = image_size;
    else {
        XCAM_ASSERT (size >= image_size);
        this->size = size;
    }

    return true;
}

bool
VideoBufferInfo::get_planar_info (
    VideoBufferPlanarInfo &planar, const uint32_t index) const
{
    planar.width = this->width;
    planar.height = this->height;
    planar.pixel_bytes = XCAM_ALIGN_UP (this->color_bits, 8) / 8;

    switch (format) {
    case V4L2_PIX_FMT_NV12:
        XCAM_ASSERT (index <= 1);
        if (index == 1) {
            planar.height = this->height / 2;
        }
        break;

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
        planar.pixel_bytes = 3;
        break;

    case V4L2_PIX_FMT_RGBA32:
    case V4L2_PIX_FMT_XBGR32:
    case V4L2_PIX_FMT_ABGR32:
    case V4L2_PIX_FMT_BGR32:
    case V4L2_PIX_FMT_RGB32:
    case V4L2_PIX_FMT_ARGB32:
    case V4L2_PIX_FMT_XRGB32:
        XCAM_ASSERT (index <= 0);
        planar.pixel_bytes = 4;
        break;

    case XCAM_PIX_FMT_RGB48:
        XCAM_ASSERT (index <= 0);
        planar.pixel_bytes = 3 * 2;
        break;

    case XCAM_PIX_FMT_RGBA64:
        planar.pixel_bytes = 4 * 2;
        break;

    case XCAM_PIX_FMT_LAB:
        planar.pixel_bytes = 3 * 4;
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
        XCAM_LOG_WARNING ("VideoBufferInfo get_planar_info failed, unsupported format:%s", xcam_fourcc_to_string (format));
        return false;
    }

    return true;
}

};
