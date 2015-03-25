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
    uint32_t alignment_w, uint32_t alignment_h)
{
    XCAM_ASSERT ((alignment_w & (alignment_w - 1)) == 0 && alignment_w != 0);
    XCAM_ASSERT ((alignment_h & (alignment_h - 1)) == 0 && alignment_h != 0);

    uint32_t final_width = XCAM_ALIGN_UP (width, alignment_w);
    uint32_t final_height = XCAM_ALIGN_UP (height, alignment_h);

    this->format = format;
    this->width = width;
    this->height = height;
    this->aligned_width = final_width;
    this->aligned_height = final_height;

    switch (format) {
    case V4L2_PIX_FMT_NV12:
        this->color_bits = 8;
        this->components = 2;
        this->strides [0] = final_width;
        this->strides [1] = this->strides [0];
        this->offsets [0] = 0;
        this->offsets [1] = this->offsets [0] + this->strides [0] * final_height;
        this->size = this->strides [0] * final_height + this->strides [1] * final_height / 2;
        break;
    case V4L2_PIX_FMT_YUYV:
        this->color_bits = 8;
        this->components = 1;
        this->strides [0] = final_width * 2;
        this->offsets [0] = 0;
        this->size = this->strides [0] * final_height;
        break;
    case V4L2_PIX_FMT_RGB24:
        this->color_bits = 8;
        this->components = 1;
        this->strides [0] = final_width * 3;
        this->offsets [0] = 0;
        this->size = this->strides [0] * final_height;
        break;
    case V4L2_PIX_FMT_RGB32:
        this->color_bits = 8;
        this->components = 1;
        this->strides [0] = final_width * 4;
        this->offsets [0] = 0;
        this->size = this->strides [0] * final_height;
        break;
    case XCAM_PIX_FMT_RGB48:
        this->color_bits = 16;
        this->components = 1;
        this->strides [0] = final_width * 3 * 2;
        this->offsets [0] = 0;
        this->size = this->strides [0] * final_height;
        break;
    case XCAM_PIX_FMT_RGBA64:
        this->color_bits = 16;
        this->components = 1;
        this->strides [0] = final_width * 4 * 2;
        this->offsets [0] = 0;
        this->size = this->strides [0] * final_height;
        break;

    case V4L2_PIX_FMT_SBGGR8:
    case V4L2_PIX_FMT_SGBRG8:
    case V4L2_PIX_FMT_SGRBG8:
    case V4L2_PIX_FMT_SRGGB8:
        this->color_bits = 8;
        this->components = 1;
        this->strides [0] = final_width;
        this->offsets [0] = 0;
        this->size = this->strides [0] * final_height;
        break;

    case V4L2_PIX_FMT_SBGGR10:
    case V4L2_PIX_FMT_SGBRG10:
    case V4L2_PIX_FMT_SGRBG10:
    case V4L2_PIX_FMT_SRGGB10:
        this->color_bits = 10;
        this->components = 1;
        this->strides [0] = final_width * 2;
        this->offsets [0] = 0;
        this->size = this->strides [0] * final_height;
        break;

    case V4L2_PIX_FMT_SBGGR12:
    case V4L2_PIX_FMT_SGBRG12:
    case V4L2_PIX_FMT_SGRBG12:
    case V4L2_PIX_FMT_SRGGB12:
        this->color_bits = 12;
        this->components = 1;
        this->strides [0] = final_width * 2;
        this->offsets [0] = 0;
        this->size = this->strides [0] * final_height;
        break;

    case V4L2_PIX_FMT_SBGGR16:
    case XCAM_PIX_FMT_SGRBG16:
        this->color_bits = 16;
        this->components = 1;
        this->strides [0] = final_width * 2;
        this->offsets [0] = 0;
        this->size = this->strides [0] * final_height;
        break;
    default:
        XCAM_LOG_WARNING ("VideoBufferInfo init failed, unsupported format:%s", xcam_fourcc_to_string (format));
        return false;
    }

    return true;
}

};
