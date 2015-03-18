/*
 * cl_memory.cpp - CL memory
 *
 *  Copyright (c) 2015 Intel Corporation
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

#include "cl_memory.h"
#include "drm_display.h"

namespace XCam {

CLMemory::CLMemory (SmartPtr<CLContext> &context)
    : _context (context)
    , _mem_id (NULL)
{
}

CLMemory::~CLMemory ()
{
    if (_mem_id) {
        _context->destroy_mem (_mem_id);
    }
}

uint32_t
CLVaImage::get_pixel_bytes (cl_image_format fmt)
{
    uint32_t a = 0, b = 0;
    switch (fmt.image_channel_order) {
    case CL_R:
    case CL_A:
    case CL_Rx:
        a = 1;
        break;
    case CL_RG:
    case CL_RA:
    case CL_RGx:
        a = 2;
        break;
    case CL_RGB:
    case CL_RGBx:
        a = 3;
        break;
    case CL_RGBA:
    case CL_BGRA:
    case CL_ARGB:
        a = 4;
        break;
    default:
        XCAM_LOG_DEBUG ("get_pixel_bytes with wrong channel_order:0x%04x", fmt.image_channel_order);
        return 0;
    }

    switch (fmt.image_channel_data_type) {
    case CL_UNORM_INT8:
    case CL_SNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
        b = 1;
        break;
    case CL_SNORM_INT16:
    case CL_UNORM_INT16:
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
    case CL_HALF_FLOAT:
        b = 2;
        break;
    case CL_UNORM_INT24:
        b = 3;
        break;
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
    case CL_FLOAT:
        b = 4;
        break;
    default:
        XCAM_LOG_DEBUG ("get_pixel_bytes with wrong channel_data_type:0x%04x", fmt.image_channel_data_type);
        return 0;
    }

    return a * b;
}

bool
CLVaImage::video_info_2_cl_image_info (
    const VideoBufferInfo & video_info,
    cl_libva_image &cl_image_info)
{
    cl_image_info.offset = 0;
    cl_image_info.width = video_info.width;
    cl_image_info.height = video_info.height;
    cl_image_info.row_pitch = video_info.strides[0];
    XCAM_ASSERT (cl_image_info.row_pitch >= cl_image_info.width);

    switch (video_info.format) {
    case XCAM_PIX_FMT_RGB48:
        //cl_image_info.fmt.image_channel_order = CL_RGB;
        //cl_image_info.fmt.image_channel_data_type = CL_UNORM_INT16;
        XCAM_LOG_WARNING (
            "video_info to cl_image_info doesn't support XCAM_PIX_FMT_RGB48, maybe try XCAM_PIX_FMT_RGBA64 instread\n"
            " **** XCAM_PIX_FMT_RGB48 need check with cl implementation ****");
        return false;
        break;

    case XCAM_PIX_FMT_RGBA64:
        cl_image_info.fmt.image_channel_order = CL_RGBA;
        cl_image_info.fmt.image_channel_data_type = CL_UNORM_INT16;
        break;

    case V4L2_PIX_FMT_RGB24:
        cl_image_info.fmt.image_channel_order = CL_RGB;
        cl_image_info.fmt.image_channel_data_type = CL_UNORM_INT8;
        break;

    case V4L2_PIX_FMT_RGB32:
        cl_image_info.fmt.image_channel_order = CL_RGBA;
        cl_image_info.fmt.image_channel_data_type = CL_UNORM_INT8;
        break;

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
        cl_image_info.fmt.image_channel_order = CL_R;
        cl_image_info.fmt.image_channel_data_type = CL_UNORM_INT16;
        break;

    case V4L2_PIX_FMT_SBGGR8:
    case V4L2_PIX_FMT_SGBRG8:
    case V4L2_PIX_FMT_SGRBG8:
    case V4L2_PIX_FMT_SRGGB8:
        cl_image_info.fmt.image_channel_order = CL_R;
        cl_image_info.fmt.image_channel_data_type = CL_UNORM_INT8;
        break;

    case V4L2_PIX_FMT_NV12:
        cl_image_info.fmt.image_channel_order = CL_R;
        cl_image_info.fmt.image_channel_data_type = CL_UNORM_INT8;
        cl_image_info.height *= 2;
        break;

    case V4L2_PIX_FMT_YUYV:
        cl_image_info.fmt.image_channel_order = CL_RGBA;
        cl_image_info.fmt.image_channel_data_type = CL_UNORM_INT8;
        cl_image_info.width /= 2;
        break;

    default:
        XCAM_LOG_WARNING (
            "video_info to cl_image_info doesn't support format:%s",
            xcam_fourcc_to_string (video_info.format));
        return false;
    }

#if 0
    pixel_bytes = get_pixel_bytes (cl_image_info.fmt);
    XCAM_FAIL_RETURN (
        WARNING,
        pixel_bytes,
        false,
        "video_info(%s) to cl_image_info failed to get pixel_bytes",
        xcam_fourcc_to_string (video_info.format));
#endif

    return true;
}

CLVaImage::CLVaImage (
    SmartPtr<CLContext> &context,
    SmartPtr<DrmBoBuffer> &bo,
    const cl_libva_image *image_info)
    : CLMemory (context)
    , _bo (bo)
{
    uint32_t bo_name = 0;

    XCAM_ASSERT (context.ptr () && context->is_valid ());

    if (image_info) {
        _image_info = *image_info;
    } else {
        const VideoBufferInfo & video_info = bo->get_video_info ();
        xcam_mem_clear (&_image_info);
        if (!video_info_2_cl_image_info (video_info, _image_info)) {
            XCAM_LOG_WARNING ("CLVaImage create va image failed on default videoinfo");
            return;
        }
    }

    if (drm_intel_bo_flink (bo->get_bo (), &bo_name) != 0) {
        XCAM_LOG_WARNING ("CLVaImage get bo flick failed");
    } else {
        _image_info.bo_name = bo_name;
        _mem_id = context->create_va_image (_image_info);
        if (_mem_id == NULL) {
            XCAM_LOG_WARNING ("create va image failed");
        }
    }
}

};
