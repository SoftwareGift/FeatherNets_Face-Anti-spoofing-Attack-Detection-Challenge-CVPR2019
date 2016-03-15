/*
 * cl_post_image_processor.cpp - CL post image processor
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "cl_post_image_processor.h"
#include "cl_context.h"

#include "cl_retinex_handler.h"
#include "cl_csc_handler.h"

#define XCAM_CL_POST_IMAGE_MAX_POOL_SIZE 6

namespace XCam {

CLPostImageProcessor::CLPostImageProcessor ()
    : CLImageProcessor ("CLPostImageProcessor")
    , _output_fourcc (V4L2_PIX_FMT_NV12)
    , _out_sample_type (OutSampleYuv)
    , _enable_retinex (false)
{
    XCAM_LOG_DEBUG ("CLPostImageProcessor constructed");
}

CLPostImageProcessor::~CLPostImageProcessor ()
{
    XCAM_LOG_DEBUG ("CLPostImageProcessor destructed");
}

bool
CLPostImageProcessor::set_output_format (uint32_t fourcc)
{
    switch (fourcc) {
    case XCAM_PIX_FMT_RGBA64:
    case V4L2_PIX_FMT_XBGR32:
    case V4L2_PIX_FMT_ABGR32:
    case V4L2_PIX_FMT_BGR32:
    case V4L2_PIX_FMT_RGB32:
    case V4L2_PIX_FMT_ARGB32:
    case V4L2_PIX_FMT_XRGB32:
        _out_sample_type = OutSampleRGB;
        break;
    case V4L2_PIX_FMT_NV12:
        _out_sample_type = OutSampleYuv;
        break;
    default:
        XCAM_LOG_WARNING (
            "cl post processor doesn't support output format: %s",
            xcam_fourcc_to_string(fourcc));
        return false;
    }

    _output_fourcc = fourcc;
    return true;
}

XCamReturn
CLPostImageProcessor::create_handlers ()
{
    SmartPtr<CLImageHandler> image_handler;
    SmartPtr<CLContext> context = get_cl_context ();

    XCAM_ASSERT (context.ptr ());

    /* retinex*/
    image_handler = create_cl_retinex_image_handler (context);
    _retinex = image_handler.dynamic_cast_ptr<CLRetinexImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _retinex.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create retinex handler failed");
    _retinex->set_kernels_enable (_enable_retinex);
    image_handler->set_pool_type (CLImageHandler::DrmBoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    image_handler = create_cl_csc_image_handler (context, CL_CSC_TYPE_NV12TORGBA);
    _csc = image_handler.dynamic_cast_ptr<CLCscImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _csc .ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create csc handler failed");
    _csc->set_kernels_enable (_out_sample_type == OutSampleRGB);
    image_handler->set_pool_type (CLImageHandler::DrmBoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    return XCAM_RETURN_NO_ERROR;
}

bool
CLPostImageProcessor::set_retinex (bool enable)
{
    _enable_retinex = enable;

    STREAM_LOCK;

    return true;
}

};
