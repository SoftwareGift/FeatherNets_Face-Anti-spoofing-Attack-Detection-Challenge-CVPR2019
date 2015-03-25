/*
 * cl_3a_image_processor.cpp - CL 3A image processor
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
#include "cl_3a_image_processor.h"
#include "cl_context.h"
#include "cl_blc_handler.h"
#include "cl_demosaic_handler.h"
#include "cl_csc_handler.h"
#include "cl_hdr_handler.h"

namespace XCam {

CL3aImageProcessor::CL3aImageProcessor ()
    : CLImageProcessor ("CL3aImageProcessor")
    , _output_fourcc (V4L2_PIX_FMT_NV12)
    , _enable_hdr (false)
{
    XCAM_LOG_DEBUG ("CL3aImageProcessor constructed");
}

CL3aImageProcessor::~CL3aImageProcessor ()
{
    XCAM_LOG_DEBUG ("CL3aImageProcessor destructed");
}

bool
CL3aImageProcessor::set_output_format (uint32_t fourcc)
{
    XCAM_FAIL_RETURN (
        WARNING,
        fourcc == V4L2_PIX_FMT_NV12 || fourcc == V4L2_PIX_FMT_RGB32,
        false,
        "CL3aImageProcessor doesn't support format(%s) output",
        xcam_fourcc_to_string (fourcc));

    _output_fourcc = fourcc;
    return true;
}

bool
CL3aImageProcessor::can_process_result (SmartPtr<X3aResult> &result)
{
    XCAM_UNUSED (result);
    return false;
}

XCamReturn
CL3aImageProcessor::apply_3a_results (X3aResultList &results)
{
    XCAM_UNUSED (results);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CL3aImageProcessor::apply_3a_result (SmartPtr<X3aResult> &result)
{
    XCAM_UNUSED (result);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CL3aImageProcessor::create_handlers ()
{
    SmartPtr<CLImageHandler> image_handler;
    SmartPtr<CLContext> context = get_cl_context ();

    XCAM_ASSERT (context.ptr ());

    /* black leve as first */
    image_handler = create_cl_blc_image_handler (context);
    _black_level = image_handler;
    XCAM_FAIL_RETURN (
        WARNING,
        image_handler.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create blc handler failed");
    add_handler (image_handler);

    /* demosaic */
    image_handler = create_cl_demosaic_image_handler (context);
    _demosaic = image_handler.dynamic_cast_ptr<CLBayer2RGBImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _demosaic.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create demosaic handler failed");
    add_handler (image_handler);

    /* hdr */
    if (_enable_hdr) {
        image_handler = create_cl_hdr_image_handler (context);
        _hdr = image_handler;
        XCAM_FAIL_RETURN (
            WARNING,
            _hdr.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create hdr handler failed");
        add_handler (image_handler);
    }

    /* color space conversion */
    if (_output_fourcc == V4L2_PIX_FMT_NV12) {
        image_handler = create_cl_csc_image_handler (context);
        _csc = image_handler.dynamic_cast_ptr<CLRgba2Nv12ImageHandler> ();
        XCAM_FAIL_RETURN (
            WARNING,
            _csc .ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create csc handler failed");
        add_handler (image_handler);
    }

    return XCAM_RETURN_NO_ERROR;
}

};
