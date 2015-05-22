/*
 * cl_hdr_handler.cpp - CL hdr handler
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
 * Author: wangfei <feix.w.wang@intel.com>
 */
#include "xcam_utils.h"
#include "cl_hdr_handler.h"

namespace XCam {

CLHdrImageKernel::CLHdrImageKernel (SmartPtr<CLContext> &context,
                                    const char *name,
                                    CLHdrType type)
    : CLImageKernel (context, name, false)
    , _type (type)
{
}

CLHdrImageHandler::CLHdrImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLHdrImageHandler::set_rgb_kernel(SmartPtr<CLHdrImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _rgb_kernel = kernel;
    return true;
}

bool
CLHdrImageHandler::set_lab_kernel(SmartPtr<CLHdrImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _lab_kernel = kernel;
    return true;
}

bool
CLHdrImageHandler::set_mode (uint32_t mode)
{
    _rgb_kernel->set_enable (mode == CL_HDR_TYPE_RGB);
    _lab_kernel->set_enable (mode == CL_HDR_TYPE_LAB);

    return true;
}

SmartPtr<CLImageHandler>
create_cl_hdr_image_handler (SmartPtr<CLContext> &context, CLHdrType type)
{
    SmartPtr<CLHdrImageHandler> hdr_handler;
    SmartPtr<CLHdrImageKernel> hdr_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    hdr_handler = new CLHdrImageHandler ("cl_handler_hdr");

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_hdr_rgb)
#include "kernel_hdr_rgb.clx"
    XCAM_CL_KERNEL_FUNC_END;
    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_hdr_lab)
#include "kernel_hdr_lab.clx"
    XCAM_CL_KERNEL_FUNC_END;

    hdr_kernel = new CLHdrImageKernel (context, "kernel_hdr_rgb", CL_HDR_TYPE_RGB);
    ret = hdr_kernel->load_from_source (kernel_hdr_rgb_body, strlen (kernel_hdr_rgb_body));
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CL image handler(%s) load source failed", hdr_kernel->get_kernel_name());

    XCAM_ASSERT (hdr_kernel->is_valid ());
    hdr_handler->set_rgb_kernel (hdr_kernel);

    hdr_kernel = new CLHdrImageKernel (context, "kernel_hdr_lab", CL_HDR_TYPE_LAB);
    ret = hdr_kernel->load_from_source (kernel_hdr_lab_body, strlen (kernel_hdr_lab_body));
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CL image handler(%s) load source failed", hdr_kernel->get_kernel_name());

    XCAM_ASSERT (hdr_kernel->is_valid ());
    hdr_handler->set_lab_kernel (hdr_kernel);

    hdr_handler->set_mode (type);
    return hdr_handler;
}

};
