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

SmartPtr<CLImageHandler>
create_cl_hdr_image_handler (SmartPtr<CLContext> &context, CLHdrType type)
{
    SmartPtr<CLImageHandler> hdr_handler;
    SmartPtr<CLImageKernel> hdr_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_hdr_rgb)
#include "kernel_hdr_rgb.cl"
    XCAM_CL_KERNEL_FUNC_END;
    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_hdr_lab)
#include "kernel_hdr_lab.cl"
    XCAM_CL_KERNEL_FUNC_END;

    if (type == CL_HDR_TYPE_RGB) {
        hdr_kernel = new CLImageKernel (context, "kernel_hdr_rgb");
        ret = hdr_kernel->load_from_source (kernel_hdr_rgb_body, strlen (kernel_hdr_rgb_body));
    }
    else if (type == CL_HDR_TYPE_LAB) {
        hdr_kernel = new CLImageKernel (context, "kernel_hdr_lab");
        ret = hdr_kernel->load_from_source (kernel_hdr_lab_body, strlen (kernel_hdr_lab_body));
    }
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CL image handler(%s) load source failed", hdr_kernel->get_kernel_name());

    XCAM_ASSERT (hdr_kernel->is_valid ());
    hdr_handler = new CLImageHandler ("cl_handler_hdr");
    hdr_handler->add_kernel  (hdr_kernel);

    return hdr_handler;
}

};
