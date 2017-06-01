/*
 * cl_biyuv_handler.cpp - CL edge enhancement handler
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
 * Author: Juan Zhao <juan.j.zhao@intel.com>
 *             Wind Yuan <feng.yuan@intel.com>
 */

#include "xcam_utils.h"
#include "cl_bilateral_handler.h"

namespace XCam {

enum {
    KernelBilateralRGB,
    KernelBilateralNV12,
};

const XCamKernelInfo kernel_bilateral_info[] = {
    {
        "kernel_bilateral_rgb",
#include "kernel_bilateral.clx"
        , 0,
    },
    {
        "kernel_bilateral_nv12",
#include "kernel_bilateral.clx"
        , 0,
    },
};

CLBilateralKernel::CLBilateralKernel (const SmartPtr<CLContext> &context)
    : CLImageKernel (context)
{
}

CLBilateralImageHandler::CLBilateralImageHandler (
    const SmartPtr<CLContext> &context, const char *name, bool is_rgb)
    : CLImageHandler (context, name)
    , _is_rgb (is_rgb)
{
}

void
CLBilateralImageHandler::set_bi_kernel (SmartPtr<CLBilateralKernel> &kernel)
{
    XCAM_ASSERT (kernel.ptr () && kernel->is_valid ());
    _kernel = kernel;
    add_kernel (kernel);
}

XCamReturn
CLBilateralImageHandler::prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();
    uint32_t imw = video_info.width;
    uint32_t imh = video_info.height;
    uint32_t vertical_offset = video_info.aligned_height;
    float sigma_r = 10.0;
    CLArgList args;
    CLWorkSize work_size;

    XCAM_ASSERT (_kernel.ptr ());
    SmartPtr<CLImage> image_in = new CLVaImage (context, input);
    SmartPtr<CLImage> image_out = new CLVaImage (context, output);

    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image handler(%s) in/out memory not available", XCAM_STR(get_name ()));

    //set args;
    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLArgumentT<float> (sigma_r));
    args.push_back (new CLArgumentT<uint32_t> (imw));
    args.push_back (new CLArgumentT<uint32_t> (imh));

    if (!_is_rgb)
        args.push_back (new CLArgumentT<uint32_t> (vertical_offset));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = imh;
    work_size.global[1] = imw;
    work_size.local[0] = (imh + 71) / 72;
    work_size.local[1] = (imw + 119) / 120;

    XCAM_ASSERT (_kernel.ptr ());
    XCamReturn ret = _kernel->set_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "bilateral kernel set arguments failed.");

    return XCAM_RETURN_NO_ERROR;
}


SmartPtr<CLImageHandler>
create_cl_bilateral_image_handler (const SmartPtr<CLContext> &context, bool is_rgb)
{
    SmartPtr<CLBilateralImageHandler> bilateral_handler;
    SmartPtr<CLBilateralKernel> bilateral_kernel;
    const char *handler_name = (is_rgb ? "cl_bilateral_rgb" : "cl_bilateral_nv12");
    int kenel_idx = (is_rgb ? KernelBilateralRGB : KernelBilateralNV12);

    bilateral_handler = new CLBilateralImageHandler (context, handler_name, is_rgb);
    bilateral_kernel = new CLBilateralKernel (context);
    XCAM_ASSERT (bilateral_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, bilateral_kernel->build_kernel (kernel_bilateral_info[kenel_idx], NULL) == XCAM_RETURN_NO_ERROR,
        NULL, "build bilateral kernel failed");
    bilateral_handler->set_bi_kernel (bilateral_kernel);

    XCAM_ASSERT (bilateral_kernel->is_valid ());

    return bilateral_handler;
}

};
