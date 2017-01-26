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

CLBilateralKernel::CLBilateralKernel (SmartPtr<CLContext> &context, bool is_rgb)
    : CLImageKernel (context)
    , _sigma_r (10.0)
    , _imw (1920)
    , _imh (1080)
    , _vertical_offset (1080)
    , _is_rgb (is_rgb)
{
}

XCamReturn
CLBilateralKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();

    _imw = video_info.width;
    _imh = video_info.height;
    //sigma_r = 0.1*100
    _sigma_r = 10.0;

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    _vertical_offset = video_info.aligned_height;

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_sigma_r;
    args[2].arg_size = sizeof (_sigma_r);
    args[3].arg_adress = &_imw;
    args[3].arg_size = sizeof (_imw);
    args[4].arg_adress = &_imh;
    args[4].arg_size = sizeof (_imh);

    if (_is_rgb)
        arg_count = 5;
    else {
        args[5].arg_adress = &_vertical_offset;
        args[5].arg_size = sizeof (_vertical_offset);
        arg_count = 6;
    }

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = _imh;
    work_size.global[1] = _imw;
    work_size.local[0] = _imh / 72;
    work_size.local[1] = _imw / 120;


    return XCAM_RETURN_NO_ERROR;
}

CLBilateralImageHandler::CLBilateralImageHandler (const char *name)
    : CLImageHandler (name)
{
}

SmartPtr<CLImageHandler>
create_cl_bilateral_image_handler (SmartPtr<CLContext> &context, bool is_rgb)
{
    SmartPtr<CLBilateralImageHandler> bilateral_handler;
    SmartPtr<CLImageKernel> bilateral_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    const char *handler_name = (is_rgb ? "cl_bilateral_rgb" : "cl_bilateral_nv12");
    int kenel_idx = (is_rgb ? KernelBilateralRGB : KernelBilateralNV12);

    bilateral_handler = new CLBilateralImageHandler (handler_name);
    bilateral_kernel = new CLBilateralKernel (context, is_rgb);
    XCAM_ASSERT (bilateral_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, bilateral_kernel->build_kernel (kernel_bilateral_info[kenel_idx], NULL) == XCAM_RETURN_NO_ERROR,
        NULL, "build bilateral kernel failed");
    bilateral_handler->add_kernel (bilateral_kernel);

    XCAM_ASSERT (bilateral_kernel->is_valid ());

    return bilateral_handler;
}

};
