/*
 * cl_denoise_handler.cpp - CL denoise handler
 *
 *  Copyright (c) 2015 Intel Corporation
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
 */



#include "xcam_utils.h"
#include "cl_denoise_handler.h"

namespace XCam {

CLDenoiseImageKernel::CLDenoiseImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_denoise")
    , _sigma_r (10.0)
    , _imw (1920)
    , _imh (1080)
{
}

XCamReturn
CLDenoiseImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();

    _imw = video_info.width;
    _imh = (video_info.size / video_info.strides[0]) / 4 * 4;
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
    arg_count = 5;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = _imh;
    work_size.global[1] = _imw;

    return XCAM_RETURN_NO_ERROR;
}


SmartPtr<CLImageHandler>
create_cl_denoise_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> denoise_handler;
    SmartPtr<CLImageKernel> denoise_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    denoise_kernel = new CLDenoiseImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_denoise)
#include "kernel_denoise.cl"
        XCAM_CL_KERNEL_FUNC_END;
        ret = denoise_kernel->load_from_source (kernel_denoise_body, strlen (kernel_denoise_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", denoise_kernel->get_kernel_name());
    }
    XCAM_ASSERT (denoise_kernel->is_valid ());
    denoise_handler = new CLImageHandler ("cl_handler_denoise");
    denoise_handler->add_kernel  (denoise_kernel);

    return denoise_handler;
}

};
