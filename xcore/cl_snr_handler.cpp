/*
 * cl_snr_handler.cpp - CL simple noise reduction handler
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
 * Author: Shincy Tu <shincy.tu@intel.com>
 */
#include "xcam_utils.h"
#include "cl_snr_handler.h"

namespace XCam {

CLSnrImageKernel::CLSnrImageKernel (SmartPtr<CLContext> &context,
                                    const char *name)
    : CLImageKernel (context, name, false)
{
}

XCamReturn
CLSnrImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    //const VideoBufferInfo & video_info = input->get_video_info ();

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
    arg_count = 2;

    const CLImageDesc out_info = _image_out->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width;
    work_size.global[1] = out_info.height;
    //printf("out_info.width = %u, out_info.height = %u\n", out_info.width, out_info.height);
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

CLSnrImageHandler::CLSnrImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLSnrImageHandler::set_simple_kernel(SmartPtr<CLSnrImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _simple_kernel = kernel;
    return true;
}

bool
CLSnrImageHandler::set_mode (uint32_t mode)
{
    _simple_kernel->set_enable (mode == CL_DENOISE_TYPE_SIMPLE);

    return true;
}

SmartPtr<CLImageHandler>
create_cl_snr_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLSnrImageHandler> snr_handler;
    SmartPtr<CLSnrImageKernel> snr_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    snr_kernel = new CLSnrImageKernel (context, "kernel_snr");
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_snr)
#include "kernel_snr.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = snr_kernel->load_from_source (kernel_snr_body, strlen (kernel_snr_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", snr_kernel->get_kernel_name());
    }
    XCAM_ASSERT (snr_kernel->is_valid ());
    snr_handler = new CLSnrImageHandler ("cl_handler_snr");
    snr_handler->set_simple_kernel  (snr_kernel);

    return snr_handler;
}

};
