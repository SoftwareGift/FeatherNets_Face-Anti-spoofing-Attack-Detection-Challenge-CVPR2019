/*
 * cl_bnr_handler.cpp - CL bayer noise reduction handler
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
#include "cl_bnr_handler.h"

namespace XCam {

CLBnrImageKernel::CLBnrImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_bnr")
{
    _bnr_config.bnr_gain = XCAM_CL_BNR_GAIN_DEFAULT;
    _bnr_config.direction = XCAM_CL_BNR_DIRECTION_DEFAULT;
}

XCamReturn
CLBnrImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

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
    args[2].arg_adress = &_bnr_config.bnr_gain;
    args[2].arg_size = sizeof (cl_float);
    args[3].arg_adress = &_bnr_config.direction;
    args[3].arg_size = sizeof (cl_float);
    arg_count = 4;

    const CLImageDesc out_info = _image_out->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width;
    work_size.global[1] = out_info.height;
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

bool
CLBnrImageKernel::set_bnr (CLBNRConfig bnr)
{
    _bnr_config = bnr;
    return true;
}

CLBnrImageHandler::CLBnrImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLBnrImageHandler::set_bnr_config (const XCam3aResultBayerNoiseReduction &bnr)
{
    CLBNRConfig _bnr_config;
    _bnr_config.bnr_gain = (float)bnr.bnr_gain;
    _bnr_config.direction = (float)bnr.direction;
    _simple_kernel->set_bnr(_bnr_config);
    return true;
}

bool
CLBnrImageHandler::set_simple_kernel(SmartPtr<CLBnrImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _simple_kernel = kernel;
    return true;
}

SmartPtr<CLImageHandler>
create_cl_bnr_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLBnrImageHandler> bnr_handler;
    SmartPtr<CLBnrImageKernel> bnr_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    bnr_kernel = new CLBnrImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_bnr)
#include "kernel_bnr.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = bnr_kernel->load_from_source (kernel_bnr_body, strlen (kernel_bnr_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", bnr_kernel->get_kernel_name());
    }
    XCAM_ASSERT (bnr_kernel->is_valid ());
    bnr_handler = new CLBnrImageHandler ("cl_handler_bnr");
    bnr_handler->set_simple_kernel (bnr_kernel);

    return bnr_handler;
}

};
