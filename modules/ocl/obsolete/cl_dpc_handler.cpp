/*
 * cl_dpc_handler.cpp - CL defect pixel correction handler
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
#include <xcam_std.h>
#include "cl_dpc_handler.h"

namespace XCam {

CLDpcImageKernel::CLDpcImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_dpc")
{
    _dpc_config.gain = XCAM_CL_DPC_DEFAULT_GAIN;
    _dpc_config.r_threshold = XCAM_CL_DPC_DEFAULT_THRESHOLD;
    _dpc_config.gr_threshold = XCAM_CL_DPC_DEFAULT_THRESHOLD;
    _dpc_config.gb_threshold = XCAM_CL_DPC_DEFAULT_THRESHOLD;
    _dpc_config.b_threshold = XCAM_CL_DPC_DEFAULT_THRESHOLD;
}

XCamReturn
CLDpcImageKernel::prepare_arguments (
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
    args[2].arg_adress = &_dpc_config.gr_threshold;
    args[2].arg_size = sizeof (cl_float);
    args[3].arg_adress = &_dpc_config.r_threshold;
    args[3].arg_size = sizeof (cl_float);
    args[4].arg_adress = &_dpc_config.b_threshold;
    args[4].arg_size = sizeof (cl_float);
    args[5].arg_adress = &_dpc_config.gb_threshold;
    args[5].arg_size = sizeof (cl_float);
    arg_count = 6;

    const CLImageDesc out_info = _image_out->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width;
    work_size.global[1] = out_info.height;
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

bool
CLDpcImageKernel::set_dpc (CLDPCConfig dpc)
{
    _dpc_config = dpc;
    return true;
}
CLDpcImageHandler::CLDpcImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLDpcImageHandler::set_dpc_config (const XCam3aResultDefectPixel &dpc)
{
    CLDPCConfig _dpc_config;
    _dpc_config.gain = 0.0f;
    _dpc_config.gr_threshold = (float)dpc.gr_threshold;
    _dpc_config.r_threshold = (float)dpc.r_threshold;
    _dpc_config.b_threshold = (float)dpc.b_threshold;
    _dpc_config.gb_threshold = (float)dpc.gb_threshold;
    _dpc_kernel->set_dpc(_dpc_config);
    return true;
}

bool
CLDpcImageHandler::set_dpc_kernel(SmartPtr<CLDpcImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _dpc_kernel = kernel;
    return true;
}

SmartPtr<CLImageHandler>
create_cl_dpc_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLDpcImageHandler> dpc_handler;
    SmartPtr<CLDpcImageKernel> dpc_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    dpc_kernel = new CLDpcImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_dpc)
#include "kernel_dpc.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = dpc_kernel->load_from_source (kernel_dpc_body, strlen (kernel_dpc_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", dpc_kernel->get_kernel_name());
    }
    XCAM_ASSERT (dpc_kernel->is_valid ());
    dpc_handler = new CLDpcImageHandler ("cl_handler_dpc");
    dpc_handler->set_dpc_kernel (dpc_kernel);

    return dpc_handler;
}

};
