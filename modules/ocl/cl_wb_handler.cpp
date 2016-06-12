/*
 * cl_wb_handler.cpp - CL white balance handler
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
#include "cl_wb_handler.h"

namespace XCam {

CLWbImageKernel::CLWbImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_wb")
{
    _wb_config.r_gain = 1.0;
    _wb_config.gr_gain = 1.0;
    _wb_config.gb_gain = 1.0;
    _wb_config.b_gain = 1.0;
}

XCamReturn
CLWbImageKernel::prepare_arguments (
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
    args[2].arg_adress = &_wb_config;
    args[2].arg_size = sizeof (CLWBConfig);
    arg_count = 3;

    const CLImageDesc out_info = _image_out->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width / 2;
    work_size.global[1] = out_info.height / 2;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

bool
CLWbImageKernel::set_wb (CLWBConfig wb)
{
    _wb_config = wb;
    return true;
}
CLWbImageHandler::CLWbImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLWbImageHandler::set_wb_config (const XCam3aResultWhiteBalance &wb)
{
    CLWBConfig _wb_config;
    _wb_config.r_gain = (float)wb.r_gain;
    _wb_config.gr_gain = (float)wb.gr_gain;
    _wb_config.gb_gain = (float)wb.gb_gain;
    _wb_config.b_gain = (float)wb.b_gain;
    _wb_kernel->set_wb(_wb_config);
    return true;
}

bool
CLWbImageHandler::set_wb_kernel(SmartPtr<CLWbImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _wb_kernel = kernel;
    return true;
}


SmartPtr<CLImageHandler>
create_cl_wb_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLWbImageHandler> wb_handler;
    SmartPtr<CLWbImageKernel> wb_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    wb_kernel = new CLWbImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_wb)
#include "kernel_wb.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = wb_kernel->load_from_source (kernel_wb_body, strlen (kernel_wb_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", wb_kernel->get_kernel_name());
    }
    XCAM_ASSERT (wb_kernel->is_valid ());
    wb_handler = new CLWbImageHandler ("cl_handler_wb");
    wb_handler->set_wb_kernel (wb_kernel);

    return wb_handler;
}

}
