/*
 * cl_demo_handler.cpp - CL demo handler
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
#include "xcam_utils.h"
#include "cl_demo_handler.h"

namespace XCam {

CLDemoImageKernel::CLDemoImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_demo")
{
}

XCamReturn
CLDemoImageKernel::prepare_arguments (
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
    arg_count = 2;

    const CLImageDesc out_info = _image_out->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width;
    work_size.global[1] = out_info.height;
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLDemoImageKernel::post_execute ()
{
    return CLImageKernel::post_execute ();
}

SmartPtr<CLImageHandler>
create_cl_demo_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> demo_handler;
    SmartPtr<CLImageKernel> demo_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    demo_kernel = new CLDemoImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_demo)
#include "kernel_demo.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = demo_kernel->load_from_source (kernel_demo_body, strlen (kernel_demo_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", demo_kernel->get_kernel_name());
    }
    XCAM_ASSERT (demo_kernel->is_valid ());
    demo_handler = new CLImageHandler ("cl_handler_demo");
    demo_handler->add_kernel  (demo_kernel);

    return demo_handler;
}

SmartPtr<CLImageHandler>
create_cl_binary_demo_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> demo_handler;
    SmartPtr<CLImageKernel> demo_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    demo_kernel = new CLDemoImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_BINARY_BEGIN(kernel_demo)
#include "kernel_demo.clx.bin"
        XCAM_CL_KERNEL_FUNC_END;
        ret = demo_kernel->load_from_binary (kernel_demo_body, sizeof (kernel_demo_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load binary failed", demo_kernel->get_kernel_name());
    }
    XCAM_ASSERT (demo_kernel->is_valid ());
    demo_handler = new CLImageHandler ("cl_handler_demo");
    demo_handler->add_kernel  (demo_kernel);

    return demo_handler;
}

};
