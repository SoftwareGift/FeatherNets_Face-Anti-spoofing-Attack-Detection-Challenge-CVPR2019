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
#include "cl_device.h"
#include "cl_kernel.h"


namespace XCam {

static const XCamKernelInfo kernel_demo_info = {
    "kernel_demo",
#include "kernel_demo.clx"
    , 0,
};

CLDemoImageHandler::CLDemoImageHandler (const SmartPtr<CLContext> &context)
    : CLImageHandler (context, "cl_demo_handler")
{
}

XCamReturn
CLDemoImageHandler::prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
    CLArgList args;
    CLWorkSize work_size;

    SmartPtr<CLImage> input_image = new CLVaImage (context, input);
    SmartPtr<CLImage> output_image = new CLVaImage (context, output);

    XCAM_ASSERT (input_image.ptr () && output_image.ptr ());
    XCAM_ASSERT (input_image->is_valid () && output_image->is_valid ());
    args.push_back (new CLMemArgument (input_image));
    args.push_back (new CLMemArgument (output_image));

    const CLImageDesc out_info = output_image->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width;
    work_size.global[1] = out_info.height;
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    _copy_kernel->set_arguments (args, work_size);

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_demo_image_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLDemoImageHandler> demo_handler;
    SmartPtr<CLImageKernel> demo_kernel;

    demo_kernel = new CLImageKernel (context);
    XCAM_ASSERT (demo_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, demo_kernel->build_kernel (kernel_demo_info, NULL) == XCAM_RETURN_NO_ERROR,
        NULL, "build demo kernel failed");

    XCAM_ASSERT (demo_kernel->is_valid ());
    demo_handler = new CLDemoImageHandler (context);
    XCAM_ASSERT (demo_handler.ptr ());
    demo_handler->set_copy_kernel (demo_kernel);

    return demo_handler;
}

SmartPtr<CLImageHandler>
create_cl_binary_demo_image_handler (const SmartPtr<CLContext> &context, const uint8_t *binary, size_t size)
{
    SmartPtr<CLDemoImageHandler> demo_handler;
    SmartPtr<CLImageKernel> demo_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    demo_kernel = new CLImageKernel (context, "kernel_demo");
    {
#if 0
        XCAM_CL_KERNEL_FUNC_BINARY_BEGIN(kernel_demo)
#include "kernel_demo.clx.bin"
        XCAM_CL_KERNEL_FUNC_END;
        ret = demo_kernel->load_from_binary (kernel_demo_body, sizeof (kernel_demo_body));
#endif
        ret = demo_kernel->load_from_binary (binary, size);
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load binary failed", demo_kernel->get_kernel_name());
    }
    XCAM_ASSERT (demo_kernel->is_valid ());
    demo_handler = new CLDemoImageHandler (context);
    demo_handler->set_copy_kernel (demo_kernel);

    return demo_handler;
}

};
