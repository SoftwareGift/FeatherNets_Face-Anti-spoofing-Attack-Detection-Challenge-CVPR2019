/*
 * cl_ee_handler.cpp - CL edge enhancement handler
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
#include "cl_ee_handler.h"

namespace XCam {

CLEeImageKernel::CLEeImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_ee")
{
    _ee_config.ee_gain = 2.0;
    _ee_config.ee_threshold = 150.0;
    _ee_config.nr_gain = 0.1;
}

XCamReturn
CLEeImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = output->get_video_info ();

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
    args[2].arg_adress = &_vertical_offset;
    args[2].arg_size = sizeof (_vertical_offset);
    args[3].arg_adress = &_ee_config;
    args[3].arg_size = sizeof (CLEeConfig);
    arg_count = 4;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = video_info.width;
    work_size.global[1] = video_info.height;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

bool
CLEeImageKernel::set_ee_ee (const XCam3aResultEdgeEnhancement &ee)
{
    _ee_config.ee_gain = ee.gain;
    _ee_config.ee_threshold = ee.threshold;
    return true;
}

bool
CLEeImageKernel::set_ee_nr (const XCam3aResultNoiseReduction &nr)
{
    _ee_config.nr_gain = nr.gain;
    return true;
}

CLEeImageHandler::CLEeImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLEeImageHandler::set_ee_config_ee (const XCam3aResultEdgeEnhancement &ee)
{
    _ee_kernel->set_ee_ee (ee);
    return true;
}

bool
CLEeImageHandler::set_ee_config_nr (const XCam3aResultNoiseReduction &nr)
{
    _ee_kernel->set_ee_nr (nr);
    return true;
}

bool
CLEeImageHandler::set_ee_kernel(SmartPtr<CLEeImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _ee_kernel = kernel;
    return true;
}

SmartPtr<CLImageHandler>
create_cl_ee_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLEeImageHandler> ee_handler;
    SmartPtr<CLEeImageKernel> ee_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    ee_kernel = new CLEeImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_ee)
#include "kernel_ee.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = ee_kernel->load_from_source (kernel_ee_body, strlen (kernel_ee_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", ee_kernel->get_kernel_name());
    }
    XCAM_ASSERT (ee_kernel->is_valid ());
    ee_handler = new CLEeImageHandler ("cl_handler_ee");
    ee_handler->set_ee_kernel (ee_kernel);

    return ee_handler;
}

}
