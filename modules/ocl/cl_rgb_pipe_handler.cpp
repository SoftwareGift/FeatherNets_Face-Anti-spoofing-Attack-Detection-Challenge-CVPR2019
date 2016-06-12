/*
 * cl_rgb_pipe_handler.cpp - CL rgb pipe handler
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
 * Author: Wei Zong <wei.zong@intel.com>
 * Author: Wangfei <feix.w.wang@intel.com>
 */
#include "xcam_utils.h"
#include "base/xcam_3a_result.h"
#include "cl_rgb_pipe_handler.h"

namespace XCam {

CLRgbPipeImageKernel::CLRgbPipeImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_rgb_pipe")
{
    _tnr_config.thr_r = 0.064;
    _tnr_config.thr_g = 0.045;
    _tnr_config.thr_b = 0.073;
}

bool
CLRgbPipeImageKernel::set_tnr_config (const XCam3aResultTemporalNoiseReduction& config)
{
    _tnr_config.gain = (float)config.gain;
    _tnr_config.thr_r  = (float)config.threshold[0];
    _tnr_config.thr_g  = (float)config.threshold[1];
    _tnr_config.thr_b  = (float)config.threshold[2];
    XCAM_LOG_DEBUG ("set TNR RGB config: _gain(%f), _thr_r(%f), _thr_g(%f), _thr_b(%f)",
                    _tnr_config.gain, _tnr_config.thr_r, _tnr_config.thr_g, _tnr_config.thr_b);

    return true;
}

XCamReturn
CLRgbPipeImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);

    if (_image_in_list.size () < 4) {
        while (_image_in_list.size () < 4) {
            _image_in_list.push_back (_image_in);
        }
    } else {
        _image_in_list.pop_front ();
        _image_in_list.push_back (_image_in);
    }
    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    args[0].arg_adress = &_image_out->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_tnr_config;
    args[1].arg_size = sizeof (CLRgbPipeTnrConfig);

    uint8_t index = 0;
    for (std::list<SmartPtr<CLImage>>::iterator it = _image_in_list.begin (); it != _image_in_list.end (); it++) {
        args[2 + index].arg_adress = &(*it)->get_mem_id ();
        args[2 + index].arg_size = sizeof (cl_mem);
        index++;
    }

    arg_count = 2 + index;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = XCAM_ALIGN_UP(video_info.width, 16);
    work_size.global[1] = XCAM_ALIGN_UP(video_info.height, 16);
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

CLRgbPipeImageHandler::CLRgbPipeImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLRgbPipeImageHandler::set_rgb_pipe_kernel(SmartPtr<CLRgbPipeImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _rgb_pipe_kernel = kernel;
    return true;
}

bool
CLRgbPipeImageHandler::set_tnr_config (const XCam3aResultTemporalNoiseReduction& config)
{
    if (!_rgb_pipe_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set config error, invalid TNR kernel !");
    }

    _rgb_pipe_kernel->set_tnr_config (config);

    return true;
}

SmartPtr<CLImageHandler>
create_cl_rgb_pipe_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLRgbPipeImageHandler> rgb_pipe_handler;
    SmartPtr<CLRgbPipeImageKernel> rgb_pipe_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    rgb_pipe_kernel = new CLRgbPipeImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_rgb_pipe)
#include "kernel_rgb_pipe.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = rgb_pipe_kernel->load_from_source (kernel_rgb_pipe_body, strlen (kernel_rgb_pipe_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", rgb_pipe_kernel->get_kernel_name());
    }
    XCAM_ASSERT (rgb_pipe_kernel->is_valid ());
    rgb_pipe_handler = new CLRgbPipeImageHandler ("cl_handler_rgb_pipe");
    rgb_pipe_handler->set_rgb_pipe_kernel  (rgb_pipe_kernel);

    return rgb_pipe_handler;
}

};
