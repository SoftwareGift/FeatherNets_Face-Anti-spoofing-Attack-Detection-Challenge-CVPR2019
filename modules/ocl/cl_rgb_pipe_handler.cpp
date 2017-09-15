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

#include "cl_utils.h"
#include "base/xcam_3a_result.h"
#include "cl_rgb_pipe_handler.h"

namespace XCam {

static const XCamKernelInfo kernel_rgb_pipe_info = {
    "kernel_rgb_pipe",
#include "kernel_rgb_pipe.clx"
    , 0,
};

CLRgbPipeImageKernel::CLRgbPipeImageKernel (const SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_rgb_pipe")
{
}

CLRgbPipeImageHandler::CLRgbPipeImageHandler (const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
{
    _tnr_config.thr_r = 0.064;
    _tnr_config.thr_g = 0.045;
    _tnr_config.thr_b = 0.073;
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

    _tnr_config.gain = (float)config.gain;
    _tnr_config.thr_r  = (float)config.threshold[0];
    _tnr_config.thr_g  = (float)config.threshold[1];
    _tnr_config.thr_b  = (float)config.threshold[2];
    XCAM_LOG_DEBUG ("set TNR RGB config: _gain(%f), _thr_r(%f), _thr_g(%f), _thr_b(%f)",
                    _tnr_config.gain, _tnr_config.thr_r, _tnr_config.thr_g, _tnr_config.thr_b);

    return true;
}

XCamReturn
CLRgbPipeImageHandler::prepare_parameters (
    SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();
    CLArgList args;
    CLWorkSize work_size;

    CLImageDesc desc;
    desc.format.image_channel_order = CL_RGBA;
    desc.format.image_channel_data_type = CL_UNORM_INT16;
    desc.width = video_info.width;
    desc.height = video_info.height;
    desc.array_size = 0;
    desc.row_pitch = video_info.strides[0];
    desc.slice_pitch = 0;

    XCAM_ASSERT (_rgb_pipe_kernel.ptr ());
    SmartPtr<CLImage> image_in = convert_to_climage (context, input, desc);
    SmartPtr<CLImage> image_out = convert_to_climage (context, output, desc);

    if (_image_in_list.size () < 4) {
        while (_image_in_list.size () < 4) {
            _image_in_list.push_back (image_in);
        }
    } else {
        _image_in_list.pop_front ();
        _image_in_list.push_back (image_in);
    }
    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image handler(%s) in/out memory not available", XCAM_STR(get_name ()));

    //set args;
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLArgumentT<CLRgbPipeTnrConfig> (_tnr_config));

    for (CLImagePtrList::iterator it = _image_in_list.begin (); it != _image_in_list.end (); it++) {
        args.push_back (new CLMemArgument (*it));
    }

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = XCAM_ALIGN_UP(video_info.width, 16);
    work_size.global[1] = XCAM_ALIGN_UP(video_info.height, 16);
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    XCAM_ASSERT (_rgb_pipe_kernel.ptr ());
    XCamReturn ret = _rgb_pipe_kernel->set_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "rgb pipe kernel set arguments failed.");

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_rgb_pipe_image_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLRgbPipeImageHandler> rgb_pipe_handler;
    SmartPtr<CLRgbPipeImageKernel> rgb_pipe_kernel;

    rgb_pipe_kernel = new CLRgbPipeImageKernel (context);
    XCAM_ASSERT (rgb_pipe_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, rgb_pipe_kernel->build_kernel (kernel_rgb_pipe_info, NULL) == XCAM_RETURN_NO_ERROR, NULL,
        "build rgb-pipe kernel(%s) failed", kernel_rgb_pipe_info.kernel_name);

    XCAM_ASSERT (rgb_pipe_kernel->is_valid ());
    rgb_pipe_handler = new CLRgbPipeImageHandler (context, "cl_handler_rgb_pipe");
    rgb_pipe_handler->set_rgb_pipe_kernel  (rgb_pipe_kernel);

    return rgb_pipe_handler;
}

};
