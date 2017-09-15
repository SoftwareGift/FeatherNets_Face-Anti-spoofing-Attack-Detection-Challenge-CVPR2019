/*
 * cl_wavelet_denoise_handler.cpp - CL wavelet denoise handler
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
 * Author: Wei Zong <wei.zong@intel.com>
 */
#include "cl_utils.h"
#include "x3a_stats_pool.h"
#include "cl_context.h"
#include "cl_device.h"
#include "cl_wavelet_denoise_handler.h"

#define WAVELET_DECOMPOSITION_LEVELS 4

namespace XCam {

static const XCamKernelInfo kernel_wavelet_denoise_info = {
    "kernel_wavelet_denoise",
#include "kernel_wavelet_denoise.clx"
    , 0,
};

CLWaveletDenoiseImageKernel::CLWaveletDenoiseImageKernel (
    const SmartPtr<CLContext> &context,
    const char *name,
    SmartPtr<CLWaveletDenoiseImageHandler> &handler,
    uint32_t channel,
    uint32_t layer)
    : CLImageKernel (context, name)
    , _channel (channel)
    , _current_layer (layer)
    , _handler (handler)
{
}

XCamReturn
CLWaveletDenoiseImageKernel::prepare_arguments (
    CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<VideoBuffer> input = _handler->get_input_buf ();
    SmartPtr<VideoBuffer> output = _handler->get_output_buf ();

    const VideoBufferInfo &video_info_in = input->get_video_info ();
    const VideoBufferInfo &video_info_out = output->get_video_info ();

    SmartPtr<CLMemory> input_image = convert_to_clbuffer (context, input);
    SmartPtr<CLMemory> reconstruct_image = convert_to_clbuffer (context, output);

    SmartPtr<CLMemory> details_image = _handler->get_details_image ();
    SmartPtr<CLMemory> approx_image = _handler->get_approx_image ();

    uint32_t decomposition_levels = WAVELET_DECOMPOSITION_LEVELS;
    float soft_threshold = _handler->get_denoise_config ().threshold[0];
    float hard_threshold = _handler->get_denoise_config ().threshold[1];

    uint32_t input_y_offset = video_info_in.offsets[0] / 4;
    uint32_t output_y_offset = video_info_out.offsets[0] / 4;

    uint32_t input_uv_offset = video_info_in.aligned_height;
    uint32_t output_uv_offset = video_info_out.aligned_height;

    XCAM_FAIL_RETURN (
        WARNING,
        input_image->is_valid () && reconstruct_image->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", XCAM_STR(get_kernel_name ()));

    //set args;
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    if (_current_layer % 2) {
        args.push_back (new CLMemArgument (input_image));
        args.push_back (new CLMemArgument (approx_image));
    } else {
        args.push_back (new CLMemArgument (approx_image));
        args.push_back (new CLMemArgument (input_image));
    }
    args.push_back (new CLMemArgument (details_image));
    args.push_back (new CLMemArgument (reconstruct_image));
    args.push_back (new CLArgumentT<uint32_t> (input_y_offset));
    args.push_back (new CLArgumentT<uint32_t> (output_y_offset));
    args.push_back (new CLArgumentT<uint32_t> (input_uv_offset));
    args.push_back (new CLArgumentT<uint32_t> (output_uv_offset));
    args.push_back (new CLArgumentT<uint32_t> (_current_layer));
    args.push_back (new CLArgumentT<uint32_t> (decomposition_levels));
    args.push_back (new CLArgumentT<float> (hard_threshold));
    args.push_back (new CLArgumentT<float> (soft_threshold));

    if (_channel & CL_IMAGE_CHANNEL_UV) {
        work_size.global[0] = video_info_in.width / 16;
        work_size.global[1] = video_info_in.height / 2;
    } else {
        work_size.global[0] = video_info_in.width / 16;
        work_size.global[1] = video_info_in.height;
    }

    return XCAM_RETURN_NO_ERROR;
}

CLWaveletDenoiseImageHandler::CLWaveletDenoiseImageHandler (
    const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
{
    _config.decomposition_levels = 5;
    _config.threshold[0] = 0.5;
    _config.threshold[1] = 5.0;
}

XCamReturn
CLWaveletDenoiseImageHandler::prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    CLImageHandler::prepare_output_buf(input, output);

    if (!_approx_image.ptr ()) {
        const VideoBufferInfo & video_info = input->get_video_info ();
        uint32_t buffer_size = video_info.width * video_info.aligned_height;

        _approx_image = new CLBuffer (get_context (), buffer_size,
                                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NULL);
    }

    if (!_details_image.ptr ()) {
        const VideoBufferInfo & video_info = input->get_video_info ();
        uint32_t buffer_size = sizeof(float) * video_info.width * video_info.height;

        _details_image = new CLBuffer (get_context (), buffer_size,
                                       CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NULL);
    }
    return ret;
}

bool
CLWaveletDenoiseImageHandler::set_denoise_config (const XCam3aResultWaveletNoiseReduction& config)

{
    _config = config;

    return true;
}

SmartPtr<CLImageHandler>
create_cl_wavelet_denoise_image_handler (const SmartPtr<CLContext> &context, uint32_t channel)
{
    SmartPtr<CLWaveletDenoiseImageHandler> wavelet_handler;
    SmartPtr<CLWaveletDenoiseImageKernel> wavelet_kernel;

    wavelet_handler = new CLWaveletDenoiseImageHandler (context, "cl_handler_wavelet_denoise");
    XCAM_ASSERT (wavelet_handler.ptr ());

    for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
        wavelet_kernel = new CLWaveletDenoiseImageKernel (
            context, "kernel_wavelet_denoise", wavelet_handler, channel, layer);
        const char *build_options =
            (channel & CL_IMAGE_CHANNEL_UV) ? "-DWAVELET_DENOISE_UV=1" : "-DWAVELET_DENOISE_UV=0";

        XCAM_ASSERT (wavelet_kernel.ptr ());
        XCAM_FAIL_RETURN (
            ERROR, wavelet_kernel->build_kernel (kernel_wavelet_denoise_info, build_options) == XCAM_RETURN_NO_ERROR, NULL,
            "build wavelet denoise kernel(%s) failed", kernel_wavelet_denoise_info.kernel_name);
        XCAM_ASSERT (wavelet_kernel->is_valid ());

        wavelet_handler->add_kernel (wavelet_kernel);
    }
    return wavelet_handler;
}

};
