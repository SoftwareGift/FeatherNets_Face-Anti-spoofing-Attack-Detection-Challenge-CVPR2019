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
#include "xcam_utils.h"
#include "x3a_stats_pool.h"
#include "cl_context.h"
#include "cl_device.h"
#include "cl_wavelet_denoise_handler.h"

namespace XCam {

CLWaveletDenoiseImageKernel::CLWaveletDenoiseImageKernel (SmartPtr<CLContext> &context,
        const char *name,
        SmartPtr<CLWaveletDenoiseImageHandler> &handler,
        uint32_t layer)
    : CLImageKernel (context, name, false)
    , _hard_threshold (0.1)
    , _soft_threshold (0.5)
    , _decomposition_levels (WAVELET_DECOMPOSITION_LEVELS)
    , _current_layer (layer)
    , _input_y_offset (0)
    , _output_y_offset (0)
    , _input_uv_offset (0)
    , _output_uv_offset (0)
    , _handler (handler)
{
}

XCamReturn
CLWaveletDenoiseImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    const VideoBufferInfo & video_info_in = input->get_video_info ();
    const VideoBufferInfo & video_info_out = output->get_video_info ();

    _buffer_in = new CLVaBuffer (context, input);
    _buffer_out = new CLVaBuffer (context, output);

    _details_image = _handler->get_details_image ();

    _decomposition_levels = WAVELET_DECOMPOSITION_LEVELS;
    _soft_threshold = _handler->get_denoise_config ().threshold[0];
    _hard_threshold = _handler->get_denoise_config ().threshold[1];

    _input_y_offset = video_info_in.offsets[0] / 4;
    _output_y_offset = video_info_out.offsets[0] / 4;

    _input_uv_offset = video_info_in.aligned_height;
    _output_uv_offset = video_info_out.aligned_height;

    XCAM_ASSERT (_buffer_in->is_valid () && _buffer_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _buffer_in->is_valid () && _buffer_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    args[0].arg_adress = &_buffer_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);

    args[1].arg_adress = &_buffer_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);

    args[2].arg_adress = &_details_image->get_mem_id ();
    args[2].arg_size = sizeof (cl_mem);

    args[3].arg_adress = &_input_y_offset;
    args[3].arg_size = sizeof (_input_y_offset);

    args[4].arg_adress = &_output_y_offset;
    args[4].arg_size = sizeof (_output_y_offset);

    args[5].arg_adress = &_input_uv_offset;
    args[5].arg_size = sizeof (_input_uv_offset);

    args[6].arg_adress = &_output_uv_offset;
    args[6].arg_size = sizeof (_output_uv_offset);

    args[7].arg_adress = &_current_layer;
    args[7].arg_size = sizeof (_current_layer);

    args[8].arg_adress = &_decomposition_levels;
    args[8].arg_size = sizeof (_decomposition_levels);

    args[9].arg_adress = &_hard_threshold;
    args[9].arg_size = sizeof (_hard_threshold);

    args[10].arg_adress = &_soft_threshold;
    args[10].arg_size = sizeof (_soft_threshold);

    work_size.global[0] = video_info_in.width / 16;
    work_size.global[1] = video_info_in.height;
    arg_count = 11;

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLWaveletDenoiseImageKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    return CLImageKernel::post_execute (output);
}

CLWaveletDenoiseImageHandler::CLWaveletDenoiseImageHandler (const char *name)
    : CLImageHandler (name)
{
    _config.decomposition_levels = 5;
    _config.threshold[0] = 0.5;
    _config.threshold[1] = 5.0;
}

XCamReturn
CLWaveletDenoiseImageHandler::prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    CLImageHandler::prepare_output_buf(input, output);

    if (!_details_image.ptr ()) {
        const VideoBufferInfo & video_info = input->get_video_info ();
        uint32_t buffer_size = sizeof(float) * video_info.width * video_info.height;

        SmartPtr<CLContext>  context = CLDevice::instance ()->get_context ();
        _details_image = new CLBuffer (context, buffer_size,
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
create_cl_wavelet_denoise_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLWaveletDenoiseImageHandler> wavelet_handler;
    SmartPtr<CLWaveletDenoiseImageKernel> wavelet_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_wavelet_denoise)
#include "kernel_wavelet_denoise.clx"
    XCAM_CL_KERNEL_FUNC_END;

    wavelet_handler = new CLWaveletDenoiseImageHandler ("cl_handler_wavelet_denoise");
    XCAM_ASSERT (wavelet_handler.ptr ());

    for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
        wavelet_kernel = new CLWaveletDenoiseImageKernel (context, "kernel_wavelet_denoise", wavelet_handler, layer);

        ret = wavelet_kernel->load_from_source (kernel_wavelet_denoise_body, strlen (kernel_wavelet_denoise_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", wavelet_kernel->get_kernel_name());

        XCAM_ASSERT (wavelet_kernel->is_valid ());

        SmartPtr<CLImageKernel> image_kernel = wavelet_kernel;
        wavelet_handler->add_kernel (image_kernel);
    }
    return wavelet_handler;
}

};
