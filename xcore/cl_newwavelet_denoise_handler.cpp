/*
 * cl_newwavelet_denoise_handler.cpp - CL wavelet denoise handler
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
#include "cl_context.h"
#include "cl_device.h"
#include "cl_newwavelet_denoise_handler.h"

#define WAVELET_DECOMPOSITION_LEVELS 4

namespace XCam {

CLNewWaveletDenoiseImageKernel::CLNewWaveletDenoiseImageKernel (SmartPtr<CLContext> &context,
        const char *name,
        SmartPtr<CLNewWaveletDenoiseImageHandler> &handler,
        CLWaveletFilterBank fb,
        uint32_t channel,
        uint32_t layer)
    : CLImageKernel (context, name, true)
    , _filter_bank (fb)
    , _decomposition_levels (WAVELET_DECOMPOSITION_LEVELS)
    , _channel (channel)
    , _current_layer (layer)
    , _hard_threshold (0.1)
    , _soft_threshold (0.5)
    , _handler (handler)
{
}

XCamReturn
CLNewWaveletDenoiseImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    const VideoBufferInfo & video_info_in = input->get_video_info ();
    const VideoBufferInfo & video_info_out = output->get_video_info ();

    _decomposition_levels = WAVELET_DECOMPOSITION_LEVELS;
    _soft_threshold = _handler->get_denoise_config ().threshold[0];
    _hard_threshold = _handler->get_denoise_config ().threshold[1];

    CLImageDesc cl_desc_in, cl_desc_out;
    cl_desc_in.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_in.format.image_channel_order = CL_RGBA;
    cl_desc_in.width = video_info_in.width / 4;
    cl_desc_in.height = video_info_in.height;
    cl_desc_in.row_pitch = video_info_in.strides[0];

    cl_desc_out.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_out.format.image_channel_order = CL_RGBA;
    cl_desc_out.width = video_info_out.width / 4;
    cl_desc_out.height = video_info_out.height;
    cl_desc_out.row_pitch = video_info_out.strides[0];

    _image_in = new CLVaImage (context, input, cl_desc_in, video_info_in.offsets[0]);
    _image_out = new CLVaImage (context, output, cl_desc_out, video_info_out.offsets[0]);

    cl_desc_in.height = video_info_in.height / 2;
    cl_desc_in.row_pitch = video_info_in.strides[1];

    cl_desc_out.height = video_info_out.height / 2;
    cl_desc_out.row_pitch = video_info_out.strides[1];

    _image_in_uv = new CLVaImage (context, input, cl_desc_in, video_info_in.offsets[1]);
    _image_out_uv = new CLVaImage (context, output, cl_desc_out, video_info_out.offsets[1]);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_in_uv->is_valid () &&
        _image_out->is_valid () && _image_out_uv->is_valid(),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    if (_channel == CL_WAVELET_CHANNEL_Y) {
        work_size.global[0] = XCAM_ALIGN_UP ((video_info_in.width >> _current_layer) / 4 , work_size.local[0]);
        work_size.global[1] = XCAM_ALIGN_UP (video_info_in.height >> _current_layer, work_size.local[1]);
    } else if (_channel == CL_WAVELET_CHANNEL_UV) {
        work_size.global[0] = XCAM_ALIGN_UP ((video_info_in.width >> _current_layer) / 4 , work_size.local[0]);
        work_size.global[1] = XCAM_ALIGN_UP (video_info_in.height >> (_current_layer + 1), work_size.local[1]);
    }

    SmartPtr<CLWaveletDecompBuffer> buffer;
    if (_current_layer == 1) {
        if (_filter_bank == CL_WAVELET_HAAR_ANALYSIS) {
            if (_channel == CL_WAVELET_CHANNEL_Y) {
                args[0].arg_adress = &_image_in->get_mem_id ();
                args[0].arg_size = sizeof (cl_mem);
            } else if (_channel == CL_WAVELET_CHANNEL_UV) {
                args[0].arg_adress = &_image_in_uv->get_mem_id ();
                args[0].arg_size = sizeof (cl_mem);
            }
        } else if (_filter_bank == CL_WAVELET_HAAR_SYNTHESIS) {
            if (_channel == CL_WAVELET_CHANNEL_Y) {
                args[0].arg_adress = &_image_out->get_mem_id ();
                args[0].arg_size = sizeof (cl_mem);
            } else if (_channel == CL_WAVELET_CHANNEL_UV) {
                args[0].arg_adress = &_image_out_uv->get_mem_id ();
                args[0].arg_size = sizeof (cl_mem);
            }
        }
    } else {
        buffer = get_decomp_buffer (_channel, _current_layer - 1);
        args[0].arg_adress = &buffer->ll->get_mem_id ();
        args[0].arg_size = sizeof (cl_mem);
    }

    buffer = get_decomp_buffer (_channel, _current_layer);

    args[1].arg_adress = &buffer->ll->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &buffer->hl->get_mem_id ();
    args[2].arg_size = sizeof (cl_mem);
    args[3].arg_adress = &buffer->lh->get_mem_id ();
    args[3].arg_size = sizeof (cl_mem);
    args[4].arg_adress = &buffer->hh->get_mem_id ();
    args[4].arg_size = sizeof (cl_mem);

    args[5].arg_adress = &_current_layer;
    args[5].arg_size = sizeof (_current_layer);

    args[6].arg_adress = &_decomposition_levels;
    args[6].arg_size = sizeof (_decomposition_levels);

    args[7].arg_adress = &_hard_threshold;
    args[7].arg_size = sizeof (_hard_threshold);

    args[8].arg_adress = &_soft_threshold;
    args[8].arg_size = sizeof (_soft_threshold);

    arg_count = 9;

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLNewWaveletDenoiseImageKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    return CLImageKernel::post_execute (output);
}

SmartPtr<CLWaveletDecompBuffer>
CLNewWaveletDenoiseImageKernel::get_decomp_buffer (uint32_t channel, int layer)
{
    SmartPtr<CLWaveletDecompBuffer> buffer;
    if (_handler.ptr ()) {
        buffer = _handler->get_decomp_buffer (channel, layer);
    }

    if (!buffer.ptr ()) {
        XCAM_LOG_ERROR ("get channel(%d) layer(%d) decomposition buffer failed!", channel, layer);
    }
    XCAM_ASSERT (buffer.ptr ());
    return buffer;
}

CLNewWaveletDenoiseImageHandler::CLNewWaveletDenoiseImageHandler (const char *name, uint32_t channel)
    : CLCloneImageHandler (name)
    , _channel (channel)
{
    _config.decomposition_levels = 5;
    _config.threshold[0] = 0.5;
    _config.threshold[1] = 5.0;
}

XCamReturn
CLNewWaveletDenoiseImageHandler::prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    CLCloneImageHandler::prepare_output_buf(input, output);

    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();
    CLImageDesc cl_desc;
    SmartPtr<CLWaveletDecompBuffer> decompBuffer;

    CLImage::video_info_2_cl_image_desc (video_info, cl_desc);

    _decompBufferList.clear ();

    if (_channel & CL_WAVELET_CHANNEL_Y) {
        for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
            decompBuffer = new CLWaveletDecompBuffer ();
            if (decompBuffer.ptr ()) {
                decompBuffer->width = video_info.width >> layer;
                decompBuffer->height = video_info.height >> layer;

                decompBuffer->channel = CL_WAVELET_CHANNEL_Y;
                decompBuffer->layer = layer;

                cl_desc.width = decompBuffer->width / 4;
                cl_desc.height = decompBuffer->height;
                cl_desc.format.image_channel_order = CL_RGBA;
                cl_desc.format.image_channel_data_type = CL_UNORM_INT8;

                decompBuffer->ll = new CLImage2D (context, cl_desc);
                decompBuffer->hl = new CLImage2D (context, cl_desc);
                decompBuffer->lh = new CLImage2D (context, cl_desc);
                decompBuffer->hh = new CLImage2D (context, cl_desc);
                _decompBufferList.push_back (decompBuffer);
            } else {
                XCAM_LOG_ERROR ("create Y decomposition buffer failed!");
                ret = XCAM_RETURN_ERROR_MEM;
            }
        }
    }

    if (_channel & CL_WAVELET_CHANNEL_UV) {
        for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
            decompBuffer = new CLWaveletDecompBuffer ();
            if (decompBuffer.ptr ()) {
                decompBuffer->width = video_info.width >> layer;
                decompBuffer->height = video_info.height >> (layer + 1);

                decompBuffer->channel = CL_WAVELET_CHANNEL_UV;
                decompBuffer->layer = layer;

                cl_desc.width = decompBuffer->width / 4;
                cl_desc.height = decompBuffer->height;
                cl_desc.format.image_channel_order = CL_RGBA;
                cl_desc.format.image_channel_data_type = CL_UNORM_INT8;

                decompBuffer->ll = new CLImage2D (context, cl_desc);
                decompBuffer->hl = new CLImage2D (context, cl_desc);
                decompBuffer->lh = new CLImage2D (context, cl_desc);
                decompBuffer->hh = new CLImage2D (context, cl_desc);
                _decompBufferList.push_back (decompBuffer);
            } else {
                XCAM_LOG_ERROR ("create UV decomposition buffer failed!");
                ret = XCAM_RETURN_ERROR_MEM;
            }
        }
    }
    return ret;
}

bool
CLNewWaveletDenoiseImageHandler::set_denoise_config (const XCam3aResultWaveletNoiseReduction& config)

{
    _config = config;

    return true;
}

SmartPtr<CLWaveletDecompBuffer>
CLNewWaveletDenoiseImageHandler::get_decomp_buffer (uint32_t channel, int layer)
{
    SmartPtr<CLWaveletDecompBuffer> buffer;

    for (CLWaveletDecompBufferList::iterator it = _decompBufferList.begin ();
            it != _decompBufferList.end (); ++it) {
        if ((channel == (*it)->channel) && (layer == (*it)->layer))
            buffer = (*it);
    }
    return buffer;
}

SmartPtr<CLNewWaveletDenoiseImageKernel>
create_kernel_haar_transform (SmartPtr<CLContext> &context,
                              SmartPtr<CLNewWaveletDenoiseImageHandler> handler,
                              uint32_t channel,
                              uint32_t layer)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLNewWaveletDenoiseImageKernel> haar_transform_kernel;

    char build_options[1024];
    xcam_mem_clear (build_options);

    snprintf (build_options, sizeof (build_options),
              " -DWAVELET_DENOISE_Y=%d "
              " -DWAVELET_DENOISE_UV=%d ",
              (channel == CL_WAVELET_CHANNEL_Y ? 1 : 0),
              (channel == CL_WAVELET_CHANNEL_UV ? 1 : 0));

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_wavelet_haar_transform)
#include "kernel_wavelet_haar_transform.clx"
    XCAM_CL_KERNEL_FUNC_END;

    haar_transform_kernel = new CLNewWaveletDenoiseImageKernel (context, "kernel_wavelet_haar_transform",
            handler, CL_WAVELET_HAAR_ANALYSIS, channel, layer);

    ret = haar_transform_kernel->load_from_source (
              kernel_wavelet_haar_transform_body, strlen (kernel_wavelet_haar_transform_body),
              NULL, NULL,
              build_options);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CL image handler(%s) load source failed", haar_transform_kernel->get_kernel_name());

    XCAM_ASSERT (haar_transform_kernel->is_valid ());

    return haar_transform_kernel;
}

SmartPtr<CLNewWaveletDenoiseImageKernel>
create_kernel_haar_reconstruction (SmartPtr<CLContext> &context,
                                   SmartPtr<CLNewWaveletDenoiseImageHandler> handler,
                                   uint32_t channel,
                                   uint32_t layer)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLNewWaveletDenoiseImageKernel> haar_reconstruction_kernel;

    char build_options[1024];
    xcam_mem_clear (build_options);

    snprintf (build_options, sizeof (build_options),
              " -DWAVELET_DENOISE_Y=%d "
              " -DWAVELET_DENOISE_UV=%d ",
              (channel == CL_WAVELET_CHANNEL_Y ? 1 : 0),
              (channel == CL_WAVELET_CHANNEL_UV ? 1 : 0));

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_wavelet_haar_reconstruction)
#include "kernel_wavelet_haar_reconstruction.clx"
    XCAM_CL_KERNEL_FUNC_END;

    haar_reconstruction_kernel = new CLNewWaveletDenoiseImageKernel (context, "kernel_wavelet_haar_reconstruction",
            handler, CL_WAVELET_HAAR_SYNTHESIS, channel, layer);

    ret = haar_reconstruction_kernel->load_from_source (
              kernel_wavelet_haar_reconstruction_body, strlen (kernel_wavelet_haar_reconstruction_body),
              NULL, NULL,
              build_options);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CL image handler(%s) load source failed", haar_reconstruction_kernel->get_kernel_name());

    XCAM_ASSERT (haar_reconstruction_kernel->is_valid ());

    return haar_reconstruction_kernel;
}

SmartPtr<CLImageHandler>
create_cl_newwavelet_denoise_image_handler (SmartPtr<CLContext> &context, uint32_t channel)
{
    SmartPtr<CLNewWaveletDenoiseImageHandler> wavelet_handler;
    SmartPtr<CLNewWaveletDenoiseImageKernel> haar_transform_kernel;
    SmartPtr<CLNewWaveletDenoiseImageKernel> haar_reconstruction_kernel;

    wavelet_handler = new CLNewWaveletDenoiseImageHandler ("cl_handler_newwavelet_denoise", channel);
    XCAM_ASSERT (wavelet_handler.ptr ());

    if (channel & CL_WAVELET_CHANNEL_Y) {
        for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
            SmartPtr<CLImageKernel> image_kernel =
                create_kernel_haar_transform (context, wavelet_handler, CL_WAVELET_CHANNEL_Y, layer);
            wavelet_handler->add_kernel (image_kernel);
        }

        for (int layer = WAVELET_DECOMPOSITION_LEVELS; layer >= 1; layer--) {
            SmartPtr<CLImageKernel> image_kernel =
                create_kernel_haar_reconstruction (context, wavelet_handler, CL_WAVELET_CHANNEL_Y, layer);
            wavelet_handler->add_kernel (image_kernel);
        }
    }

    if (channel & CL_WAVELET_CHANNEL_UV) {
        for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
            SmartPtr<CLImageKernel> image_kernel =
                create_kernel_haar_transform (context, wavelet_handler, CL_WAVELET_CHANNEL_UV, layer);
            wavelet_handler->add_kernel (image_kernel);
        }

        for (int layer = WAVELET_DECOMPOSITION_LEVELS; layer >= 1; layer--) {
            SmartPtr<CLImageKernel> image_kernel =
                create_kernel_haar_reconstruction (context, wavelet_handler, CL_WAVELET_CHANNEL_UV, layer);
            wavelet_handler->add_kernel (image_kernel);
        }
    }

    if ((channel & CL_WAVELET_CHANNEL_Y) &&
            (channel & CL_WAVELET_CHANNEL_UV)) {
        wavelet_handler->set_clone_flags (SwappedBuffer::SwapY | SwappedBuffer::SwapUV);
    } else if (channel & CL_WAVELET_CHANNEL_UV) {
        wavelet_handler->set_clone_flags (SwappedBuffer::SwapUV);
    } else if (channel & CL_WAVELET_CHANNEL_Y) {
        wavelet_handler->set_clone_flags (SwappedBuffer::SwapY);
    }

    return wavelet_handler;
}

};
