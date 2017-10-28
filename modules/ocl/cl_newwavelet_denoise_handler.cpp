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

#include "cl_utils.h"
#include "cl_context.h"
#include "cl_device.h"
#include "cl_newwavelet_denoise_handler.h"

#define WAVELET_DECOMPOSITION_LEVELS 4

namespace XCam {

enum {
    KernelWaveletDecompose = 0,
    KernelWaveletReconstruct,
    KernelWaveletNoiseEstimate,
    KernelWaveletThreshold,
};

static const XCamKernelInfo kernel_new_wavelet_info[] = {
    {
        "kernel_wavelet_haar_decomposition",
#include "kernel_wavelet_haar.clx"
        , 0,
    },
    {
        "kernel_wavelet_haar_reconstruction",
#include "kernel_wavelet_haar.clx"
        , 0,
    },
    {
        "kernel_wavelet_coeff_variance",
#include "kernel_wavelet_coeff.clx"
        , 0,
    },
    {
        "kernel_wavelet_coeff_thresholding",
#include "kernel_wavelet_coeff.clx"
        , 0,
    },
};


CLWaveletNoiseEstimateKernel::CLWaveletNoiseEstimateKernel (
    const SmartPtr<CLContext> &context,
    const char *name,
    SmartPtr<CLNewWaveletDenoiseImageHandler> &handler,
    uint32_t channel,
    uint32_t subband,
    uint32_t layer)
    : CLImageKernel (context, name)
    , _decomposition_levels (WAVELET_DECOMPOSITION_LEVELS)
    , _channel (channel)
    , _subband (subband)
    , _current_layer (layer)
    , _analog_gain (-1.0)
    , _handler (handler)
{
}

SmartPtr<CLImage>
CLWaveletNoiseEstimateKernel::get_input_buffer ()
{
    SmartPtr<VideoBuffer> input = _handler->get_input_buf ();
    const VideoBufferInfo & video_info = input->get_video_info ();

    SmartPtr<CLImage> image;
    SmartPtr<CLWaveletDecompBuffer> buffer = _handler->get_decomp_buffer (_channel, _current_layer);
    XCAM_ASSERT (buffer.ptr ());

    if (_subband == CL_WAVELET_SUBBAND_HL) {
        image = buffer->hl[0];
    } else if (_subband == CL_WAVELET_SUBBAND_LH) {
        image = buffer->lh[0];
    } else if (_subband == CL_WAVELET_SUBBAND_HH) {
        image = buffer->hh[0];
    } else {
        image = buffer->ll;
    }

    float current_ag = _handler->get_denoise_config ().analog_gain;
    if ((_analog_gain == -1.0f) ||
            (fabs(_analog_gain - current_ag) > 0.2)) {

        if ((_current_layer == 1) && (_subband == CL_WAVELET_SUBBAND_HH)) {
            _analog_gain = current_ag;
            estimate_noise_variance (video_info, buffer->hh[0], buffer->noise_variance);
            _handler->set_estimated_noise_variation (buffer->noise_variance);
        } else {
            _handler->get_estimated_noise_variation (buffer->noise_variance);
        }
    } else {
        _handler->get_estimated_noise_variation (buffer->noise_variance);
    }
    return image;
}

SmartPtr<CLImage>
CLWaveletNoiseEstimateKernel::get_output_buffer ()
{
    SmartPtr<CLImage> image;
    SmartPtr<CLWaveletDecompBuffer> buffer = _handler->get_decomp_buffer (_channel, _current_layer);
    XCAM_ASSERT (buffer.ptr ());

    if (_subband == CL_WAVELET_SUBBAND_HL) {
        image = buffer->hl[1];
    } else if (_subband == CL_WAVELET_SUBBAND_LH) {
        image = buffer->lh[1];
    } else if (_subband == CL_WAVELET_SUBBAND_HH) {
        image = buffer->hh[1];
    } else {
        image = buffer->ll;
    }
    return image;
}

XCamReturn
CLWaveletNoiseEstimateKernel::prepare_arguments (
    CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> image_in = get_input_buffer ();
    SmartPtr<CLImage> image_out = get_output_buffer ();

    CLImageDesc cl_desc = image_in->get_image_desc ();
    uint32_t cl_width = XCAM_ALIGN_UP (cl_desc.width, 2);
    uint32_t cl_height = XCAM_ALIGN_UP (cl_desc.height, 2);

    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLArgumentT<uint32_t> (_current_layer));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 8;
    work_size.global[0] = XCAM_ALIGN_UP (cl_width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLWaveletNoiseEstimateKernel::estimate_noise_variance (const VideoBufferInfo & video_info, SmartPtr<CLImage> image, float* noise_var)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    SmartPtr<CLEvent> map_event = new CLEvent;
    void *buf_ptr = NULL;

    CLImageDesc cl_desc = image->get_image_desc ();
    uint32_t cl_width = XCAM_ALIGN_UP (cl_desc.width, 2);
    uint32_t cl_height = XCAM_ALIGN_UP (cl_desc.height, 2);

    uint32_t image_width = cl_width << 2;
    uint32_t image_height = cl_height;

    size_t origin[3] = {0, 0, 0};
    size_t row_pitch = cl_desc.row_pitch;
    size_t slice_pitch = 0;
    size_t region[3] = {cl_width, cl_height, 1};

    ret = image->enqueue_map (buf_ptr,
                              origin, region,
                              &row_pitch, &slice_pitch,
                              CL_MAP_READ,
                              CLEvent::EmptyList,
                              map_event);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("wavelet noise variance buffer enqueue map failed");
    }
    XCAM_ASSERT (map_event->get_event_id ());

    ret = map_event->wait ();
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("wavelet noise variance buffer enqueue map event wait failed");
    }

    uint8_t* pixel = (uint8_t*)buf_ptr;
    uint32_t pixel_count = image_width * image_height;
    uint32_t pixel_sum = 0;

    uint32_t median_thresh = pixel_count >> 1;
    float median = 0;
    float noise_std_deviation = 0;

    uint32_t hist_bin_count = 1 << video_info.color_bits;
    uint32_t hist_y[128] = {0};
    uint32_t hist_u[128] = {0};
    uint32_t hist_v[128] = {0};

    if (_channel == CL_IMAGE_CHANNEL_Y) {
        for (uint32_t i = 0; i < image_width; i++) {
            for (uint32_t j = 0; j < image_height; j++) {
                uint8_t base = (pixel[i + j * row_pitch] <= 127) ? 127 : 128;
                hist_y[abs(pixel[i + j * row_pitch] - base)]++;
            }
        }
        pixel_sum = 0;
        median = 0;
        for (uint32_t i = 0; i < (hist_bin_count - 1); i++) {
            pixel_sum += hist_y[i];
            if (pixel_sum >= median_thresh) {
                median = i;
                break;
            }
        }
        noise_std_deviation = median / 0.6745;
        noise_var[0] = noise_std_deviation * noise_std_deviation;
    }
    if (_channel == CL_IMAGE_CHANNEL_UV) {
        for (uint32_t i = 0; i < (image_width / 2); i++) {
            for (uint32_t j = 0; j < image_height; j++) {
                uint8_t base = (pixel[2 * i + j * row_pitch] <= 127) ? 127 : 128;
                hist_u[abs(pixel[2 * i + j * row_pitch] - base)]++;
                base = (pixel[2 * i + 1 + j * row_pitch] <= 127) ? 127 : 128;
                hist_v[abs(pixel[2 * i + 1 + j * row_pitch] - base)]++;
            }
        }
        pixel_sum = 0;
        median = 0;
        for (uint32_t i = 0; i < (hist_bin_count - 1); i++) {
            pixel_sum += hist_u[i];
            if (pixel_sum >= median_thresh >> 1) {
                median = i;
                break;
            }
        }
        noise_std_deviation = median / 0.6745;
        noise_var[1] = noise_std_deviation * noise_std_deviation;

        pixel_sum = 0;
        median = 0;
        for (uint32_t i = 0; i < (hist_bin_count - 1); i++) {
            pixel_sum += hist_v[i];
            if (pixel_sum >= median_thresh >> 1) {
                median = i;
                break;
            }
        }
        noise_std_deviation = median / 0.6745;
        noise_var[2] = noise_std_deviation * noise_std_deviation;
    }

    map_event.release ();

    SmartPtr<CLEvent> unmap_event = new CLEvent;
    ret = image->enqueue_unmap (buf_ptr, CLEvent::EmptyList, unmap_event);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("wavelet noise variance buffer enqueue unmap failed");
    }
    XCAM_ASSERT (unmap_event->get_event_id ());

    ret = unmap_event->wait ();
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("wavelet noise variance buffer enqueue unmap event wait failed");
    }
    unmap_event.release ();

    return ret;
}

CLWaveletThresholdingKernel::CLWaveletThresholdingKernel (
    const SmartPtr<CLContext> &context,
    const char *name,
    SmartPtr<CLNewWaveletDenoiseImageHandler> &handler,
    uint32_t channel,
    uint32_t layer)
    : CLImageKernel (context, name, true)
    , _decomposition_levels (WAVELET_DECOMPOSITION_LEVELS)
    , _channel (channel)
    , _current_layer (layer)
    , _handler (handler)
{
}

XCamReturn
CLWaveletThresholdingKernel::prepare_arguments (
    CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    float noise_variance[2];

    xcam_mem_clear (noise_variance);
    _decomposition_levels = WAVELET_DECOMPOSITION_LEVELS;
    float soft_threshold = _handler->get_denoise_config ().threshold[0];
    float hard_threshold = _handler->get_denoise_config ().threshold[1];
    float anolog_gain_weight = 1.0 + 100 *  _handler->get_denoise_config ().analog_gain;

    SmartPtr<CLWaveletDecompBuffer> buffer;
    buffer = _handler->get_decomp_buffer (_channel, _current_layer);

    CLImageDesc cl_desc = buffer->ll->get_image_desc ();

    float weight = 4;
    if (_channel == CL_IMAGE_CHANNEL_Y) {
        noise_variance[0] = buffer->noise_variance[0] * weight;
        noise_variance[1] = buffer->noise_variance[0] * weight;
    } else {
        noise_variance[0] = buffer->noise_variance[1] * weight;
        noise_variance[1] = buffer->noise_variance[2] * weight;
    }
#if 0
    {
        SmartPtr<CLImage> save_image = buffer->hh[0];
        _handler->dump_coeff (save_image, _channel, _current_layer, CL_WAVELET_SUBBAND_HH);
    }
#endif
    if (_channel == CL_IMAGE_CHANNEL_Y) {
        args.push_back (new CLArgumentT<float> (noise_variance[0]));
        args.push_back (new CLArgumentT<float> (noise_variance[0]));
    } else {
        args.push_back (new CLArgumentT<float> (noise_variance[0]));
        args.push_back (new CLArgumentT<float> (noise_variance[1]));
    }

    args.push_back (new CLMemArgument (buffer->hl[0]));
    args.push_back (new CLMemArgument (buffer->hl[1]));
    args.push_back (new CLMemArgument (buffer->hl[2]));

    args.push_back (new CLMemArgument (buffer->lh[0]));
    args.push_back (new CLMemArgument (buffer->lh[1]));
    args.push_back (new CLMemArgument (buffer->lh[2]));

    args.push_back (new CLMemArgument (buffer->hh[0]));
    args.push_back (new CLMemArgument (buffer->hh[1]));
    args.push_back (new CLMemArgument (buffer->hh[2]));

    args.push_back (new CLArgumentT<uint32_t> (_current_layer));
    args.push_back (new CLArgumentT<uint32_t> (_decomposition_levels));
    args.push_back (new CLArgumentT<float> (hard_threshold));
    args.push_back (new CLArgumentT<float> (soft_threshold));
    args.push_back (new CLArgumentT<float> (anolog_gain_weight));

    uint32_t cl_width = XCAM_ALIGN_UP (cl_desc.width, 2);
    uint32_t cl_height = XCAM_ALIGN_UP (cl_desc.height, 2);

    //set args;
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (cl_width , work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLWaveletTransformKernel::CLWaveletTransformKernel (
    const SmartPtr<CLContext> &context,
    const char *name,
    SmartPtr<CLNewWaveletDenoiseImageHandler> &handler,
    CLWaveletFilterBank fb,
    uint32_t channel,
    uint32_t layer,
    bool bayes_shrink)
    : CLImageKernel (context, name, true)
    , _filter_bank (fb)
    , _decomposition_levels (WAVELET_DECOMPOSITION_LEVELS)
    , _channel (channel)
    , _current_layer (layer)
    , _bayes_shrink (bayes_shrink)
    , _handler (handler)
{
}

XCamReturn
CLWaveletTransformKernel::prepare_arguments (
    CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<VideoBuffer> input = _handler->get_input_buf ();
    SmartPtr<VideoBuffer> output = _handler->get_output_buf ();
    SmartPtr<CLContext> context = get_context ();

    const VideoBufferInfo & video_info_in = input->get_video_info ();
    const VideoBufferInfo & video_info_out = output->get_video_info ();

    _decomposition_levels = WAVELET_DECOMPOSITION_LEVELS;
    float soft_threshold = _handler->get_denoise_config ().threshold[0];
    float hard_threshold = _handler->get_denoise_config ().threshold[1];

    CLImageDesc cl_desc_in, cl_desc_out;
    cl_desc_in.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_in.format.image_channel_order = CL_RGBA;
    cl_desc_in.width = XCAM_ALIGN_UP (video_info_in.width, 4) / 4;
    cl_desc_in.height = video_info_in.height;
    cl_desc_in.row_pitch = video_info_in.strides[0];

    cl_desc_out.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_out.format.image_channel_order = CL_RGBA;
    cl_desc_out.width = XCAM_ALIGN_UP (video_info_out.width, 4) / 4;
    cl_desc_out.height = video_info_out.height;
    cl_desc_out.row_pitch = video_info_out.strides[0];

    SmartPtr<CLImage> image_in = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[0]);
    SmartPtr<CLImage> image_out = convert_to_climage (context, output, cl_desc_out, video_info_out.offsets[0]);

    cl_desc_in.height = XCAM_ALIGN_UP (video_info_in.height, 2) / 2;
    cl_desc_in.row_pitch = video_info_in.strides[1];

    cl_desc_out.height = XCAM_ALIGN_UP (video_info_out.height, 2) / 2;
    cl_desc_out.row_pitch = video_info_out.strides[1];

    SmartPtr<CLImage> image_in_uv = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[1]);
    SmartPtr<CLImage> image_out_uv = convert_to_climage (context, output, cl_desc_out, video_info_out.offsets[1]);

    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_in_uv->is_valid () &&
        image_out->is_valid () && image_out_uv->is_valid(),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    if (_channel == CL_IMAGE_CHANNEL_Y) {
        work_size.global[0] = XCAM_ALIGN_UP ((video_info_in.width >> _current_layer) / 4 , work_size.local[0]);
        work_size.global[1] = XCAM_ALIGN_UP (video_info_in.height >> _current_layer, work_size.local[1]);
    } else if (_channel == CL_IMAGE_CHANNEL_UV) {
        work_size.global[0] = XCAM_ALIGN_UP ((video_info_in.width >> _current_layer) / 4 , work_size.local[0]);
        work_size.global[1] = XCAM_ALIGN_UP (video_info_in.height >> (_current_layer + 1), work_size.local[1]);
    }

    SmartPtr<CLWaveletDecompBuffer> buffer;
    if (_current_layer == 1) {
        if (_filter_bank == CL_WAVELET_HAAR_ANALYSIS) {
            if (_channel == CL_IMAGE_CHANNEL_Y) {
                args.push_back (new CLMemArgument (image_in));
            } else if (_channel == CL_IMAGE_CHANNEL_UV) {
                args.push_back (new CLMemArgument (image_in_uv));
            }
        } else if (_filter_bank == CL_WAVELET_HAAR_SYNTHESIS) {
            if (_channel == CL_IMAGE_CHANNEL_Y) {
                args.push_back (new CLMemArgument (image_out));
            } else if (_channel == CL_IMAGE_CHANNEL_UV) {
                args.push_back (new CLMemArgument (image_out_uv));
            }
        }
    } else {
        buffer = get_decomp_buffer (_channel, _current_layer - 1);
        args.push_back (new CLMemArgument (buffer->ll));
    }

    buffer = get_decomp_buffer (_channel, _current_layer);
    args.push_back (new CLMemArgument (buffer->ll));

    if (_bayes_shrink == true) {
        if (_filter_bank == CL_WAVELET_HAAR_ANALYSIS) {
            args.push_back (new CLMemArgument (buffer->hl[0]));
            args.push_back (new CLMemArgument (buffer->lh[0]));
            args.push_back (new CLMemArgument (buffer->hh[0]));
        } else if (_filter_bank == CL_WAVELET_HAAR_SYNTHESIS) {
            args.push_back (new CLMemArgument (buffer->hl[2]));
            args.push_back (new CLMemArgument (buffer->lh[2]));
            args.push_back (new CLMemArgument (buffer->hh[2]));
        }
    } else {
        args.push_back (new CLMemArgument (buffer->hl[0]));
        args.push_back (new CLMemArgument (buffer->lh[0]));
        args.push_back (new CLMemArgument (buffer->hh[0]));
    }

    args.push_back (new CLArgumentT<uint32_t> (_current_layer));
    args.push_back (new CLArgumentT<uint32_t> (_decomposition_levels));
    args.push_back (new CLArgumentT<float> (hard_threshold));
    args.push_back (new CLArgumentT<float> (soft_threshold));

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLWaveletDecompBuffer>
CLWaveletTransformKernel::get_decomp_buffer (uint32_t channel, int layer)
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

CLNewWaveletDenoiseImageHandler::CLNewWaveletDenoiseImageHandler (
    const SmartPtr<CLContext> &context, const char *name, uint32_t channel)
    : CLImageHandler (context, name)
    , _channel (channel)
{
    _config.decomposition_levels = 5;
    _config.threshold[0] = 0.5;
    _config.threshold[1] = 5.0;
    xcam_mem_clear (_noise_variance);
}

XCamReturn
CLNewWaveletDenoiseImageHandler::prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    CLImageHandler::prepare_output_buf(input, output);

    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();
    CLImageDesc cl_desc;
    SmartPtr<CLWaveletDecompBuffer> decompBuffer;

    CLImage::video_info_2_cl_image_desc (video_info, cl_desc);

    _decompBufferList.clear ();

    if (_channel & CL_IMAGE_CHANNEL_Y) {
        for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
            decompBuffer = new CLWaveletDecompBuffer ();
            if (decompBuffer.ptr ()) {
                decompBuffer->width = XCAM_ALIGN_UP (video_info.width, 1 << layer) >> layer;
                decompBuffer->height = XCAM_ALIGN_UP (video_info.height, 1 << layer) >> layer;
                decompBuffer->width = XCAM_ALIGN_UP (decompBuffer->width, 4);
                decompBuffer->height = XCAM_ALIGN_UP (decompBuffer->height, 2);

                decompBuffer->channel = CL_IMAGE_CHANNEL_Y;
                decompBuffer->layer = layer;
                decompBuffer->noise_variance[0] = 0;

                cl_desc.width = decompBuffer->width / 4;
                cl_desc.height = decompBuffer->height;
                cl_desc.slice_pitch = 0;
                cl_desc.format.image_channel_order = CL_RGBA;
                cl_desc.format.image_channel_data_type = CL_UNORM_INT8;

                decompBuffer->ll = new CLImage2D (context, cl_desc);

                decompBuffer->hl[0] = new CLImage2D (context, cl_desc);
                decompBuffer->lh[0] = new CLImage2D (context, cl_desc);
                decompBuffer->hh[0] = new CLImage2D (context, cl_desc);
                /*
                                uint32_t width = decompBuffer->width / 4;
                                uint32_t height = decompBuffer->height;
                                SmartPtr<CLBuffer> hh_buffer = new CLBuffer (
                                    context, sizeof(uint8_t) * width * height,
                                    CL_MEM_READ_WRITE, NULL);
                                CLImageDesc hh_desc;
                                hh_desc.format = {CL_RGBA, CL_UNORM_INT8};
                                hh_desc.width = width;
                                hh_desc.height = height;
                                hh_desc.row_pitch = sizeof(uint8_t) * width;
                                hh_desc.slice_pitch = 0;
                                hh_desc.size = 0;
                                hh_desc.array_size = 0;

                                decompBuffer->hh[0] = new CLImage2D (
                                    context, hh_desc, 0, hh_buffer);
                */

                cl_desc.format.image_channel_data_type = CL_UNORM_INT16;
                decompBuffer->hl[1] = new CLImage2D (context, cl_desc);
                decompBuffer->lh[1] = new CLImage2D (context, cl_desc);
                decompBuffer->hh[1] = new CLImage2D (context, cl_desc);

                cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
                decompBuffer->hl[2] = new CLImage2D (context, cl_desc);
                decompBuffer->lh[2] = new CLImage2D (context, cl_desc);
                decompBuffer->hh[2] = new CLImage2D (context, cl_desc);

                _decompBufferList.push_back (decompBuffer);
            } else {
                XCAM_LOG_ERROR ("create Y decomposition buffer failed!");
                ret = XCAM_RETURN_ERROR_MEM;
            }
        }
    }

    if (_channel & CL_IMAGE_CHANNEL_UV) {
        for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
            decompBuffer = new CLWaveletDecompBuffer ();
            if (decompBuffer.ptr ()) {
                decompBuffer->width = XCAM_ALIGN_UP (video_info.width, 1 << layer) >> layer;
                decompBuffer->height = XCAM_ALIGN_UP (video_info.height, 1 << (layer + 1)) >> (layer + 1);
                decompBuffer->width = XCAM_ALIGN_UP (decompBuffer->width, 4);
                decompBuffer->height = XCAM_ALIGN_UP (decompBuffer->height, 2);

                decompBuffer->channel = CL_IMAGE_CHANNEL_UV;
                decompBuffer->layer = layer;
                decompBuffer->noise_variance[1] = 0;
                decompBuffer->noise_variance[2] = 0;

                cl_desc.width = decompBuffer->width / 4;
                cl_desc.height = decompBuffer->height;
                cl_desc.slice_pitch = 0;
                cl_desc.format.image_channel_order = CL_RGBA;
                cl_desc.format.image_channel_data_type = CL_UNORM_INT8;

                decompBuffer->ll = new CLImage2D (context, cl_desc);

                decompBuffer->hl[0] = new CLImage2D (context, cl_desc);
                decompBuffer->lh[0] = new CLImage2D (context, cl_desc);
                decompBuffer->hh[0] = new CLImage2D (context, cl_desc);
                /*
                                uint32_t width = decompBuffer->width / 4;
                                uint32_t height = decompBuffer->height;
                                SmartPtr<CLBuffer> hh_buffer = new CLBuffer (
                                    context, sizeof(uint8_t) * width * height,
                                    CL_MEM_READ_WRITE, NULL);
                                CLImageDesc hh_desc;
                                hh_desc.format = {CL_RGBA, CL_UNORM_INT8};
                                hh_desc.width = width;
                                hh_desc.height = height;
                                hh_desc.row_pitch = sizeof(uint8_t) * width;
                                hh_desc.slice_pitch = 0;
                                hh_desc.size = 0;
                                hh_desc.array_size = 0;
                                decompBuffer->hh[0] = new CLImage2D (
                                    context, hh_desc, 0, hh_buffer);
                */
                cl_desc.format.image_channel_data_type = CL_UNORM_INT16;
                decompBuffer->hl[1] = new CLImage2D (context, cl_desc);
                decompBuffer->lh[1] = new CLImage2D (context, cl_desc);
                decompBuffer->hh[1] = new CLImage2D (context, cl_desc);

                cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
                decompBuffer->hl[2] = new CLImage2D (context, cl_desc);
                decompBuffer->lh[2] = new CLImage2D (context, cl_desc);
                decompBuffer->hh[2] = new CLImage2D (context, cl_desc);

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

void
CLNewWaveletDenoiseImageHandler::set_estimated_noise_variation (float* noise_var)
{
    if (noise_var == NULL) {
        XCAM_LOG_ERROR ("invalid input noise variation!");
        return;
    }
    _noise_variance[0] = noise_var[0];
    _noise_variance[1] = noise_var[1];
    _noise_variance[2] = noise_var[2];
}

void
CLNewWaveletDenoiseImageHandler::get_estimated_noise_variation (float* noise_var)
{
    if (noise_var == NULL) {
        XCAM_LOG_ERROR ("invalid output parameters!");
        return;
    }
    noise_var[0] = _noise_variance[0];
    noise_var[1] = _noise_variance[1];
    noise_var[2] = _noise_variance[2];
}

void
CLNewWaveletDenoiseImageHandler::dump_coeff (SmartPtr<CLImage> image, uint32_t channel, uint32_t layer, uint32_t subband)
{
    FILE *file;

    void *buf_ptr = NULL;
    SmartPtr<CLEvent> map_event = new CLEvent;

    CLImageDesc cl_desc = image->get_image_desc ();

    uint32_t cl_width = XCAM_ALIGN_UP (cl_desc.width, 2);
    uint32_t cl_height = XCAM_ALIGN_UP (cl_desc.height, 2);

    size_t origin[3] = {0, 0, 0};
    size_t row_pitch = cl_desc.row_pitch;
    size_t slice_pitch = 0;
    size_t region[3] = {cl_width, cl_height, 1};

    image->enqueue_map (buf_ptr,
                        origin, region,
                        &row_pitch, &slice_pitch,
                        CL_MAP_READ,
                        CLEvent::EmptyList,
                        map_event);
    XCAM_ASSERT (map_event->get_event_id ());

    map_event->wait ();

    uint8_t* pixel = (uint8_t*)buf_ptr;
    uint32_t pixel_count = row_pitch * cl_height;

    char file_name[512];
    snprintf (file_name, sizeof(file_name),
              "wavelet_cl_coeff_"
              "channel%d_"
              "layer%d_"
              "subband%d_"
              "rowpitch%d_"
              "width%dxheight%d"
              ".raw",
              channel, layer, subband, (uint32_t)row_pitch, cl_width, cl_height);
    file = fopen(file_name, "wb");

    if (file != NULL) {
        if (fwrite (pixel, pixel_count, 1, file) <= 0) {
            XCAM_LOG_WARNING ("write frame failed.");
        }
        fclose (file);
    }
    map_event.release ();

    SmartPtr<CLEvent> unmap_event = new CLEvent;
    image->enqueue_unmap (buf_ptr, CLEvent::EmptyList, unmap_event);
    XCAM_ASSERT (unmap_event->get_event_id ());

    unmap_event->wait ();
    unmap_event.release ();
}

static SmartPtr<CLWaveletTransformKernel>
create_kernel_haar_decomposition (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLNewWaveletDenoiseImageHandler> handler,
    uint32_t channel,
    uint32_t layer,
    bool bayes_shrink)
{
    SmartPtr<CLWaveletTransformKernel> haar_decomp_kernel;

    char build_options[1024];
    xcam_mem_clear (build_options);

    snprintf (build_options, sizeof (build_options),
              " -DWAVELET_DENOISE_Y=%d "
              " -DWAVELET_DENOISE_UV=%d ",
              (channel == CL_IMAGE_CHANNEL_Y ? 1 : 0),
              (channel == CL_IMAGE_CHANNEL_UV ? 1 : 0));

    haar_decomp_kernel = new CLWaveletTransformKernel (context, "kernel_wavelet_haar_decomposition",
            handler, CL_WAVELET_HAAR_ANALYSIS, channel, layer, bayes_shrink);

    XCAM_ASSERT (haar_decomp_kernel.ptr ());
    XCAM_FAIL_RETURN (
        WARNING,
        haar_decomp_kernel->build_kernel (kernel_new_wavelet_info[KernelWaveletDecompose], build_options) == XCAM_RETURN_NO_ERROR,
        NULL,
        "wavelet denoise build kernel(%s) failed", kernel_new_wavelet_info[KernelWaveletDecompose].kernel_name);
    XCAM_ASSERT (haar_decomp_kernel->is_valid ());

    return haar_decomp_kernel;
}

static SmartPtr<CLWaveletTransformKernel>
create_kernel_haar_reconstruction (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLNewWaveletDenoiseImageHandler> handler,
    uint32_t channel,
    uint32_t layer,
    bool bayes_shrink)
{
    SmartPtr<CLWaveletTransformKernel> haar_reconstruction_kernel;

    char build_options[1024];
    xcam_mem_clear (build_options);
    snprintf (build_options, sizeof (build_options),
              " -DWAVELET_DENOISE_Y=%d "
              " -DWAVELET_DENOISE_UV=%d "
              " -DWAVELET_BAYES_SHRINK=%d",
              (channel == CL_IMAGE_CHANNEL_Y ? 1 : 0),
              (channel == CL_IMAGE_CHANNEL_UV ? 1 : 0),
              (bayes_shrink == true ? 1 : 0));

    haar_reconstruction_kernel = new CLWaveletTransformKernel (context, "kernel_wavelet_haar_reconstruction",
            handler, CL_WAVELET_HAAR_SYNTHESIS, channel, layer, bayes_shrink);

    XCAM_ASSERT (haar_reconstruction_kernel.ptr ());
    XCAM_FAIL_RETURN (
        WARNING,
        haar_reconstruction_kernel->build_kernel (kernel_new_wavelet_info[KernelWaveletReconstruct], build_options) == XCAM_RETURN_NO_ERROR,
        NULL,
        "wavelet denoise build kernel(%s) failed", kernel_new_wavelet_info[KernelWaveletReconstruct].kernel_name);
    XCAM_ASSERT (haar_reconstruction_kernel->is_valid ());

    return haar_reconstruction_kernel;
}

static SmartPtr<CLWaveletNoiseEstimateKernel>
create_kernel_noise_estimation (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLNewWaveletDenoiseImageHandler> handler,
    uint32_t channel, uint32_t subband, uint32_t layer)
{
    SmartPtr<CLWaveletNoiseEstimateKernel> estimation_kernel;

    char build_options[1024];
    xcam_mem_clear (build_options);

    snprintf (build_options, sizeof (build_options),
              " -DWAVELET_DENOISE_Y=%d "
              " -DWAVELET_DENOISE_UV=%d ",
              (channel == CL_IMAGE_CHANNEL_Y ? 1 : 0),
              (channel == CL_IMAGE_CHANNEL_UV ? 1 : 0));

    estimation_kernel = new CLWaveletNoiseEstimateKernel (
        context, "kernel_wavelet_coeff_variance", handler, channel, subband, layer);
    XCAM_ASSERT (estimation_kernel.ptr ());
    XCAM_FAIL_RETURN (
        WARNING,
        estimation_kernel->build_kernel (kernel_new_wavelet_info[KernelWaveletNoiseEstimate], build_options) == XCAM_RETURN_NO_ERROR,
        NULL,
        "wavelet denoise build kernel(%s) failed", kernel_new_wavelet_info[KernelWaveletNoiseEstimate].kernel_name);
    XCAM_ASSERT (estimation_kernel->is_valid ());

    return estimation_kernel;
}

static SmartPtr<CLWaveletThresholdingKernel>
create_kernel_thresholding (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLNewWaveletDenoiseImageHandler> handler,
    uint32_t channel, uint32_t layer)
{
    SmartPtr<CLWaveletThresholdingKernel> threshold_kernel;

    char build_options[1024];
    xcam_mem_clear (build_options);

    snprintf (build_options, sizeof (build_options),
              " -DWAVELET_DENOISE_Y=%d "
              " -DWAVELET_DENOISE_UV=%d ",
              (channel == CL_IMAGE_CHANNEL_Y ? 1 : 0),
              (channel == CL_IMAGE_CHANNEL_UV ? 1 : 0));

    threshold_kernel = new CLWaveletThresholdingKernel (context,
            "kernel_wavelet_coeff_thresholding",
            handler, channel, layer);
    XCAM_ASSERT (threshold_kernel.ptr ());
    XCAM_FAIL_RETURN (
        WARNING,
        threshold_kernel->build_kernel (kernel_new_wavelet_info[KernelWaveletThreshold], build_options) == XCAM_RETURN_NO_ERROR,
        NULL,
        "wavelet denoise build kernel(%s) failed", kernel_new_wavelet_info[KernelWaveletThreshold].kernel_name);
    XCAM_ASSERT (threshold_kernel->is_valid ());

    return threshold_kernel;
}

SmartPtr<CLImageHandler>
create_cl_newwavelet_denoise_image_handler (
    const SmartPtr<CLContext> &context, uint32_t channel, bool bayes_shrink)
{
    SmartPtr<CLNewWaveletDenoiseImageHandler> wavelet_handler;
    SmartPtr<CLWaveletTransformKernel> haar_decomposition_kernel;
    SmartPtr<CLWaveletTransformKernel> haar_reconstruction_kernel;

    wavelet_handler = new CLNewWaveletDenoiseImageHandler (context, "cl_newwavelet_denoise_handler", channel);
    XCAM_ASSERT (wavelet_handler.ptr ());

    if (channel & CL_IMAGE_CHANNEL_Y) {
        for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
            SmartPtr<CLImageKernel> image_kernel =
                create_kernel_haar_decomposition (context, wavelet_handler, CL_IMAGE_CHANNEL_Y, layer, bayes_shrink);
            wavelet_handler->add_kernel (image_kernel);
        }

        if (bayes_shrink) {
            for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
                SmartPtr<CLImageKernel> image_kernel;

                image_kernel = create_kernel_noise_estimation (context, wavelet_handler,
                               CL_IMAGE_CHANNEL_Y, CL_WAVELET_SUBBAND_HH, layer);
                wavelet_handler->add_kernel (image_kernel);

                image_kernel = create_kernel_noise_estimation (context, wavelet_handler,
                               CL_IMAGE_CHANNEL_Y, CL_WAVELET_SUBBAND_LH, layer);
                wavelet_handler->add_kernel (image_kernel);

                image_kernel = create_kernel_noise_estimation (context, wavelet_handler,
                               CL_IMAGE_CHANNEL_Y, CL_WAVELET_SUBBAND_HL, layer);
                wavelet_handler->add_kernel (image_kernel);
            }
            for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
                SmartPtr<CLImageKernel> image_kernel;
                image_kernel = create_kernel_thresholding (context, wavelet_handler, CL_IMAGE_CHANNEL_Y, layer);
                wavelet_handler->add_kernel (image_kernel);
            }
        }

        for (int layer = WAVELET_DECOMPOSITION_LEVELS; layer >= 1; layer--) {
            SmartPtr<CLImageKernel> image_kernel =
                create_kernel_haar_reconstruction (context, wavelet_handler, CL_IMAGE_CHANNEL_Y, layer, bayes_shrink);
            wavelet_handler->add_kernel (image_kernel);
        }
    }

    if (channel & CL_IMAGE_CHANNEL_UV) {
        for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
            SmartPtr<CLImageKernel> image_kernel =
                create_kernel_haar_decomposition (context, wavelet_handler, CL_IMAGE_CHANNEL_UV, layer, bayes_shrink);
            wavelet_handler->add_kernel (image_kernel);
        }

        if (bayes_shrink) {
            for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
                SmartPtr<CLImageKernel> image_kernel;

                image_kernel = create_kernel_noise_estimation (context, wavelet_handler,
                               CL_IMAGE_CHANNEL_UV, CL_WAVELET_SUBBAND_HH, layer);
                wavelet_handler->add_kernel (image_kernel);

                image_kernel = create_kernel_noise_estimation (context, wavelet_handler,
                               CL_IMAGE_CHANNEL_UV, CL_WAVELET_SUBBAND_LH, layer);
                wavelet_handler->add_kernel (image_kernel);

                image_kernel = create_kernel_noise_estimation (context, wavelet_handler,
                               CL_IMAGE_CHANNEL_UV, CL_WAVELET_SUBBAND_HL, layer);
                wavelet_handler->add_kernel (image_kernel);
            }
            for (int layer = 1; layer <= WAVELET_DECOMPOSITION_LEVELS; layer++) {
                SmartPtr<CLImageKernel> image_kernel;
                image_kernel = create_kernel_thresholding (context, wavelet_handler, CL_IMAGE_CHANNEL_UV, layer);
                wavelet_handler->add_kernel (image_kernel);
            }
        }

        for (int layer = WAVELET_DECOMPOSITION_LEVELS; layer >= 1; layer--) {
            SmartPtr<CLImageKernel> image_kernel =
                create_kernel_haar_reconstruction (context, wavelet_handler, CL_IMAGE_CHANNEL_UV, layer, bayes_shrink);
            wavelet_handler->add_kernel (image_kernel);
        }
    }

    return wavelet_handler;
}

};
