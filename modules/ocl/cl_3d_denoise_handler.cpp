/*
 * cl_3d_denoise_handler.cpp - CL 3D noise reduction handler
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
#include "cl_3d_denoise_handler.h"

namespace XCam {

#define CL_3D_DENOISE_MAX_REFERENCE_FRAME_COUNT  3
#define CL_3D_DENOISE_REFERENCE_FRAME_COUNT      3
#define CL_3D_DENOISE_WG_WIDTH   4
#define CL_3D_DENOISE_WG_HEIGHT  16

#define CL_3D_DENOISE_ENABLE_SUBGROUP 1
#define CL_3D_DENOISE_IIR_FILTERING   1

#if CL_3D_DENOISE_ENABLE_SUBGROUP
#define KERNEL_3D_DENOISE_NAME "kernel_3d_denoise"
#else
#define KERNEL_3D_DENOISE_NAME "kernel_3d_denoise_slm"
#endif

enum {
    Kernel3DDenoise,
    Kernel3DDenoiseSLM,
};

const XCamKernelInfo kernel_3d_denoise_info[] = {
    {
        "kernel_3d_denoise",
#include "kernel_3d_denoise.clx"
        , 0,
    },

    {
        "kernel_3d_denoise_slm",
#include "kernel_3d_denoise_slm.clx"
        , 0,
    },
};

CL3DDenoiseImageKernel::CL3DDenoiseImageKernel (
    const SmartPtr<CLContext> &context,
    const char *name,
    uint32_t channel,
    SmartPtr<CL3DDenoiseImageHandler> &handler)
    : CLImageKernel (context, name)
    , _channel (channel)
    , _ref_count (CL_3D_DENOISE_REFERENCE_FRAME_COUNT)
    , _handler (handler)
{
}

XCamReturn
CL3DDenoiseImageKernel::prepare_arguments (
    CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<VideoBuffer> input = _handler->get_input_buf ();
    SmartPtr<VideoBuffer> output = _handler->get_output_buf ();

    const VideoBufferInfo & video_info_in = input->get_video_info ();
    const VideoBufferInfo & video_info_out = output->get_video_info ();

    uint32_t info_index = 0;
    if (_channel == CL_IMAGE_CHANNEL_Y) {
        info_index = 0;
    } else if (_channel == CL_IMAGE_CHANNEL_UV) {
        info_index = 1;
    }

    CLImageDesc cl_desc_in, cl_desc_out;
    cl_desc_in.format.image_channel_order = CL_RGBA;
#if CL_3D_DENOISE_ENABLE_SUBGROUP
    cl_desc_in.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc_in.width = XCAM_ALIGN_UP (video_info_in.width, 8) / 8;
#else
    cl_desc_in.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_in.width = XCAM_ALIGN_UP (video_info_in.width, 4) / 4;
#endif
    cl_desc_in.height = video_info_in.height >> info_index;
    cl_desc_in.row_pitch = video_info_in.strides[info_index];

    cl_desc_out.format.image_channel_order = CL_RGBA;
#if CL_3D_DENOISE_ENABLE_SUBGROUP
    cl_desc_out.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc_out.width = XCAM_ALIGN_UP (video_info_out.width, 8) / 8;
#else
    cl_desc_out.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_out.width = XCAM_ALIGN_UP (video_info_out.width, 4) / 4;
#endif
    cl_desc_out.height = video_info_out.height >> info_index;
    cl_desc_out.row_pitch = video_info_out.strides[info_index];

    _ref_count = _handler->get_ref_framecount ();
    float gain = 5.0f / (_handler->get_denoise_config ().gain + 0.0001f);
    float threshold = 2.0f * _handler->get_denoise_config ().threshold[info_index];

    SmartPtr<CLImage> image_in = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[info_index]);
    SmartPtr<CLImage> image_out = convert_to_climage (context, output, cl_desc_out, video_info_out.offsets[info_index]);
    XCAM_ASSERT (image_in->is_valid () && image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    if (_image_in_list.size () < _ref_count) {
        while (_image_in_list.size () < _ref_count) {
            _image_in_list.push_back (image_in);
        }
    } else {
        _image_in_list.pop_back ();
        _image_in_list.push_front (image_in);
    }

    if (!_image_out_prev.ptr ()) {
        _image_out_prev = image_in;
    }

    //set args;
    args.push_back (new CLArgumentT<float> (gain));
    args.push_back (new CLArgumentT<float> (threshold));
    args.push_back (new CLMemArgument (_image_out_prev));
    args.push_back (new CLMemArgument (image_out));

    uint8_t image_list_count = _image_in_list.size ();
    for (std::list<SmartPtr<CLImage>>::iterator it = _image_in_list.begin (); it != _image_in_list.end (); it++) {
        args.push_back (new CLMemArgument (*it));
    }

    //backup enough buffers for kernel
    for (; image_list_count < CL_3D_DENOISE_MAX_REFERENCE_FRAME_COUNT; ++image_list_count) {
        args.push_back (new CLMemArgument (image_in));
    }

    //set worksize
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
#if CL_3D_DENOISE_ENABLE_SUBGROUP
    work_size.local[0] = CL_3D_DENOISE_WG_WIDTH;
    work_size.local[1] = CL_3D_DENOISE_WG_HEIGHT;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_in.width, work_size.local[0]);
    work_size.global[1] = (cl_desc_in.height +  work_size.local[1] - 1) / work_size.local[1] * work_size.local[1];
#else
    work_size.local[0] = 8;
    work_size.local[1] = 1;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_in.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP(cl_desc_in.height / 8, 8 * work_size.local[1]);
#endif

    _image_out_prev = image_out;

    return XCAM_RETURN_NO_ERROR;
}

CL3DDenoiseImageHandler::CL3DDenoiseImageHandler (const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
    , _ref_count (CL_3D_DENOISE_REFERENCE_FRAME_COUNT - 2)
{
    _config.gain = 1.0f;
    _config.threshold[0] = 0.05f;
    _config.threshold[1] = 0.05f;
}

bool
CL3DDenoiseImageHandler::set_ref_framecount (const uint8_t count)
{
    _ref_count = count;

    return true;
}

bool
CL3DDenoiseImageHandler::set_denoise_config (const XCam3aResultTemporalNoiseReduction& config)
{
    _config = config;

    return true;
}

XCamReturn
CL3DDenoiseImageHandler::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    _input_buf = input;
    _output_buf = output;
    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<CLImageKernel>
create_3d_denoise_kernel (
    const SmartPtr<CLContext> &context, SmartPtr<CL3DDenoiseImageHandler> handler,
    uint32_t channel, uint8_t ref_count)
{
    char build_options[1024];
    xcam_mem_clear (build_options);

    snprintf (build_options, sizeof (build_options),
              " -DREFERENCE_FRAME_COUNT=%d"
              " -DWORKGROUP_WIDTH=%d"
              " -DWORKGROUP_HEIGHT=%d"
              " -DENABLE_IIR_FILERING=%d",
              ref_count,
              CL_3D_DENOISE_WG_WIDTH,
              CL_3D_DENOISE_WG_HEIGHT,
              CL_3D_DENOISE_IIR_FILTERING);

#if CL_3D_DENOISE_ENABLE_SUBGROUP
    int kernel_index = Kernel3DDenoise;
#else
    int kernel_index = Kernel3DDenoiseSLM;
#endif

    SmartPtr<CLImageKernel> kernel =
        new CL3DDenoiseImageKernel (context, KERNEL_3D_DENOISE_NAME, channel, handler);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, kernel->build_kernel (kernel_3d_denoise_info[kernel_index], build_options) == XCAM_RETURN_NO_ERROR,
        NULL, "build 3d denoise kernel failed");
    return kernel;
}

SmartPtr<CLImageHandler>
create_cl_3d_denoise_image_handler (
    const SmartPtr<CLContext> &context, uint32_t channel, uint8_t ref_count)
{
    SmartPtr<CL3DDenoiseImageHandler> denoise_handler;
    SmartPtr<CLImageKernel> denoise_kernel;

    denoise_handler = new CL3DDenoiseImageHandler (context, "cl_3d_denoise_handler");
    XCAM_ASSERT (denoise_handler.ptr ());
    denoise_handler->set_ref_framecount (ref_count);

    if (channel & CL_IMAGE_CHANNEL_Y) {
        denoise_kernel = create_3d_denoise_kernel (context, denoise_handler, CL_IMAGE_CHANNEL_Y, ref_count);
        XCAM_FAIL_RETURN (
            ERROR, denoise_kernel.ptr (), NULL, "3D denoise handler create Y channel kernel failed.");

        denoise_handler->add_kernel (denoise_kernel);
    }

    if (channel & CL_IMAGE_CHANNEL_UV) {
        denoise_kernel = create_3d_denoise_kernel (context, denoise_handler, CL_IMAGE_CHANNEL_UV, ref_count);
        XCAM_FAIL_RETURN (
            ERROR, denoise_kernel.ptr (), NULL, "3D denoise handler create UV channel kernel failed.");

        denoise_handler->add_kernel (denoise_kernel);
    }

    return denoise_handler;
}
};
