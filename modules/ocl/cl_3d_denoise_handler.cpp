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

CL3DDenoiseImageKernel::CL3DDenoiseImageKernel (SmartPtr<CLContext> &context,
        const char *name,
        uint32_t channel,
        SmartPtr<CL3DDenoiseImageHandler> &handler)
    : CLImageKernel (context, name)
    , _channel (channel)
    , _gain (1.0f)
    , _thr_y (0.05f)
    , _thr_uv (0.05f)
    , _ref_count (CL_3D_DENOISE_REFERENCE_FRAME_COUNT)
    , _handler (handler)
{
}

XCamReturn
CL3DDenoiseImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

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
    _gain = 6.0 / _handler->get_denoise_config ().gain;
    _thr_y = _handler->get_denoise_config ().threshold[0];
    _thr_uv = _handler->get_denoise_config ().threshold[1];

    _image_in = new CLVaImage (context, input, cl_desc_in, video_info_in.offsets[info_index]);
    if (_image_in_list.size () < _ref_count) {
        while (_image_in_list.size () < _ref_count) {
            _image_in_list.push_back (_image_in);
        }
    } else {
        _image_in_list.pop_front ();
        _image_in_list.push_back (_image_in);
    }
    _image_out = new CLVaImage (context, output, cl_desc_out, video_info_out.offsets[info_index]);

    if (!_image_out_prev.ptr ()) {
        _image_out_prev = _image_in;
    }
    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
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

    args[0].arg_adress = &_gain;
    args[0].arg_size = sizeof (_gain);

    if (_channel == CL_IMAGE_CHANNEL_Y) {
        args[1].arg_adress = &_thr_y;
        args[1].arg_size = sizeof (_thr_y);
    } else if (_channel == CL_IMAGE_CHANNEL_UV) {
        args[1].arg_adress = &_thr_uv;
        args[1].arg_size = sizeof (_thr_uv);
    }
    args[2].arg_adress = &_image_out_prev->get_mem_id ();
    args[2].arg_size = sizeof (cl_mem);
    args[3].arg_adress = &_image_out->get_mem_id ();
    args[3].arg_size = sizeof (cl_mem);

    uint8_t image_list_count = _image_in_list.size ();
    uint8_t image_index = image_list_count;
    for (std::list<SmartPtr<CLImage>>::iterator it = _image_in_list.begin (); it != _image_in_list.end (); it++) {
        args[3 + image_index].arg_adress = &(*it)->get_mem_id ();
        args[3 + image_index].arg_size = sizeof (cl_mem);
        image_index--;
    }

    if (image_list_count < CL_3D_DENOISE_MAX_REFERENCE_FRAME_COUNT) {
        int append = CL_3D_DENOISE_MAX_REFERENCE_FRAME_COUNT - image_list_count;
        for (int i = 1; i <= append; i++) {
            args[3 + image_list_count + i].arg_adress = &_image_in->get_mem_id ();
            args[3 + image_list_count + i].arg_size = sizeof (cl_mem);
        }
    }

    arg_count = 4 + CL_3D_DENOISE_MAX_REFERENCE_FRAME_COUNT;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CL3DDenoiseImageKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    _image_out_prev = _image_out;
    return CLImageKernel::post_execute (output);
}

CL3DDenoiseImageHandler::CL3DDenoiseImageHandler (const char *name)
    : CLImageHandler (name)
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

SmartPtr<CLImageHandler>
create_cl_3d_denoise_image_handler (SmartPtr<CLContext> &context, uint32_t channel, uint8_t ref_count)
{
    SmartPtr<CL3DDenoiseImageHandler> denoise_handler;
    SmartPtr<CLImageKernel> denoise_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

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

    denoise_handler = new CL3DDenoiseImageHandler ("cl_3d_denoise_handler");
    XCAM_ASSERT (denoise_handler.ptr ());
    denoise_handler->set_ref_framecount (ref_count);

//#if CL_3D_DENOISE_ENABLE_SUBGROUP
    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_3d_denoise)
#if CL_3D_DENOISE_ENABLE_SUBGROUP
#include "kernel_3d_denoise.clx"
#else
#include "kernel_3d_denoise_slm.clx"
#endif
    XCAM_CL_KERNEL_FUNC_END;
//#else
//   XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_3d_denoise)
//#include "kernel_3d_denoise_slm.clx"
//    XCAM_CL_KERNEL_FUNC_END;
//#endif

    if (channel & CL_IMAGE_CHANNEL_Y) {
        denoise_kernel = new CL3DDenoiseImageKernel (context, KERNEL_3D_DENOISE_NAME, CL_IMAGE_CHANNEL_Y, denoise_handler);
        ret = denoise_kernel->load_from_source (
                  kernel_3d_denoise_body, strlen (kernel_3d_denoise_body),
                  NULL, NULL,
                  build_options);
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", denoise_kernel->get_kernel_name());

        denoise_handler->add_kernel (denoise_kernel);
    }

    if (channel & CL_IMAGE_CHANNEL_UV) {
        denoise_kernel = new CL3DDenoiseImageKernel (context, KERNEL_3D_DENOISE_NAME, CL_IMAGE_CHANNEL_UV, denoise_handler);
        ret = denoise_kernel->load_from_source (
                  kernel_3d_denoise_body, strlen (kernel_3d_denoise_body),
                  NULL, NULL,
                  build_options);
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", denoise_kernel->get_kernel_name());

        denoise_handler->add_kernel (denoise_kernel);
    }

    return denoise_handler;
}
};
