/*
 * cl_image_warp_handler.cpp - CL image warping handler
 *
 *  Copyright (c) 2016 Intel Corporation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "cl_utils.h"
#include "cl_image_warp_handler.h"

namespace XCam {

#define CL_IMAGE_WARP_WG_WIDTH   8
#define CL_IMAGE_WARP_WG_HEIGHT  4


static const XCamKernelInfo kernel_image_warp_info [] = {
    {
        "kernel_image_warp_8_pixel",
#include "kernel_image_warp.clx"
        , 0,
    },
    {
        "kernel_image_warp_1_pixel",
#include "kernel_image_warp.clx"
        , 0,
    }
};

CLImageWarpKernel::CLImageWarpKernel (
    const SmartPtr<CLContext> &context,
    const char *name,
    uint32_t channel,
    SmartPtr<CLImageHandler> &handler)
    : CLImageKernel (context, name)
    , _channel (channel)
{
    _handler = handler.dynamic_cast_ptr<CLImageWarpHandler> ();
}

XCamReturn
CLImageWarpKernel::prepare_arguments (
    CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<VideoBuffer> input = _handler->get_warp_input_buf ();
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
    cl_desc_in.format.image_channel_order = info_index == 0 ? CL_R : CL_RG;
    cl_desc_in.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_in.width = video_info_in.width >> info_index;
    cl_desc_in.height = video_info_in.height >> info_index;
    cl_desc_in.row_pitch = video_info_in.strides[info_index];

#if CL_IMAGE_WARP_WRITE_UINT
    cl_desc_out.format.image_channel_data_type = info_index == 0 ? CL_UNSIGNED_INT16 : CL_UNSIGNED_INT32;
    cl_desc_out.format.image_channel_order = CL_RGBA;
    cl_desc_out.width = XCAM_ALIGN_DOWN (video_info_out.width >> info_index, 8) / 8;
    cl_desc_out.height = video_info_out.height >> info_index;
#else
    cl_desc_out.format.image_channel_order = info_index == 0 ? CL_R : CL_RG;
    cl_desc_out.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_out.width = video_info_out.width >> info_index;
    cl_desc_out.height = video_info_out.height >> info_index;
#endif

    cl_desc_out.row_pitch = video_info_out.strides[info_index];
    SmartPtr<CLImage> image_in = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[info_index]);

    CLWarpConfig warp_config = _handler->get_warp_config ();
    if ((warp_config.trim_ratio > 0.5f) || (warp_config.trim_ratio < 0.0f)) {
        warp_config.trim_ratio = 0.0f;
    }

    float sample_rate_x = (float)warp_config.width / (float)video_info_in.width;
    float sample_rate_y = (float)warp_config.height / (float)video_info_in.height;
    XCAM_LOG_DEBUG ("warp analyze image sample rate(%fx%f)", sample_rate_x, sample_rate_y);
    warp_config.proj_mat[2] = warp_config.proj_mat[2] / sample_rate_x;
    warp_config.proj_mat[5] = warp_config.proj_mat[5] / sample_rate_y;
    warp_config.proj_mat[6] = warp_config.proj_mat[6] * sample_rate_x;
    warp_config.proj_mat[7] = warp_config.proj_mat[7] * sample_rate_y;

    /*
       For NV12 image (YUV420), UV plane has half horizontal & vertical coordinate size of Y plane,
       need to adjust the projection matrix as:
       H(uv) = [0.5, 0, 0; 0, 0.5, 0; 0, 0, 1] * H(y) * [2, 0, 0; 0, 2, 0; 0, 0, 1]
    */
    if (_channel == CL_IMAGE_CHANNEL_UV) {
        warp_config.proj_mat[2] = 0.5 * warp_config.proj_mat[2];
        warp_config.proj_mat[5] = 0.5 * warp_config.proj_mat[5];
        warp_config.proj_mat[6] = 2.0 * warp_config.proj_mat[6];
        warp_config.proj_mat[7] = 2.0 * warp_config.proj_mat[7];
    }

    /*
      Trim image: shift toward origin then scale up
      Trim Matrix (TMat)
      TMat = [ scale_x, 0.0f,    shift_x;
               0.0f,    scale_y, shift_y;
               1.0f,    1.0f,    1.0f;   ]

      Warp Perspective Matrix = TMat * HMat
    */
#if CL_IMAGE_WARP_WRITE_UINT
    float shift_x = warp_config.trim_ratio * cl_desc_out.width * 8.0f;
#else
    float shift_x = warp_config.trim_ratio * cl_desc_out.width;
#endif
    float shift_y = warp_config.trim_ratio * cl_desc_out.height;
    float scale_x = 1.0f - 2.0f * warp_config.trim_ratio;
    float scale_y = 1.0f - 2.0f * warp_config.trim_ratio;

    warp_config.proj_mat[0] = scale_x * warp_config.proj_mat[0] + shift_x * warp_config.proj_mat[6];
    warp_config.proj_mat[1] = scale_x * warp_config.proj_mat[1] + shift_x * warp_config.proj_mat[7];
    warp_config.proj_mat[2] = scale_x * warp_config.proj_mat[2] + shift_x * warp_config.proj_mat[8];
    warp_config.proj_mat[3] = scale_y * warp_config.proj_mat[3] + shift_y * warp_config.proj_mat[6];
    warp_config.proj_mat[4] = scale_y * warp_config.proj_mat[4] + shift_y * warp_config.proj_mat[7];
    warp_config.proj_mat[5] = scale_y * warp_config.proj_mat[5] + shift_y * warp_config.proj_mat[8];

    XCAM_LOG_DEBUG ("warp config image size(%dx%d)", warp_config.width, warp_config.height);
    XCAM_LOG_DEBUG ("proj_mat[%d]=(%f, %f, %f, %f, %f, %f, %f, %f, %f);", warp_config.frame_id,
                    warp_config.proj_mat[0], warp_config.proj_mat[1], warp_config.proj_mat[2],
                    warp_config.proj_mat[3], warp_config.proj_mat[4], warp_config.proj_mat[5],
                    warp_config.proj_mat[6], warp_config.proj_mat[7], warp_config.proj_mat[8]);

    SmartPtr<CLImage> image_out = convert_to_climage (context, output, cl_desc_out, video_info_out.offsets[info_index]);
    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = CL_IMAGE_WARP_WG_WIDTH;
    work_size.local[1] = CL_IMAGE_WARP_WG_HEIGHT;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP(cl_desc_out.height, work_size.local[1]);

    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLArgumentT<CLWarpConfig> (warp_config));

    return XCAM_RETURN_NO_ERROR;
}

CLImageWarpHandler::CLImageWarpHandler (const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
{
}

bool
CLImageWarpHandler::is_ready ()
{
    bool ret = !_warp_config_list.empty ();
    return ret && CLImageHandler::is_ready ();
}

XCamReturn
CLImageWarpHandler::execute_done (SmartPtr<VideoBuffer> &output)
{
    XCAM_UNUSED (output);
    if (!_warp_config_list.empty ()) {
        _warp_config_list.pop_front ();
    }

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<VideoBuffer>
CLImageWarpHandler::get_warp_input_buf ()
{
    return CLImageHandler::get_input_buf ();
}

bool
CLImageWarpHandler::set_warp_config (const XCamDVSResult& config)
{
    CLWarpConfig warp_config;
    warp_config.frame_id = config.frame_id;
    warp_config.width = config.frame_width;
    warp_config.height = config.frame_height;
    for( int i = 0; i < 9; i++ ) {
        warp_config.proj_mat[i] = config.proj_mat[i];
    }
    XCAM_LOG_DEBUG ("warp_mat{%d}=[%f, %f, %f; %f, %f, %f; %f, %f, %f]", warp_config.frame_id + 1,
                    warp_config.proj_mat[0], warp_config.proj_mat[1], warp_config.proj_mat[2],
                    warp_config.proj_mat[3], warp_config.proj_mat[4], warp_config.proj_mat[5],
                    warp_config.proj_mat[6], warp_config.proj_mat[7], warp_config.proj_mat[8]);
#if 0
    printf ("warp_mat{%d}=[%f, %f, %f; %f, %f, %f; %f, %f, %f]; \n", warp_config.frame_id + 1,
            warp_config.proj_mat[0], warp_config.proj_mat[1], warp_config.proj_mat[2],
            warp_config.proj_mat[3], warp_config.proj_mat[4], warp_config.proj_mat[5],
            warp_config.proj_mat[6], warp_config.proj_mat[7], warp_config.proj_mat[8]);
#endif
    _warp_config_list.push_back (warp_config);

    return true;
}

CLWarpConfig
CLImageWarpHandler::get_warp_config ()
{
    CLWarpConfig warp_config;

    if (_warp_config_list.size () > 0) {
        warp_config = *(_warp_config_list.begin ());
    } else {
        warp_config.frame_id = -1;
        warp_config.proj_mat[0] = 1.0f;
        warp_config.proj_mat[1] = 0.0f;
        warp_config.proj_mat[2] = 0.0f;
        warp_config.proj_mat[3] = 0.0f;
        warp_config.proj_mat[4] = 1.0f;
        warp_config.proj_mat[5] = 0.0f;
        warp_config.proj_mat[6] = 0.0f;
        warp_config.proj_mat[7] = 0.0f;
        warp_config.proj_mat[8] = 1.0f;
    }

    return warp_config;
}

static SmartPtr<CLImageWarpKernel>
create_kernel_image_warp (
    const SmartPtr<CLContext> &context,
    uint32_t channel,
    SmartPtr<CLImageHandler> handler)
{
    SmartPtr<CLImageWarpKernel> warp_kernel;

    const char *name = (channel == CL_IMAGE_CHANNEL_Y ? "kernel_image_warp_y" : "kernel_image_warp_uv");
    char build_options[1024];
    xcam_mem_clear (build_options);

    snprintf (build_options, sizeof (build_options),
              " -DWARP_Y=%d ",
              (channel == CL_IMAGE_CHANNEL_Y ? 1 : 0));

    warp_kernel = new CLImageWarpKernel (context, name, channel, handler);
    XCAM_ASSERT (warp_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, warp_kernel->build_kernel (kernel_image_warp_info[KernelImageWarp], build_options) == XCAM_RETURN_NO_ERROR,
        NULL, "build image warp kernel failed");
    XCAM_ASSERT (warp_kernel->is_valid ());

    return warp_kernel;
}

SmartPtr<CLImageHandler>
create_cl_image_warp_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageWarpHandler> warp_handler;
    SmartPtr<CLImageKernel> warp_kernel;

    warp_handler = new CLImageWarpHandler (context);
    XCAM_ASSERT (warp_handler.ptr ());

    warp_kernel = create_kernel_image_warp (context, CL_IMAGE_CHANNEL_Y, warp_handler);
    XCAM_ASSERT (warp_kernel.ptr ());
    warp_handler->add_kernel (warp_kernel);

    warp_kernel = create_kernel_image_warp (context, CL_IMAGE_CHANNEL_UV, warp_handler);
    XCAM_ASSERT (warp_kernel.ptr ());
    warp_handler->add_kernel (warp_kernel);

    return warp_handler;
}

};
