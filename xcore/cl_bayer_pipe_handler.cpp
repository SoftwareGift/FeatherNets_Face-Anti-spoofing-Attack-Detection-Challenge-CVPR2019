/*
 * cl_bayer_pipe_handler.cpp - CL bayer pipe handler
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: wangfei <feix.w.wang@intel.com>
 * Author: Shincy Tu <shincy.tu@intel.com>
 */
#include "xcam_utils.h"
#include "cl_bayer_pipe_handler.h"

#define WORKGROUP_PIXEL_WIDTH 16
#define WORKGROUP_PIXEL_HEIGHT 16

#define BAYER_LOCAL_X_SIZE 8
#define BAYER_LOCAL_Y_SIZE 4

float guass_2_0_table[XCAM_GUASS_TABLE_SIZE] = {
    3.978874, 3.966789, 3.930753, 3.871418, 3.789852, 3.687501, 3.566151, 3.427876, 3.274977, 3.109920,
    2.935268, 2.753622, 2.567547, 2.379525, 2.191896, 2.006815, 1.826218, 1.651792, 1.484965, 1.326889,
    1.178449, 1.040267, 0.912718, 0.795950, 0.689911, 0.594371, 0.508957, 0.433173, 0.366437, 0.308103,
    0.257483, 0.213875, 0.176575, 0.144896, 0.118179, 0.095804, 0.077194, 0.061822, 0.049210, 0.038934,
    0.030617, 0.023930, 0.018591, 0.014355, 0.011017, 0.008404, 0.006372, 0.004802, 0.003597, 0.002678,
    0.001981, 0.001457, 0.001065, 0.000774, 0.000559, 0.000401, 0.000286, 0.000203, 0.000143, 0.000100,
    0.000070, 0.000048, 0.000033, 0.000023
};

namespace XCam {

CLBayerPipeImageKernel::CLBayerPipeImageKernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLBayerPipeImageHandler> &handler)
    : CLImageKernel (context, "kernel_bayer_pipe")
    , _input_height (0)
    , _output_height (0)
    , _enable_denoise (0)
    , _handler (handler)
{
    memcpy(_guass_table, guass_2_0_table, sizeof(float)*XCAM_GUASS_TABLE_SIZE);
}

bool
CLBayerPipeImageKernel::enable_denoise (bool enable)
{
    _enable_denoise = (enable ? 1 : 0);
    return true;
}

XCamReturn
CLBayerPipeImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & in_video_info = input->get_video_info ();
    const VideoBufferInfo & out_video_info = output->get_video_info ();

    CLImageDesc in_image_info;
    in_image_info.format.image_channel_order = CL_RGBA;
    in_image_info.format.image_channel_data_type = CL_UNORM_INT16; //CL_UNSIGNED_INT32;
    in_image_info.width = in_video_info.width / 4; // 960/4
    in_image_info.height = in_video_info.aligned_height * 4;  //540
    in_image_info.row_pitch = in_video_info.strides[0];

    _image_in = new CLVaImage (context, input, in_image_info);
    _image_out = new CLVaImage (context, output);
    _input_height = in_video_info.aligned_height;
    _output_height = out_video_info.aligned_height;

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    _guass_table_buffer = new CLBuffer(
        context, sizeof(float) * 64,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &_guass_table);

    //set args;
    arg_count = 0;
    args[arg_count].arg_adress = &_image_in->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_input_height;
    args[arg_count].arg_size = sizeof (_input_height);
    ++arg_count;

    args[arg_count].arg_adress = &_image_out->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_output_height;
    args[arg_count].arg_size = sizeof (_output_height);
    ++arg_count;

    args[arg_count].arg_adress = &_guass_table_buffer->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_enable_denoise;
    args[arg_count].arg_size = sizeof (_enable_denoise);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = BAYER_LOCAL_X_SIZE;
    work_size.local[1] = BAYER_LOCAL_Y_SIZE;
    work_size.global[0] = (XCAM_ALIGN_UP(out_video_info.width, WORKGROUP_PIXEL_WIDTH) / WORKGROUP_PIXEL_WIDTH) *
                          work_size.local[0];
    work_size.global[1] = (XCAM_ALIGN_UP(out_video_info.height, WORKGROUP_PIXEL_HEIGHT) / WORKGROUP_PIXEL_HEIGHT) *
                          work_size.local[1];

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLBayerPipeImageKernel::post_execute ()
{
    _image_in.release ();
    _image_out.release ();
    _guass_table_buffer.release ();

    return XCAM_RETURN_NO_ERROR;
}

CLBayerPipeImageHandler::CLBayerPipeImageHandler (const char *name)
    : CLImageHandler (name)
    , _output_format (XCAM_PIX_FMT_RGB48_planar)
{
}

bool
CLBayerPipeImageHandler::set_output_format (uint32_t fourcc)
{
    XCAM_FAIL_RETURN (
        WARNING,
        XCAM_PIX_FMT_RGB48_planar == fourcc || XCAM_PIX_FMT_RGB24_planar == fourcc,
        false,
        "CL image handler(%s) doesn't support format(%s) settings",
        get_name (), xcam_fourcc_to_string (fourcc));

    _output_format = fourcc;
    return true;
}

bool
CLBayerPipeImageHandler::set_bayer_kernel (SmartPtr<CLBayerPipeImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _bayer_kernel = kernel;
    return true;
}

bool
CLBayerPipeImageHandler::enable_denoise (bool enable)
{
    return _bayer_kernel->enable_denoise (enable);
}

XCamReturn
CLBayerPipeImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    uint32_t format = _output_format;
    uint32_t width = input.width;
    uint32_t height = input.height;
    if (input.format == XCAM_PIX_FMT_SGRBG16_planar) {
        width *= 2;
        height *= 2;
    }
    bool format_inited = output.init (format, width, height);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) ouput format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (format));

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_bayer_pipe_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLBayerPipeImageHandler> bayer_pipe_handler;
    SmartPtr<CLBayerPipeImageKernel> bayer_pipe_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    bayer_pipe_handler = new CLBayerPipeImageHandler ("cl_handler_bayer_pipe");
    bayer_pipe_kernel = new CLBayerPipeImageKernel (context, bayer_pipe_handler);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_bayer_pipe)
#include "kernel_bayer_pipe.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = bayer_pipe_kernel->load_from_source (kernel_bayer_pipe_body, strlen (kernel_bayer_pipe_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", bayer_pipe_kernel->get_kernel_name());
    }
    XCAM_ASSERT (bayer_pipe_kernel->is_valid ());
    bayer_pipe_handler->set_bayer_kernel (bayer_pipe_kernel);

    return bayer_pipe_handler;
}

};
