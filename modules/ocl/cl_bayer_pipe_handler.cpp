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

#define WORKGROUP_PIXEL_WIDTH 128
#define WORKGROUP_PIXEL_HEIGHT 8

#define BAYER_LOCAL_X_SIZE 64
#define BAYER_LOCAL_Y_SIZE 2

float table[XCAM_BNR_TABLE_SIZE] = {
    63.661991, 60.628166, 52.366924, 41.023067, 29.146584, 18.781729, 10.976704,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000
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
    memcpy(_bnr_table, table, sizeof(float)*XCAM_BNR_TABLE_SIZE);
    _ee_config.ee_gain = 0.8;
    _ee_config.ee_threshold = 0.025;
}

bool
CLBayerPipeImageKernel::enable_denoise (bool enable)
{
    _enable_denoise = (enable ? 1 : 0);
    return true;
}

bool
CLBayerPipeImageKernel::set_ee (const XCam3aResultEdgeEnhancement &ee)
{
    _ee_config.ee_gain = (float)ee.gain;
    _ee_config.ee_threshold = (float)ee.threshold;
    return true;
}

bool
CLBayerPipeImageKernel::set_bnr (const XCam3aResultBayerNoiseReduction &bnr)
{
    for(int i = 0; i < XCAM_BNR_TABLE_SIZE; i++)
        _bnr_table[i] = (float)bnr.table[i];
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

    _bnr_table_buffer = new CLBuffer(
        context, sizeof(float) * XCAM_BNR_TABLE_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &_bnr_table);

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

    args[arg_count].arg_adress = &_bnr_table_buffer->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_enable_denoise;
    args[arg_count].arg_size = sizeof (_enable_denoise);
    ++arg_count;

    args[arg_count].arg_adress = &_ee_config;
    args[arg_count].arg_size = sizeof (_ee_config);
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
CLBayerPipeImageKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (output);

    _image_in.release ();
    _image_out.release ();
    _bnr_table_buffer.release ();

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

bool
CLBayerPipeImageHandler::set_ee_config (const XCam3aResultEdgeEnhancement &ee)
{
    _bayer_kernel->set_ee (ee);
    return true;
}
bool
CLBayerPipeImageHandler::set_bnr_config (const XCam3aResultBayerNoiseReduction &bnr)
{
    _bayer_kernel->set_bnr (bnr);
    return true;
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
        "CL image handler(%s) output format(%s) unsupported",
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
