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

#include "cl_utils.h"
#include "cl_bayer_pipe_handler.h"

#define WORKGROUP_PIXEL_WIDTH 128
#define WORKGROUP_PIXEL_HEIGHT 8

#define BAYER_LOCAL_X_SIZE 64
#define BAYER_LOCAL_Y_SIZE 2

namespace XCam {

static const float table [XCAM_BNR_TABLE_SIZE] = {
    63.661991f, 60.628166f, 52.366924f, 41.023067f, 29.146584f, 18.781729f, 10.976704f,
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f,
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f,
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f,
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f,
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f,
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f,
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f,
    6.000000f,
};

static const XCamKernelInfo kernel_bayer_pipe_info = {
    "kernel_bayer_pipe",
#include "kernel_bayer_pipe.clx"
    , 0,
};

CLBayerPipeImageKernel::CLBayerPipeImageKernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLBayerPipeImageHandler> &handler)
    : CLImageKernel (context, "kernel_bayer_pipe")
    , _handler (handler)
{

}

CLBayerPipeImageHandler::CLBayerPipeImageHandler (const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
    , _output_format (XCAM_PIX_FMT_RGB48_planar)
    , _enable_denoise (0)
{
    memcpy(_bnr_table, table, sizeof(float)*XCAM_BNR_TABLE_SIZE);
    _ee_config.ee_gain = 0.8;
    _ee_config.ee_threshold = 0.025;
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
    _enable_denoise = (enable ? 1 : 0);
    return true;

}

bool
CLBayerPipeImageHandler::set_ee_config (const XCam3aResultEdgeEnhancement &ee)
{
    _ee_config.ee_gain = (float)ee.gain;
    _ee_config.ee_threshold = (float)ee.threshold;
    return true;
}
bool
CLBayerPipeImageHandler::set_bnr_config (const XCam3aResultBayerNoiseReduction &bnr)
{
    for(int i = 0; i < XCAM_BNR_TABLE_SIZE; i++)
        _bnr_table[i] = (float)bnr.table[i];
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

XCamReturn
CLBayerPipeImageHandler::prepare_parameters (
    SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & in_video_info = input->get_video_info ();
    const VideoBufferInfo & out_video_info = output->get_video_info ();
    CLArgList args;
    CLWorkSize work_size;

    XCAM_ASSERT (_bayer_kernel.ptr ());

    CLImageDesc in_desc;
    in_desc.format.image_channel_order = CL_RGBA;
    in_desc.format.image_channel_data_type = CL_UNORM_INT16; //CL_UNSIGNED_INT32;
    in_desc.width = in_video_info.width / 4; // 960/4
    in_desc.height = in_video_info.aligned_height * 4;  //540
    in_desc.row_pitch = in_video_info.strides[0];

    SmartPtr<CLImage> image_in = convert_to_climage (context, input, in_desc);

    CLImageDesc out_desc;
    out_desc.format.image_channel_order = CL_RGBA;
    if (XCAM_PIX_FMT_RGB48_planar == out_video_info.format)
        out_desc.format.image_channel_data_type = CL_UNORM_INT16;
    else
        out_desc.format.image_channel_data_type = CL_UNORM_INT8;
    out_desc.width = out_video_info.aligned_width / 4;
    out_desc.height = out_video_info.aligned_height * 3;
    out_desc.row_pitch = out_video_info.strides[0];
    out_desc.array_size = 3;
    out_desc.slice_pitch = out_video_info.strides [0] * out_video_info.aligned_height;

    SmartPtr<CLImage> image_out = convert_to_climage (context, output, out_desc);

    uint input_height = in_video_info.aligned_height;
    uint output_height = out_video_info.aligned_height;

    XCAM_ASSERT (image_in.ptr () && image_out.ptr ());
    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", _bayer_kernel->get_kernel_name ());

    SmartPtr<CLBuffer> bnr_table_buffer = new CLBuffer(
        context, sizeof(float) * XCAM_BNR_TABLE_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &_bnr_table);

    //set args;
    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLArgumentT<uint> (input_height));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLArgumentT<uint> (output_height));
    args.push_back (new CLMemArgument (bnr_table_buffer));
    args.push_back (new CLArgumentT<uint32_t> (_enable_denoise));
    args.push_back (new CLArgumentT<CLEeConfig> (_ee_config));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = BAYER_LOCAL_X_SIZE;
    work_size.local[1] = BAYER_LOCAL_Y_SIZE;
    work_size.global[0] = (XCAM_ALIGN_UP(out_video_info.width, WORKGROUP_PIXEL_WIDTH) / WORKGROUP_PIXEL_WIDTH) *
                          work_size.local[0];
    work_size.global[1] = (XCAM_ALIGN_UP(out_video_info.height, WORKGROUP_PIXEL_HEIGHT) / WORKGROUP_PIXEL_HEIGHT) *
                          work_size.local[1];

    XCAM_ASSERT (_bayer_kernel.ptr ());
    XCamReturn ret = _bayer_kernel->set_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "bayer pipe kernel set arguments failed.");

    return XCAM_RETURN_NO_ERROR;
}


SmartPtr<CLImageHandler>
create_cl_bayer_pipe_image_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLBayerPipeImageHandler> bayer_pipe_handler;
    SmartPtr<CLBayerPipeImageKernel> bayer_pipe_kernel;

    bayer_pipe_handler = new CLBayerPipeImageHandler (context, "cl_handler_bayer_pipe");
    bayer_pipe_kernel = new CLBayerPipeImageKernel (context, bayer_pipe_handler);
    XCAM_ASSERT (bayer_pipe_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, bayer_pipe_kernel->build_kernel (kernel_bayer_pipe_info, NULL) == XCAM_RETURN_NO_ERROR, NULL,
        "build bayer-pipe kernel(%s) failed", kernel_bayer_pipe_info.kernel_name);

    XCAM_ASSERT (bayer_pipe_kernel->is_valid ());
    bayer_pipe_handler->set_bayer_kernel (bayer_pipe_kernel);

    return bayer_pipe_handler;
}

};
