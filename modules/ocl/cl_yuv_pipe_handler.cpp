/*
 * cl_yuv_pipe_handler.cpp - CL YuvPipe Pipe handler
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
 * Author: Wangfei <feix.w.wang@intel.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "cl_utils.h"
#include "cl_yuv_pipe_handler.h"

#define USE_BUFFER_OBJECT 0

namespace XCam {

static const XCamKernelInfo kernel_yuv_pipe_info = {
    "kernel_yuv_pipe",
#include "kernel_yuv_pipe.clx"
    , 0,
};

float default_matrix[XCAM_COLOR_MATRIX_SIZE] = {
    0.299f, 0.587f, 0.114f,
    -0.14713f, -0.28886f, 0.436f,
    0.615f, -0.51499f, -0.10001f,
};
float default_macc[XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE] = {
    1.000000f, 0.000000f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f,
    1.000000f, 0.000000f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f,
    1.000000f, 0.000000f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f,
    1.000000f, 0.000000f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f,
    1.000000f, 0.000000f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f,
    1.000000f, 0.000000f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f,
    1.000000f, 0.000000f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f,
    1.000000f, 0.000000f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 1.000000f,
};

CLYuvPipeImageKernel::CLYuvPipeImageKernel (const SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_yuv_pipe")

{
}

CLYuvPipeImageHandler::CLYuvPipeImageHandler (const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
    , _output_format(V4L2_PIX_FMT_NV12)
    , _enable_tnr_yuv (0)
    , _gain_yuv (1.0)
    , _thr_y (0.05)
    , _thr_uv (0.05)
    ,  _enable_tnr_yuv_state (0)

{
    memcpy(_macc_table, default_macc, sizeof(float)*XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE);
    memcpy(_rgbtoyuv_matrix, default_matrix, sizeof(float)*XCAM_COLOR_MATRIX_SIZE);
}

bool
CLYuvPipeImageHandler::set_macc_table (const XCam3aResultMaccMatrix &macc)
{
    for(int i = 0; i < XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE; i++)
        _macc_table[i] = (float)macc.table[i];
    return true;
}

bool
CLYuvPipeImageHandler::set_rgbtoyuv_matrix (const XCam3aResultColorMatrix &matrix)
{
    for (int i = 0; i < XCAM_COLOR_MATRIX_SIZE; i++)
        _rgbtoyuv_matrix[i] = (float)matrix.matrix[i];
    return true;
}

XCamReturn
CLYuvPipeImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    bool format_inited = output.init (_output_format, input.width, input.height);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) output format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (_output_format));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLYuvPipeImageHandler::prepare_parameters (
    SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info_in = input->get_video_info ();
    const VideoBufferInfo & video_info_out = output->get_video_info ();
    CLArgList args;
    CLWorkSize work_size;

    XCAM_ASSERT (_yuv_pipe_kernel.ptr ());
    SmartPtr<CLMemory> buffer_in, buffer_out, buffer_out_UV;

#if !USE_BUFFER_OBJECT
    CLImageDesc in_image_info;
    in_image_info.format.image_channel_order = CL_RGBA;
    in_image_info.format.image_channel_data_type = CL_UNSIGNED_INT32;
    in_image_info.width = video_info_in.aligned_width / 8;
    in_image_info.height = video_info_in.aligned_height * 3;
    in_image_info.row_pitch = video_info_in.strides[0];

    CLImageDesc out_image_info;
    out_image_info.format.image_channel_order = CL_RGBA;
    out_image_info.format.image_channel_data_type = CL_UNSIGNED_INT16;
    out_image_info.width = video_info_out.width / 8;
    out_image_info.height = video_info_out.aligned_height;
    out_image_info.row_pitch = video_info_out.strides[0];

    buffer_in = convert_to_climage (context, input, in_image_info);
    buffer_out = convert_to_climage (context, output, out_image_info, video_info_out.offsets[0]);

    out_image_info.height = video_info_out.aligned_height / 2;
    out_image_info.row_pitch = video_info_out.strides[1];
    buffer_out_UV = convert_to_climage (context, output, out_image_info, video_info_out.offsets[1]);
#else
    buffer_in = convert_to_clbuffer (context, input);
    buffer_out = convert_to_clbuffer (context, output);
#endif
    SmartPtr<CLBuffer> matrix_buffer = new CLBuffer (
        context, sizeof(float)*XCAM_COLOR_MATRIX_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_rgbtoyuv_matrix);
    SmartPtr<CLBuffer> macc_table_buffer = new CLBuffer(
        context, sizeof(float)*XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_macc_table);

    uint32_t plannar_offset = video_info_in.aligned_height;

    if (!_buffer_out_prev.ptr ()) {
        _buffer_out_prev = buffer_out;
        _buffer_out_prev_UV = buffer_out_UV;
        _enable_tnr_yuv_state = _enable_tnr_yuv;
        _enable_tnr_yuv = 0;
    }
    else {
        if (_enable_tnr_yuv == 0)
            _enable_tnr_yuv = _enable_tnr_yuv_state;
    }
    XCAM_FAIL_RETURN (
        WARNING,
        buffer_in->is_valid () && buffer_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image handler(%s) in/out memory not available", XCAM_STR (get_name ()));

    //set args;
    args.push_back (new CLMemArgument (buffer_out));

#if !USE_BUFFER_OBJECT
    args.push_back (new CLMemArgument (buffer_out_UV));
#endif

    args.push_back (new CLMemArgument (_buffer_out_prev));

#if !USE_BUFFER_OBJECT
    args.push_back (new CLMemArgument (_buffer_out_prev_UV));
#else
    uint32_t vertical_offset = video_info_out.aligned_height;
    args.push_back (new CLArgumentT<uint32_t> (vertical_offset));
#endif
    args.push_back (new CLArgumentT<uint32_t> (plannar_offset));
    args.push_back (new CLMemArgument (matrix_buffer));
    args.push_back (new CLMemArgument (macc_table_buffer));
    args.push_back (new CLArgumentT<float> (_gain_yuv));
    args.push_back (new CLArgumentT<float> (_thr_y));
    args.push_back (new CLArgumentT<float> (_thr_uv));
    args.push_back (new CLArgumentT<uint32_t> (_enable_tnr_yuv));
    args.push_back (new CLMemArgument (buffer_in));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = video_info_out.width / 8 ;
    work_size.global[1] = video_info_out.aligned_height / 2 ;
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    XCAM_ASSERT (_yuv_pipe_kernel.ptr ());
    XCamReturn ret = _yuv_pipe_kernel->set_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "yuv pipe kernel set arguments failed.");

    if (buffer_out->is_valid ()) {
        _buffer_out_prev = buffer_out;
        _buffer_out_prev_UV = buffer_out_UV;
    }

    return XCAM_RETURN_NO_ERROR;
}

bool
CLYuvPipeImageHandler::set_yuv_pipe_kernel(SmartPtr<CLYuvPipeImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _yuv_pipe_kernel = kernel;
    return true;
}

bool
CLYuvPipeImageHandler::set_tnr_yuv_config (const XCam3aResultTemporalNoiseReduction& config)
{
    if (!_yuv_pipe_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set config error, invalid YUV-Pipe kernel !");
    }

    _gain_yuv = (float)config.gain;
    _thr_y = (float)config.threshold[0];
    _thr_uv = (float)config.threshold[1];
    XCAM_LOG_DEBUG ("set TNR YUV config: _gain(%f), _thr_y(%f), _thr_uv(%f)",
                    _gain_yuv, _thr_y, _thr_uv);
    return true;
}

bool
CLYuvPipeImageHandler::set_tnr_enable (bool enable_tnr_yuv)
{
    _enable_tnr_yuv = (enable_tnr_yuv ? 1 : 0);
    return true;
}

SmartPtr<CLImageHandler>
create_cl_yuv_pipe_image_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLYuvPipeImageHandler> yuv_pipe_handler;
    SmartPtr<CLYuvPipeImageKernel> yuv_pipe_kernel;

    yuv_pipe_kernel = new CLYuvPipeImageKernel (context);
    XCAM_ASSERT (yuv_pipe_kernel.ptr ());
    const char * options = USE_BUFFER_OBJECT ? "-DUSE_BUFFER_OBJECT=1" : "-DUSE_BUFFER_OBJECT=0";
    XCAM_FAIL_RETURN (
        ERROR, yuv_pipe_kernel->build_kernel (kernel_yuv_pipe_info, options) == XCAM_RETURN_NO_ERROR, NULL,
        "build yuv-pipe kernel(%s) failed", kernel_yuv_pipe_info.kernel_name);

    XCAM_ASSERT (yuv_pipe_kernel->is_valid ());
    yuv_pipe_handler = new CLYuvPipeImageHandler (context, "cl_handler_pipe_yuv");
    yuv_pipe_handler->set_yuv_pipe_kernel (yuv_pipe_kernel);

    return yuv_pipe_handler;
}

};
