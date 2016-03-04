/*
 * cl_retinex_handler.cpp - CL retinex handler
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
 * Author: wangfei <feix.w.wang@intel.com>
 */
#include "xcam_utils.h"
#include "cl_retinex_handler.h"
#include <algorithm>

namespace XCam {

CLRetinexScalerImageKernel::CLRetinexScalerImageKernel (SmartPtr<CLContext> &context,
        CLImageScalerMemoryLayout mem_layout,
        SmartPtr<CLRetinexImageHandler> &scaler)
    :  CLScalerKernel (context, mem_layout),
       _scaler(scaler)
{
}

XCamReturn
CLRetinexScalerImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo &input_info = input->get_video_info ();
    _pixel_format = input_info.format;

    XCAM_UNUSED (output);
    SmartPtr<DrmBoBuffer> scaler_buf = _scaler->get_scaler_buf ();
    XCAM_ASSERT (scaler_buf.ptr ());

    const VideoBufferInfo & output_info = scaler_buf->get_video_info ();
    CLImageDesc output_imageDesc;
    uint32_t channel_bits = XCAM_ALIGN_UP (output_info.color_bits, 8);
    if (channel_bits == 8)
        output_imageDesc.format.image_channel_data_type = CL_UNSIGNED_INT8;
    else if (channel_bits == 16)
        output_imageDesc.format.image_channel_data_type = CL_UNSIGNED_INT16;

    if ((CL_IMAGE_SCALER_NV12_UV == get_mem_layout ()) && (V4L2_PIX_FMT_NV12 == input_info.format)) {
        output_imageDesc.format.image_channel_order = CL_RG;
        output_imageDesc.width = output_info.width / 2;
        output_imageDesc.height = output_info.height / 2;
        output_imageDesc.row_pitch = output_info.strides[1];

        _cl_image_out = new CLVaImage (context, scaler_buf, output_imageDesc, output_info.offsets[1]);
        _output_width = output_info.width / 2;
        _output_height = output_info.height / 2;
    } else {
        output_imageDesc.format.image_channel_order = CL_R;
        output_imageDesc.width = output_info.width;
        output_imageDesc.height = output_info.height;
        output_imageDesc.row_pitch = output_info.strides[0];

        _cl_image_out = new CLVaImage (context, scaler_buf, output_imageDesc, 0);
        _output_width = output_info.width;
        _output_height = output_info.height;
    }

    CLImageDesc input_imageDesc;
    channel_bits = XCAM_ALIGN_UP (input_info.color_bits, 8);
    if (channel_bits == 8)
        input_imageDesc.format.image_channel_data_type = CL_UNSIGNED_INT8;
    else if (channel_bits == 16)
        input_imageDesc.format.image_channel_data_type = CL_UNSIGNED_INT16;

    if ((CL_IMAGE_SCALER_NV12_UV == get_mem_layout ()) && (V4L2_PIX_FMT_NV12 == input_info.format)) {
        input_imageDesc.format.image_channel_order = CL_RG;
        input_imageDesc.width = input_info.width / 2;
        input_imageDesc.height = input_info.height / 2;
        input_imageDesc.row_pitch = input_info.strides[1];

        _image_in = new CLVaImage (context, input, input_imageDesc, input_info.offsets[1]);
    } else {
        input_imageDesc.format.image_channel_order = CL_R;
        input_imageDesc.width = input_info.width;
        input_imageDesc.height = input_info.height;
        input_imageDesc.row_pitch = input_info.strides[0];

        _image_in = new CLVaImage (context, input, input_imageDesc, 0);
    }

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_cl_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_output_width;
    args[2].arg_size = sizeof (_output_width);
    args[3].arg_adress = &_output_height;
    args[3].arg_size = sizeof (_output_height);
    arg_count = 4;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = _output_width;
    work_size.global[1] = _output_height;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return ret;
}

XCamReturn
CLRetinexScalerImageKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (output);
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if ((V4L2_PIX_FMT_NV12 != get_pixel_format ()) ||
            ((CL_IMAGE_SCALER_NV12_UV == get_mem_layout ()) && (V4L2_PIX_FMT_NV12 == get_pixel_format ()))) {
        get_context ()->finish();
        _image_in.release ();
    }
    return ret;
}

void
CLRetinexScalerImageKernel::pre_stop ()
{
    if (_scaler.ptr ())
        _scaler.ptr ()->pre_stop ();
}

CLRetinexGaussImageKernel::CLRetinexGaussImageKernel (SmartPtr<CLContext> &context,
        SmartPtr<CLRetinexImageHandler> &scaler)
    :  CLGaussImageKernel (context),
       _scaler(scaler)
{
}

XCamReturn
CLRetinexGaussImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_UNUSED (input);
    XCAM_UNUSED (output);

    SmartPtr<CLContext> context = get_context ();

    SmartPtr<DrmBoBuffer> scaler_buf = _scaler->get_scaler_buf ();
    XCAM_ASSERT (scaler_buf.ptr ());

    const VideoBufferInfo & buf_info = scaler_buf->get_video_info ();

    _image_in = new CLVaImage (context, scaler_buf);
    _image_out = new CLVaImage (context, scaler_buf);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    _vertical_offset_in = buf_info.aligned_height;
    _vertical_offset_out = buf_info.aligned_height;

    _g_table_buffer = new CLBuffer(
        context, sizeof(float)*XCAM_GAUSS_TABLE_SIZE * XCAM_GAUSS_TABLE_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_g_table);

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_vertical_offset_in;
    args[2].arg_size = sizeof (_vertical_offset_in);
    args[3].arg_adress = &_vertical_offset_out;
    args[3].arg_size = sizeof (_vertical_offset_out);
    args[4].arg_adress = &_g_table_buffer->get_mem_id();
    args[4].arg_size = sizeof (cl_mem);
    arg_count = 5;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = buf_info.width;
    work_size.global[1] = buf_info.height;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return ret;
}

CLRetinexImageKernel::CLRetinexImageKernel (SmartPtr<CLContext> &context, SmartPtr<CLRetinexImageHandler> &scaler)
    : CLImageKernel (context, "kernel_retinex"),
      _scaler(scaler)
{
}

XCamReturn
CLRetinexImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info_in = input->get_video_info ();
    const VideoBufferInfo & video_info_out = output->get_video_info ();
    _retinex_config.log_min = -0.1;
    _retinex_config.log_max = 0.3;
    _retinex_config.gain = 255.0 / (_retinex_config.log_max - _retinex_config.log_min);
    _retinex_config.width = (float)video_info_in.width;
    _retinex_config.height = (float)video_info_in.height;

    SmartPtr<DrmBoBuffer> scaler_buf = _scaler->get_scaler_buf ();
    XCAM_ASSERT (scaler_buf.ptr ());

    _image_in_ga = new CLVaImage (context, scaler_buf);
    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid () && _image_in->is_valid());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid () && _image_in->is_valid(),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    _vertical_offset_in = video_info_in.aligned_height;
    _vertical_offset_out = video_info_out.aligned_height;

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_in_ga->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_image_out->get_mem_id ();
    args[2].arg_size = sizeof (cl_mem);
    args[3].arg_adress = &_vertical_offset_in;
    args[3].arg_size = sizeof (_vertical_offset_in);
    args[4].arg_adress = &_vertical_offset_out;
    args[4].arg_size = sizeof (_vertical_offset_out);
    args[5].arg_adress = &_retinex_config;
    args[5].arg_size = sizeof (CLRetinexConfig);

    arg_count = 6;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = video_info_in.width;
    work_size.global[1] = video_info_in.height;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

CLRetinexImageHandler::CLRetinexImageHandler (const char *name)
    : CLImageHandler (name)
    , _scaler_factor(0.5)
{
}

void
CLRetinexImageHandler::pre_stop ()
{
    if (_scaler_buf_pool.ptr ())
        _scaler_buf_pool->stop ();
}

XCamReturn
CLRetinexImageHandler::prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    CLImageHandler::prepare_output_buf(input, output);
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    ret = prepare_scaler_buf (input->get_video_info (), _scaler_buf);
    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "CLImageScalerKernel prepare scaled video buf failed");

    _scaler_buf->set_timestamp (input->get_timestamp ());

    return XCAM_RETURN_NO_ERROR;

}

XCamReturn
CLRetinexImageHandler::prepare_scaler_buf (const VideoBufferInfo &video_info, SmartPtr<DrmBoBuffer> &output)
{
    SmartPtr<BufferProxy> buffer;
    SmartPtr<DrmDisplay> display;

    if (!_scaler_buf_pool.ptr ()) {
        VideoBufferInfo scaler_video_info;
        uint32_t new_width = XCAM_ALIGN_UP ((uint32_t)(video_info.width * _scaler_factor),
                                            2 * XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE);
        uint32_t new_height = XCAM_ALIGN_UP ((uint32_t)(video_info.height * _scaler_factor),
                                             2 * XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE);

        scaler_video_info.init (video_info.format, new_width, new_height);

        display = DrmDisplay::instance ();
        XCAM_ASSERT (display.ptr ());
        _scaler_buf_pool = new ScaledVideoBufferPool (display);
        _scaler_buf_pool->set_video_info (scaler_video_info);
        _scaler_buf_pool->reserve (6);
    }

    buffer = _scaler_buf_pool->get_buffer (_scaler_buf_pool);
    XCAM_ASSERT (buffer.ptr ());

    output = buffer.dynamic_cast_ptr<DrmBoBuffer> ();
    XCAM_ASSERT (output.ptr ());
    return XCAM_RETURN_NO_ERROR;
}

bool
CLRetinexImageHandler::set_retinex_kernel(SmartPtr<CLRetinexImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _retinex_kernel = kernel;
    return true;
}

bool
CLRetinexImageHandler::set_retinex_scaler_kernel(SmartPtr<CLRetinexScalerImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _retinex_scaler_kernel = kernel;
    return true;
}

bool
CLRetinexImageHandler::set_retinex_gauss_kernel(SmartPtr<CLRetinexGaussImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _retinex_gauss_kernel = kernel;
    return true;
}

SmartPtr<CLImageHandler>
create_cl_retinex_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLRetinexImageHandler> retinex_handler;

    SmartPtr<CLRetinexScalerImageKernel> retinex_scaler_kernel;
    SmartPtr<CLRetinexGaussImageKernel> retinex_gauss_kernel;
    SmartPtr<CLRetinexImageKernel> retinex_kernel;

    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    retinex_handler = new CLRetinexImageHandler ("cl_handler_retinex");

    retinex_scaler_kernel = new CLRetinexScalerImageKernel (context, CL_IMAGE_SCALER_NV12_Y, retinex_handler);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_image_scaler)
#include "kernel_image_scaler.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = retinex_scaler_kernel->load_from_source (kernel_image_scaler_body, strlen (kernel_image_scaler_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", retinex_scaler_kernel->get_kernel_name());
    }
    XCAM_ASSERT (retinex_scaler_kernel->is_valid ());
    retinex_handler->set_retinex_scaler_kernel (retinex_scaler_kernel);

    retinex_gauss_kernel = new CLRetinexGaussImageKernel (context, retinex_handler);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_gauss)
#include "kernel_gauss.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = retinex_gauss_kernel->load_from_source (kernel_gauss_body, strlen (kernel_gauss_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", retinex_gauss_kernel->get_kernel_name());
    }
    XCAM_ASSERT (retinex_gauss_kernel->is_valid ());
    retinex_handler->set_retinex_gauss_kernel (retinex_gauss_kernel);

    retinex_kernel = new CLRetinexImageKernel (context, retinex_handler);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_retinex)
#include "kernel_retinex.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = retinex_kernel->load_from_source (kernel_retinex_body, strlen (kernel_retinex_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", retinex_kernel->get_kernel_name());
    }
    XCAM_ASSERT (retinex_kernel->is_valid ());
    retinex_handler->set_retinex_kernel (retinex_kernel);

    return retinex_handler;
}

}
