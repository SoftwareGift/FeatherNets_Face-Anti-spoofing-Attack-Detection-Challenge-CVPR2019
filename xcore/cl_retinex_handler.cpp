/*
 * cl_retinex_handler.cpp - CL retinex handler
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
 * Author: wangfei <feix.w.wang@intel.com>
 *             Wind Yuan <feng.yuan@intel.com>
 */

#include "xcam_utils.h"
#include "cl_retinex_handler.h"
#include <algorithm>
#include "cl_device.h"
#include "cl_image_bo_buffer.h"

#define RETINEX_SCALER_FACTOR 0.4

namespace XCam {

CLRetinexScalerImageKernel::CLRetinexScalerImageKernel (SmartPtr<CLContext> &context,
        CLImageScalerMemoryLayout mem_layout,
        SmartPtr<CLRetinexImageHandler> &scaler)
    :  CLScalerKernel (context, mem_layout),
       _scaler(scaler)
{
}

SmartPtr<DrmBoBuffer>
CLRetinexScalerImageKernel::get_output_parameter (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    return _scaler->get_scaler_buf1 ();
}

void
CLRetinexScalerImageKernel::pre_stop ()
{
    if (_scaler.ptr ())
        _scaler.ptr ()->pre_stop ();
}

CLRetinexGaussImageKernel::CLRetinexGaussImageKernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLRetinexImageHandler> &scaler,
    uint32_t radius, float sigma)
    : CLGaussImageKernel (context, radius, sigma)
    , _scaler(scaler)
{
}

SmartPtr<DrmBoBuffer>
CLRetinexGaussImageKernel::get_input_parameter (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    return _scaler->get_scaler_buf1 ();
}
SmartPtr<DrmBoBuffer>
CLRetinexGaussImageKernel::get_output_parameter (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    return _scaler->get_scaler_buf2 ();
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

    CLImageDesc cl_desc_in, cl_desc_out, cl_desc_ga;

    cl_desc_in.format.image_channel_data_type = CL_UNORM_INT8; //CL_UNSIGNED_INT32;
    cl_desc_in.format.image_channel_order = CL_RGBA;
    cl_desc_in.width = video_info_in.width / 4; // 16;
    cl_desc_in.height = video_info_in.height;
    cl_desc_in.row_pitch = video_info_in.strides[0];
    _image_in = new CLVaImage (context, input, cl_desc_in, video_info_in.offsets[0]);

    cl_desc_in.height = video_info_in.height / 2;
    cl_desc_in.row_pitch = video_info_in.strides[1];
    _image_in_uv = new CLVaImage (context, input, cl_desc_in, video_info_in.offsets[1]);

    cl_desc_out.format.image_channel_data_type = CL_UNORM_INT8; //CL_UNSIGNED_INT32;
    cl_desc_out.format.image_channel_order = CL_RGBA;
    cl_desc_out.width = video_info_out.width / 4; // 16;
    cl_desc_out.height = video_info_out.height;
    cl_desc_out.row_pitch = video_info_out.strides[0];
    _image_out = new CLVaImage (context, output, cl_desc_out, video_info_out.offsets[0]);

    cl_desc_out.height = video_info_out.height / 2;
    cl_desc_out.row_pitch = video_info_out.strides[1];
    _image_out_uv = new CLVaImage (context, output, cl_desc_out, video_info_out.offsets[1]);


    SmartPtr<DrmBoBuffer> scaler_buf = _scaler->get_scaler_buf2 ();
    XCAM_ASSERT (scaler_buf.ptr ());
    const VideoBufferInfo & video_info_scale = scaler_buf->get_video_info ();

    cl_desc_ga.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_ga.format.image_channel_order = CL_R;
    cl_desc_ga.width = video_info_scale.width;
    cl_desc_ga.height = video_info_scale.height;
    cl_desc_ga.row_pitch = video_info_scale.strides[0];
    _image_in_ga = new CLVaImage (context, scaler_buf, cl_desc_ga, video_info_scale.offsets[0]);

    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_in_uv->is_valid () &&
        _image_out->is_valid () && _image_out_uv->is_valid() &&
        _image_in_ga->is_valid(),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());


    _retinex_config.log_min = -0.1f;
    _retinex_config.log_max = 0.2f;
    _retinex_config.gain = 1.0f / (_retinex_config.log_max - _retinex_config.log_min);
    _retinex_config.width = (float)video_info_in.width;
    _retinex_config.height = (float)video_info_in.height;


    //set args;
    arg_count = 0;
    args[arg_count].arg_adress = &_image_in->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_image_in_uv->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_image_in_ga->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_image_out->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_image_out_uv->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;


    args[arg_count].arg_adress = &_retinex_config;
    args[arg_count].arg_size = sizeof (CLRetinexConfig);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = video_info_out.width / 4;
    work_size.global[1] = video_info_out.height;
    work_size.local[0] = 16;
    work_size.local[1] = 2;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLRetinexImageKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    _image_in_ga.release ();
    _image_in_uv.release ();
    _image_out_uv.release ();
    return CLImageKernel::post_execute (output);
}

CLRetinexImageHandler::CLRetinexImageHandler (const char *name)
    : CLImageHandler (name)
    , _scaler_factor(RETINEX_SCALER_FACTOR)
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
    ret = prepare_scaler_buf (input->get_video_info ());
    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "CLImageScalerKernel prepare scaled video buf failed");

    return XCAM_RETURN_NO_ERROR;

}

XCamReturn
CLRetinexImageHandler::prepare_scaler_buf (const VideoBufferInfo &video_info)
{
    SmartPtr<BufferProxy> buffer;


    if (!_scaler_buf_pool.ptr ()) {
        SmartPtr<DrmDisplay> display = DrmDisplay::instance ();
        SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
        VideoBufferInfo scaler_video_info;
        uint32_t new_width = XCAM_ALIGN_UP ((uint32_t)(video_info.width * _scaler_factor), 8);
        uint32_t new_height = XCAM_ALIGN_UP ((uint32_t)(video_info.height * _scaler_factor), 4);

        scaler_video_info.init (video_info.format, new_width, new_height);

        XCAM_ASSERT (display.ptr ());
        _scaler_buf_pool = new CLBoBufferPool (display, context);
        XCAM_ASSERT (_scaler_buf_pool.ptr ());
        _scaler_buf_pool->set_video_info (scaler_video_info);
        _scaler_buf_pool->reserve (4);

        _scaler_buf1 = _scaler_buf_pool->get_buffer (_scaler_buf_pool).dynamic_cast_ptr<DrmBoBuffer> ();
        _scaler_buf2 = _scaler_buf_pool->get_buffer (_scaler_buf_pool).dynamic_cast_ptr<DrmBoBuffer> ();
        XCAM_ASSERT (_scaler_buf1.ptr () && _scaler_buf2.ptr ());
    }

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

SmartPtr<CLRetinexScalerImageKernel>
create_kernel_retinex_scaler (SmartPtr<CLContext> &context, SmartPtr<CLRetinexImageHandler> handler)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLRetinexScalerImageKernel> kernel;

    kernel = new CLRetinexScalerImageKernel (context, CL_IMAGE_SCALER_NV12_Y, handler);
    XCAM_ASSERT (kernel.ptr ());
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_image_scaler)
#include "kernel_image_scaler.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = kernel->load_from_source (kernel_image_scaler_body, strlen (kernel_image_scaler_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", kernel->get_kernel_name());
    }
    return kernel;
}

SmartPtr<CLRetinexGaussImageKernel>
create_kernel_retinex_gaussian (
    SmartPtr<CLContext> &context,
    SmartPtr<CLRetinexImageHandler> handler,
    uint32_t radius, float sigma)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLRetinexGaussImageKernel> kernel;

    kernel = new CLRetinexGaussImageKernel (context, handler, radius, sigma);
    {
        char build_options[1024];
        xcam_mem_clear (build_options);
        snprintf (build_options, sizeof (build_options), " -DGAUSS_RADIUS=%d ", radius);

        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_gauss)
#include "kernel_gauss.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = kernel->load_from_source (
                  kernel_gauss_body, strlen (kernel_gauss_body),
                  NULL, NULL, build_options);
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", kernel->get_kernel_name());
    }
    return kernel;
}

SmartPtr<CLRetinexImageKernel>
create_kernel_retinex (SmartPtr<CLContext> &context, SmartPtr<CLRetinexImageHandler> handler)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLRetinexImageKernel> kernel;

    kernel = new CLRetinexImageKernel (context, handler);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_retinex)
#include "kernel_retinex.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = kernel->load_from_source (kernel_retinex_body, strlen (kernel_retinex_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", kernel->get_kernel_name());
    }
    return kernel;
}

SmartPtr<CLImageHandler>
create_cl_retinex_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLRetinexImageHandler> retinex_handler;

    SmartPtr<CLRetinexScalerImageKernel> retinex_scaler_kernel;
    SmartPtr<CLRetinexGaussImageKernel> retinex_gauss_kernel;
    SmartPtr<CLRetinexImageKernel> retinex_kernel;

    retinex_handler = new CLRetinexImageHandler ("cl_handler_retinex");
    retinex_scaler_kernel = create_kernel_retinex_scaler (context, retinex_handler);
    XCAM_FAIL_RETURN (
        ERROR,
        retinex_scaler_kernel.ptr () && retinex_scaler_kernel->is_valid (),
        NULL,
        "Retinex handler create scaler kernel failed");
    retinex_handler->set_retinex_scaler_kernel (retinex_scaler_kernel);

    retinex_gauss_kernel = create_kernel_retinex_gaussian (context, retinex_handler, 2, 2.0);
    XCAM_FAIL_RETURN (
        ERROR,
        retinex_gauss_kernel.ptr () && retinex_gauss_kernel->is_valid (),
        NULL,
        "Retinex handler create gaussian kernel failed");
    retinex_handler->set_retinex_gauss_kernel (retinex_gauss_kernel);

    retinex_kernel = create_kernel_retinex (context, retinex_handler);
    XCAM_FAIL_RETURN (
        ERROR,
        retinex_kernel.ptr () && retinex_kernel->is_valid (),
        NULL,
        "Retinex handler create retinex kernel failed");
    retinex_handler->set_retinex_kernel (retinex_kernel);

    return retinex_handler;
}

}
