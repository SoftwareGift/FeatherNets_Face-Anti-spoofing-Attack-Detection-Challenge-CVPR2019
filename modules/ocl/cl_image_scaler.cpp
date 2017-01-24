/*
 * cl_image_scaler.cpp - CL image scaler
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "xcam_utils.h"
#include "cl_image_scaler.h"

namespace XCam {

CLScalerKernel::CLScalerKernel (
    SmartPtr<CLContext> &context,
    CLImageScalerMemoryLayout mem_layout
)
    : CLImageKernel (context, "kernel_image_scaler")
    , _pixel_format (V4L2_PIX_FMT_NV12)
    , _mem_layout (mem_layout)
    , _output_width (0)
    , _output_height (0)
{
}

SmartPtr<DrmBoBuffer>
CLScalerKernel::get_input_parameter (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (output);
    return input;
}

SmartPtr<DrmBoBuffer>
CLScalerKernel::get_output_parameter (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (input);
    return output;
}

XCamReturn
CLScalerKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<DrmBoBuffer> input_buf = get_input_parameter (input, output);
    SmartPtr<DrmBoBuffer> output_buf = get_output_parameter (input, output);

    XCAM_FAIL_RETURN (
        WARNING,
        input_buf.ptr () && output_buf.ptr (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) get input/output buffer failed", get_kernel_name ());

    const VideoBufferInfo &input_info = input_buf->get_video_info ();
    const VideoBufferInfo &output_info = output_buf->get_video_info ();

    _pixel_format = input_info.format;

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

        _image_out = new CLVaImage (context, output_buf, output_imageDesc, output_info.offsets[1]);
        _output_width = output_info.width / 2;
        _output_height = output_info.height / 2;
    } else {
        output_imageDesc.format.image_channel_order = CL_R;
        output_imageDesc.width = output_info.width;
        output_imageDesc.height = output_info.height;
        output_imageDesc.row_pitch = output_info.strides[0];

        _image_out = new CLVaImage (context, output_buf, output_imageDesc, output_info.offsets[0]);
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

        _image_in = new CLVaImage (context, input_buf, input_imageDesc, input_info.offsets[1]);
    } else {
        input_imageDesc.format.image_channel_order = CL_R;
        input_imageDesc.width = input_info.width;
        input_imageDesc.height = input_info.height;
        input_imageDesc.row_pitch = input_info.strides[0];

        _image_in = new CLVaImage (context, input_buf, input_imageDesc, input_info.offsets[0]);
    }

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_output_width;
    args[2].arg_size = sizeof (_output_width);
    args[3].arg_adress = &_output_height;
    args[3].arg_size = sizeof (_output_height);
    arg_count = 4;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = XCAM_ALIGN_UP (_output_width, XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE0);
    work_size.global[1] = XCAM_ALIGN_UP (_output_height, XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE1);
    work_size.local[0] = XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE0;
    work_size.local[1] = XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE1;

    return ret;
}


CLImageScalerKernel::CLImageScalerKernel (
    SmartPtr<CLContext> &context,
    CLImageScalerMemoryLayout mem_layout,
    SmartPtr<CLImageScaler> &scaler
)
    : CLScalerKernel (context, mem_layout)
    , _scaler (scaler)
{
}

SmartPtr<DrmBoBuffer>
CLImageScalerKernel::get_output_parameter (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    return _scaler->get_scaler_buf ();
}

XCamReturn
CLImageScalerKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if ((V4L2_PIX_FMT_NV12 != get_pixel_format ()) ||
            ((CL_IMAGE_SCALER_NV12_UV == get_mem_layout ()) && (V4L2_PIX_FMT_NV12 == get_pixel_format ()))) {
        SmartPtr<DrmBoBuffer> buffer;
        get_context ()->finish();

        _image_in.release ();

        buffer = _scaler->get_scaler_buf ();
        XCAM_ASSERT (buffer.ptr ());

        //post buffer out
        ret = _scaler->post_buffer (buffer);
    }

    CLScalerKernel::post_execute (output);
    return ret;
}

void
CLImageScalerKernel::pre_stop ()
{
    if (_scaler.ptr ())
        _scaler.ptr ()->pre_stop ();
}

CLImageScaler::CLImageScaler ()
    : CLImageHandler ("CLImageScaler")
    , _h_scaler_factor (0.5)
    , _v_scaler_factor (0.5)
{
}

void
CLImageScaler::pre_stop ()
{
    if (_scaler_buf_pool.ptr ())
        _scaler_buf_pool->stop ();
}

bool
CLImageScaler::set_scaler_factor (const double factor)
{
    _h_scaler_factor = factor;
    _v_scaler_factor = factor;

    return true;
}

double
CLImageScaler::get_scaler_factor () const
{
    if (_h_scaler_factor != _v_scaler_factor)
        XCAM_LOG_WARNING ("non-uniform scale, return horizontal scale factor");

    return _h_scaler_factor;
};

bool
CLImageScaler::set_scaler_factor (const double h_factor, const double v_factor)
{
    _h_scaler_factor = h_factor;
    _v_scaler_factor = v_factor;

    return true;
}

bool
CLImageScaler::get_scaler_factor (double &h_factor, double &v_factor) const
{
    h_factor = _h_scaler_factor;
    v_factor = _v_scaler_factor;

    return true;
};

XCamReturn
CLImageScaler::prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    output = input;

    ret = prepare_scaler_buf (input->get_video_info (), _scaler_buf);
    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "CLImageScalerKernel prepare scaled video buf failed");

    _scaler_buf->set_timestamp (input->get_timestamp ());

    return ret;
}

XCamReturn
CLImageScaler::prepare_scaler_buf (const VideoBufferInfo &video_info, SmartPtr<DrmBoBuffer> &output)
{
    SmartPtr<BufferProxy> buffer;
    SmartPtr<DrmDisplay> display;

    if (!_scaler_buf_pool.ptr ()) {
        VideoBufferInfo scaler_video_info;
        uint32_t new_width = XCAM_ALIGN_UP ((uint32_t)(video_info.width * _h_scaler_factor),
                                            2 * XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE0);
        uint32_t new_height = XCAM_ALIGN_UP ((uint32_t)(video_info.height * _v_scaler_factor),
                                             2 * XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE1);

        scaler_video_info.init (video_info.format, new_width, new_height);

        display = DrmDisplay::instance ();
        XCAM_ASSERT (display.ptr ());
        _scaler_buf_pool = new DrmBoBufferPool (display);
        _scaler_buf_pool->set_video_info (scaler_video_info);
        _scaler_buf_pool->reserve (6);
    }

    buffer = _scaler_buf_pool->get_buffer (_scaler_buf_pool);
    XCAM_ASSERT (buffer.ptr ());

    output = buffer.dynamic_cast_ptr<DrmBoBuffer> ();
    XCAM_ASSERT (output.ptr ());
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageScaler::post_buffer (const SmartPtr<DrmBoBuffer> &buffer)
{
    if (_scaler_callback.ptr ())
        return _scaler_callback->scaled_image_ready (buffer);

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_image_scaler_handler (SmartPtr<CLContext> &context, const uint32_t format)
{
    SmartPtr<CLImageScaler> scaler_handler;
    SmartPtr<CLImageKernel> scaler_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    scaler_handler = new CLImageScaler ();
    XCAM_ASSERT (scaler_handler.ptr ());

    if (V4L2_PIX_FMT_NV12 == format) {
        scaler_kernel = new CLImageScalerKernel (context, CL_IMAGE_SCALER_NV12_Y, scaler_handler);
    } else if (XCAM_PIX_FMT_RGBA64 == format) {
        scaler_kernel = new CLImageScalerKernel (context, CL_IMAGE_SCALER_RGBA, scaler_handler);
    } else {
        XCAM_LOG_ERROR ("create cl image scaler failed, unknown format:0x%08x", format);
        return NULL;
    }

    XCAM_ASSERT (scaler_kernel.ptr ());
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_image_scaler)
#include "kernel_image_scaler.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = scaler_kernel->load_from_source (kernel_image_scaler_body, strlen (kernel_image_scaler_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", scaler_kernel->get_kernel_name());
    }
    XCAM_ASSERT (scaler_kernel->is_valid ());
    scaler_handler->add_kernel (scaler_kernel);

    if (V4L2_PIX_FMT_NV12 == format) {
        SmartPtr<CLImageKernel> uv_scaler_kernel =
            new CLImageScalerKernel (context, CL_IMAGE_SCALER_NV12_UV, scaler_handler);
        {
            XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_image_scaler)
#include "kernel_image_scaler.clx"
            XCAM_CL_KERNEL_FUNC_END;
            ret = uv_scaler_kernel->load_from_source (kernel_image_scaler_body, strlen (kernel_image_scaler_body));
            XCAM_FAIL_RETURN (
                WARNING,
                ret == XCAM_RETURN_NO_ERROR,
                NULL,
                "CL image handler(%s) load source failed", uv_scaler_kernel->get_kernel_name());
        }
        XCAM_ASSERT (uv_scaler_kernel->is_valid ());
        scaler_handler->add_kernel (uv_scaler_kernel);
    }
    return scaler_handler;
}

};
