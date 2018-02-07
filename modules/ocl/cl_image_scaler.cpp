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

#include "cl_utils.h"
#include "cl_image_scaler.h"

namespace XCam {

static const XCamKernelInfo kernel_scale_info = {
    "kernel_image_scaler",
#include "kernel_image_scaler.clx"
    , 0,
};

CLScalerKernel::CLScalerKernel (
    const SmartPtr<CLContext> &context,
    CLImageScalerMemoryLayout mem_layout
)
    : CLImageKernel (context, "kernel_image_scaler")
    , _mem_layout (mem_layout)
{
}

XCamReturn
CLScalerKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<VideoBuffer> input = get_input_buffer ();
    SmartPtr<VideoBuffer> output = get_output_buffer ();
    SmartPtr<CLImage> image_in, image_out;

    XCAM_FAIL_RETURN (
        WARNING,
        input.ptr () && output.ptr (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) get input/output buffer failed", XCAM_STR(get_kernel_name ()));

    const VideoBufferInfo &input_info = input->get_video_info ();
    const VideoBufferInfo &output_info = output->get_video_info ();

    uint32_t output_width = 0, output_height = 0;
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

        image_out = convert_to_climage (context, output, output_imageDesc, output_info.offsets[1]);
        output_width = output_info.width / 2;
        output_height = output_info.height / 2;
    } else {
        output_imageDesc.format.image_channel_order = CL_R;
        output_imageDesc.width = output_info.width;
        output_imageDesc.height = output_info.height;
        output_imageDesc.row_pitch = output_info.strides[0];

        image_out = convert_to_climage (context, output, output_imageDesc, output_info.offsets[0]);
        output_width = output_info.width;
        output_height = output_info.height;
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

        image_in = convert_to_climage (context, input, input_imageDesc, input_info.offsets[1]);
    } else {
        input_imageDesc.format.image_channel_order = CL_R;
        input_imageDesc.width = input_info.width;
        input_imageDesc.height = input_info.height;
        input_imageDesc.row_pitch = input_info.strides[0];

        image_in = convert_to_climage (context, input, input_imageDesc, input_info.offsets[0]);
    }

    //set args;
    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLArgumentT<uint32_t> (output_width));
    args.push_back (new CLArgumentT<uint32_t> (output_height));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = XCAM_ALIGN_UP (output_width, XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE0);
    work_size.global[1] = XCAM_ALIGN_UP (output_height, XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE1);
    work_size.local[0] = XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE0;
    work_size.local[1] = XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE1;

    return ret;
}

CLImageScalerKernel::CLImageScalerKernel (
    const SmartPtr<CLContext> &context,
    CLImageScalerMemoryLayout mem_layout,
    SmartPtr<CLImageScaler> &scaler
)
    : CLScalerKernel (context, mem_layout)
    , _scaler (scaler)
{
}

SmartPtr<VideoBuffer>
CLImageScalerKernel::get_input_buffer ()
{
    return _scaler->get_input_buf ();
}

SmartPtr<VideoBuffer>
CLImageScalerKernel::get_output_buffer ()
{
    return _scaler->get_scaler_buf ();
}

CLImageScaler::CLImageScaler (const SmartPtr<CLContext> &context)
    : CLImageHandler (context, "CLImageScaler")
    , _h_scaler_factor (0.5)
    , _v_scaler_factor (0.5)
{
}

void
CLImageScaler::emit_stop ()
{
    if (_scaler_buf_pool.ptr ())
        _scaler_buf_pool->stop ();
}

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
CLImageScaler::prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
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
CLImageScaler::execute_done (SmartPtr<VideoBuffer> &output)
{
    XCAM_UNUSED (output);
    get_context ()->finish();
    XCAM_ASSERT (_scaler_buf.ptr ());

    //post buffer out
    return post_buffer (_scaler_buf);
}

XCamReturn
CLImageScaler::prepare_scaler_buf (const VideoBufferInfo &video_info, SmartPtr<VideoBuffer> &output)
{
    if (!_scaler_buf_pool.ptr ()) {
        VideoBufferInfo scaler_video_info;
        uint32_t new_width = XCAM_ALIGN_UP ((uint32_t)(video_info.width * _h_scaler_factor),
                                            2 * XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE0);
        uint32_t new_height = XCAM_ALIGN_UP ((uint32_t)(video_info.height * _v_scaler_factor),
                                             2 * XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE1);

        scaler_video_info.init (video_info.format, new_width, new_height);

        SmartPtr<BufferPool> pool = new CLVideoBufferPool ();
        XCAM_ASSERT (pool.ptr ());
        pool->set_video_info (scaler_video_info);
        pool->reserve (6);
        _scaler_buf_pool = pool;
    }

    output = _scaler_buf_pool->get_buffer (_scaler_buf_pool);
    XCAM_ASSERT (output.ptr ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageScaler::post_buffer (const SmartPtr<VideoBuffer> &buffer)
{
    if (_scaler_callback.ptr ())
        return _scaler_callback->scaled_image_ready (buffer);

    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<CLImageKernel>
create_scale_kernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLImageScaler> &handler, CLImageScalerMemoryLayout layout)
{
    SmartPtr<CLImageKernel> kernel;
    kernel = new CLImageScalerKernel (context, layout, handler);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, kernel->build_kernel (kernel_scale_info, NULL) == XCAM_RETURN_NO_ERROR, NULL,
        "build scaler kernel(%s) failed", kernel_scale_info.kernel_name);
    XCAM_ASSERT (kernel->is_valid ());
    return kernel;
}

SmartPtr<CLImageHandler>
create_cl_image_scaler_handler (const SmartPtr<CLContext> &context, const uint32_t format)
{
    SmartPtr<CLImageScaler> scaler_handler;
    SmartPtr<CLImageKernel> scaler_kernel;

    scaler_handler = new CLImageScaler (context);
    XCAM_ASSERT (scaler_handler.ptr ());

    if (V4L2_PIX_FMT_NV12 == format) {
        //Y
        scaler_kernel = create_scale_kernel (context, scaler_handler, CL_IMAGE_SCALER_NV12_Y);
        XCAM_FAIL_RETURN (ERROR, scaler_kernel.ptr (), NULL, "build CL_IMAGE_SCALER_NV12_Y kernel failed");
        scaler_handler->add_kernel (scaler_kernel);
        //UV
        scaler_kernel = create_scale_kernel (context, scaler_handler, CL_IMAGE_SCALER_NV12_UV);
        XCAM_FAIL_RETURN (ERROR, scaler_kernel.ptr (), NULL, "build CL_IMAGE_SCALER_NV12_UV kernel failed");
        scaler_handler->add_kernel (scaler_kernel);
    } else if (XCAM_PIX_FMT_RGBA64 == format) {
        scaler_kernel = create_scale_kernel (context, scaler_handler, CL_IMAGE_SCALER_RGBA);
        XCAM_FAIL_RETURN (ERROR, scaler_kernel.ptr (), NULL, "build CL_IMAGE_SCALER_RGBA kernel failed");
        scaler_handler->add_kernel (scaler_kernel);
    } else {
        XCAM_LOG_ERROR ("create cl image scaler failed, unknown format:0x%08x", format);
        return NULL;
    }

    return scaler_handler;
}

};
