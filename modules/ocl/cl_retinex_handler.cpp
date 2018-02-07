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

#include "cl_utils.h"
#include "cl_retinex_handler.h"
#include <algorithm>
#include "cl_device.h"

namespace XCam {

static uint32_t retinex_gauss_scale [3] = {2, 8, 20}; //{20, 60, 150};
static float retinex_gauss_sigma [3] = {2.0f, 8.0f, 20.0f}; //{12.0f, 40.0f, 120.0f};
static float retinex_config_log_min = -0.12f; // -0.18f
static float retinex_config_log_max = 0.18f;  //0.2f

enum {
    KernelScaler = 0,
    KernelGaussian,
    KernelRetinex,
};

const static XCamKernelInfo kernel_retinex_info [] = {
    {
        "kernel_image_scaler",
#include "kernel_image_scaler.clx"
        , 0,
    },
    {
        "kernel_gauss",
#include "kernel_gauss.clx"
        , 0,
    },
    {
        "kernel_retinex",
#include "kernel_retinex.clx"
        , 0,
    },
};

CLRetinexScalerImageKernel::CLRetinexScalerImageKernel (
    const SmartPtr<CLContext> &context,
    const CLImageScalerMemoryLayout mem_layout,
    SmartPtr<CLRetinexImageHandler> &retinex)
    : CLScalerKernel (context, mem_layout)
    , _retinex(retinex)
{
}

SmartPtr<VideoBuffer>
CLRetinexScalerImageKernel::get_input_buffer ()
{
    return _retinex->get_input_buf ();
}

SmartPtr<VideoBuffer>
CLRetinexScalerImageKernel::get_output_buffer ()
{
    return _retinex->get_scaler_buf1 ();
}

CLRetinexGaussImageKernel::CLRetinexGaussImageKernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLRetinexImageHandler> &retinex,
    uint32_t index,
    uint32_t radius, float sigma)
    : CLGaussImageKernel (context, radius, sigma)
    , _retinex (retinex)
    , _index (index)
{
}

SmartPtr<VideoBuffer>
CLRetinexGaussImageKernel::get_input_buf ()
{
    return _retinex->get_scaler_buf1 ();
}

SmartPtr<VideoBuffer>
CLRetinexGaussImageKernel::get_output_buf ()
{
    return _retinex->get_gaussian_buf (_index);
}

CLRetinexImageKernel::CLRetinexImageKernel (const SmartPtr<CLContext> &context, SmartPtr<CLRetinexImageHandler> &retinex)
    : CLImageKernel (context, "kernel_retinex"),
      _retinex (retinex)
{
}

XCamReturn
CLRetinexImageKernel::prepare_arguments (
    CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<VideoBuffer> input = _retinex->get_input_buf ();
    SmartPtr<VideoBuffer> output = _retinex->get_output_buf ();

    const VideoBufferInfo & video_info_in = input->get_video_info ();
    const VideoBufferInfo & video_info_out = output->get_video_info ();
    SmartPtr<CLImage> image_in, image_in_uv;
    SmartPtr<CLImage> image_out, image_out_uv;
    SmartPtr<CLImage> image_in_ga[XCAM_RETINEX_MAX_SCALE];

    CLImageDesc cl_desc_in, cl_desc_out, cl_desc_ga;

    cl_desc_in.format.image_channel_data_type = CL_UNORM_INT8; //CL_UNSIGNED_INT32;
    cl_desc_in.format.image_channel_order = CL_RGBA;
    cl_desc_in.width = video_info_in.width / 4; // 16;
    cl_desc_in.height = video_info_in.height;
    cl_desc_in.row_pitch = video_info_in.strides[0];
    image_in = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[0]);

    cl_desc_in.height = video_info_in.height / 2;
    cl_desc_in.row_pitch = video_info_in.strides[1];
    image_in_uv = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[1]);

    cl_desc_out.format.image_channel_data_type = CL_UNORM_INT8; //CL_UNSIGNED_INT32;
    cl_desc_out.format.image_channel_order = CL_RGBA;
    cl_desc_out.width = video_info_out.width / 4; // 16;
    cl_desc_out.height = video_info_out.height;
    cl_desc_out.row_pitch = video_info_out.strides[0];
    image_out = convert_to_climage (context, output, cl_desc_out, video_info_out.offsets[0]);

    cl_desc_out.height = video_info_out.height / 2;
    cl_desc_out.row_pitch = video_info_out.strides[1];
    image_out_uv = convert_to_climage (context, output, cl_desc_out, video_info_out.offsets[1]);

    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_in_uv->is_valid () &&
        image_out->is_valid () && image_out_uv->is_valid(),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    for (uint32_t i = 0; i < XCAM_RETINEX_MAX_SCALE; ++i) {
        SmartPtr<VideoBuffer> gaussian_buf = _retinex->get_gaussian_buf (i);
        XCAM_ASSERT (gaussian_buf.ptr ());

        const VideoBufferInfo & video_info_gauss = gaussian_buf->get_video_info ();

        cl_desc_ga.format.image_channel_data_type = CL_UNORM_INT8;
        cl_desc_ga.format.image_channel_order = CL_R;
        cl_desc_ga.width = video_info_gauss.width;
        cl_desc_ga.height = video_info_gauss.height;
        cl_desc_ga.row_pitch = video_info_gauss.strides[0];
        image_in_ga[i] = convert_to_climage (context, gaussian_buf, cl_desc_ga, video_info_gauss.offsets[0]);

        XCAM_FAIL_RETURN (
            WARNING,
            image_in_ga[i]->is_valid (),
            XCAM_RETURN_ERROR_MEM,
            "cl image kernel(%s) gauss memory[%d] is invalid", get_kernel_name (), i);
    }
    CLRetinexConfig retinex_config;
    retinex_config.log_min = retinex_config_log_min;
    retinex_config.log_max = retinex_config_log_max;
    retinex_config.gain = 1.0f / (retinex_config.log_max - retinex_config.log_min);
    retinex_config.width = (float)video_info_in.width;
    retinex_config.height = (float)video_info_in.height;

    //set args;
    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_in_uv));
    for (uint32_t i = 0; i < XCAM_RETINEX_MAX_SCALE; ++i) {
        args.push_back (new CLMemArgument (image_in_ga[i]));
    }
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLMemArgument (image_out_uv));
    args.push_back (new CLArgumentT<CLRetinexConfig> (retinex_config));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = video_info_out.width / 4;
    work_size.global[1] = video_info_out.height;
    work_size.local[0] = 16;
    work_size.local[1] = 2;

    return XCAM_RETURN_NO_ERROR;
}

CLRetinexImageHandler::CLRetinexImageHandler (const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
    , _scaler_factor(XCAM_RETINEX_SCALER_FACTOR)
{
}

void
CLRetinexImageHandler::emit_stop ()
{
    if (_scaler_buf_pool.ptr ())
        _scaler_buf_pool->stop ();
}

XCamReturn
CLRetinexImageHandler::prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    CLImageHandler::prepare_output_buf(input, output);
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    ret = prepare_scaler_buf (input->get_video_info ());
    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "CLRetinexImageHandler prepare scaled video buf failed");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLRetinexImageHandler::prepare_scaler_buf (const VideoBufferInfo &video_info)
{
    if (!_scaler_buf_pool.ptr ()) {
        SmartPtr<CLContext> context = get_context ();
        VideoBufferInfo scaler_video_info;
        uint32_t new_width = XCAM_ALIGN_UP ((uint32_t)(video_info.width * _scaler_factor), 8);
        uint32_t new_height = XCAM_ALIGN_UP ((uint32_t)(video_info.height * _scaler_factor), 4);

        scaler_video_info.init (video_info.format, new_width, new_height);

        SmartPtr<BufferPool> pool = new CLVideoBufferPool ();
        XCAM_ASSERT (pool.ptr ());
        pool->set_video_info (scaler_video_info);
        pool->reserve (XCAM_RETINEX_MAX_SCALE + 1);
        _scaler_buf_pool = pool;

        _scaler_buf1 = _scaler_buf_pool->get_buffer (_scaler_buf_pool);
        XCAM_ASSERT (_scaler_buf1.ptr ());

        for (int i = 0; i < XCAM_RETINEX_MAX_SCALE; ++i) {
            _gaussian_buf[i] = _scaler_buf_pool->get_buffer (_scaler_buf_pool);
            XCAM_ASSERT (_gaussian_buf[i].ptr ());
        }
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

static SmartPtr<CLRetinexScalerImageKernel>
create_kernel_retinex_scaler (
    const SmartPtr<CLContext> &context, SmartPtr<CLRetinexImageHandler> handler)
{
    SmartPtr<CLRetinexScalerImageKernel> kernel;

    kernel = new CLRetinexScalerImageKernel (context, CL_IMAGE_SCALER_NV12_Y, handler);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, kernel->build_kernel (kernel_retinex_info[KernelScaler], NULL) == XCAM_RETURN_NO_ERROR, NULL,
        "build retinex scaler kernel(%s) failed", kernel_retinex_info[KernelScaler].kernel_name);

    XCAM_ASSERT (kernel->is_valid ());
    return kernel;
}

static SmartPtr<CLRetinexGaussImageKernel>
create_kernel_retinex_gaussian (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLRetinexImageHandler> handler,
    uint32_t index,
    uint32_t radius, float sigma)
{
    SmartPtr<CLRetinexGaussImageKernel> kernel;
    char build_options[1024];

    xcam_mem_clear (build_options);
    snprintf (build_options, sizeof (build_options), " -DGAUSS_RADIUS=%d ", radius);

    kernel = new CLRetinexGaussImageKernel (context, handler, index, radius, sigma);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, kernel->build_kernel (kernel_retinex_info[KernelGaussian], build_options) == XCAM_RETURN_NO_ERROR, NULL,
        "build retinex gaussian kernel(%s) failed", kernel_retinex_info[KernelGaussian].kernel_name);

    XCAM_ASSERT (kernel->is_valid ());

    return kernel;
}

static SmartPtr<CLRetinexImageKernel>
create_kernel_retinex (const SmartPtr<CLContext> &context, SmartPtr<CLRetinexImageHandler> handler)
{
    SmartPtr<CLRetinexImageKernel> kernel;
    char build_options[1024];

    xcam_mem_clear (build_options);
    snprintf (build_options, sizeof (build_options), " -DRETINEX_SCALE_SIZE=%d ", XCAM_RETINEX_MAX_SCALE);

    kernel = new CLRetinexImageKernel (context, handler);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, kernel->build_kernel (kernel_retinex_info[KernelRetinex], build_options) == XCAM_RETURN_NO_ERROR, NULL,
        "build retinex kernel(%s) failed", kernel_retinex_info[KernelRetinex].kernel_name);

    XCAM_ASSERT (kernel->is_valid ());
    return kernel;
}

SmartPtr<CLImageHandler>
create_cl_retinex_image_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLRetinexImageHandler> retinex_handler;

    SmartPtr<CLRetinexScalerImageKernel> retinex_scaler_kernel;
    SmartPtr<CLRetinexImageKernel> retinex_kernel;

    retinex_handler = new CLRetinexImageHandler (context, "cl_handler_retinex");
    retinex_scaler_kernel = create_kernel_retinex_scaler (context, retinex_handler);
    XCAM_FAIL_RETURN (
        ERROR,
        retinex_scaler_kernel.ptr () && retinex_scaler_kernel->is_valid (),
        NULL,
        "Retinex handler create scaler kernel failed");
    retinex_handler->set_retinex_scaler_kernel (retinex_scaler_kernel);

    for (uint32_t i = 0; i < XCAM_RETINEX_MAX_SCALE; ++i) {
        SmartPtr<CLImageKernel> retinex_gauss_kernel;
        retinex_gauss_kernel = create_kernel_retinex_gaussian (
                                   context, retinex_handler, i, retinex_gauss_scale [i], retinex_gauss_sigma [i]);
        XCAM_FAIL_RETURN (
            ERROR,
            retinex_gauss_kernel.ptr () && retinex_gauss_kernel->is_valid (),
            NULL,
            "Retinex handler create gaussian kernel failed");
        retinex_handler->add_kernel (retinex_gauss_kernel);
    }

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
