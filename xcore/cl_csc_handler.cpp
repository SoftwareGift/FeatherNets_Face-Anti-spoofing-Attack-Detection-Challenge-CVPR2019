/*
 * cl_csc_handler.cpp - CL csc handler
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */
#include "xcam_utils.h"
#include "cl_csc_handler.h"

namespace XCam {

CLCscImageKernel::CLCscImageKernel (SmartPtr<CLContext> &context, const char *name)
    : CLImageKernel (context, name)
    , _vertical_offset (0)
{
}

XCamReturn
CLCscImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = output->get_video_info ();

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);
    _vertical_offset = video_info.aligned_height;

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_vertical_offset;
    args[2].arg_size = sizeof (_vertical_offset);


    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    if (video_info.format == V4L2_PIX_FMT_NV12) {
        work_size.global[0] = video_info.width / 2;
        work_size.global[1] = video_info.height / 2;
        arg_count = 3;
    }
    else if ((video_info.format == XCAM_PIX_FMT_LAB) || (video_info.format == V4L2_PIX_FMT_RGBA32)) {
        work_size.global[0] = video_info.width;
        work_size.global[1] = video_info.height;
        arg_count = 2;
    }
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

CLCscImageHandler::CLCscImageHandler (const char *name, CLCscType type)
    : CLImageHandler (name)
    , _output_format (V4L2_PIX_FMT_NV12)
    , _csc_type (type)
{
    switch (type) {
    case CL_CSC_TYPE_RGBATONV12:
        _output_format = V4L2_PIX_FMT_NV12;
        break;
    case CL_CSC_TYPE_RGBATOLAB:
        _output_format = XCAM_PIX_FMT_LAB;
        break;
    case CL_CSC_TYPE_RGBA64TORGBA:
        _output_format = V4L2_PIX_FMT_RGBA32;
        break;
    default:
        break;
    }
}

XCamReturn
CLCscImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    bool format_inited = output.init (_output_format, input.width, input.height);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) ouput format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (_output_format));

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_csc_image_handler (SmartPtr<CLContext> &context, CLCscType type)
{
    SmartPtr<CLImageHandler> csc_handler;
    SmartPtr<CLImageKernel> csc_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;


    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_csc_rgbatonv12)
#include "kernel_csc_rgbatonv12.cl"
    XCAM_CL_KERNEL_FUNC_END;

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_csc_rgbatolab)
#include "kernel_csc_rgbatolab.cl"
    XCAM_CL_KERNEL_FUNC_END;

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_csc_rgba64torgba)
#include "kernel_csc_rgba64torgba.cl"
    XCAM_CL_KERNEL_FUNC_END;


    if (type == CL_CSC_TYPE_RGBATONV12) {
        csc_kernel = new CLCscImageKernel (context, "kernel_csc_rgbatonv12");
        ret = csc_kernel->load_from_source (kernel_csc_rgbatonv12_body, strlen (kernel_csc_rgbatonv12_body));
    }
    else if (type == CL_CSC_TYPE_RGBATOLAB) {
        csc_kernel = new CLCscImageKernel (context, "kernel_csc_rgbatolab");
        ret = csc_kernel->load_from_source (kernel_csc_rgbatolab_body, strlen (kernel_csc_rgbatolab_body));
    }
    else if (type == CL_CSC_TYPE_RGBA64TORGBA) {
        csc_kernel = new CLCscImageKernel (context, "kernel_csc_rgba64torgba");
        ret = csc_kernel->load_from_source (kernel_csc_rgba64torgba_body, strlen (kernel_csc_rgba64torgba_body));
    }

    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CL image handler(%s) load source failed", csc_kernel->get_kernel_name());

    XCAM_ASSERT (csc_kernel->is_valid ());

    csc_handler = new CLCscImageHandler ("cl_handler_csc", type);
    csc_handler->add_kernel (csc_kernel);

    return csc_handler;
}

};
