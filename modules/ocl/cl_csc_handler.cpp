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
#include "cl_utils.h"
#include "cl_csc_handler.h"
#include "cl_device.h"
#include "cl_kernel.h"

static const XCamKernelInfo kernel_csc_info[] = {
    {
        "kernel_csc_rgbatonv12",
#include "kernel_csc.clx"
        , 0,
    },
    {
        "kernel_csc_rgbatolab",
#include "kernel_csc.clx"
        , 0,
    },
    {
        "kernel_csc_rgba64torgba",
#include "kernel_csc.clx"
        , 0,
    },
    {
        "kernel_csc_yuyvtorgba",
#include "kernel_csc.clx"
        , 0,
    },
    {
        "kernel_csc_nv12torgba",
#include "kernel_csc.clx"
        , 0,
    },
};


float default_rgbtoyuv_matrix[XCAM_COLOR_MATRIX_SIZE] = {
    0.299f, 0.587f, 0.114f,
    -0.14713f, -0.28886f, 0.436f,
    0.615f, -0.51499f, -0.10001f
};

namespace XCam {

CLCscImageKernel::CLCscImageKernel (const SmartPtr<CLContext> &context, CLCscType type)
    : CLImageKernel (context)
    , _kernel_csc_type (type)
{
}

CLCscImageHandler::CLCscImageHandler (
    const SmartPtr<CLContext> &context, const char *name, CLCscType type)
    : CLImageHandler (context, name)
    , _output_format (V4L2_PIX_FMT_NV12)
    , _csc_type (type)
{
    memcpy (_rgbtoyuv_matrix, default_rgbtoyuv_matrix, sizeof (_rgbtoyuv_matrix));

    switch (type) {
    case CL_CSC_TYPE_RGBATONV12:
        _output_format = V4L2_PIX_FMT_NV12;
        break;
    case CL_CSC_TYPE_RGBATOLAB:
        _output_format = XCAM_PIX_FMT_LAB;
        break;
    case CL_CSC_TYPE_RGBA64TORGBA:
    case CL_CSC_TYPE_YUYVTORGBA:
    case CL_CSC_TYPE_NV12TORGBA:
        _output_format = V4L2_PIX_FMT_RGBA32;
        break;
    default:
        break;
    }
}

bool
CLCscImageHandler::set_csc_kernel (SmartPtr<CLCscImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _csc_kernel = kernel;
    return true;
}

bool
CLCscImageHandler::set_matrix (const XCam3aResultColorMatrix &matrix)
{
    for (int i = 0; i < XCAM_COLOR_MATRIX_SIZE; i++)
        _rgbtoyuv_matrix[i] = (float)matrix.matrix[i];
    return true;
}

bool
CLCscImageHandler::set_output_format (uint32_t fourcc)
{
    XCAM_FAIL_RETURN (
        WARNING,
        V4L2_PIX_FMT_XBGR32 == fourcc || V4L2_PIX_FMT_NV12 == fourcc,
        false,
        "CL csc handler doesn't support format: (%s)",
        xcam_fourcc_to_string (fourcc));

    _output_format = fourcc;
    return true;
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
        "CL image handler(%s) output format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (_output_format));

    return XCAM_RETURN_NO_ERROR;
}

static bool
ensure_image_desc (const VideoBufferInfo &info, CLImageDesc &desc)
{
    desc.array_size = 0;
    desc.slice_pitch = 0;
    if (info.format == XCAM_PIX_FMT_RGB48_planar || info.format == XCAM_PIX_FMT_RGB24_planar)
        desc.height = info.aligned_height * 3;

    return true;
}

XCamReturn
CLCscImageHandler::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    SmartPtr<CLContext> context = get_context ();

    const VideoBufferInfo &in_video_info = input->get_video_info ();
    const VideoBufferInfo &out_video_info = output->get_video_info ();
    CLArgList args;
    CLWorkSize work_size;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_csc_kernel.ptr ());

    CLImageDesc in_desc, out_desc;
    CLImage::video_info_2_cl_image_desc (in_video_info, in_desc);
    CLImage::video_info_2_cl_image_desc (out_video_info, out_desc);
    ensure_image_desc (in_video_info, in_desc);
    ensure_image_desc (out_video_info, out_desc);

    SmartPtr<CLImage> image_in  = convert_to_climage (context, input, in_desc, in_video_info.offsets[0]);
    SmartPtr<CLImage> image_out  = convert_to_climage (context, output, out_desc, out_video_info.offsets[0]);
    SmartPtr<CLBuffer> matrix_buffer = new CLBuffer (
        context, sizeof(float)*XCAM_COLOR_MATRIX_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_rgbtoyuv_matrix);

    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid () && matrix_buffer->is_valid(),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", _csc_kernel->get_kernel_name ());

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_out));

    do {
        if ((_csc_type == CL_CSC_TYPE_RGBATOLAB)
                || (_csc_type == CL_CSC_TYPE_RGBA64TORGBA)
                || (_csc_type == CL_CSC_TYPE_YUYVTORGBA)) {
            work_size.global[0] = out_video_info.width;
            work_size.global[1] = out_video_info.height;
            break;
        }

        SmartPtr<CLImage> image_uv;
        if(_csc_type == CL_CSC_TYPE_NV12TORGBA) {
            in_desc.height /= 2;
            image_uv = convert_to_climage (context, input, in_desc, in_video_info.offsets[1]);
            args.push_back (new CLMemArgument (image_uv));

            work_size.global[0] = out_video_info.width / 2;
            work_size.global[1] = out_video_info.height / 2;
            break;
        }

        if (_csc_type == CL_CSC_TYPE_RGBATONV12) {
            out_desc.height /= 2;
            image_uv = convert_to_climage (context, output, out_desc, out_video_info.offsets[1]);
            args.push_back (new CLMemArgument (image_uv));
            args.push_back (new CLMemArgument (matrix_buffer));

            work_size.global[0] = out_video_info.width / 2;
            work_size.global[1] = out_video_info.height / 2;
            break;
        }
    } while (0);

    XCAM_ASSERT (_csc_kernel.ptr ());
    ret = _csc_kernel->set_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "csc kernel set arguments failed.");

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_csc_image_handler (const SmartPtr<CLContext> &context, CLCscType type)
{
    SmartPtr<CLCscImageHandler> csc_handler;
    SmartPtr<CLCscImageKernel> csc_kernel;

    XCAM_ASSERT (type < CL_CSC_TYPE_MAX);
    csc_kernel = new CLCscImageKernel (context, type);
    XCAM_ASSERT (csc_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, csc_kernel->build_kernel (kernel_csc_info[type], NULL) == XCAM_RETURN_NO_ERROR, NULL,
        "build csc kernel(%s) failed", kernel_csc_info[type].kernel_name);

    XCAM_ASSERT (csc_kernel->is_valid ());

    csc_handler = new CLCscImageHandler (context, "cl_handler_csc", type);
    csc_handler->set_csc_kernel (csc_kernel);

    return csc_handler;
}

};
