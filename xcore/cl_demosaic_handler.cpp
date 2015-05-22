/*
 * cl_demosaic_handler.cpp - CL demosaic handler
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
 */
#include "xcam_utils.h"
#include "cl_demosaic_handler.h"

namespace XCam {

CLDemosaicImageKernel::CLDemosaicImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_demosaic")
{
}

XCamReturn
CLDemosaicImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = output->get_video_info ();

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);

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
    arg_count = 2;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = video_info.width / 2;
    work_size.global[1] = video_info.height / 2;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

CLBayer2RGBImageHandler::CLBayer2RGBImageHandler (const char *name)
    : CLImageHandler (name)
    , _output_format (XCAM_PIX_FMT_RGBA64)
{
}

bool
CLBayer2RGBImageHandler::set_output_format (uint32_t fourcc)
{
    XCAM_FAIL_RETURN (
        WARNING,
        fourcc == XCAM_PIX_FMT_RGBA64 || fourcc == V4L2_PIX_FMT_RGB24 ||
        fourcc == V4L2_PIX_FMT_XBGR32 || fourcc == V4L2_PIX_FMT_ABGR32 || V4L2_PIX_FMT_BGR32 ||
        //fourcc == V4L2_PIX_FMT_RGB32 || fourcc == V4L2_PIX_FMT_ARGB32 || V4L2_PIX_FMT_XRGB32 ||
        fourcc == V4L2_PIX_FMT_RGBA32,
        false,
        "CL image handler(%s) doesn't support format(%s) settings",
        get_name (), xcam_fourcc_to_string (fourcc));

    _output_format = fourcc;
    return true;
}

XCamReturn
CLBayer2RGBImageHandler::prepare_buffer_pool_video_info (
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
create_cl_demosaic_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> demosaic_handler;
    SmartPtr<CLImageKernel> demosaic_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    demosaic_kernel = new CLDemosaicImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_demosaic)
#include "kernel_demosaic.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = demosaic_kernel->load_from_source (kernel_demosaic_body, strlen (kernel_demosaic_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", demosaic_kernel->get_kernel_name());
    }
    XCAM_ASSERT (demosaic_kernel->is_valid ());
    demosaic_handler = new CLBayer2RGBImageHandler ("cl_handler_demosaic");
    demosaic_handler->add_kernel (demosaic_kernel);

    return demosaic_handler;
}

};
