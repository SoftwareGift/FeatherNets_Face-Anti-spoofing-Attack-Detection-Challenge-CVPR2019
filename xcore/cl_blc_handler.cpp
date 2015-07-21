/*
 * cl_blc_handler.cpp - CL black level correction handler
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
 * Author: Shincy Tu <shincy.tu@intel.com>
 */
#include "xcam_utils.h"
#include "cl_blc_handler.h"

namespace XCam {

CLBlcImageKernel::CLBlcImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_blc")
{
    _blc_config.level_gr = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_r = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_b = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_gb = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.color_bits = 0;
}

XCamReturn
CLBlcImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & in_video_info = input->get_video_info ();
    const VideoBufferInfo & out_video_info = output->get_video_info ();
    CLImageDesc image_info;
    uint32_t channel_bits = XCAM_ALIGN_UP (in_video_info.color_bits, 8);
    _blc_config.color_bits = in_video_info.color_bits;

    image_info.format.image_channel_order = CL_R;
    if (channel_bits == 8)
        image_info.format.image_channel_data_type = CL_UNSIGNED_INT8;
    else if (channel_bits == 16)
        image_info.format.image_channel_data_type = CL_UNSIGNED_INT16;
    image_info.width = in_video_info.width;
    image_info.height = in_video_info.height;
    image_info.row_pitch = in_video_info.strides[0];

    _image_in = new CLVaImage (context, input, image_info, 0);

    image_info.format.image_channel_data_type = CL_UNSIGNED_INT16;
    image_info.row_pitch = out_video_info.strides[0];
    _image_out = new CLVaImage (context, output, image_info, 0);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    /* This is a temporary workaround to hard code black level for 12bit raw data.
        And it should be removed once tunning is finished.   */
    if (_blc_config.color_bits == 12) {
        _blc_config.level_gr = 240 / (double)pow(2, 12);
        _blc_config.level_r = 240 / (double)pow(2, 12);
        _blc_config.level_b = 240 / (double)pow(2, 12);
        _blc_config.level_gb = 240 / (double)pow(2, 12);
    }

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_blc_config;
    args[2].arg_size = sizeof (CLBLCConfig);
    arg_count = 3;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = image_info.width / 2;
    work_size.global[1] = image_info.height / 2;
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

bool
CLBlcImageKernel::set_blc (const XCam3aResultBlackLevel &blc)
{
    _blc_config.level_r = (float)blc.r_level;
    _blc_config.level_gr = (float)blc.gr_level;
    _blc_config.level_gb = (float)blc.gb_level;
    _blc_config.level_b = (float)blc.b_level;
    _blc_config.color_bits = 0;
    return true;
}
CLBlcImageHandler::CLBlcImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLBlcImageHandler::set_blc_config (const XCam3aResultBlackLevel &blc)
{
    return _blc_kernel->set_blc(blc);
}

bool
CLBlcImageHandler::set_blc_kernel(SmartPtr<CLBlcImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _blc_kernel = kernel;
    return true;
}

XCamReturn
CLBlcImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    const uint32_t format = XCAM_PIX_FMT_SGRBG16;
    bool format_inited = output.init (format, input.width, input.height);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) prepare ouput format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (format));

    return XCAM_RETURN_NO_ERROR;
}


SmartPtr<CLImageHandler>
create_cl_blc_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLBlcImageHandler> blc_handler;
    SmartPtr<CLBlcImageKernel> blc_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    blc_kernel = new CLBlcImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_blc)
#include "kernel_blc.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = blc_kernel->load_from_source (kernel_blc_body, strlen (kernel_blc_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", blc_kernel->get_kernel_name());
    }
    XCAM_ASSERT (blc_kernel->is_valid ());
    blc_handler = new CLBlcImageHandler ("cl_handler_blc");
    blc_handler->set_blc_kernel (blc_kernel);

    return blc_handler;
}

}
