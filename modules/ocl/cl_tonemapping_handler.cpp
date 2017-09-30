/*
 * cl_tonemapping_handler.cpp - CL tonemapping handler
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
 *  Author: Wu Junkai <junkai.wu@intel.com>
 */

#include "cl_utils.h"
#include "cl_tonemapping_handler.h"

namespace XCam {

static const XCamKernelInfo kernel_tonemapping_info = {
    "kernel_tonemapping",
#include "kernel_tonemapping.clx"
    , 0,
};

CLTonemappingImageKernel::CLTonemappingImageKernel (
    const SmartPtr<CLContext> &context, const char *name)
    : CLImageKernel (context, name)
{
}

CLTonemappingImageHandler::CLTonemappingImageHandler (
    const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
    , _output_format (XCAM_PIX_FMT_SGRBG16_planar)
{
    _wb_config.r_gain = 1.0;
    _wb_config.gr_gain = 1.0;
    _wb_config.gb_gain = 1.0;
    _wb_config.b_gain = 1.0;
}

bool
CLTonemappingImageHandler::set_tonemapping_kernel(SmartPtr<CLTonemappingImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _tonemapping_kernel = kernel;
    return true;
}

bool
CLTonemappingImageHandler::set_wb_config (const XCam3aResultWhiteBalance &wb)
{
    _wb_config.r_gain = (float)wb.r_gain;
    _wb_config.gr_gain = (float)wb.gr_gain;
    _wb_config.gb_gain = (float)wb.gb_gain;
    _wb_config.b_gain = (float)wb.b_gain;
    return true;
}

XCamReturn
CLTonemappingImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    bool format_inited = output.init (_output_format, input.width, input.height);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) output format(%s) unsupported",
        XCAM_STR(get_name ()), xcam_fourcc_to_string (_output_format));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLTonemappingImageHandler::prepare_parameters (
    SmartPtr<VideoBuffer> &input,
    SmartPtr<VideoBuffer> &output)
{
    SmartPtr<CLContext> context = get_context ();
    float y_max = 0.0f, y_target = 0.0f;
    CLArgList args;
    CLWorkSize work_size;
    XCAM_ASSERT (_tonemapping_kernel.ptr ());

    const VideoBufferInfo &video_info = input->get_video_info ();

    CLImageDesc desc;
    desc.format.image_channel_order = CL_RGBA;
    desc.format.image_channel_data_type = CL_UNORM_INT16;
    desc.width = video_info.aligned_width / 4;
    desc.height = video_info.aligned_height * 4;
    desc.row_pitch = video_info.strides[0];
    desc.array_size = 4;
    desc.slice_pitch = video_info.strides [0] * video_info.aligned_height;

    SmartPtr<CLImage> image_in = convert_to_climage (context, input, desc);
    SmartPtr<CLImage> image_out = convert_to_climage (context, output, desc);
    int image_height = video_info.aligned_height;

    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image handler(%s) in/out memory not available", XCAM_STR(get_name ()));

    SmartPtr<X3aStats> stats;
    SmartPtr<CLVideoBuffer> cl_buf = input.dynamic_cast_ptr<CLVideoBuffer> ();
    if (cl_buf.ptr ()) {
        stats = cl_buf->find_3a_stats ();
    }
#if HAVE_LIBDRM
    else {
        SmartPtr<DrmBoBuffer> bo_buf = input.dynamic_cast_ptr<DrmBoBuffer> ();
        stats = bo_buf->find_3a_stats ();
    }
#endif
    XCAM_FAIL_RETURN (
        ERROR,
        stats.ptr (),
        XCAM_RETURN_ERROR_MEM,
        "CLTonemappingImageKernel find_3a_stats failed");
    XCam3AStats *stats_ptr = stats->get_stats ();
    XCAM_ASSERT (stats_ptr);

    int pixel_totalnum = stats_ptr->info.aligned_width * stats_ptr->info.aligned_height;
    int pixel_num = 0;
    int hist_bin_count = 1 << stats_ptr->info.bit_depth;
    int64_t cumulative_value = 0;
    int saturated_thresh = pixel_totalnum * 0.003f;
    int percent_90_thresh = pixel_totalnum * 0.1f;
    int medium_thresh = pixel_totalnum * 0.5f;
    float y_saturated = 0;
    float y_percent_90 = 0;
    float y_average = 0;
    float y_medium = 0;

    for (int i = (hist_bin_count - 1); i >= 0; i--)
    {
        pixel_num += stats_ptr->hist_y[i];
        if ((y_saturated == 0) && (pixel_num >= saturated_thresh))
        {
            y_saturated = i;
        }
        if ((y_percent_90 == 0) && (pixel_num >= percent_90_thresh))
        {
            y_percent_90 = i;
        }
        if ((y_medium == 0) && (pixel_num >= medium_thresh))
        {
            y_medium = i;
        }
        cumulative_value += i * stats_ptr->hist_y[i];
    }

    y_average = cumulative_value / pixel_totalnum;

    if (y_saturated < (hist_bin_count - 1)) {
        y_saturated = y_saturated + 1;
    }

    y_target =  (hist_bin_count / y_saturated) * (1.5 * y_medium + 0.5 * y_average) / 2;

    if (y_target < 4) {
        y_target = 4;
    }
    if ((y_target > y_saturated) || (y_saturated < 4)) {
        y_target = y_saturated / 4;
    }

    y_max = hist_bin_count * (2 * y_saturated + y_target) / y_saturated - y_saturated - y_target;

    y_target = y_target / pow(2, stats_ptr->info.bit_depth - 8);
    y_max = y_max / pow(2, stats_ptr->info.bit_depth - 8);

    //set args;
    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLArgumentT<float> (y_max));
    args.push_back (new CLArgumentT<float> (y_target));
    args.push_back (new CLArgumentT<int> (image_height));

    const CLImageDesc out_info = image_out->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width;
    work_size.global[1] = out_info.height / 4;
    work_size.local[0] = 8;
    work_size.local[1] = 8;

    XCAM_ASSERT (_tonemapping_kernel.ptr ());
    XCamReturn ret = _tonemapping_kernel->set_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "tone mapping kernel set arguments failed.");

    return XCAM_RETURN_NO_ERROR;
}


SmartPtr<CLImageHandler>
create_cl_tonemapping_image_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLTonemappingImageHandler> tonemapping_handler;
    SmartPtr<CLTonemappingImageKernel> tonemapping_kernel;

    tonemapping_kernel = new CLTonemappingImageKernel (context, "kernel_tonemapping");
    XCAM_ASSERT (tonemapping_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, tonemapping_kernel->build_kernel (kernel_tonemapping_info, NULL) == XCAM_RETURN_NO_ERROR, NULL,
        "build tonemapping kernel(%s) failed", kernel_tonemapping_info.kernel_name);

    XCAM_ASSERT (tonemapping_kernel->is_valid ());
    tonemapping_handler = new CLTonemappingImageHandler(context, "cl_handler_tonemapping");
    tonemapping_handler->set_tonemapping_kernel(tonemapping_kernel);

    return tonemapping_handler;
}

};
