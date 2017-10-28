/*
 * cl_defog_dcp_handler.cpp - CL defog dark channel prior handler
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "cl_utils.h"
#include "cl_defog_dcp_handler.h"
#include <algorithm>
#include "cl_device.h"

enum {
    KernelDarkChannel = 0,
    KernelMinFilter,
    KernelBiFilter,
    KernelDefogRecover,
};

const static XCamKernelInfo kernels_info [] = {
    {
        "kernel_dark_channel",
#include "kernel_defog_dcp.clx"
        , 0,
    },
    {
        "kernel_min_filter",
#include "kernel_min_filter.clx"
        , 0,
    },
    {
        "kernel_bi_filter",
#include "kernel_bi_filter.clx"
        , 0,
    },
    {
        "kernel_defog_recover",
#include "kernel_defog_dcp.clx"
        , 0,
    },
};

namespace XCam {

CLDarkChannelKernel::CLDarkChannelKernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLDefogDcpImageHandler> &defog_handler)
    : CLImageKernel (context)
    , _defog_handler (defog_handler)
{
}

XCamReturn
CLDarkChannelKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<VideoBuffer> &input = _defog_handler->get_input_buf ();

    const VideoBufferInfo & video_info_in = input->get_video_info ();

    CLImageDesc cl_desc_in;

    cl_desc_in.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc_in.format.image_channel_order = CL_RGBA;
    cl_desc_in.width = video_info_in.width / 8;
    cl_desc_in.height = video_info_in.height;
    cl_desc_in.row_pitch = video_info_in.strides[0];
    SmartPtr<CLImage> image_in_y = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[0]);

    cl_desc_in.height = video_info_in.height / 2;
    cl_desc_in.row_pitch = video_info_in.strides[1];
    SmartPtr<CLImage> image_in_uv = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[1]);

    args.push_back (new CLMemArgument (image_in_y));
    args.push_back (new CLMemArgument (image_in_uv));

    SmartPtr<CLImage> &dark_channel = _defog_handler->get_dark_map (XCAM_DEFOG_DC_ORIGINAL);
    args.push_back (new CLMemArgument (dark_channel));

    // R, G, B channel
    for (uint32_t i = 0; i < XCAM_DEFOG_MAX_CHANNELS; ++i) {
        SmartPtr<CLImage> &rgb_image = _defog_handler->get_rgb_channel (i);
        args.push_back (new CLMemArgument (rgb_image));
    }

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 2;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_in.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_in.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLMinFilterKernel::CLMinFilterKernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLDefogDcpImageHandler> &defog_handler,
    int index)
    : CLImageKernel (context)
    , _defog_handler (defog_handler)
    , _buf_index (index)
{
    XCAM_ASSERT (XCAM_DEFOG_DC_MIN_FILTER_V == _buf_index || XCAM_DEFOG_DC_MIN_FILTER_H == _buf_index);
}

XCamReturn
CLMinFilterKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLImage> &dark_channel_in = _defog_handler->get_dark_map (_buf_index - 1);
    SmartPtr<CLImage> &dark_channel_out = _defog_handler->get_dark_map (_buf_index);

    args.push_back (new CLMemArgument (dark_channel_in));
    args.push_back (new CLMemArgument (dark_channel_out));

    const CLImageDesc &cl_desc = dark_channel_in->get_image_desc ();

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    if (XCAM_DEFOG_DC_MIN_FILTER_V == _buf_index) {
        work_size.local[0] = 16;
        work_size.local[1] = 4;
        work_size.global[0] = XCAM_ALIGN_UP (cl_desc.width, work_size.local[0]);
        work_size.global[1] = XCAM_ALIGN_UP (cl_desc.height / 2, work_size.local[1]);
    } else {
        work_size.local[0] = 16;
        work_size.local[1] = 4;
        work_size.global[0] = XCAM_ALIGN_UP (cl_desc.width, work_size.local[0]);
        work_size.global[1] = XCAM_ALIGN_UP (cl_desc.height, work_size.local[1]);
    }

    return XCAM_RETURN_NO_ERROR;
}

CLBiFilterKernel::CLBiFilterKernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLDefogDcpImageHandler> &defog_handler)
    : CLImageKernel (context)
    , _defog_handler (defog_handler)
{
}

XCamReturn
CLBiFilterKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<VideoBuffer> &input = _defog_handler->get_input_buf ();
    const VideoBufferInfo & video_info_in = input->get_video_info ();

    CLImageDesc cl_desc_in;
    cl_desc_in.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc_in.format.image_channel_order = CL_RGBA;
    cl_desc_in.width = video_info_in.width / 8;
    cl_desc_in.height = video_info_in.height;
    cl_desc_in.row_pitch = video_info_in.strides[0];
    SmartPtr<CLImage> image_in_y = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[0]);

    SmartPtr<CLImage> &dark_channel_in = _defog_handler->get_dark_map (XCAM_DEFOG_DC_ORIGINAL);
    SmartPtr<CLImage> &dark_channel_out = _defog_handler->get_dark_map (XCAM_DEFOG_DC_BI_FILTER);

    args.push_back (new CLMemArgument (image_in_y));
    args.push_back (new CLMemArgument (dark_channel_in));
    args.push_back (new CLMemArgument (dark_channel_out));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 2;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_in.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_in.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLDefogRecoverKernel::CLDefogRecoverKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLDefogDcpImageHandler> &defog_handler)
    : CLImageKernel (context)
    , _defog_handler (defog_handler)
    , _max_r (255.0f)
    , _max_g (255.0f)
    , _max_b (255.0f)
    , _max_i (255.0f)
{
}

float
CLDefogRecoverKernel::get_max_value (SmartPtr<VideoBuffer> &buf)
{
    float ret = 255.0f;
    const float max_percent = 1.0f;

    SmartPtr<X3aStats> stats;
    SmartPtr<CLVideoBuffer> cl_buf = buf.dynamic_cast_ptr<CLVideoBuffer> ();
    if (cl_buf.ptr ()) {
        stats = cl_buf->find_3a_stats ();
    }
#if HAVE_LIBDRM
    else {
        SmartPtr<DrmBoBuffer> bo_buf = buf.dynamic_cast_ptr<DrmBoBuffer> ();
        stats = bo_buf->find_3a_stats ();
    }
#endif

    _max_r = 230.0f;
    _max_g = 230.0f;
    _max_b = 230.0f;
    _max_i = XCAM_MAX (_max_r, _max_g);
    _max_i = XCAM_MAX (_max_i, _max_b);
    if (!stats.ptr ())
        return ret;

    XCam3AStats *stats_ptr = stats->get_stats ();
    if (!stats_ptr || !stats_ptr->hist_y)
        return ret;

    uint32_t his_bins = stats_ptr->info.histogram_bins;
    uint32_t pixel_count = stats_ptr->info.width * stats_ptr->info.height;
    uint32_t max_expect_count = (uint32_t)(max_percent * pixel_count / 100.0f);
    uint32_t sum_count = 0;
    int32_t i = (int32_t)(his_bins - 1);

    for (; i >= 0; --i) {
        sum_count += stats_ptr->hist_y[i];
        if (sum_count >= max_expect_count)
            break;
    }
    ret = (float)i * 256.0f / (1 << stats_ptr->info.bit_depth);
    ret = XCAM_MAX (ret, 1.0f);
    return ret;
}

XCamReturn
CLDefogRecoverKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<VideoBuffer> &input = _defog_handler->get_input_buf ();
    SmartPtr<VideoBuffer> &output = _defog_handler->get_output_buf ();
    SmartPtr<CLImage> &dark_map = _defog_handler->get_dark_map (XCAM_DEFOG_DC_BI_FILTER);
    get_max_value (input);

    args.push_back (new CLMemArgument (dark_map));
    args.push_back (new CLArgumentT<float> (_max_i));
    args.push_back (new CLArgumentT<float> (_max_r));
    args.push_back (new CLArgumentT<float> (_max_g));
    args.push_back (new CLArgumentT<float> (_max_b));

    for (int i = 0; i < XCAM_DEFOG_MAX_CHANNELS; ++i) {
        SmartPtr<CLImage> &input_color = _defog_handler->get_rgb_channel (i);
        args.push_back (new CLMemArgument (input_color));
    }

    const VideoBufferInfo & video_info_out = output->get_video_info ();

    CLImageDesc cl_desc_out;
    cl_desc_out.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc_out.format.image_channel_order = CL_RGBA;
    cl_desc_out.width = video_info_out.width / 8;
    cl_desc_out.height = video_info_out.height;
    cl_desc_out.row_pitch = video_info_out.strides[0];
    SmartPtr<CLImage> image_out_y = convert_to_climage (context, output, cl_desc_out, video_info_out.offsets[0]);

    cl_desc_out.height = video_info_out.height / 2;
    cl_desc_out.row_pitch = video_info_out.strides[1];
    SmartPtr<CLImage> image_out_uv = convert_to_climage (context, output, cl_desc_out, video_info_out.offsets[1]);

    args.push_back (new CLMemArgument (image_out_y));
    args.push_back (new CLMemArgument (image_out_uv));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 8;

    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out.height, work_size.local[1]); // uv height

    return XCAM_RETURN_NO_ERROR;
}

CLDefogDcpImageHandler::CLDefogDcpImageHandler (
    const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
{
}

XCamReturn
CLDefogDcpImageHandler::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCAM_UNUSED (output);
    XCamReturn ret = allocate_transmit_bufs (input->get_video_info ());
    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "CLDefogDcpImageHandler allocate transmit buffers failed");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLDefogDcpImageHandler::execute_done (SmartPtr<VideoBuffer> &output)
{
    XCAM_UNUSED (output);
#if 0
    dump_buffer ();
#endif

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLDefogDcpImageHandler::allocate_transmit_bufs (const VideoBufferInfo &video_info)
{
    int i;
    CLImageDesc cl_rgb_desc, cl_dark_desc;
    SmartPtr<CLContext> context = get_context ();

    cl_rgb_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_rgb_desc.format.image_channel_order = CL_RGBA;
    cl_rgb_desc.width = video_info.width / 8;
    cl_rgb_desc.height = video_info.height;

    for (i = 0; i < XCAM_DEFOG_MAX_CHANNELS; ++i) {
        _rgb_buf[i] = new CLImage2D (context, cl_rgb_desc);
        XCAM_FAIL_RETURN(
            WARNING,
            _rgb_buf[i]->is_valid (),
            XCAM_RETURN_ERROR_MEM,
            "CLDefogDcpImageHandler allocate RGB buffers failed");
    }

    cl_dark_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_dark_desc.format.image_channel_order = CL_RGBA;
    cl_dark_desc.width = video_info.width / 8;
    cl_dark_desc.height = video_info.height;

    for (i = 0; i < XCAM_DEFOG_DC_MAX_BUF; ++i) {
        _dark_channel_buf[i] = new CLImage2D (context, cl_dark_desc);
        XCAM_FAIL_RETURN(
            WARNING,
            _dark_channel_buf[i]->is_valid (),
            XCAM_RETURN_ERROR_MEM,
            "CLDefogDcpImageHandler allocate dark channel buffers failed");
    }

    return XCAM_RETURN_NO_ERROR;
}

void
CLDefogDcpImageHandler::dump_buffer ()
{
    SmartPtr<CLImage> image;
    CLImageDesc desc;
    uint32_t width, height;
    char file_name[1024];

    // dump dark channel bi-filtered map
    image = _dark_channel_buf[XCAM_DEFOG_DC_BI_FILTER];
    desc = image->get_image_desc ();
    width = image->get_pixel_bytes () * desc.width;
    height = desc.height;

    snprintf (file_name, 1024, "dark-channel-map_%dx%d.y", width, height);
    dump_image (image, file_name);
}

static SmartPtr<CLDarkChannelKernel>
create_kernel_dark_channel (const SmartPtr<CLContext> &context, SmartPtr<CLDefogDcpImageHandler> handler)
{
    SmartPtr<CLDarkChannelKernel> kernel;

    kernel = new CLDarkChannelKernel (context, handler);
    XCAM_FAIL_RETURN (
        WARNING,
        kernel->build_kernel (kernels_info[KernelDarkChannel], NULL) == XCAM_RETURN_NO_ERROR,
        NULL,
        "Defog build kernel(%s) failed", kernels_info[KernelDarkChannel].kernel_name);
    return kernel;
}

static SmartPtr<CLMinFilterKernel>
create_kernel_min_filter (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLDefogDcpImageHandler> handler,
    int index)
{
    SmartPtr<CLMinFilterKernel> kernel;

    char build_options[1024];
    xcam_mem_clear (build_options);
    snprintf (
        build_options, sizeof (build_options),
        " -DVERTICAL_MIN_KERNEL=%d ", (XCAM_DEFOG_DC_MIN_FILTER_V == index ? 1 : 0));

    kernel = new CLMinFilterKernel (context, handler, index);
    XCAM_FAIL_RETURN (
        WARNING,
        kernel->build_kernel (kernels_info[KernelMinFilter], build_options) == XCAM_RETURN_NO_ERROR,
        NULL,
        "Defog build kernel(%s) failed", kernels_info[KernelMinFilter].kernel_name);

    return kernel;
}

static SmartPtr<CLBiFilterKernel>
create_kernel_bi_filter (
    const SmartPtr<CLContext> &context, SmartPtr<CLDefogDcpImageHandler> handler)
{
    SmartPtr<CLBiFilterKernel> kernel;

    kernel = new CLBiFilterKernel (context, handler);
    XCAM_FAIL_RETURN (
        WARNING,
        kernel->build_kernel (kernels_info[KernelBiFilter], NULL) == XCAM_RETURN_NO_ERROR,
        NULL,
        "Defog build kernel(%s) failed", kernels_info[KernelBiFilter].kernel_name);

    return kernel;
}

static SmartPtr<CLDefogRecoverKernel>
create_kernel_defog_recover (
    const SmartPtr<CLContext> &context, SmartPtr<CLDefogDcpImageHandler> handler)
{
    SmartPtr<CLDefogRecoverKernel> kernel;

    kernel = new CLDefogRecoverKernel (context, handler);
    XCAM_FAIL_RETURN (
        WARNING,
        kernel->build_kernel (kernels_info[KernelDefogRecover], NULL) == XCAM_RETURN_NO_ERROR,
        NULL,
        "Defog build kernel(%s) failed", kernels_info[KernelDefogRecover].kernel_name);
    return kernel;
}

SmartPtr<CLImageHandler>
create_cl_defog_dcp_image_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLDefogDcpImageHandler> defog_handler;

    SmartPtr<CLImageKernel> kernel;

    defog_handler = new CLDefogDcpImageHandler (context, "cl_handler_defog_dcp");
    kernel = create_kernel_dark_channel (context, defog_handler);
    XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "defog handler create dark channel kernel failed");
    defog_handler->add_kernel (kernel);

#if 0
    for (int i = XCAM_DEFOG_DC_MIN_FILTER_V; i <= XCAM_DEFOG_DC_MIN_FILTER_H; ++i) {
        SmartPtr<CLImageKernel> min_kernel;
        min_kernel = create_kernel_min_filter (context, defog_handler, i);
        XCAM_FAIL_RETURN (ERROR, min_kernel.ptr (), NULL, "defog handler create min filter kernel failed");
        defog_handler->add_kernel (min_kernel);
    }
#endif

    kernel = create_kernel_bi_filter (context, defog_handler);
    XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "defog handler create bilateral filter kernel failed");
    defog_handler->add_kernel (kernel);

    kernel = create_kernel_defog_recover (context, defog_handler);
    XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "defog handler create defog recover kernel failed");
    defog_handler->add_kernel (kernel);

    return defog_handler;
}

}
