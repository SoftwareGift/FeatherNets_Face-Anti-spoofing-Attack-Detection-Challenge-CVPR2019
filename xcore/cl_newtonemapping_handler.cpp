/*
 * cl_newtonemapping_handler.cpp - CL tonemapping handler
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
#include "xcam_utils.h"
#include "cl_newtonemapping_handler.h"

namespace XCam {

CLNewTonemappingImageKernel::CLNewTonemappingImageKernel (SmartPtr<CLContext> &context,
        const char *name)
    : CLImageKernel (context, name)
{
}

XCamReturn
CLNewTonemappingImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);

    const VideoBufferInfo & in_video_info = input->get_video_info ();
    _image_height = in_video_info.aligned_height;

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    SmartPtr<X3aStats> stats = input->find_3a_stats ();
    XCam3AStats *stats_ptr = stats->get_stats ();
    int pixel_totalnum = stats_ptr->info.aligned_width * stats_ptr->info.aligned_height;
    ushort *image_buf = (ushort *)input->map();
    float *y = new float[pixel_totalnum];
    _y_max = 0;
    _y_min = 65536;
    for(int i = 0; i < pixel_totalnum; i++)
    {
        float r = image_buf[i + pixel_totalnum];
        float g = (image_buf[i] + image_buf[i + pixel_totalnum * 3]) / 2.0f;
        float b = image_buf[i + pixel_totalnum * 2];
        y[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        if(_y_max < y[i]) _y_max = (int)y[i];
        if(_y_min > y[i]) _y_min = (int)y[i];
    }
    int hist_log[65536];
    float t = 0.01f;
    for(int i = 0; i < 65536; i++)
    {
        hist_log[i] = 0;
    }
    for(int i = 0; i < pixel_totalnum; i++)
    {
        int index = (int)(65535 * (log(y[i] / _y_max + t) - log(_y_min / _y_max + t)) / (log(1.0f + t) - log(_y_min / _y_max + t)));
        hist_log[index]++;
    }
    int avg_binnum = pixel_totalnum / 256;
    int acc_num = 0;
    int j = 1;
    _hist_leq[0] = 0;
    for(int i = 0; i < 65536 && j < 256; i++)
    {
        acc_num += hist_log[i];
        while(acc_num > avg_binnum && j < 256)
        {
            _hist_leq[j] = 0.5 * (i / 65535.0f - j / 255.0f) + j / 255.0f;
            j++;
            acc_num -= avg_binnum;
        }
    }
    for(int i = j; i < 256; i++)
    {
        _hist_leq[i] = i / 255.0f;
    }
    input->unmap();

    _hist_leq_buffer = new CLBuffer(
        context, sizeof(float) * 256,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &_hist_leq);

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_hist_leq_buffer->get_mem_id ();
    args[2].arg_size = sizeof (cl_mem);
    args[3].arg_adress = &_image_height;
    args[3].arg_size = sizeof (int);
    args[4].arg_adress = &_y_max;
    args[4].arg_size = sizeof (int);
    args[5].arg_adress = &_y_min;
    args[5].arg_size = sizeof (int);

    arg_count = 6;

    const CLImageDesc out_info = _image_out->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width;
    work_size.global[1] = out_info.height / 4;
    work_size.local[0] = 8;
    work_size.local[1] = 8;

    return XCAM_RETURN_NO_ERROR;
}

CLNewTonemappingImageHandler::CLNewTonemappingImageHandler (const char *name)
    : CLImageHandler (name)
    , _output_format (XCAM_PIX_FMT_SGRBG16_planar)
{
}

bool
CLNewTonemappingImageHandler::set_tonemapping_kernel(SmartPtr<CLNewTonemappingImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _tonemapping_kernel = kernel;
    return true;
}

XCamReturn
CLNewTonemappingImageHandler::prepare_buffer_pool_video_info (
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
create_cl_newtonemapping_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLNewTonemappingImageHandler> tonemapping_handler;
    SmartPtr<CLNewTonemappingImageKernel> tonemapping_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    tonemapping_kernel = new CLNewTonemappingImageKernel (context, "kernel_newtonemapping");
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_newtonemapping)
#include "kernel_newtonemapping.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = tonemapping_kernel->load_from_source (kernel_newtonemapping_body, strlen (kernel_newtonemapping_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", tonemapping_kernel->get_kernel_name());
    }
    XCAM_ASSERT (tonemapping_kernel->is_valid ());
    tonemapping_handler = new CLNewTonemappingImageHandler("cl_handler_newtonemapping");
    tonemapping_handler->set_tonemapping_kernel(tonemapping_kernel);

    return tonemapping_handler;
}

};
