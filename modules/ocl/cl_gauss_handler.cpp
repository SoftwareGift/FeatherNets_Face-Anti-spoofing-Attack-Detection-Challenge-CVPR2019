/*
 * cl_gauss_handler.cpp - CL gauss handler
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
#include "cl_gauss_handler.h"
#include <algorithm>

#define XCAM_GAUSS_SCALE(radius) ((radius) * 2 + 1)

namespace XCam {

const static XCamKernelInfo kernel_gauss_info = {
    "kernel_gauss",
#include "kernel_gauss.clx"
    , 0,
};

class CLGaussImageKernelImpl
    : public CLGaussImageKernel
{
public:
    CLGaussImageKernelImpl (
        SmartPtr<CLGaussImageHandler> &handler,
        const SmartPtr<CLContext> &context, uint32_t radius, float sigma);

    virtual SmartPtr<VideoBuffer> get_input_buf ();
    virtual SmartPtr<VideoBuffer> get_output_buf ();
private:
    SmartPtr<CLGaussImageHandler> _handler;
};

CLGaussImageKernelImpl::CLGaussImageKernelImpl (
    SmartPtr<CLGaussImageHandler> &handler,
    const SmartPtr<CLContext> &context,
    uint32_t radius,
    float sigma
)
    : CLGaussImageKernel (context, radius, sigma)
    , _handler (handler)
{
}

SmartPtr<VideoBuffer>
CLGaussImageKernelImpl::get_input_buf ()
{
    return _handler->get_input_buf ();
}
SmartPtr<VideoBuffer>
CLGaussImageKernelImpl::get_output_buf ()
{
    return _handler->get_output_buf ();;
}

CLGaussImageKernel::CLGaussImageKernel (
    const SmartPtr<CLContext> &context, uint32_t radius, float sigma)
    : CLImageKernel (context, "kernel_gauss")
    , _g_radius (radius)
    , _g_table (NULL)
{
    set_gaussian(radius, sigma);
}

CLGaussImageKernel::~CLGaussImageKernel ()
{
    xcam_free (_g_table);
}

bool
CLGaussImageKernel::set_gaussian (uint32_t radius, float sigma)
{
    uint32_t i, j;
    uint32_t scale = XCAM_GAUSS_SCALE (radius);
    float dis = 0.0f, sum = 0.0f;
    uint32_t scale_size = scale * scale * sizeof (_g_table[0]);

    xcam_free (_g_table);
    _g_table_buffer.release ();
    _g_radius = radius;
    _g_table = (float*) xcam_malloc0 (scale_size);
    XCAM_ASSERT (_g_table);

    for(i = 0; i < scale; i++)  {
        for(j = 0; j < scale; j++) {
            dis = ((float)i - radius) * ((float)i - radius) + ((float)j - radius) * ((float)j - radius);
            _g_table[i * scale + j] = exp(-dis / (2.0f * sigma * sigma));
            sum += _g_table[i * scale + j];
        }
    }

    for(i = 0; i < scale * scale; i++) {
        _g_table[i] = _g_table[i] / sum;
    }

    _g_table_buffer = new CLBuffer(
        get_context (), scale_size,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , _g_table);

    return true;
}

XCamReturn
CLGaussImageKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<VideoBuffer> input = get_input_buf ();
    SmartPtr<VideoBuffer> output = get_output_buf ();

    XCAM_FAIL_RETURN (
        WARNING,
        input.ptr () && output.ptr (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) get input/output buffer failed", get_kernel_name ());

    const VideoBufferInfo & video_info_in = input->get_video_info ();
    const VideoBufferInfo & video_info_out = output->get_video_info ();
    CLImageDesc cl_desc_in, cl_desc_out;

    cl_desc_in.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_in.format.image_channel_order = CL_R;
    cl_desc_in.width = video_info_in.width;
    cl_desc_in.height = video_info_in.height;
    cl_desc_in.row_pitch = video_info_in.strides[0];
    SmartPtr<CLImage> image_in = convert_to_climage (context, input, cl_desc_in, video_info_in.offsets[0]);

    cl_desc_out.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_out.format.image_channel_order = CL_RGBA;
    cl_desc_out.width = video_info_out.width / 4;
    cl_desc_out.height = video_info_out.height;
    cl_desc_out.row_pitch = video_info_out.strides[0];
    SmartPtr<CLImage> image_out = convert_to_climage (context, output, cl_desc_out, video_info_out.offsets[0]);

    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLMemArgument (_g_table_buffer));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = XCAM_ALIGN_UP(cl_desc_out.width, 8);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out.height / 2, 4);
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

CLGaussImageHandler::CLGaussImageHandler (const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
{
}

bool
CLGaussImageHandler::set_gaussian_table (int size, float sigma)
{
    _gauss_kernel->set_gaussian (size, sigma);
    return true;
}

bool
CLGaussImageHandler::set_gauss_kernel(SmartPtr<CLGaussImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _gauss_kernel = kernel;
    return true;
}

SmartPtr<CLImageHandler>
create_cl_gauss_image_handler (const SmartPtr<CLContext> &context, uint32_t radius, float sigma)
{
    SmartPtr<CLGaussImageHandler> gauss_handler;
    SmartPtr<CLGaussImageKernel> gauss_kernel;
    char build_options[1024];

    xcam_mem_clear (build_options);
    snprintf (build_options, sizeof (build_options), " -DGAUSS_RADIUS=%d ", radius);

    gauss_handler = new CLGaussImageHandler (context, "cl_handler_gauss");
    gauss_kernel = new CLGaussImageKernelImpl (gauss_handler, context, radius, sigma);
    XCAM_ASSERT (gauss_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, gauss_kernel->build_kernel (kernel_gauss_info, build_options) == XCAM_RETURN_NO_ERROR, NULL,
        "build gaussian kernel(%s) failed", kernel_gauss_info.kernel_name);

    XCAM_ASSERT (gauss_kernel->is_valid ());
    gauss_handler->set_gauss_kernel (gauss_kernel);

    return gauss_handler;
}

}
