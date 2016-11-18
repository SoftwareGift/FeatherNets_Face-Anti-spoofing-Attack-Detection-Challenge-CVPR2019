/*
 * cl_fisheye_handler.cpp - CL fisheye handler
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

#include "xcam_utils.h"
#include "cl_fisheye_handler.h"
#include "cl_device.h"

namespace XCam {

#define PI 3.1415926f
#define degree2radian(degree) ((degree) * PI / 180.0f)

enum {
    KernelFisheye2GPS,
};

const XCamKernelInfo kernel_fisheye_info[] = {
    {
        "kernel_fisheye_2_gps",
#include "kernel_fisheye.clx"
        , 0,
    },
};

CLFisheyeInfo::CLFisheyeInfo ()
    : center_x (0.0f)
    , center_y (0.0f)
    , wide_angle (0.0f)
    , radius (0.0f)
    , rotate_angle (0.0f)
{
}

bool
CLFisheyeInfo::is_valid () const
{
    return wide_angle >= 1.0f && radius >= 1.0f;
}

CLFisheye2GPSKernel::CLFisheye2GPSKernel (
    SmartPtr<CLContext> &context, SmartPtr<CLFisheyeHandler> &handler)
    : CLImageKernel (context)
    , _handler (handler)
{
    XCAM_ASSERT (handler.ptr ());
    xcam_mem_clear (_input_y_size);
    xcam_mem_clear (_out_center);
    xcam_mem_clear (_radian_per_pixel);
}

XCamReturn
CLFisheye2GPSKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);

    SmartPtr<CLImage> input_y = _handler->get_input_image (CLNV12PlaneY);
    SmartPtr<CLImage> input_uv = _handler->get_input_image (CLNV12PlaneUV);
    SmartPtr<CLImage> output_y = _handler->get_output_image (CLNV12PlaneY);
    SmartPtr<CLImage> output_uv = _handler->get_output_image (CLNV12PlaneUV);
    const CLImageDesc &input_y_desc = input_y->get_image_desc ();
    const CLImageDesc &outuv_desc = output_uv->get_image_desc ();

    _input_y_size[0] = input_y_desc.width;
    _input_y_size[1] = input_y_desc.height;

    uint32_t dst_w, dst_h;
    float dst_range_x, dst_range_y;
    _handler->get_ouput_size (dst_w, dst_h);
    _out_center[0] = (float)dst_w / 2.0f;
    _out_center[1] = (float)dst_h / 2.0f;

    _handler->get_dst_range (dst_range_x, dst_range_y);
    _radian_per_pixel[0] = degree2radian (dst_range_x) / (float)dst_w;
    _radian_per_pixel[1] = degree2radian (dst_range_y) / (float)dst_h;

    _fisheye_info = _handler->get_fisheye_info ();
    _fisheye_info.wide_angle = degree2radian (_fisheye_info.wide_angle);
    _fisheye_info.rotate_angle = degree2radian (_fisheye_info.rotate_angle);

    XCAM_LOG_DEBUG ("@CLFisheye2GPSKernel input size(%d, %d), out_center:(%d, %d), range:(%d,%d)",
                    (int)_input_y_size[0], (int)_input_y_size[1],
                    (int)_out_center[0], (int)_out_center[1],
                    (int)dst_range_x, (int)dst_range_y);

    arg_count = 0;
    args[arg_count].arg_adress = &input_y->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &input_uv->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_input_y_size;
    args[arg_count].arg_size = sizeof (_input_y_size);
    ++arg_count;

    args[arg_count].arg_adress = &_fisheye_info;
    args[arg_count].arg_size = sizeof (_fisheye_info);
    ++arg_count;

    args[arg_count].arg_adress = &output_y->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &output_uv->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_out_center;
    args[arg_count].arg_size = sizeof (_out_center);
    ++arg_count;

    args[arg_count].arg_adress = &_radian_per_pixel;
    args[arg_count].arg_size = sizeof (_radian_per_pixel);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (outuv_desc.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (outuv_desc.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLFisheyeHandler::CLFisheyeHandler ()
    : CLImageHandler ("CLFisheyeHandler")
    , _output_width (0)
    , _output_height (0)
    , _range_longitude (180.0f)
    , _range_latitude (180.0f)
{
}

void
CLFisheyeHandler::set_output_size (uint32_t width, uint32_t height)
{
    _output_width = width;
    _output_height = height;
}

void
CLFisheyeHandler::get_ouput_size (uint32_t &width, uint32_t &height) const
{
    width = _output_width;
    height = _output_height;
}

void
CLFisheyeHandler::set_dst_range (float longitude, float latitude)
{
    _range_longitude = longitude;
    _range_latitude = latitude;
}

void
CLFisheyeHandler::get_dst_range (float &longitude, float &latitude) const
{
    longitude = _range_longitude;
    latitude = _range_latitude;
}

void
CLFisheyeHandler::set_fisheye_info (const CLFisheyeInfo &info)
{
    _fisheye_info = info;
}

XCamReturn
CLFisheyeHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    XCAM_FAIL_RETURN (
        WARNING, input.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "CLFisheyeHandler(%s) input buffer format(%s) is not supported, try NV12",
        get_name (), xcam_fourcc_to_string (input.format));

    if (!_output_width || !_output_height) {
        return XCAM_RETURN_ERROR_PARAM;
    }
    XCAM_FAIL_RETURN (
        WARNING, _output_width && _output_height, XCAM_RETURN_ERROR_PARAM,
        "CLFisheyeHandler output size(%d, %d) should > 0",
        _output_width, _output_height);

    output.init (
        input.format, _output_width, _output_height,
        XCAM_ALIGN_UP (_output_width, 16), XCAM_ALIGN_UP (_output_height, 16));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLFisheyeHandler::prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    const VideoBufferInfo &in_info = input->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();
    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
    uint32_t input_image_w = XCAM_ALIGN_DOWN (in_info.width, 2);
    uint32_t input_image_h = XCAM_ALIGN_DOWN (in_info.height, 2);

    XCAM_FAIL_RETURN (
        WARNING, _fisheye_info.is_valid (), XCAM_RETURN_ERROR_PARAM,
        "CLFisheyeHandler fisheye info is not valid, please check");

    CLImageDesc cl_desc;
    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc.format.image_channel_order = CL_R;
    cl_desc.width = input_image_w;
    cl_desc.height = input_image_h;
    cl_desc.row_pitch = in_info.strides[CLNV12PlaneY];
    _input[CLNV12PlaneY] = new CLVaImage (context, input, cl_desc, in_info.offsets[CLNV12PlaneY]);

    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc.format.image_channel_order = CL_RG;
    cl_desc.width = input_image_w / 2;
    cl_desc.height = input_image_h / 2;
    cl_desc.row_pitch = in_info.strides[CLNV12PlaneUV];
    _input[CLNV12PlaneUV] = new CLVaImage (context, input, cl_desc, in_info.offsets[CLNV12PlaneUV]);

    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT8;
    cl_desc.format.image_channel_order = CL_RGBA;
    cl_desc.width = XCAM_ALIGN_DOWN (out_info.width, 4) / 4; //CL_RGBA * CL_UNSIGNED_INT8 = 4
    cl_desc.height = XCAM_ALIGN_DOWN (out_info.height, 2);
    cl_desc.row_pitch = out_info.strides[CLNV12PlaneY];
    _output[CLNV12PlaneY] = new CLVaImage (context, output, cl_desc, out_info.offsets[CLNV12PlaneY]);
    cl_desc.height /= 2;
    cl_desc.row_pitch = out_info.strides[CLNV12PlaneUV];
    _output[CLNV12PlaneUV] = new CLVaImage (context, output, cl_desc, out_info.offsets[CLNV12PlaneUV]);

    XCAM_ASSERT (
        _input[CLNV12PlaneY].ptr () && _input[CLNV12PlaneY]->is_valid () &&
        _input[CLNV12PlaneUV].ptr () && _input[CLNV12PlaneUV]->is_valid () &&
        _output[CLNV12PlaneY].ptr () && _output[CLNV12PlaneY]->is_valid () &&
        _output[CLNV12PlaneUV].ptr () && _output[CLNV12PlaneUV]->is_valid ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLFisheyeHandler::execute_done (SmartPtr<DrmBoBuffer> &output)
{
    for (int i = 0; i < CLNV12PlaneMax; ++i) {
        _input[i].release ();
        _output[i].release ();
    }
    return CLImageHandler::execute_done (output);
}

SmartPtr<CLImageHandler>
create_fisheye_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLFisheyeHandler> handler;
    SmartPtr<CLImageKernel> kernel;

    handler = new CLFisheyeHandler ();
    XCAM_ASSERT (handler.ptr ());

    kernel = new CLFisheye2GPSKernel (context, handler);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, kernel->build_kernel (kernel_fisheye_info[KernelFisheye2GPS], NULL) == XCAM_RETURN_NO_ERROR,
        NULL, "build fisheye kernel failed");
    handler->add_kernel (kernel);

    return handler;
}


}
