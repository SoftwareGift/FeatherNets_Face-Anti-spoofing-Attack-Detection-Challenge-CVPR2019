/*
 * cl_geo_map_handler.cpp - CL geometry map handler
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
#include "cl_geo_map_handler.h"
#include "cl_device.h"

namespace XCam {

const XCamKernelInfo kernel_geo_map_info = {
    "kernel_geo_map",
#include "kernel_geo_map.clx"
    , 0,
};

// GEO_MAP_CHANNEL for CL_RGBA channel
#define GEO_MAP_CHANNEL 4

CLGeoMapKernel::CLGeoMapKernel (SmartPtr<CLContext> &context, SmartPtr<CLGeoMapHandler> &handler)
    : CLImageKernel (context)
    , _handler (handler)
{
    XCAM_ASSERT (handler.ptr ());
    xcam_mem_clear (_geo_scale_size);
    xcam_mem_clear (_out_size);
}

XCamReturn
CLGeoMapKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);

    SmartPtr<CLImage> input_y = _handler->get_input_image (CLGeoPlaneY);
    SmartPtr<CLImage> input_uv = _handler->get_input_image (CLGeoPlaneUV);
    SmartPtr<CLImage> output_y = _handler->get_output_image (CLGeoPlaneY);
    SmartPtr<CLImage> output_uv = _handler->get_output_image (CLGeoPlaneUV);
    const CLImageDesc &outuv_desc = output_uv->get_image_desc ();
    SmartPtr<CLImage> geo_image = _handler->get_geo_map_image ();
    _handler->get_geo_equivalent_out_size (_geo_scale_size[0], _geo_scale_size[1]);
    _out_size[0] = _handler->get_output_width ();
    _out_size[1] = _handler->get_output_height ();

    arg_count = 0;
    args[arg_count].arg_adress = &input_y->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &input_uv->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &geo_image->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_geo_scale_size;
    args[arg_count].arg_size = sizeof (_geo_scale_size);
    ++arg_count;

    args[arg_count].arg_adress = &output_y->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &output_uv->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_out_size;
    args[arg_count].arg_size = sizeof (_out_size);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (outuv_desc.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (outuv_desc.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLGeoMapHandler::CLGeoMapHandler ()
    : CLImageHandler ("CLGeoMapHandler")
    , _output_width (0)
    , _output_height (0)
    , _map_width (0)
    , _map_height (0)
    , _uint_x (1.0f)
    , _uint_y (1.0f)
    , _geo_map_normalized (false)
{
}

bool
CLGeoMapHandler::set_map_data (GeoPos *data, uint32_t width, uint32_t height)
{
    uint32_t size = width * height * GEO_MAP_CHANNEL * sizeof (float); // 4 for CL_RGBA,
    float *map_ptr = NULL;

    XCAM_ASSERT (width && height);
    if (width != _map_width || height != _map_height) {
        SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
        _map_width = width;
        _map_height = height;
        XCAM_ASSERT (context.ptr ());
        _geo_map = new CLBuffer (context, size);
    }

    XCAM_ASSERT (_geo_map.ptr () && _geo_map->is_valid ());

    XCamReturn ret = _geo_map->enqueue_map ((void *&)map_ptr, 0, size);
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, false, "CLGeoMapHandler(%s) map buffer failed", get_name ());
    for (uint32_t i = 0; i < width * height; ++i) {
        map_ptr [i * GEO_MAP_CHANNEL] = data [i].x;
        map_ptr [i * GEO_MAP_CHANNEL + 1] = data [i].y;
    }
    _geo_map->enqueue_unmap ((void *&)map_ptr);
    _geo_map_normalized = false;
    return true;
}

void
CLGeoMapHandler::get_geo_equivalent_out_size (float &width, float &height) const
{
    width = _map_width * _uint_x;
    height = _map_height * _uint_y;
}

XCamReturn
CLGeoMapHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input, VideoBufferInfo &output)
{
    XCAM_FAIL_RETURN (
        WARNING, input.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "CLGeoMapHandler(%s) input buffer format(%s) not NV12", get_name (), xcam_fourcc_to_string (input.format));

    if (!_output_width || !_output_height) {
        _output_width = input.width;
        _output_height = input.height;
    }
    output.init (
        input.format, _output_width, _output_height,
        XCAM_ALIGN_UP (_output_width, 16), XCAM_ALIGN_UP (_output_height, 16));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLGeoMapHandler::normalize_geo_map (uint32_t image_w, uint32_t image_h)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    uint32_t size = _map_width * _map_height * GEO_MAP_CHANNEL * sizeof (float);
    float *map_ptr = NULL;

    if (_geo_map_normalized)
        return XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (image_w && image_h);
    XCAM_FAIL_RETURN (
        ERROR, _geo_map.ptr () && _geo_map->is_valid (),
        XCAM_RETURN_ERROR_PARAM, "geo map was not set");

    ret = _geo_map->enqueue_map ((void *&)map_ptr, 0, size);
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, ret, "CLGeoMapHandler map buffer failed");
    for (uint32_t i = 0; i < _map_width * _map_height; ++i) {
        map_ptr [i * GEO_MAP_CHANNEL] /= image_w;      // x
        map_ptr [i * GEO_MAP_CHANNEL + 1] /= image_h;  //y
    }
    _geo_map->enqueue_unmap ((void *&)map_ptr);

    _geo_map_normalized = true;
    return ret;
}

XCamReturn
CLGeoMapHandler::prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    const VideoBufferInfo &in_info = input->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();
    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
    uint32_t input_image_w = XCAM_ALIGN_DOWN (in_info.width, 2);
    uint32_t input_image_h = XCAM_ALIGN_DOWN (in_info.height, 2);

    CLImageDesc cl_desc;
    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc.format.image_channel_order = CL_R;
    cl_desc.width = input_image_w;
    cl_desc.height = input_image_h;
    cl_desc.row_pitch = in_info.strides[CLGeoPlaneY];
    _input[CLGeoPlaneY] = new CLVaImage (context, input, cl_desc, in_info.offsets[CLGeoPlaneY]);

    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc.format.image_channel_order = CL_RG;
    cl_desc.width = input_image_w / 2;
    cl_desc.height = input_image_h / 2;
    cl_desc.row_pitch = in_info.strides[CLGeoPlaneUV];
    _input[CLGeoPlaneUV] = new CLVaImage (context, input, cl_desc, in_info.offsets[CLGeoPlaneUV]);

    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;
    cl_desc.width = XCAM_ALIGN_DOWN (out_info.width, 4) / 8; //CL_RGBA * CL_UNSIGNED_INT16 = 8
    cl_desc.height = XCAM_ALIGN_DOWN (out_info.height, 2);
    cl_desc.row_pitch = out_info.strides[CLGeoPlaneY];
    _output[CLGeoPlaneY] = new CLVaImage (context, output, cl_desc, out_info.offsets[CLGeoPlaneY]);
    cl_desc.height /= 2;
    cl_desc.row_pitch = out_info.strides[CLGeoPlaneUV];
    _output[CLGeoPlaneUV] = new CLVaImage (context, output, cl_desc, out_info.offsets[CLGeoPlaneUV]);

    XCAM_ASSERT (
        _input[CLGeoPlaneY].ptr () && _input[CLGeoPlaneY]->is_valid () &&
        _input[CLGeoPlaneUV].ptr () && _input[CLGeoPlaneUV]->is_valid () &&
        _output[CLGeoPlaneY].ptr () && _output[CLGeoPlaneY]->is_valid () &&
        _output[CLGeoPlaneUV].ptr () && _output[CLGeoPlaneUV]->is_valid ());

    XCamReturn ret = normalize_geo_map (input_image_w, input_image_h);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR,
        ret, "normalized geo map failed");

    CLImageDesc cl_geo_desc;
    cl_geo_desc.format.image_channel_data_type = CL_FLOAT;
    cl_geo_desc.format.image_channel_order = CL_RGBA; // CL_FLOAT need co-work with CL_RGBA
    cl_geo_desc.width = _map_width;
    cl_geo_desc.height = _map_height;
    cl_geo_desc.row_pitch = _map_width * CLImage::calculate_pixel_bytes (cl_geo_desc.format);
    _geo_image = new CLImage2D (context, cl_geo_desc, 0, _geo_map);
    XCAM_ASSERT (_geo_image.ptr () && _geo_image->is_valid ());

    return CLImageHandler::prepare_parameters (input, output);
}

XCamReturn
CLGeoMapHandler::execute_done (SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (output);

    for (int i = 0; i < CLGeoPlaneMax; ++i) {
        _input[i].release ();
        _output[i].release ();
    }
    _geo_image.release ();

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_geo_map_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLGeoMapHandler> handler;
    SmartPtr<CLImageKernel> kernel;

    handler = new CLGeoMapHandler ();
    XCAM_ASSERT (handler.ptr ());

    kernel = new CLGeoMapKernel (context, handler);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, kernel->build_kernel (kernel_geo_map_info, NULL) == XCAM_RETURN_NO_ERROR,
        NULL, "build geo map kernel failed");
    handler->add_kernel (kernel);

    return handler;
}

}
