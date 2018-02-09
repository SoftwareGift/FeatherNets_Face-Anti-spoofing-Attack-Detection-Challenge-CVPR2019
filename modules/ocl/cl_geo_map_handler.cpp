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

#include "cl_utils.h"
#include "cl_geo_map_handler.h"
#include "cl_device.h"

namespace XCam {

static const XCamKernelInfo kernel_geo_map_info = {
    "kernel_geo_map",
#include "kernel_geo_map.clx"
    , 0,
};

// GEO_MAP_CHANNEL for CL_RGBA channel
#define GEO_MAP_CHANNEL 4  /* only use channel_0, channel_1 */

CLGeoMapKernel::CLGeoMapKernel (
    const SmartPtr<CLContext> &context, const SmartPtr<GeoKernelParamCallback> handler, bool need_lsc, bool need_scale)
    : CLImageKernel (context)
    , _handler (handler)
    , _need_lsc (need_lsc)
    , _need_scale (need_scale)
{
    XCAM_ASSERT (handler.ptr ());
}

XCamReturn
CLGeoMapKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLImage> input_y = _handler->get_geo_input_image (NV12PlaneYIdx);
    SmartPtr<CLImage> input_uv = _handler->get_geo_input_image (NV12PlaneUVIdx);
    SmartPtr<CLImage> output_y = _handler->get_geo_output_image (NV12PlaneYIdx);
    SmartPtr<CLImage> output_uv = _handler->get_geo_output_image (NV12PlaneUVIdx);
    const CLImageDesc &outuv_desc = output_uv->get_image_desc ();
    SmartPtr<CLImage> geo_image = _handler->get_geo_map_table ();

    float geo_scale_size[2]; //width/height
    float out_size[2];
    _handler->get_geo_equivalent_out_size (geo_scale_size[0], geo_scale_size[1]);
    _handler->get_geo_pixel_out_size (out_size[0], out_size[1]);

    args.push_back (new CLMemArgument (input_y));
    args.push_back (new CLMemArgument (input_uv));
    args.push_back (new CLMemArgument (geo_image));
    args.push_back (new CLArgumentTArray<float, 2> (geo_scale_size));

    if (_need_scale) {
        PointFloat2 left_scale_factor = _handler->get_left_scale_factor ();
        PointFloat2 right_scale_factor = _handler->get_right_scale_factor ();
        float stable_y_start = _handler->get_stable_y_start ();

        args.push_back (new CLArgumentT<PointFloat2> (left_scale_factor));
        args.push_back (new CLArgumentT<PointFloat2> (right_scale_factor));
        args.push_back (new CLArgumentT<float> (stable_y_start));
    }

    if (_need_lsc) {
        SmartPtr<CLImage> lsc_image = _handler->get_lsc_table ();
        float *gray_threshold = _handler->get_lsc_gray_threshold ();
        XCAM_FAIL_RETURN (
            ERROR,
            lsc_image.ptr() && lsc_image->is_valid () && gray_threshold,
            XCAM_RETURN_ERROR_PARAM,
            "CLGeoMapHandler::lsc table or gray threshold was not found");
        args.push_back (new CLMemArgument (lsc_image));
        args.push_back (new CLArgumentTArray<float, 2> (gray_threshold));
    }
    args.push_back (new CLMemArgument (output_y));
    args.push_back (new CLMemArgument (output_uv));
    args.push_back (new CLArgumentTArray<float, 2> (out_size));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (outuv_desc.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (outuv_desc.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLGeoMapHandler::CLGeoMapHandler (const SmartPtr<CLContext> &context)
    : CLImageHandler (context, "CLGeoMapHandler")
    , _output_width (0)
    , _output_height (0)
    , _map_width (0)
    , _map_height (0)
    , _map_aligned_width (0)
    , _uint_x (0.0f)
    , _uint_y (0.0f)
    , _geo_map_normalized (false)
{
}

void
CLGeoMapHandler::get_geo_equivalent_out_size (float &width, float &height)
{
    width = _map_width * _uint_x;
    height = _map_height * _uint_y;
}

void
CLGeoMapHandler::get_geo_pixel_out_size (float &width, float &height)
{
    width = _output_width;
    height = _output_height;
}

bool
CLGeoMapHandler::set_map_uint (float uint_x, float uint_y)
{
    _uint_x = uint_x;
    _uint_y = uint_y;
    return true;
}

bool
CLGeoMapHandler::set_map_data (GeoPos *data, uint32_t width, uint32_t height)
{
    uint32_t size = width * height * GEO_MAP_CHANNEL * sizeof (float); // 4 for CL_RGBA,
    float *map_ptr = NULL;

    XCAM_FAIL_RETURN (
        ERROR, check_geo_map_buf (width, height), false,
        "CLGeoMapKernel check geo map buffer failed");

    XCamReturn ret = _geo_map->enqueue_map ((void *&)map_ptr, 0, size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, false,
        "CLGeoMapKernel map buffer failed");

    uint32_t start, idx;
    for (uint32_t h = 0; h < height; ++h) {
        for (uint32_t w = 0; w < width; ++w) {
            start = (h * _map_aligned_width + w) * GEO_MAP_CHANNEL;
            idx = h * width + w;

            map_ptr [start] = data [idx].x;
            map_ptr [start + 1] = data [idx].y;
        }
    }
    _geo_map->enqueue_unmap ((void *&)map_ptr);
    _geo_map_normalized = false;
    return true;
}

bool
CLGeoMapHandler::check_geo_map_buf (uint32_t width, uint32_t height)
{
    XCAM_ASSERT (width && height);
    if (width == _map_width && height == _map_height && _geo_map.ptr ()) {
        return true; // geo memory already created
    }

    uint32_t aligned_width = XCAM_ALIGN_UP (width, XCAM_CL_IMAGE_ALIGNMENT_X);  // 4 channel for CL_RGBA, but only use RG
    uint32_t row_pitch = aligned_width * GEO_MAP_CHANNEL * sizeof (float);
    uint32_t size = row_pitch * height;
    SmartPtr<CLContext> context = get_context ();
    XCAM_ASSERT (context.ptr ());
    _geo_map = new CLBuffer (context, size);

    if (!_geo_map.ptr () || !_geo_map->is_valid ()) {
        XCAM_LOG_ERROR ("CLGeoMapKernel create geo map buffer failed.");
        _geo_map.release ();
        return false;
    }

    CLImageDesc cl_geo_desc;
    cl_geo_desc.format.image_channel_data_type = CL_FLOAT;
    cl_geo_desc.format.image_channel_order = CL_RGBA; // CL_FLOAT need co-work with CL_RGBA
    cl_geo_desc.width = width;
    cl_geo_desc.height = height;
    cl_geo_desc.row_pitch = row_pitch;
    _geo_image = new CLImage2D (context, cl_geo_desc, 0, _geo_map);
    if (!_geo_image.ptr () || !_geo_image->is_valid ()) {
        XCAM_LOG_ERROR ("CLGeoMapKernel convert geo map buffer to image2d failed.");
        _geo_map.release ();
        _geo_image.release ();
        return false;
    }

    _map_width = width;
    _map_height = height;
    _map_aligned_width = aligned_width;
    return true;
}


bool
CLGeoMapHandler::normalize_geo_map (uint32_t image_w, uint32_t image_h)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    uint32_t row_pitch = _map_aligned_width * GEO_MAP_CHANNEL * sizeof (float);  // 4 channel for CL_RGBA, but only use RG
    uint32_t size = row_pitch * _map_height;
    float *map_ptr = NULL;

    XCAM_ASSERT (image_w && image_h);
    XCAM_FAIL_RETURN (
        ERROR, _geo_map.ptr () && _geo_map->is_valid (),
        false, "CLGeoMapKernel geo_map was not initialized");

    ret = _geo_map->enqueue_map ((void *&)map_ptr, 0, size);
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, false, "CLGeoMapKernel map buffer failed");
    uint32_t idx = 0;
    for (uint32_t h = 0; h < _map_height; ++h) {
        for (uint32_t w = 0; w < _map_width; ++w) {
            idx = (h * _map_aligned_width + w) * GEO_MAP_CHANNEL;

            map_ptr [idx] /= image_w;
            map_ptr [idx + 1] /= image_h;
        }
    }
    _geo_map->enqueue_unmap ((void *&)map_ptr);

    return true;
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
CLGeoMapHandler::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    const VideoBufferInfo &in_info = input->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();
    SmartPtr<CLContext> context = get_context ();
    uint32_t input_image_w = XCAM_ALIGN_DOWN (in_info.width, 2);
    uint32_t input_image_h = XCAM_ALIGN_DOWN (in_info.height, 2);

    CLImageDesc cl_desc;
    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc.format.image_channel_order = CL_R;
    cl_desc.width = input_image_w;
    cl_desc.height = input_image_h;
    cl_desc.row_pitch = in_info.strides[NV12PlaneYIdx];
    _input[NV12PlaneYIdx] = convert_to_climage (context, input, cl_desc, in_info.offsets[NV12PlaneYIdx]);

    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc.format.image_channel_order = CL_RG;
    cl_desc.width = input_image_w / 2;
    cl_desc.height = input_image_h / 2;
    cl_desc.row_pitch = in_info.strides[NV12PlaneUVIdx];
    _input[NV12PlaneUVIdx] = convert_to_climage (context, input, cl_desc, in_info.offsets[NV12PlaneUVIdx]);

    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;
    cl_desc.width = XCAM_ALIGN_DOWN (out_info.width, 4) / 8; //CL_RGBA * CL_UNSIGNED_INT16 = 8
    cl_desc.height = XCAM_ALIGN_DOWN (out_info.height, 2);
    cl_desc.row_pitch = out_info.strides[NV12PlaneYIdx];
    _output[NV12PlaneYIdx] = convert_to_climage (context, output, cl_desc, out_info.offsets[NV12PlaneYIdx]);
    cl_desc.height /= 2;
    cl_desc.row_pitch = out_info.strides[NV12PlaneUVIdx];
    _output[NV12PlaneUVIdx] = convert_to_climage (context, output, cl_desc, out_info.offsets[NV12PlaneUVIdx]);

    XCAM_ASSERT (
        _input[NV12PlaneYIdx].ptr () && _input[NV12PlaneYIdx]->is_valid () &&
        _input[NV12PlaneUVIdx].ptr () && _input[NV12PlaneUVIdx]->is_valid () &&
        _output[NV12PlaneYIdx].ptr () && _output[NV12PlaneYIdx]->is_valid () &&
        _output[NV12PlaneUVIdx].ptr () && _output[NV12PlaneUVIdx]->is_valid ());

    XCAM_FAIL_RETURN (
        ERROR, _geo_map.ptr () && _geo_map->is_valid (),
        XCAM_RETURN_ERROR_PARAM, "CLGeoMapHandler map data was not set");

    //calculate kernel map unit_x, unit_y.
    float uint_x, uint_y;
    get_map_uint (uint_x, uint_y);
    if (uint_x < 1.0f && uint_y < 1.0f) {
        uint_x = out_info.width / (float)_map_width;
        uint_y = out_info.height / (float)_map_height;
        set_map_uint (uint_x, uint_y);
    }

    if (!_geo_map_normalized) {
        XCAM_FAIL_RETURN (
            ERROR, normalize_geo_map (input_image_w, input_image_h),
            XCAM_RETURN_ERROR_PARAM, "CLGeoMapHandler normalized geo map failed");
        _geo_map_normalized = true;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLGeoMapHandler::execute_done (SmartPtr<VideoBuffer> &output)
{
    XCAM_UNUSED (output);

    for (int i = 0; i < NV12PlaneMax; ++i) {
        _input[i].release ();
        _output[i].release ();
    }

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageKernel>
create_geo_map_kernel (
    const SmartPtr<CLContext> &context, SmartPtr<GeoKernelParamCallback> param_cb, bool need_lsc, bool need_scale)
{
    SmartPtr<CLImageKernel> kernel;
    kernel = new CLGeoMapKernel (context, param_cb, need_lsc, need_scale);
    XCAM_ASSERT (kernel.ptr ());

    char build_options[1024];
    snprintf (build_options, sizeof(build_options), "-DENABLE_LSC=%d -DENABLE_SCALE=%d", need_lsc ? 1 : 0, need_scale ? 1 : 0);
    XCAM_FAIL_RETURN (
        ERROR, kernel->build_kernel (kernel_geo_map_info, build_options) == XCAM_RETURN_NO_ERROR,
        NULL, "build geo map kernel failed");

    return kernel;
}

SmartPtr<CLImageHandler>
create_geo_map_handler (const SmartPtr<CLContext> &context, bool need_lsc, bool need_scale)
{
    SmartPtr<CLGeoMapHandler> handler;
    SmartPtr<CLImageKernel> kernel;

    handler = new CLGeoMapHandler (context);
    XCAM_ASSERT (handler.ptr ());

    kernel = create_geo_map_kernel (context, handler, need_lsc, need_scale);
    XCAM_FAIL_RETURN (
        ERROR, kernel.ptr (), NULL, "CLMapHandler build geo map kernel failed");
    handler->add_kernel (kernel);

    return handler;
}

}
