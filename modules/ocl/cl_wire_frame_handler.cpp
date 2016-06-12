/*
  * cl_wire_frame_handler.cpp - CL wire frame handler
  *
  *  Copyright (c) 2016 Intel Corporation
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  * 	 http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  *
  * Author: Yinhang Liu <yinhangx.liu@intel.com>
  */

#include "cl_wire_frame_handler.h"

static float border_y = 120.0f;
static float border_u = -58.0f;
static float border_v = -104.0f;
static uint32_t border_size = 2;

namespace XCam {

CLWireFrameImageKernel::CLWireFrameImageKernel (SmartPtr<CLContext> &context, const char *name)
    : CLImageKernel (context, name)
    , _wire_frames_num (0)
    , _wire_frames_coords_num (0)
    , _wire_frames_coords (NULL)
{
}

bool
CLWireFrameImageKernel::set_wire_frame_config (const XCamFDResult *config, double scaler_factor)
{
    if (!config) {
        XCAM_LOG_ERROR ("set wire frame config error, invalid config parameters !");
        return false;
    }

    _wire_frames_num = config->face_num;
    xcam_mem_clear (_wire_frames);
    for (uint32_t i = 0; i < _wire_frames_num; i++) {
        _wire_frames [i].pos_x = (uint32_t)(config->faces [i].pos_x / scaler_factor / 2) * 2;
        _wire_frames [i].pos_y = (uint32_t)(config->faces [i].pos_y / scaler_factor / 2) * 2;
        _wire_frames [i].width = (uint32_t)(config->faces [i].width / scaler_factor / 2) * 2;
        _wire_frames [i].height = (uint32_t)(config->faces [i].height / scaler_factor / 2) * 2;
    }

    return true;
}

bool
CLWireFrameImageKernel::check_wire_frames_validity (uint32_t image_width, uint32_t image_height)
{
    for (uint32_t i = 0; i < _wire_frames_num; i++) {
        if (_wire_frames [i].pos_x > image_width) {
            XCAM_LOG_ERROR ("check_wire_frames_validity: invalid pos_x (%d)", _wire_frames [i].pos_x);
            return false;
        }
        if (_wire_frames [i].pos_y > image_height) {
            XCAM_LOG_ERROR ("check_wire_frames_validity: invalid pos_y (%d)", _wire_frames [i].pos_y);
            return false;
        }
        if (_wire_frames [i].pos_x + _wire_frames [i].width > image_width) {
            XCAM_LOG_ERROR ("check_wire_frames_validity: invalid width (%d)", _wire_frames [i].width);
            return false;
        }
        if (_wire_frames [i].pos_y + _wire_frames [i].height > image_width) {
            XCAM_LOG_ERROR ("check_wire_frames_validity: invalid height (%d)", _wire_frames [i].height);
            return false;
        }
    }

    return true;
}

uint32_t
CLWireFrameImageKernel::get_border_coordinates_num ()
{
    uint32_t coords_num = 0;
    for (uint32_t i = 0; i < _wire_frames_num; i++) {
        coords_num += _wire_frames [i].width * _wire_frames [i].height
                      - (_wire_frames [i].width - 2 * border_size) * (_wire_frames [i].height - 2 * border_size);
    }

    return coords_num / 2;
}

bool
CLWireFrameImageKernel::get_border_coordinates (uint32_t *coords)
{
    uint32_t index = 0;
    for (uint32_t i = 0; i < _wire_frames_num; i++) {
        for (uint32_t j = 0; j < border_size; j++) {
            for (uint32_t k = 0; k < _wire_frames [i].width; k += 2) {
                coords [index++] = _wire_frames [i].pos_x + k;
                coords [index++] = _wire_frames [i].pos_y + j;
            }
        }

        for (uint32_t j = 0; j < border_size; j++) {
            for (uint32_t k = 0; k < _wire_frames [i].width; k += 2) {
                coords [index++] = _wire_frames [i].pos_x + k;
                coords [index++] = _wire_frames [i].pos_y + _wire_frames [i].height - border_size + j;
            }
        }

        for (uint32_t j = 0; j < _wire_frames [i].height - 2 * border_size; j++) {
            for (uint32_t k = 0; k < border_size; k += 2) {
                coords [index++] = _wire_frames [i].pos_x + k;
                coords [index++] = _wire_frames [i].pos_y + border_size + j;
            }
        }

        for (uint32_t j = 0; j < _wire_frames [i].height - 2 * border_size; j++) {
            for (uint32_t k = 0; k < border_size; k += 2) {
                coords [index++] = _wire_frames [i].pos_x + _wire_frames [i].width - border_size + k;
                coords [index++] = _wire_frames [i].pos_y + border_size + j;
            }
        }
    }

    return true;
}

XCamReturn
CLWireFrameImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);

    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo &video_info_out = output->get_video_info ();
    CLImageDesc cl_desc_out;

    cl_desc_out.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_out.format.image_channel_order = CL_RG;
    cl_desc_out.width = video_info_out.width / 2;
    cl_desc_out.height = video_info_out.height;
    cl_desc_out.row_pitch = video_info_out.strides [0];
    _image_out = new CLVaImage (context, output, cl_desc_out, video_info_out.offsets [0]);

    cl_desc_out.height = video_info_out.height / 2;
    cl_desc_out.row_pitch = video_info_out.strides [1];
    _image_out_uv = new CLVaImage (context, output, cl_desc_out, video_info_out.offsets [1]);

    XCAM_FAIL_RETURN (
        WARNING,
        _image_out->is_valid () && _image_out_uv->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel (%s) in/out memory not available", get_kernel_name ());

    XCAM_FAIL_RETURN (
        ERROR,
        check_wire_frames_validity (video_info_out.width, video_info_out.height),
        XCAM_RETURN_ERROR_PARAM,
        "prepare_arguments: invalid wire frames parameters");
    _wire_frames_coords_num = get_border_coordinates_num ();
    _wire_frames_coords = (uint32_t *) xcam_malloc0 (_wire_frames_coords_num * sizeof (uint32_t) * 2 + 1);
    XCAM_ASSERT (_wire_frames_coords);
    get_border_coordinates (_wire_frames_coords);

    _wire_frames_coords_buf = new CLBuffer (
        context,
        _wire_frames_coords_num * sizeof (uint32_t) * 2 + 1,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        _wire_frames_coords);

    /* set args */
    arg_count = 0;
    args [arg_count].arg_adress = &_image_out->get_mem_id ();
    args [arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args [arg_count].arg_adress = &_image_out_uv->get_mem_id ();
    args [arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args [arg_count].arg_adress = &_wire_frames_coords_buf->get_mem_id ();
    args [arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args [arg_count].arg_adress = &_wire_frames_coords_num;
    args [arg_count].arg_size = sizeof (_wire_frames_coords_num);
    ++arg_count;

    args [arg_count].arg_adress = &border_y;
    args [arg_count].arg_size = sizeof (border_y);
    ++arg_count;

    args [arg_count].arg_adress = &border_u;
    args [arg_count].arg_size = sizeof (border_u);
    ++arg_count;

    args [arg_count].arg_adress = &border_v;
    args [arg_count].arg_size = sizeof (border_v);
    ++arg_count;

    work_size.dim = 1;
    work_size.local [0] = 16;
    work_size.global [0] = _wire_frames_coords_num ? XCAM_ALIGN_UP (_wire_frames_coords_num, work_size.local [0]) : work_size.local [0];

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLWireFrameImageKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    if (_wire_frames_coords) {
        xcam_free (_wire_frames_coords);
        _wire_frames_coords = NULL;
    }

    _image_out_uv.release ();
    _wire_frames_coords_buf.release ();

    return CLImageKernel::post_execute (output);
}

CLWireFrameImageHandler::CLWireFrameImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLWireFrameImageHandler::set_wire_frame_kernel (SmartPtr<CLWireFrameImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _wire_frame_kernel = kernel;
    return true;
}

bool
CLWireFrameImageHandler::set_wire_frame_config (const XCamFDResult *config, double scaler_factor)
{
    if (!_wire_frame_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set wire frame config error, invalid kernel !");
        return false;
    }

    return _wire_frame_kernel->set_wire_frame_config (config, scaler_factor);
}

XCamReturn
CLWireFrameImageHandler::prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    output = input;
    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_wire_frame_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLWireFrameImageHandler> wire_frame_handler;
    SmartPtr<CLWireFrameImageKernel> wire_frame_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    wire_frame_kernel = new CLWireFrameImageKernel (context, "kernel_wire_frame");
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN (kernel_wire_frame)
#include "kernel_wire_frame.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = wire_frame_kernel->load_from_source (
                  kernel_wire_frame_body, strlen (kernel_wire_frame_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed",
            wire_frame_kernel->get_kernel_name ());
    }
    XCAM_ASSERT (wire_frame_kernel->is_valid ());
    wire_frame_handler = new CLWireFrameImageHandler ("cl_handler_wire_frame");
    wire_frame_handler->set_wire_frame_kernel (wire_frame_kernel);

    return wire_frame_handler;
}

};
