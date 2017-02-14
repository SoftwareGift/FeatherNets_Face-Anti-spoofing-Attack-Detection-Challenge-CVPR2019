/*
 * cl_blender.cpp - CL blender
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

#include "cl_blender.h"
#include "cl_device.h"
#include "cl_image_bo_buffer.h"

namespace XCam {

CLBlender::CLBlender (const char *name, bool need_uv, CLBlenderScaleMode scale_mode)
    : CLImageHandler (name)
    , _output_width (0)
    , _output_height (0)
    , _need_uv (need_uv)
    , _swap_input_index (false)
    , _scale_mode (scale_mode)
{
}

bool
CLBlender::set_merge_window (const Rect &window) {
    _merge_window = window;
    _merge_window.pos_x = XCAM_ALIGN_AROUND (_merge_window.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    _merge_window.width = XCAM_ALIGN_AROUND (_merge_window.width, XCAM_BLENDER_ALIGNED_WIDTH);
    XCAM_ASSERT (_merge_window.width >= XCAM_BLENDER_ALIGNED_WIDTH);
    XCAM_LOG_DEBUG(
        "CLBlender(%s) merge window:(x:%d, width:%d), blend_width:%d",
        XCAM_STR (get_name()),
        _merge_window.pos_x, _merge_window.width, _output_width);
    return true;
}

bool
CLBlender::set_input_valid_area (const Rect &area, uint32_t index)
{
    XCAM_ASSERT (index < XCAM_CL_BLENDER_IMAGE_NUM);
    _input_valid_area[index] = area;

    _input_valid_area[index].pos_x = XCAM_ALIGN_DOWN (_input_valid_area[index].pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    _input_valid_area[index].width = XCAM_ALIGN_UP (_input_valid_area[index].width, XCAM_BLENDER_ALIGNED_WIDTH);

    XCAM_LOG_DEBUG(
        "CLBlender(%s) buf(%d) valid area:(x:%d, width:%d)",
        XCAM_STR(get_name()), index,
        _input_valid_area[index].pos_x, _input_valid_area[index].width);
    return true;
}

bool
CLBlender::set_input_merge_area (const Rect &area, uint32_t index)
{
    XCAM_ASSERT (index < XCAM_CL_BLENDER_IMAGE_NUM);
    if (!is_merge_window_set ()) {
        XCAM_LOG_ERROR ("set_input_merge_area(idx:%d) failed, need set merge window first", index);
        return false;
    }

    if (_scale_mode == CLBlenderScaleGlobal)
        XCAM_ASSERT (fabs((int32_t)(area.width - _merge_window.width)) < XCAM_BLENDER_ALIGNED_WIDTH);

    _input_merge_area[index] = area;
    _input_merge_area[index].pos_x = XCAM_ALIGN_AROUND (_input_merge_area[index].pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    if (_scale_mode == CLBlenderScaleGlobal)
        _input_merge_area[index].width = _merge_window.width;

    XCAM_LOG_DEBUG(
        "CLBlender(%s) buf(%d) merge area:(x:%d, width:%d)",
        XCAM_STR(get_name()), index,
        _input_merge_area[index].pos_x, _input_merge_area[index].width);

    return true;
}

XCamReturn
CLBlender::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    uint32_t output_width = _output_width;
    uint32_t output_height = input.height;

    // aligned at least XCAM_BLENDER_ALIGNED_WIDTH
    uint32_t aligned_width = XCAM_MAX (16, XCAM_BLENDER_ALIGNED_WIDTH);
    output.init (
        input.format, output_width, output_height,
        XCAM_ALIGN_UP(output_width, aligned_width), XCAM_ALIGN_UP(output_height, 16));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLBlender::prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (input.ptr () && output.ptr ());
    SmartPtr<DrmBoBuffer> input0, input1;

    SmartPtr<DrmBoBuffer> next = input->find_typed_attach<DrmBoBuffer> ();
    XCAM_FAIL_RETURN(
        WARNING,
        next.ptr (),
        XCAM_RETURN_ERROR_PARAM,
        "CLBlender(%s) does NOT find second buffer in attachment", get_name());

    if (_swap_input_index) {
        input0 = next;
        input1 = input;
    } else {
        input0 = input;
        input1 = next;
    }

    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();

    if (!_input_valid_area[0].width) {
        Rect area;
        area.width = in0_info.width;
        area.height = in0_info.height;
        set_input_valid_area (area, 0);
    }
    if (!_input_valid_area[1].width) {
        Rect area;
        area.width = in1_info.width;
        area.height = in1_info.height;
        set_input_valid_area (area, 1);
    }

    if (!is_merge_window_set ()) {
        Rect merge_window;
        XCAM_FAIL_RETURN (
            WARNING,
            calculate_merge_window (get_input_valid_area(0).width, get_input_valid_area(1).width, out_info.width, merge_window),
            XCAM_RETURN_ERROR_PARAM,
            "CLBlender(%s) auto calculate merge window failed", get_name ());

        merge_window.pos_y = 0;
        merge_window.height = out_info.height;
        set_merge_window (merge_window);

        Rect area;
        area.width = merge_window.width;
        area.height = merge_window.height;
        area.pos_x = merge_window.pos_x;
        set_input_merge_area (area, 0);
        area.pos_x = 0;
        set_input_merge_area (area, 1);
    }

    ret = allocate_cl_buffers (context, input0, input1, output);
    return ret;
}

bool
CLBlender::calculate_merge_window (
    uint32_t width0, uint32_t width1, uint32_t blend_width,
    Rect &out_window)
{
    out_window.pos_x = blend_width - width1;
    out_window.width = (width0 + width1 - blend_width) / 2;

    out_window.pos_x = XCAM_ALIGN_AROUND (out_window.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    out_window.width = XCAM_ALIGN_AROUND (out_window.width, XCAM_BLENDER_ALIGNED_WIDTH);
    if ((int)blend_width < out_window.pos_x + out_window.width)
        out_window.width = blend_width - out_window.pos_x;

    XCAM_ASSERT (out_window.width > 0 && out_window.width <= (int)blend_width);
    XCAM_ASSERT (out_window.pos_x >= 0 && out_window.pos_x <= (int)blend_width);

    return true;
}

};

