/*
 * blender.h - blender interface
 *
 *  Copyright (c) 2017 Intel Corporation
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

#include "blender.h"

namespace XCam {

Blender::Blender (uint32_t alignment_x, uint32_t alignment_y)
    : _alignment_x (alignment_x)
    , _alignment_y (alignment_y)
    , _out_width (0)
    , _out_height (0)
{
}

Blender::~Blender ()
{
}

void
Blender::set_output_size (uint32_t width, uint32_t height) {
    _out_width = XCAM_ALIGN_UP (width, get_alignment_x ());
    _out_height = XCAM_ALIGN_UP (height, get_alignment_y ());
}

bool
Blender::set_merge_window (const Rect &window) {
    uint32_t alignmend_x = get_alignment_x ();

    _merge_window = window;
    _merge_window.pos_x = XCAM_ALIGN_AROUND (_merge_window.pos_x, alignmend_x);
    _merge_window.width = XCAM_ALIGN_AROUND (_merge_window.width, alignmend_x);
    XCAM_ASSERT (_merge_window.width >= (int32_t)alignmend_x);
    XCAM_LOG_DEBUG(
        "Blender merge window:(x:%d, width:%d), blend_width:%d",
        _merge_window.pos_x, _merge_window.width, _out_width);
    return true;
}

bool
Blender::set_input_valid_area (const Rect &area, uint32_t index)
{
    XCAM_ASSERT (index < XCAM_BLENDER_IMAGE_NUM);
    _input_valid_area[index] = area;

    uint32_t alignmend_x = get_alignment_x ();
    _input_valid_area[index].pos_x = XCAM_ALIGN_DOWN (_input_valid_area[index].pos_x, alignmend_x);
    _input_valid_area[index].width = XCAM_ALIGN_UP (_input_valid_area[index].width, alignmend_x);

    XCAM_LOG_DEBUG(
        "Blender buf(%d) valid area:(x:%d, width:%d)",
        index, _input_valid_area[index].pos_x, _input_valid_area[index].width);
    return true;
}

bool
Blender::set_input_merge_area (const Rect &area, uint32_t index)
{
    XCAM_ASSERT (index < XCAM_BLENDER_IMAGE_NUM);
    if (!is_merge_window_set ()) {
        XCAM_LOG_ERROR ("set_input_merge_area(idx:%d) failed, need set merge window first", index);
        return false;
    }

    _input_merge_area[index] = area;
    _input_merge_area[index].pos_x = XCAM_ALIGN_AROUND (_input_merge_area[index].pos_x, get_alignment_x ());
    _input_merge_area[index].pos_y = XCAM_ALIGN_AROUND (_input_merge_area[index].pos_y, get_alignment_y ());

    XCAM_LOG_DEBUG(
        "Blender buf(%d) merge area:(x:%d, width:%d)",
        index, _input_merge_area[index].pos_x, _input_merge_area[index].width);

    return true;
}

bool
Blender::auto_calc_merge_window (
    uint32_t width0, uint32_t width1, uint32_t blend_width,
    Rect &out_window)
{
    out_window.pos_x = blend_width - width1;
    out_window.width = (width0 + width1 - blend_width) / 2;

    out_window.pos_x = XCAM_ALIGN_AROUND (out_window.pos_x, get_alignment_x ());
    out_window.width = XCAM_ALIGN_AROUND (out_window.width, get_alignment_x ());
    if ((int)blend_width < out_window.pos_x + out_window.width)
        out_window.width = blend_width - out_window.pos_x;

    XCAM_ASSERT (out_window.width > 0 && out_window.width <= (int)blend_width);
    XCAM_ASSERT (out_window.pos_x >= 0 && out_window.pos_x <= (int)blend_width);

    return true;
}

XCamReturn
Blender::blend (
    const SmartPtr<VideoBuffer> &,
    const SmartPtr<VideoBuffer> &,
    SmartPtr<VideoBuffer> &)
{
    XCAM_LOG_ERROR ("Blender interface blend must be derived.");
    return XCAM_RETURN_ERROR_UNKNOWN;
}

}
