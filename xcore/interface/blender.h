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

#ifndef XCAM_INTERFACE_BLENDER_H
#define XCAM_INTERFACE_BLENDER_H

#include <xcam_std.h>
#include <video_buffer.h>
#include <interface/data_types.h>

#define XCAM_BLENDER_IMAGE_NUM 2

namespace XCam {

class Blender;

class Blender
{
public:
    explicit Blender (uint32_t alignment_x, uint32_t alignment_y);
    virtual ~Blender ();
    static SmartPtr<Blender> create_ocl_blender ();
    static SmartPtr<Blender> create_soft_blender ();

    void set_output_size (uint32_t width, uint32_t height);
    void get_output_size (uint32_t &width, uint32_t &height) const {
        width = _out_width;
        height = _out_height;
    }
    bool set_input_valid_area (const Rect &area, uint32_t index);
    bool set_merge_window (const Rect &window);
    virtual bool set_input_merge_area (const Rect &area, uint32_t index);

    const Rect &get_merge_window () const {
        return _merge_window;
    }

    const Rect &get_input_merge_area (uint32_t index) const {
        return _input_merge_area[index];
    }
    const Rect &get_input_valid_area (uint32_t index) const {
        return _input_valid_area[index];
    }

    bool is_merge_window_set () const {
        return _merge_window.pos_x || _merge_window.width;
    }

    uint32_t get_alignment_x () const {
        return _alignment_x;
    }
    uint32_t get_alignment_y () const {
        return _alignment_y;
    }

    virtual XCamReturn blend (
        const SmartPtr<VideoBuffer> &in0,
        const SmartPtr<VideoBuffer> &in1,
        SmartPtr<VideoBuffer> &out_buf);

protected:
    bool auto_calc_merge_window (
        uint32_t width0, uint32_t width1, uint32_t blend_width, Rect &out_window);

private:
    XCAM_DEAD_COPY (Blender);

private:
    uint32_t                         _alignment_x, _alignment_y;
    uint32_t                         _out_width, _out_height;
    Rect                             _input_valid_area[XCAM_BLENDER_IMAGE_NUM];
    Rect                             _merge_window;  // for output buffer

protected:
    Rect                             _input_merge_area[XCAM_BLENDER_IMAGE_NUM];
};

}

#endif //XCAM_INTERFACE_BLENDER_H
