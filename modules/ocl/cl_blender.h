/*
 * cl_blender.h - CL blender
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

#ifndef XCAM_CL_BLENDER_H
#define XCAM_CL_BLENDER_H

#include "xcam_utils.h"
#include "ocl/cl_image_handler.h"

#define XCAM_CL_BLENDER_IMAGE_NUM  2
#define XCAM_BLENDER_ALIGNED_WIDTH 8

namespace XCam {

enum CLBlenderScaleMode {
    CLBlenderScaleLocal = 0,
    CLBlenderScaleGlobal,
    CLBlenderScaleMax
};

enum {
    CLBlenderPlaneY = 0,
    CLBlenderPlaneUV,
    CLBlenderPlaneMax,
};

struct Rect {
    int32_t pos_x, pos_y;
    int32_t width, height;

    Rect () : pos_x (0), pos_y (0), width (0), height (0) {}
};

class CLBlenderScaleKernel
    : public CLImageKernel
{
public:
    explicit CLBlenderScaleKernel (const SmartPtr<CLContext> &context, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (
        CLArgList &args, CLWorkSize &work_size);

    virtual SmartPtr<CLImage> get_input_image () = 0;
    virtual SmartPtr<CLImage> get_output_image () = 0;

    virtual bool get_output_info (uint32_t &out_width, uint32_t &out_height, int &out_offset_x) = 0;

private:
    XCAM_DEAD_COPY (CLBlenderScaleKernel);

protected:
    bool                               _is_uv;
};

class CLBlender
    : public CLImageHandler
{
public:
    explicit CLBlender (
        const SmartPtr<CLContext> &context, const char *name,
        bool need_uv, CLBlenderScaleMode scale_mode);

    void set_output_size (uint32_t width, uint32_t height) {
        _output_width = width; //XCAM_ALIGN_UP (width, XCAM_BLENDER_ALIGNED_WIDTH);
        _output_height = height;
    }

    bool set_input_valid_area (const Rect &area, uint32_t index);
    bool set_merge_window (const Rect &window);
    bool set_input_merge_area (const Rect &area, uint32_t index);

    const Rect &get_merge_window () const {
        return _merge_window;
    }
    const Rect &get_input_merge_area (uint32_t index) const {
        return _input_merge_area[index];
    }
    const Rect &get_input_valid_area (uint32_t index) const {
        return _input_valid_area[index];
    }

    bool need_uv () const {
        return _need_uv;
    }
    bool is_merge_window_set () const {
        return _merge_window.pos_x || _merge_window.width;
    }

    CLBlenderScaleMode get_scale_mode () const {
        return _scale_mode;
    }

    void swap_input_idx (bool flag) {
        _swap_input_index = flag;
    }

protected:
    virtual XCamReturn prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

    bool calculate_merge_window (uint32_t width0, uint32_t width1, uint32_t blend_width, Rect &out_window);

    //abstract virtual functions
    virtual XCamReturn allocate_cl_buffers (
        SmartPtr<CLContext> context, SmartPtr<DrmBoBuffer> &input0,
        SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output) = 0;

private:
    XCAM_DEAD_COPY (CLBlender);

private:
    uint32_t                         _output_width;
    uint32_t                         _output_height;
    Rect                             _input_valid_area[XCAM_CL_BLENDER_IMAGE_NUM];
    Rect                             _input_merge_area[XCAM_CL_BLENDER_IMAGE_NUM];
    Rect                             _merge_window;  // for output buffer
    bool                             _need_uv;
    bool                             _swap_input_index;
    CLBlenderScaleMode               _scale_mode;
};

SmartPtr<CLImageHandler>
create_linear_blender (const SmartPtr<CLContext> &context, bool need_uv = true);

SmartPtr<CLImageHandler>
create_pyramid_blender (
    const SmartPtr<CLContext> &context, int layer = 1, bool need_uv = true,
    bool need_seam = true, CLBlenderScaleMode scale_mode = CLBlenderScaleLocal);

};

#endif //XCAM_CL_BLENDER_H
