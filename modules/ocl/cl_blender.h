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

#include <xcam_std.h>
#include <interface/data_types.h>
#include <interface/blender.h>
#include <ocl/cl_image_handler.h>

#define XCAM_CL_BLENDER_ALIGNMENT_X 8
#define XCAM_CL_BLENDER_ALIGNMENT_Y 1

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
    : public CLImageHandler, public Blender
{
public:
    explicit CLBlender (
        const SmartPtr<CLContext> &context, const char *name,
        bool need_uv, CLBlenderScaleMode scale_mode);

    //derived from Blender
    virtual bool set_input_merge_area (const Rect &area, uint32_t index);

    bool need_uv () const {
        return _need_uv;
    }

    CLBlenderScaleMode get_scale_mode () const {
        return _scale_mode;
    }

    void swap_input_idx (bool flag) {
        _swap_input_index = flag;
    }

protected:
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

    //abstract virtual functions
    virtual XCamReturn allocate_cl_buffers (
        SmartPtr<CLContext> context, SmartPtr<VideoBuffer> &input0,
        SmartPtr<VideoBuffer> &input1, SmartPtr<VideoBuffer> &output) = 0;

private:
    XCAM_DEAD_COPY (CLBlender);

private:
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
