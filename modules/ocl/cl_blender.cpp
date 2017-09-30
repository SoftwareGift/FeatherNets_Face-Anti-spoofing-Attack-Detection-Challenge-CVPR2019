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

namespace XCam {

CLBlenderScaleKernel::CLBlenderScaleKernel (const SmartPtr<CLContext> &context, bool is_uv)
    : CLImageKernel (context)
    , _is_uv (is_uv)
{
}

XCamReturn
CLBlenderScaleKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> image_in = get_input_image ();
    SmartPtr<CLImage> image_out = get_output_image ();
    XCAM_ASSERT (image_in.ptr () && image_out.ptr ());
    int output_offset_x;
    uint32_t output_width, output_height;
    get_output_info (output_width, output_height, output_offset_x);

    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLArgumentT<int> (output_offset_x));
    args.push_back (new CLArgumentT<uint32_t> (output_width));
    args.push_back (new CLArgumentT<uint32_t> (output_height));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (output_width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (output_height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLBlender::CLBlender (
    const SmartPtr<CLContext> &context, const char *name,
    bool need_uv, CLBlenderScaleMode scale_mode)
    : CLImageHandler (context, name)
    , Blender (XCAM_CL_BLENDER_ALIGNMENT_X, XCAM_CL_BLENDER_ALIGNMENT_Y)
    , _need_uv (need_uv)
    , _swap_input_index (false)
    , _scale_mode (scale_mode)
{
    XCAM_ASSERT (get_alignment_x () == XCAM_CL_BLENDER_ALIGNMENT_X);
    XCAM_ASSERT (get_alignment_y () == XCAM_CL_BLENDER_ALIGNMENT_Y);
}

bool
CLBlender::set_input_merge_area (const Rect &area, uint32_t index)
{
    Rect tmp_area = area;
    if (_scale_mode == CLBlenderScaleGlobal)
        tmp_area.width = get_merge_window ().width;

    bool ret = Blender::set_input_merge_area (tmp_area, index);

    if (ret && _scale_mode == CLBlenderScaleGlobal) {
        XCAM_ASSERT (fabs((int32_t)(area.width - get_merge_window ().width)) < XCAM_CL_BLENDER_ALIGNMENT_X);
    }

    return ret;
}

XCamReturn
CLBlender::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    uint32_t output_width, output_height;
    get_output_size (output_width, output_height);
    XCAM_ASSERT (output_height == input.height);

    // aligned at least XCAM_BLENDER_ALIGNED_WIDTH
    uint32_t aligned_width = XCAM_MAX (16, XCAM_CL_BLENDER_ALIGNMENT_X);
    output.init (
        input.format, output_width, output_height,
        XCAM_ALIGN_UP(output_width, aligned_width), XCAM_ALIGN_UP(output_height, 16));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLBlender::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (input.ptr () && output.ptr ());
    SmartPtr<VideoBuffer> input0, input1;

    SmartPtr<VideoBuffer> next = input->find_typed_attach<VideoBuffer> ();
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

    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();

    if (!get_input_valid_area (0).width) {
        Rect area;
        area.width = in0_info.width;
        area.height = in0_info.height;
        set_input_valid_area (area, 0);
    }
    if (!get_input_valid_area (1).width) {
        Rect area;
        area.width = in1_info.width;
        area.height = in1_info.height;
        set_input_valid_area (area, 1);
    }

    if (!is_merge_window_set ()) {
        Rect merge_window;
        XCAM_FAIL_RETURN (
            WARNING,
            auto_calc_merge_window (get_input_valid_area(0).width, get_input_valid_area(1).width, out_info.width, merge_window),
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

SmartPtr<Blender>
create_ocl_blender ()
{
    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
    XCAM_FAIL_RETURN (
        ERROR, context.ptr (), NULL,
        "create ocl blender failed to get cl context");
    SmartPtr<CLBlender> blender = create_pyramid_blender (context, 2, true, false).dynamic_cast_ptr<CLBlender> ();
    XCAM_FAIL_RETURN (
        ERROR, blender.ptr (), NULL,
        "create ocl blender failed to get pyramid blender");
    return blender;
}

};

