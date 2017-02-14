/*
 * cl_image_360_stitch.cpp - CL Image 360 stitch
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

#include "cl_image_360_stitch.h"

namespace XCam {

CLImage360Stitch::CLImage360Stitch (CLBlenderScaleMode scale_mode)
    : CLMultiImageHandler ("CLImage360Stitch")
    , _output_width (0)
    , _output_height (0)
    , _scale_mode (scale_mode)
{
}

bool
CLImage360Stitch::set_left_blender (SmartPtr<CLBlender> blender)
{
    _left_blender = blender;

    SmartPtr<CLImageHandler> handler = blender;
    return add_image_handler (handler);
}

bool
CLImage360Stitch::set_right_blender (SmartPtr<CLBlender> blender)
{
    _right_blender = blender;

    SmartPtr<CLImageHandler> handler = blender;
    return add_image_handler (handler);
}

bool
CLImage360Stitch::set_image_overlap (const int idx, const Rect &overlap0, const Rect &overlap1)
{
    XCAM_ASSERT (idx < ImageIdxCount);
    _overlaps[idx][0] = overlap0;
    _overlaps[idx][1] = overlap1;
    return true;
}

XCamReturn
CLImage360Stitch::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    uint32_t output_width = _output_width;
    uint32_t output_height = _output_height;

    XCAM_FAIL_RETURN(
        WARNING,
        output_width && output_height,
        XCAM_RETURN_ERROR_PARAM,
        "CLImage360Stitch(%s) prepare buffer pool info failed since width:%d height:%d was not set correctly",
        XCAM_STR(get_name()), output_width, output_height);

    // aligned at least XCAM_BLENDER_ALIGNED_WIDTH
    uint32_t aligned_width = XCAM_MAX (16, XCAM_BLENDER_ALIGNED_WIDTH);
    output.init (
        input.format, output_width, output_height,
        XCAM_ALIGN_UP(output_width, aligned_width), XCAM_ALIGN_UP(output_height, 16));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImage360Stitch::prepare_global_scale_blender_parameters (
    SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();

    XCAM_ASSERT (in0_info.height == in1_info.height);
    XCAM_ASSERT (in0_info.width <= out_info.width && in1_info.width <= out_info.width);

    Rect main_left = get_image_overlap (ImageIdxMain, 0);
    Rect main_right = get_image_overlap (ImageIdxMain, 1);
    Rect scnd_left = get_image_overlap (ImageIdxSecondary, 1);
    Rect scnd_right = get_image_overlap (ImageIdxSecondary, 0);
    int main_mid = XCAM_ALIGN_DOWN (in0_info.width / 2, XCAM_BLENDER_ALIGNED_WIDTH);
    int scnd_mid = 0;
    int out_mid = XCAM_ALIGN_DOWN (out_info.width / 2, XCAM_BLENDER_ALIGNED_WIDTH);
    Rect area, out_merge_window;
    area.pos_y = out_merge_window.pos_y = 0;
    area.height = out_merge_window.pos_y = out_info.height;

    //calculate left stitching area(input)
    int32_t prev_pos = main_left.pos_x;
    main_left.pos_x = XCAM_ALIGN_AROUND (main_left.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    main_left.width = XCAM_ALIGN_UP (main_left.width, XCAM_BLENDER_ALIGNED_WIDTH);
    scnd_left.pos_x += main_left.pos_x - prev_pos;
    scnd_left.pos_x = XCAM_ALIGN_AROUND (scnd_left.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    scnd_left.width = main_left.width;

    //calculate right stitching area(input)
    prev_pos = main_right.pos_x;
    main_right.pos_x = XCAM_ALIGN_AROUND (main_right.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    main_right.width = XCAM_ALIGN_UP (main_right.width, XCAM_BLENDER_ALIGNED_WIDTH);
    scnd_right.pos_x += main_right.pos_x - prev_pos;
    scnd_right.pos_x = XCAM_ALIGN_AROUND (scnd_right.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    scnd_right.width = main_right.width;

    //find scnd_mid
    scnd_mid = scnd_left.pos_x + (main_mid - main_left.pos_x) - out_mid;
    if (scnd_mid < scnd_right.pos_x + scnd_right.width)
        scnd_mid = scnd_right.pos_x + scnd_right.width;

    // set left blender
    area.pos_x = scnd_mid;
    area.width = scnd_left.pos_x + scnd_left.width - scnd_mid;
    _left_blender->set_input_valid_area (area, 0);

    area.pos_x = main_left.pos_x;
    area.width = main_mid - main_left.pos_x;
    _left_blender->set_input_valid_area (area, 1);

    out_merge_window.width = main_left.width;
    out_merge_window.pos_x = out_mid - (main_mid - main_left.pos_x);
    _left_blender->set_merge_window (out_merge_window);
    _left_blender->set_input_merge_area (scnd_left, 0);
    _left_blender->set_input_merge_area (main_left, 1);

    // set right blender
    area.pos_x = main_mid;
    area.width = main_right.pos_x + main_right.width - main_mid;
    _right_blender->set_input_valid_area (area, 0);

    area.pos_x = scnd_right.pos_x;
    area.width = scnd_mid - scnd_right.pos_x;
    _right_blender->set_input_valid_area (area, 1);

    out_merge_window.pos_x = out_mid + (main_right.pos_x - main_mid);
    out_merge_window.width = main_right.width;
    _right_blender->set_merge_window (out_merge_window);
    _right_blender->set_input_merge_area (main_right, 0);
    _right_blender->set_input_merge_area (scnd_right, 1);

    return ret;
}

XCamReturn
CLImage360Stitch::prepare_local_scale_blender_parameters (
    SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();

    XCAM_ASSERT (in0_info.height == in1_info.height);
    XCAM_ASSERT (in0_info.width <= out_info.width && in1_info.width <= out_info.width);

    Rect main_left = get_image_overlap (ImageIdxMain, 0);
    Rect main_right = get_image_overlap (ImageIdxMain, 1);
    Rect scnd_left = get_image_overlap (ImageIdxSecondary, 1);
    Rect scnd_right = get_image_overlap (ImageIdxSecondary, 0);

    int main_mid = XCAM_ALIGN_DOWN (in0_info.width / 2, XCAM_BLENDER_ALIGNED_WIDTH);
    int scnd_mid = XCAM_ALIGN_DOWN (in1_info.width / 2, XCAM_BLENDER_ALIGNED_WIDTH);
    int out_mid = XCAM_ALIGN_DOWN (out_info.width / 2, XCAM_BLENDER_ALIGNED_WIDTH);
    Rect area, out_merge_window;
    area.pos_y = out_merge_window.pos_y = 0;
    area.height = out_merge_window.pos_y = out_info.height;

    //calculate left stitching area(input)
    int32_t prev_pos = main_left.pos_x;
    main_left.pos_x = XCAM_ALIGN_AROUND (main_left.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    main_left.width = XCAM_ALIGN_UP (main_left.width, XCAM_BLENDER_ALIGNED_WIDTH);
    scnd_left.pos_x += main_left.pos_x - prev_pos;
    scnd_left.pos_x = XCAM_ALIGN_AROUND (scnd_left.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    scnd_left.width = main_left.width;

    //calculate right stitching area(input)
    prev_pos = main_right.pos_x;
    main_right.pos_x = XCAM_ALIGN_AROUND (main_right.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    main_right.width = XCAM_ALIGN_UP (main_right.width, XCAM_BLENDER_ALIGNED_WIDTH);
    scnd_right.pos_x += main_right.pos_x - prev_pos;
    scnd_right.pos_x = XCAM_ALIGN_AROUND (scnd_right.pos_x, XCAM_BLENDER_ALIGNED_WIDTH);
    scnd_right.width = main_right.width;

    // set left blender
    area.pos_x = scnd_mid;
    area.width = scnd_left.pos_x + scnd_left.width - scnd_mid;
    _left_blender->set_input_valid_area (area, 0);

    area.pos_x = main_left.pos_x;
    area.width = main_mid - main_left.pos_x;
    _left_blender->set_input_valid_area (area, 1);

    int delta_width = out_mid - (main_mid - main_left.pos_x) - (scnd_left.pos_x - scnd_mid);
    out_merge_window.width = main_left.width + delta_width;
    out_merge_window.pos_x = scnd_left.pos_x - scnd_mid;
    _left_blender->set_merge_window (out_merge_window);
    _left_blender->set_input_merge_area (scnd_left, 0);
    _left_blender->set_input_merge_area (main_left, 1);

    // set right blender
    area.pos_x = main_mid;
    area.width = main_right.pos_x + main_right.width - main_mid;
    _right_blender->set_input_valid_area (area, 0);

    area.pos_x = scnd_right.pos_x;
    area.width = scnd_mid - scnd_right.pos_x;
    _right_blender->set_input_valid_area (area, 1);

    delta_width = out_mid - (scnd_mid - scnd_right.pos_x) - (main_right.pos_x - main_mid);
    out_merge_window.width = main_right.width + delta_width;
    out_merge_window.pos_x = out_mid + (main_right.pos_x - main_mid);
    _right_blender->set_merge_window (out_merge_window);
    _right_blender->set_input_merge_area (main_right, 0);
    _right_blender->set_input_merge_area (scnd_right, 1);

    return ret;
}

XCamReturn
CLImage360Stitch::prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    SmartPtr<DrmBoBuffer> input1 = input->find_typed_attach<DrmBoBuffer> ();
    XCAM_FAIL_RETURN(
        WARNING,
        input1.ptr (),
        XCAM_RETURN_ERROR_PARAM,
        "CLImage360Stitch(%s) does NOT find second buffer in attachment", get_name());

    if (_scale_mode == CLBlenderScaleLocal)
        ret = prepare_local_scale_blender_parameters (input, input1, output);
    else
        ret = prepare_global_scale_blender_parameters (input, input1, output);
    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        XCAM_RETURN_ERROR_PARAM,
        "CLImage360Stitch(%s) failed to prepare blender parameters", get_name());

    return CLMultiImageHandler::prepare_parameters (input, output);
}

SmartPtr<CLImageHandler>
create_image_360_stitch (SmartPtr<CLContext> &context, bool need_seam, CLBlenderScaleMode scale_mode)
{
    const int layer = 2;
    const bool need_uv = true;
    SmartPtr<CLBlender>  left_blender, right_blender;
    SmartPtr<CLImage360Stitch> stitch = new CLImage360Stitch (scale_mode);
    XCAM_ASSERT (stitch.ptr ());

    left_blender = create_pyramid_blender (context, layer, need_uv, need_seam, scale_mode).dynamic_cast_ptr<CLBlender> ();
    XCAM_FAIL_RETURN (ERROR, left_blender.ptr (), NULL, "image_360_stitch create left blender failed");
    left_blender->disable_buf_pool (true);
    left_blender->swap_input_idx (true);
    stitch->set_left_blender (left_blender);

    right_blender = create_pyramid_blender (context, layer, need_uv, need_seam, scale_mode).dynamic_cast_ptr<CLBlender> ();
    XCAM_FAIL_RETURN (ERROR, right_blender.ptr (), NULL, "image_360_stitch create right blender failed");
    right_blender->disable_buf_pool (true);
    stitch->set_right_blender (right_blender);

    return stitch;
}

}

