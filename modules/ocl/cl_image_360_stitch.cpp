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
#if HAVE_OPENCV
#include "cv_feature_match.h"
#endif

#define XCAM_BLENDER_GLOBAL_SCALE_EXT_WIDTH 64

#define STITCH_CHECK(ret, msg, ...) \
    if ((ret) != XCAM_RETURN_NO_ERROR) {        \
        XCAM_LOG_WARNING (msg, ## __VA_ARGS__); \
        return ret;                             \
    }

namespace XCam {

CLBlenderGlobalScaleKernel::CLBlenderGlobalScaleKernel (SmartPtr<CLContext> &context, bool is_uv)
    : CLBlenderScaleKernel (context, is_uv)
{
}

SmartPtr<CLImage>
CLBlenderGlobalScaleKernel::get_input_image (SmartPtr<DrmBoBuffer> &input) {
    SmartPtr<CLContext> context = get_context ();

    CLImageDesc cl_desc;
    SmartPtr<CLImage> cl_image;
    const VideoBufferInfo &buf_info = input->get_video_info ();

    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    if (_is_uv) {
        cl_desc.format.image_channel_order = CL_RG;
        cl_desc.width = buf_info.width / 2;
        cl_desc.height = buf_info.height / 2;
        cl_desc.row_pitch = buf_info.strides[1];
        cl_image = new CLVaImage (context, input, cl_desc, buf_info.offsets[1]);
    } else {
        cl_desc.format.image_channel_order = CL_R;
        cl_desc.width = buf_info.width;
        cl_desc.height = buf_info.height;
        cl_desc.row_pitch = buf_info.strides[0];
        cl_image = new CLVaImage (context, input, cl_desc, buf_info.offsets[0]);
    }

    return cl_image;
}

SmartPtr<CLImage>
CLBlenderGlobalScaleKernel::get_output_image (SmartPtr<DrmBoBuffer> &output) {
    SmartPtr<CLContext> context = get_context ();

    CLImageDesc cl_desc;
    SmartPtr<CLImage> cl_image;
    const VideoBufferInfo &buf_info = output->get_video_info ();

    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;
    if (_is_uv) {
        cl_desc.width = buf_info.width / 8;
        cl_desc.height = buf_info.height / 2;
        cl_desc.row_pitch = buf_info.strides[1];
        cl_image = new CLVaImage (context, output, cl_desc, buf_info.offsets[1]);
    } else {
        cl_desc.width = buf_info.width / 8;
        cl_desc.height = buf_info.height;
        cl_desc.row_pitch = buf_info.strides[0];
        cl_image = new CLVaImage (context, output, cl_desc, buf_info.offsets[0]);
    }

    return cl_image;
}

bool
CLBlenderGlobalScaleKernel::get_output_info (
    SmartPtr<DrmBoBuffer> &output,
    uint32_t &out_width, uint32_t &out_height, int &out_offset_x)
{
    const VideoBufferInfo &output_info = output->get_video_info ();

    out_width = output_info.width / 8;
    out_height = _is_uv ? output_info.height / 2 : output_info.height;
    out_offset_x = 0;

    return true;
}

static CLStitchInfo
get_default_stitch_info ()
{
    CLStitchInfo stitch_info;

    stitch_info.merge_width[0] = 56;
    stitch_info.merge_width[1] = 56;

    stitch_info.crop[0].left = 96;
    stitch_info.crop[0].right = 96;
    stitch_info.crop[0].top = 0;
    stitch_info.crop[0].bottom = 0;
    stitch_info.crop[1].left = 96;
    stitch_info.crop[1].right = 96;
    stitch_info.crop[1].top = 0;
    stitch_info.crop[1].bottom = 0;

    stitch_info.fisheye_info[0].center_x = 480.0f;
    stitch_info.fisheye_info[0].center_y = 480.0f;
    stitch_info.fisheye_info[0].wide_angle = 202.8f;
    stitch_info.fisheye_info[0].radius = 480.0f;
    stitch_info.fisheye_info[0].rotate_angle = -90.0f;
    stitch_info.fisheye_info[1].center_x = 1440.0f;
    stitch_info.fisheye_info[1].center_y = 480.0f;
    stitch_info.fisheye_info[1].wide_angle = 202.8f;
    stitch_info.fisheye_info[1].radius = 480.0f;
    stitch_info.fisheye_info[1].rotate_angle = 89.4f;

    return stitch_info;
}

CLImage360Stitch::CLImage360Stitch (SmartPtr<CLContext> &context, CLBlenderScaleMode scale_mode)
    : CLMultiImageHandler ("CLImage360Stitch")
    , _context (context)
    , _fisheye_width (0)
    , _fisheye_height (0)
    , _output_width (0)
    , _output_height (0)
    , _is_stitch_inited (false)
    , _scale_mode (scale_mode)
{
    xcam_mem_clear (_merge_width);

#if HAVE_OPENCV
    init_opencv_ocl (context);
#endif
}

bool
CLImage360Stitch::set_fisheye_handler (SmartPtr<CLFisheyeHandler> fisheye, int index)
{
    XCAM_ASSERT (index < ImageIdxCount);

    _fisheye[index] = fisheye;
    SmartPtr<CLImageHandler> handler = fisheye;
    return add_image_handler (handler);
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
CLImage360Stitch::init_stitch_info (CLStitchInfo stitch_info)
{
    if (_is_stitch_inited) {
        XCAM_LOG_WARNING ("stitching info was initialized and can't be set twice");
        return false;
    }

    for (int index = 0; index < ImageIdxCount; ++index) {
        _merge_width[index] = stitch_info.merge_width[index];
        _fisheye[index]->set_fisheye_info (stitch_info.fisheye_info[index]);
        _crop_info[index] = stitch_info.crop[index];
    }

    _is_stitch_inited = true;

    return true;
}

bool
CLImage360Stitch::set_image_overlap (const int idx, const Rect &overlap0, const Rect &overlap1)
{
    XCAM_ASSERT (idx < ImageIdxCount);
    _overlaps[idx][0] = overlap0;
    _overlaps[idx][1] = overlap1;
    return true;
}

void
CLImage360Stitch::calc_fisheye_initial_info (SmartPtr<DrmBoBuffer> &output)
{
    const VideoBufferInfo &out_info = output->get_video_info ();
    _fisheye_width = (out_info.width + _merge_width[0] + _merge_width[1]
                      + _crop_info[0].left + _crop_info[0].right
                      + _crop_info[1].left + _crop_info[1].right) / 2;
    _fisheye_width = XCAM_ALIGN_UP (_fisheye_width, 16);
    _fisheye_height = out_info.height + _crop_info[0].top + _crop_info[0].bottom;
    XCAM_LOG_INFO (
        "fisheye correction output size width:%d height:%d",
        _fisheye_width, _fisheye_height);

    float max_dst_angle = 180.0f * _fisheye_width / _fisheye_height;
    for (int index = 0; index < ImageIdxCount; ++index) {
        _fisheye[index]->set_dst_range (max_dst_angle, 180.0f);
        _fisheye[index]->set_output_size (_fisheye_width, _fisheye_height);
    }
}

void
CLImage360Stitch::update_image_overlap ()
{
    static bool is_stitch_info_inited = false;
    if (!is_stitch_info_inited) {
        _img_merge_info[0].merge_left.pos_x = _crop_info[0].left;
        _img_merge_info[0].merge_left.pos_y = _crop_info[0].top;
        _img_merge_info[0].merge_left.width = _merge_width[0];
        _img_merge_info[0].merge_left.height = _fisheye_height - _crop_info[0].top - _crop_info[0].bottom;
        _img_merge_info[0].merge_right.pos_x = _fisheye_width - _crop_info[0].right - _merge_width[1];
        _img_merge_info[0].merge_right.pos_y = _crop_info[0].top;
        _img_merge_info[0].merge_right.width = _merge_width[1];
        _img_merge_info[0].merge_right.height = _fisheye_height - _crop_info[0].top - _crop_info[0].bottom;

        _img_merge_info[1].merge_left.pos_x = _crop_info[1].left;
        _img_merge_info[1].merge_left.pos_y = _crop_info[1].top;
        _img_merge_info[1].merge_left.width = _merge_width[1];
        _img_merge_info[1].merge_left.height = _fisheye_height - _crop_info[1].top - _crop_info[1].bottom;
        _img_merge_info[1].merge_right.pos_x = _fisheye_width - _crop_info[1].right - _merge_width[0];
        _img_merge_info[1].merge_right.pos_y = _crop_info[1].top;
        _img_merge_info[1].merge_right.width = _merge_width[0];
        _img_merge_info[1].merge_right.height = _fisheye_height - _crop_info[1].top - _crop_info[1].bottom;

        is_stitch_info_inited = true;
    }

    set_image_overlap (0, _img_merge_info[0].merge_left, _img_merge_info[0].merge_right);
    set_image_overlap (1, _img_merge_info[1].merge_left, _img_merge_info[1].merge_right);
}

XCamReturn
CLImage360Stitch::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input, VideoBufferInfo &output)
{
    if (_output_width == 0 || _output_height == 0) {
        _output_width = input.width;
        _output_height = XCAM_ALIGN_UP (input.width / 2, 16);
    }
    XCAM_FAIL_RETURN(
        WARNING,
        _output_width && _output_height && (_output_width == _output_height * 2),
        XCAM_RETURN_ERROR_PARAM,
        "CLImage360Stitch(%s) prepare buffer pool info failed since width:%d height:%d was not set correctly",
        XCAM_STR(get_name()), _output_width, _output_height);

    // aligned at least XCAM_BLENDER_ALIGNED_WIDTH
    uint32_t aligned_width = XCAM_MAX (16, XCAM_BLENDER_ALIGNED_WIDTH);
    output.init (
        input.format, _output_width, _output_height,
        XCAM_ALIGN_UP(_output_width, aligned_width), XCAM_ALIGN_UP(_output_height, 16));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImage360Stitch::prepare_fisheye_parameters (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (input);

    static bool is_fisheye_inited = false;
    if (!is_fisheye_inited) {
        calc_fisheye_initial_info (output);
        is_fisheye_inited = true;
    }

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
CLImage360Stitch::execute_self_prepare_parameters (
    SmartPtr<CLImageHandler> specified_handler, SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCAM_ASSERT (specified_handler.ptr ());

    for (HandlerList::iterator i_handler = _handler_list.begin ();
            i_handler != _handler_list.end (); ++i_handler) {
        SmartPtr<CLImageHandler> &handler = *i_handler;
        XCAM_ASSERT (handler.ptr ());
        if (specified_handler.ptr () != handler.ptr ())
            continue;

        XCamReturn ret = handler->prepare_parameters (input, output);
        if (ret == XCAM_RETURN_BYPASS)
            return ret;

        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            ret,
            "CLImage360Stitch(%s) prepare parameters failed on handler(%s)",
            XCAM_STR (get_name ()), XCAM_STR (handler->get_name ()));
    }

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<DrmBoBuffer>
CLImage360Stitch::create_bo_buffer (uint32_t width, uint32_t height)
{
    VideoBufferInfo buf_info;
    width = XCAM_ALIGN_UP (width, 16);
    buf_info.init (V4L2_PIX_FMT_NV12, width, height,
                   XCAM_ALIGN_UP (width, 16), XCAM_ALIGN_UP (height, 16));

    SmartPtr<DrmDisplay> display = DrmDisplay::instance ();
    SmartPtr<BufferPool> buf_pool = new DrmBoBufferPool (display);
    XCAM_ASSERT (buf_pool.ptr ());
    buf_pool->set_video_info (buf_info);
    if (!buf_pool->reserve (1)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return NULL;
    }

    return buf_pool->get_buffer (buf_pool).dynamic_cast_ptr<DrmBoBuffer> ();
}

XCamReturn
CLImage360Stitch::reset_buffer_info (SmartPtr<DrmBoBuffer> &input)
{
    VideoBufferInfo reset_info;
    const VideoBufferInfo &buf_info = input->get_video_info ();

    Rect img0_left = get_image_overlap (ImageIdxMain, 0);
    Rect img0_right = get_image_overlap (ImageIdxMain, 1);
    Rect img1_left = get_image_overlap (ImageIdxSecondary, 0);
    Rect img1_right = get_image_overlap (ImageIdxSecondary, 1);

    uint32_t reset_width = img0_right.pos_x - img0_left.pos_x + img1_right.pos_x - img1_left.pos_x;
    reset_width = XCAM_ALIGN_UP (reset_width, XCAM_BLENDER_ALIGNED_WIDTH);
    reset_info.init (buf_info.format, reset_width, buf_info.height,
                     buf_info.aligned_width, buf_info.aligned_height);

    input->set_video_info (reset_info);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImage360Stitch::prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (!_is_stitch_inited)
        init_stitch_info (get_default_stitch_info ());

    ret = prepare_fisheye_parameters (input, output);
    STITCH_CHECK (ret, "prepare fisheye parameters failed");
    _fisheye_buf0 = create_bo_buffer (_fisheye_width, _fisheye_height);
    _fisheye_buf1 = create_bo_buffer (_fisheye_width, _fisheye_height);
    XCAM_ASSERT (_fisheye_buf0.ptr () && _fisheye_buf1.ptr ());

    ret = execute_self_prepare_parameters (_fisheye[0], input, _fisheye_buf0);
    STITCH_CHECK (ret, "execute first fisheye prepare_parameters failed");
    ret = execute_self_prepare_parameters (_fisheye[1], input, _fisheye_buf1);
    STITCH_CHECK (ret, "execute second fisheye prepare_parameters failed");

    _fisheye_buf0->attach_buffer (_fisheye_buf1);
    update_image_overlap ();

    if (_scale_mode == CLBlenderScaleLocal) {
        ret = prepare_local_scale_blender_parameters (_fisheye_buf0, _fisheye_buf1, output);
        STITCH_CHECK (ret, "prepare local scale blender parameters failed");

        ret = execute_self_prepare_parameters (_left_blender, _fisheye_buf0, output);
        STITCH_CHECK (ret, "left blender: execute prepare_parameters failed");
        ret = execute_self_prepare_parameters (_right_blender, _fisheye_buf0, output);
        STITCH_CHECK (ret, "right blender: execute prepare_parameters failed");
    } else {
        const VideoBufferInfo &buf_info = output->get_video_info ();
        SmartPtr<DrmBoBuffer> scale_input = create_bo_buffer (buf_info.width + XCAM_BLENDER_GLOBAL_SCALE_EXT_WIDTH,
                                                              buf_info.height);
        XCAM_ASSERT (scale_input.ptr ());

        ret = prepare_global_scale_blender_parameters (_fisheye_buf0, _fisheye_buf1, scale_input);
        STITCH_CHECK (ret, "prepare global scale blender parameters failed");

        ret = execute_self_prepare_parameters (_left_blender, _fisheye_buf0, scale_input);
        STITCH_CHECK (ret, "left blender: execute prepare_parameters failed");
        ret = execute_self_prepare_parameters (_right_blender, _fisheye_buf0, scale_input);
        STITCH_CHECK (ret, "right blender: execute prepare_parameters failed");

        input = scale_input;
        reset_buffer_info (input);
    }

    return XCAM_RETURN_NO_ERROR;
}

#if HAVE_OPENCV
static void
convert_to_cv_rect (
    ImageMergeInfo merge_info0, ImageMergeInfo merge_info1,
    cv::Rect &crop_area1, cv::Rect &crop_area2, cv::Rect &crop_area3, cv::Rect &crop_area4)
{
    crop_area1.x = merge_info0.merge_left.pos_x;
    crop_area1.y = merge_info0.merge_left.pos_y + merge_info0.merge_left.height / 3;
    crop_area1.width = merge_info0.merge_left.width;
    crop_area1.height = merge_info0.merge_left.height / 3;
    crop_area2.x = merge_info0.merge_right.pos_x;
    crop_area2.y = merge_info0.merge_right.pos_y + merge_info0.merge_right.height / 3;
    crop_area2.width = merge_info0.merge_right.width;
    crop_area2.height = merge_info0.merge_right.height / 3;

    crop_area3.x = merge_info1.merge_left.pos_x;
    crop_area3.y = merge_info1.merge_left.pos_y + merge_info1.merge_left.height / 3;
    crop_area3.width = merge_info1.merge_left.width;
    crop_area3.height = merge_info1.merge_left.height / 3;
    crop_area4.x = merge_info1.merge_right.pos_x;
    crop_area4.y = merge_info1.merge_right.pos_y + merge_info1.merge_right.height / 3;
    crop_area4.width = merge_info1.merge_right.width;
    crop_area4.height = merge_info1.merge_right.height / 3;
}

static void
convert_to_xcam_rect (
    cv::Rect crop_area1, cv::Rect crop_area2, cv::Rect crop_area3, cv::Rect crop_area4,
    ImageMergeInfo &merge_info0, ImageMergeInfo &merge_info1)
{
    merge_info0.merge_left.pos_x = crop_area1.x;
    merge_info0.merge_left.width = crop_area1.width;
    merge_info0.merge_right.pos_x = crop_area2.x;
    merge_info0.merge_right.width = crop_area2.width;

    merge_info1.merge_left.pos_x = crop_area3.x;
    merge_info1.merge_left.width = crop_area3.width;
    merge_info1.merge_right.pos_x = crop_area4.x;
    merge_info1.merge_right.width = crop_area4.width;
}

static void
calc_feature_match (
    SmartPtr<CLContext> context, int fisheye_width,
    SmartPtr<DrmBoBuffer> fisheye_buf0, SmartPtr<DrmBoBuffer> fisheye_buf1,
    ImageMergeInfo &merge_info0, ImageMergeInfo &merge_info1)
{
    cv::Rect crop_area1, crop_area2, crop_area3, crop_area4;

    convert_to_cv_rect (merge_info0, merge_info1, crop_area1, crop_area2, crop_area3, crop_area4);
    optical_flow_feature_match (context, fisheye_width, fisheye_buf0, fisheye_buf1,
                                crop_area1, crop_area2, crop_area3, crop_area4);
    convert_to_xcam_rect (crop_area1, crop_area2, crop_area3, crop_area4, merge_info0, merge_info1);
}
#endif

XCamReturn
CLImage360Stitch::sub_handler_execute_done (SmartPtr<CLImageHandler> &handler)
{
#if HAVE_OPENCV
    XCAM_ASSERT (handler.ptr ());

    if (handler.ptr () == _fisheye[ImageIdxCount - 1].ptr ()) {
        const VideoBufferInfo &buf_info = _fisheye_buf0->get_video_info ();

        calc_feature_match (_context, buf_info.width, _fisheye_buf0, _fisheye_buf1,
                            _img_merge_info[0], _img_merge_info[1]);
    }
#else
    XCAM_UNUSED (handler);
#endif

    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<CLImageKernel>
create_blender_global_scale_kernel (SmartPtr<CLContext> &context, bool is_uv)
{
    char transform_option[1024];
    snprintf (transform_option, sizeof(transform_option), "-DPYRAMID_UV=%d", is_uv ? 1 : 0);

    const XCamKernelInfo &kernel_info = {
        "kernel_pyramid_scale",
#include "kernel_gauss_lap_pyramid.clx"
        , 0
    };

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLBlenderGlobalScaleKernel (context, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernel_info, transform_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load blender global scaling kernel(%s) failed", is_uv ? "UV" : "Y");

    return kernel;
}

SmartPtr<CLImageHandler>
create_image_360_stitch (SmartPtr<CLContext> &context, bool need_seam, CLBlenderScaleMode scale_mode)
{
    const int layer = 2;
    const bool need_uv = true;
    SmartPtr<CLFisheyeHandler> fisheye;
    SmartPtr<CLBlender>  left_blender, right_blender;
    SmartPtr<CLImage360Stitch> stitch = new CLImage360Stitch (context, scale_mode);
    XCAM_ASSERT (stitch.ptr ());

    for (int index = 0; index < ImageIdxCount; ++index) {
        fisheye = create_fisheye_handler (context).dynamic_cast_ptr<CLFisheyeHandler> ();
        XCAM_FAIL_RETURN (ERROR, fisheye.ptr (), NULL, "image_360_stitch create fisheye handler failed");
        fisheye->disable_buf_pool (true);
        stitch->set_fisheye_handler (fisheye, index);
    }

    left_blender = create_pyramid_blender (context, layer, need_uv, need_seam, scale_mode).dynamic_cast_ptr<CLBlender> ();
    XCAM_FAIL_RETURN (ERROR, left_blender.ptr (), NULL, "image_360_stitch create left blender failed");
    left_blender->disable_buf_pool (true);
    left_blender->swap_input_idx (true);
    stitch->set_left_blender (left_blender);

    right_blender = create_pyramid_blender (context, layer, need_uv, need_seam, scale_mode).dynamic_cast_ptr<CLBlender> ();
    XCAM_FAIL_RETURN (ERROR, right_blender.ptr (), NULL, "image_360_stitch create right blender failed");
    right_blender->disable_buf_pool (true);
    stitch->set_right_blender (right_blender);

    if (scale_mode == CLBlenderScaleGlobal) {
        int max_plane = need_uv ? 2 : 1;
        bool uv_status[2] = {false, true};
        for (int plane = 0; plane < max_plane; ++plane) {
            SmartPtr<CLImageKernel> kernel = create_blender_global_scale_kernel (context, uv_status[plane]);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create blender global scaling kernel failed");
            stitch->add_kernel (kernel);
        }
    }

    return stitch;
}

}

