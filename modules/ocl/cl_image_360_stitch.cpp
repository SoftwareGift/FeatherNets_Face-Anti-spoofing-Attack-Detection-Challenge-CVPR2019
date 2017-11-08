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

#include "cl_utils.h"
#include "cl_image_360_stitch.h"

#define XCAM_BLENDER_GLOBAL_SCALE_EXT_WIDTH 64

#define STITCH_CHECK(ret, msg, ...) \
    if ((ret) != XCAM_RETURN_NO_ERROR) {        \
        XCAM_LOG_WARNING (msg, ## __VA_ARGS__); \
        return ret;                             \
    }

namespace XCam {

CLBlenderGlobalScaleKernel::CLBlenderGlobalScaleKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLImage360Stitch> &stitch, bool is_uv)
    : CLBlenderScaleKernel (context, is_uv)
    , _stitch (stitch)
{
}

SmartPtr<CLImage>
CLBlenderGlobalScaleKernel::get_input_image () {
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<VideoBuffer> input = _stitch->get_global_scale_input ();

    CLImageDesc cl_desc;
    SmartPtr<CLImage> cl_image;
    const VideoBufferInfo &buf_info = input->get_video_info ();

    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    if (_is_uv) {
        cl_desc.format.image_channel_order = CL_RG;
        cl_desc.width = buf_info.width / 2;
        cl_desc.height = buf_info.height / 2;
        cl_desc.row_pitch = buf_info.strides[1];
        cl_image = convert_to_climage (context, input, cl_desc, buf_info.offsets[1]);
    } else {
        cl_desc.format.image_channel_order = CL_R;
        cl_desc.width = buf_info.width;
        cl_desc.height = buf_info.height;
        cl_desc.row_pitch = buf_info.strides[0];
        cl_image = convert_to_climage (context, input, cl_desc, buf_info.offsets[0]);
    }

    return cl_image;
}

SmartPtr<CLImage>
CLBlenderGlobalScaleKernel::get_output_image () {
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<VideoBuffer> output = _stitch->get_global_scale_output ();

    CLImageDesc cl_desc;
    SmartPtr<CLImage> cl_image;
    const VideoBufferInfo &buf_info = output->get_video_info ();

    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;
    if (_is_uv) {
        cl_desc.width = buf_info.width / 8;
        cl_desc.height = buf_info.height / 2;
        cl_desc.row_pitch = buf_info.strides[1];
        cl_image = convert_to_climage (context, output, cl_desc, buf_info.offsets[1]);
    } else {
        cl_desc.width = buf_info.width / 8;
        cl_desc.height = buf_info.height;
        cl_desc.row_pitch = buf_info.strides[0];
        cl_image = convert_to_climage (context, output, cl_desc, buf_info.offsets[0]);
    }

    return cl_image;
}

bool
CLBlenderGlobalScaleKernel::get_output_info (
    uint32_t &out_width, uint32_t &out_height, int &out_offset_x)
{
    SmartPtr<VideoBuffer> output = _stitch->get_global_scale_output ();
    const VideoBufferInfo &output_info = output->get_video_info ();

    out_width = output_info.width / 8;
    out_height = _is_uv ? output_info.height / 2 : output_info.height;
    out_offset_x = 0;

    return true;
}

#if HAVE_OPENCV
static CVFMConfig
get_fm_default_config (StitchResMode res_mode)
{
    CVFMConfig config;

    switch (res_mode) {
    case StitchRes1080P: {
        config.sitch_min_width = 56;
        config.min_corners = 8;
        config.offset_factor = 0.8f;
        config.delta_mean_offset = 5.0f;
        config.max_adjusted_offset = 12.0f;

        break;
    }
    case StitchRes1080P4: {
        config.sitch_min_width = 96;
        config.min_corners = 8;
        config.offset_factor = 0.8f;
        config.delta_mean_offset = 5.0f;
        config.max_adjusted_offset = 12.0f;

        break;
    }
    case StitchRes4K: {
        config.sitch_min_width = 160;
        config.min_corners = 8;
        config.offset_factor = 0.8f;
        config.delta_mean_offset = 5.0f;
        config.max_adjusted_offset = 12.0f;

        break;
    }
    default:
        XCAM_LOG_DEBUG ("unknown reslution mode (%d)", res_mode);
        break;
    }

    return config;
}
#endif

static StitchInfo
get_default_stitch_info (StitchResMode res_mode)
{
    StitchInfo stitch_info;

    switch (res_mode) {
    case StitchRes1080P: {
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
        break;
    }
    case StitchRes1080P4: {
        stitch_info.merge_width[0] = 288;
        stitch_info.merge_width[1] = 288;
        stitch_info.merge_width[2] = 288;
        stitch_info.merge_width[3] = 288;

        stitch_info.crop[0].left = 0;
        stitch_info.crop[0].right = 0;
        stitch_info.crop[0].top = 0;
        stitch_info.crop[0].bottom = 0;
        stitch_info.crop[1].left = 0;
        stitch_info.crop[1].right = 0;
        stitch_info.crop[1].top = 0;
        stitch_info.crop[1].bottom = 0;
        stitch_info.crop[2].left = 0;
        stitch_info.crop[2].right = 0;
        stitch_info.crop[2].top = 0;
        stitch_info.crop[2].bottom = 0;
        stitch_info.crop[3].left = 0;
        stitch_info.crop[3].right = 0;
        stitch_info.crop[3].top = 0;
        stitch_info.crop[3].bottom = 0;

        stitch_info.fisheye_info[0].center_x = 640.0f;
        stitch_info.fisheye_info[0].center_y = 400.0f;
        stitch_info.fisheye_info[0].wide_angle = 120.0f;
        stitch_info.fisheye_info[0].radius = 640.0f;
        stitch_info.fisheye_info[0].rotate_angle = 0.0f;
        stitch_info.fisheye_info[1].center_x = 640.0f;
        stitch_info.fisheye_info[1].center_y = 400.0f;
        stitch_info.fisheye_info[1].wide_angle = 120.0f;
        stitch_info.fisheye_info[1].radius = 640.0f;
        stitch_info.fisheye_info[1].rotate_angle = 0.0f;
        stitch_info.fisheye_info[2].center_x = 640.0f;
        stitch_info.fisheye_info[2].center_y = 400.0f;
        stitch_info.fisheye_info[2].wide_angle = 120.0f;
        stitch_info.fisheye_info[2].radius = 640.0f;
        stitch_info.fisheye_info[2].rotate_angle = 0.0f;
        stitch_info.fisheye_info[3].center_x = 640.0f;
        stitch_info.fisheye_info[3].center_y = 400.0f;
        stitch_info.fisheye_info[3].wide_angle = 120.0f;
        stitch_info.fisheye_info[3].radius = 640.0f;
        stitch_info.fisheye_info[3].rotate_angle = 0.0f;
        break;
    }
    case StitchRes4K: {
        stitch_info.merge_width[0] = 160;
        stitch_info.merge_width[1] = 160;

        stitch_info.crop[0].left = 64;
        stitch_info.crop[0].right = 64;
        stitch_info.crop[0].top = 0;
        stitch_info.crop[0].bottom = 0;
        stitch_info.crop[1].left = 64;
        stitch_info.crop[1].right = 64;
        stitch_info.crop[1].top = 0;
        stitch_info.crop[1].bottom = 0;

        stitch_info.fisheye_info[0].center_x = 1024.0f;
        stitch_info.fisheye_info[0].center_y = 1024.0f;
        stitch_info.fisheye_info[0].wide_angle = 195.0f;
        stitch_info.fisheye_info[0].radius = 1040.0f;
        stitch_info.fisheye_info[0].rotate_angle = 0.0f;

        stitch_info.fisheye_info[1].center_x = 3072.0f;
        stitch_info.fisheye_info[1].center_y = 1016.0f;
        stitch_info.fisheye_info[1].wide_angle = 192.0f;
        stitch_info.fisheye_info[1].radius = 1040.0f;
        stitch_info.fisheye_info[1].rotate_angle = 0.4f;
        break;
    }
    default:
        XCAM_LOG_DEBUG ("unknown reslution mode (%d)", res_mode);
        break;
    }

    return stitch_info;
}

CLImage360Stitch::CLImage360Stitch (
    const SmartPtr<CLContext> &context, CLBlenderScaleMode scale_mode, SurroundMode surround_mode,
    StitchResMode res_mode, int fisheye_num, bool all_in_one_img)
    : CLMultiImageHandler (context, "CLImage360Stitch")
    , _context (context)
    , _output_width (0)
    , _output_height (0)
    , _scale_mode (scale_mode)
    , _surround_mode (surround_mode)
    , _res_mode (res_mode)
    , _is_stitch_inited (false)
    , _fisheye_num (fisheye_num)
    , _all_in_one_img (all_in_one_img)
{
#if HAVE_OPENCV
    for (int i = 0; i < fisheye_num; i++) {
        _feature_match[i] = new CVFeatureMatch ();
        XCAM_ASSERT (_feature_match[i].ptr ());
        _feature_match[i]->set_config (get_fm_default_config (res_mode));
        _feature_match[i]->set_fm_index (i);
    }
#endif
}

bool
CLImage360Stitch::set_stitch_info (StitchInfo stitch_info)
{
    if (_is_stitch_inited) {
        XCAM_LOG_WARNING ("stitching info was initialized and can't be set twice");
        return false;
    }

    for (int index = 0; index < _fisheye_num; ++index) {
        _fisheye[index].handler->set_fisheye_info (stitch_info.fisheye_info[index]);
    }

    _stitch_info = stitch_info;
    _is_stitch_inited = true;

    return true;
}

StitchInfo
CLImage360Stitch::get_stitch_info ()
{
    if (!_is_stitch_inited) {
        XCAM_LOG_WARNING ("stitch-info was not initialized, return default parameters");
        return get_default_stitch_info (_res_mode);
    }

    return _stitch_info;
}

bool
CLImage360Stitch::set_fisheye_handler (SmartPtr<CLFisheyeHandler> fisheye, int index)
{
    XCAM_ASSERT (index < _fisheye_num);

    _fisheye[index].handler = fisheye;
    SmartPtr<CLImageHandler> handler = fisheye;
    return add_image_handler (handler);
}

bool
CLImage360Stitch::set_blender (SmartPtr<CLBlender> blender, int idx)
{
    _blender[idx] = blender;

    SmartPtr<CLImageHandler> handler = blender;
    return add_image_handler (handler);
}

void
CLImage360Stitch::set_fisheye_intrinsic (IntrinsicParameter intrinsic_param, int index)
{
    _fisheye[index].handler->set_intrinsic_param(intrinsic_param);
}

void
CLImage360Stitch::set_fisheye_extrinsic (ExtrinsicParameter extrinsic_param, int index)
{
    _fisheye[index].handler->set_extrinsic_param(extrinsic_param);
}

const BowlDataConfig &
CLImage360Stitch::get_fisheye_bowl_config (int index)
{
    XCAM_ASSERT (index < _fisheye_num);
    return _fisheye[index].handler->get_bowl_config ();
}

bool
CLImage360Stitch::set_image_overlap (const int idx, const Rect &overlap0, const Rect &overlap1)
{
    XCAM_ASSERT (idx < _fisheye_num);
    _overlaps[idx][0] = overlap0;
    _overlaps[idx][1] = overlap1;
    return true;
}

void
CLImage360Stitch::set_feature_match_ocl (bool fm_ocl)
{
#if HAVE_OPENCV
    for (int i = 0; i < _fisheye_num; i++) {
        _feature_match[i]->set_ocl (fm_ocl);
    }
#else
    XCAM_UNUSED (fm_ocl);
    XCAM_LOG_WARNING ("non-OpenCV mode, failed to set ocl for feature match");
#endif
}

#if HAVE_OPENCV
void
CLImage360Stitch::set_feature_match_config (const int idx, CVFMConfig config)
{
    _feature_match[idx]->set_config (config);
}

CVFMConfig
CLImage360Stitch::get_feature_match_config (const int idx)
{
    return _feature_match[idx]->get_config ();
}
#endif

void
CLImage360Stitch::calc_fisheye_initial_info (SmartPtr<VideoBuffer> &output)
{
    const VideoBufferInfo &out_info = output->get_video_info ();

    if(_surround_mode == SphereView) {
        uint32_t fisheye_width_sum = out_info.width;
        for (int i = 0; i < _fisheye_num; i++) {
            fisheye_width_sum += _stitch_info.merge_width[i] + _stitch_info.crop[i].left + _stitch_info.crop[i].right;
        }
        _fisheye[0].width = fisheye_width_sum / _fisheye_num;
        _fisheye[0].width = XCAM_ALIGN_UP (_fisheye[0].width, 16);
        _fisheye[0].height = out_info.height + _stitch_info.crop[0].top + _stitch_info.crop[0].bottom;
        XCAM_LOG_INFO (
            "fisheye correction output size width:%d height:%d",
            _fisheye[0].width, _fisheye[0].height);

        for (int i = 1; i < _fisheye_num; i++) {
            _fisheye[i].width = _fisheye[0].width;
            _fisheye[i].height = _fisheye[0].height;
        }

        float max_dst_longitude, max_dst_latitude;
        for (int i = 0; i < _fisheye_num; ++i) {
            max_dst_latitude = (_stitch_info.fisheye_info[i].wide_angle > 180.0f) ?
                               180.0f : _stitch_info.fisheye_info[i].wide_angle;
            max_dst_longitude = max_dst_latitude * _fisheye[i].width / _fisheye[i].height;

            _fisheye[i].handler->set_dst_range (max_dst_longitude, max_dst_latitude);
            _fisheye[i].handler->set_output_size (_fisheye[i].width, _fisheye[i].height);
        }
    } else {
        _fisheye[0].height = out_info.height + _stitch_info.crop[0].top + _stitch_info.crop[0].bottom;

        float view_angle[XCAM_STITCH_FISHEYE_MAX_NUM];
        float car_length = 4500.0f;
        float max_z = 3000.0f;

        view_angle[0] = 68.0f / 360.0f * 2 * PI;
        _fisheye[0].width = view_angle[0] / (2 * PI) * out_info.width;
        _fisheye[0].width = XCAM_ALIGN_UP (_fisheye[0].width, 32);

        view_angle[1] = 152.0f / 360.0f * 2 * PI;
        _fisheye[1].width = view_angle[1] / (2 * PI) * out_info.width;
        _fisheye[1].width = XCAM_ALIGN_UP (_fisheye[1].width, 32);

        view_angle[2] = 68.0f / 360.0f * 2 * PI;
        _fisheye[2].width = view_angle[2] / (2 * PI) * out_info.width;
        _fisheye[2].width = XCAM_ALIGN_UP (_fisheye[2].width, 32);

        view_angle[3] = 152.0f / 360.0f * 2 * PI;
        _fisheye[3].width = view_angle[3] / (2 * PI) * out_info.width;
        _fisheye[3].width = XCAM_ALIGN_UP (_fisheye[3].width, 32);

        XCAM_LOG_INFO (
            "fisheye correction output size width:%d height:%d",
            _fisheye[0].width, _fisheye[0].height);

        BowlDataConfig bowl_data_config[XCAM_STITCH_FISHEYE_MAX_NUM];

        float ground_length = bowl_data_config[0].a * sqrt(1 - (max_z / 2) * (max_z / 2) / (bowl_data_config[0].c * bowl_data_config[0].c)) - car_length / 2;
        bowl_data_config[0].wall_image_height = (int)(max_z / (ground_length + max_z) * _fisheye[0].height);
        bowl_data_config[0].ground_image_height = _fisheye[0].height - bowl_data_config[0].wall_image_height;

        bowl_data_config[0].angle_start = view_angle[0] / 2;
        bowl_data_config[0].angle_end = -view_angle[0] / 2;

        for (int i = 1; i < _fisheye_num; i++) {
            _fisheye[i].height = _fisheye[0].height;
            bowl_data_config[i].wall_image_height = bowl_data_config[0].wall_image_height;
            bowl_data_config[i].ground_image_height = bowl_data_config[0].ground_image_height;
            bowl_data_config[i].angle_start = 2 * PI / _fisheye_num * (_fisheye_num - i) + view_angle[i] / 2;
            bowl_data_config[i].angle_end = 2 * PI / _fisheye_num * (_fisheye_num - i) - view_angle[i] / 2;
        }

        for(int i = 0; i < _fisheye_num; i++) {
            _fisheye[i].handler->set_bowl_config(bowl_data_config[i]);
            _fisheye[i].handler->set_output_size (_fisheye[i].width, _fisheye[i].height);
        }
    }
}

void
CLImage360Stitch::update_image_overlap ()
{
    static bool is_merge_info_inited = false;
    if (!is_merge_info_inited) {
        int idx_next = 1;
        for (int i = 0; i < _fisheye_num; i++) {
            idx_next = (i == (_fisheye_num - 1)) ? 0 : (i + 1);

            _img_merge_info[i].left.pos_x = _stitch_info.crop[i].left;
            _img_merge_info[i].left.pos_y = _stitch_info.crop[i].top;
            _img_merge_info[i].left.width = _stitch_info.merge_width[i];
            _img_merge_info[i].left.height = _fisheye[i].height - _stitch_info.crop[i].top
                                             - _stitch_info.crop[i].bottom;

            _img_merge_info[i].right.pos_x = _fisheye[i].width - _stitch_info.crop[i].right
                                             - _stitch_info.merge_width[idx_next];
            _img_merge_info[i].right.pos_y = _stitch_info.crop[i].top;
            _img_merge_info[i].right.width = _stitch_info.merge_width[idx_next];
            _img_merge_info[i].right.height = _fisheye[i].height - _stitch_info.crop[i].top
                                              - _stitch_info.crop[i].bottom;
        }

        is_merge_info_inited = true;
    }

    for (int i = 0; i < _fisheye_num; i++) {
        set_image_overlap (i, _img_merge_info[i].left, _img_merge_info[i].right);
    }
}

XCamReturn
CLImage360Stitch::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input, VideoBufferInfo &output)
{
    if (_output_width == 0 || _output_height == 0) {
        XCAM_LOG_ERROR ("incorrect output size: width:%d height:%d", _output_width, _output_height);
        return XCAM_RETURN_ERROR_PARAM;
    }

    // aligned at least XCAM_CL_BLENDER_ALIGNMENT_X
    uint32_t aligned_width = XCAM_MAX (16, XCAM_CL_BLENDER_ALIGNMENT_X);
    output.init (
        input.format, _output_width, _output_height,
        XCAM_ALIGN_UP(_output_width, aligned_width), XCAM_ALIGN_UP(_output_height, 16));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImage360Stitch::ensure_fisheye_parameters (
    SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    static bool is_fisheye_inited = false;

    if (!is_fisheye_inited) {
        calc_fisheye_initial_info (output);
        is_fisheye_inited = true;
    }

    SmartPtr<VideoBuffer> pre_buf;
    SmartPtr<VideoBuffer> cur_buf = input;
    for (int i = 0; i < _fisheye_num; i++) {
        if (!_fisheye[i].pool.ptr ())
            create_buffer_pool (_fisheye[i].pool, _fisheye[i].width, _fisheye[i].height);

        _fisheye[i].buf = _fisheye[i].pool->get_buffer (_fisheye[i].pool);
        XCAM_ASSERT (_fisheye[i].buf.ptr ());

        XCamReturn ret = ensure_handler_parameters (_fisheye[i].handler, cur_buf, _fisheye[i].buf);
        STITCH_CHECK (ret, "execute fisheye prepare_parameters failed");

        if (!_all_in_one_img) {
            pre_buf = cur_buf;
            cur_buf = cur_buf->find_typed_attach<VideoBuffer> ();
            if (!cur_buf.ptr () && (i != (_fisheye_num - 1))) {
                XCAM_LOG_ERROR ("conflicting attached buffers and fisheye number");
                return XCAM_RETURN_ERROR_PARAM;
            }
            pre_buf->detach_buffer (cur_buf);
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImage360Stitch::prepare_global_scale_blender_parameters (
    SmartPtr<VideoBuffer> &input0, SmartPtr<VideoBuffer> &input1, SmartPtr<VideoBuffer> &output,
    int idx, int idx_next, int &cur_start_pos)
{
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();

    XCAM_ASSERT (in0_info.height == in1_info.height);
    XCAM_ASSERT (in0_info.width <= out_info.width && in1_info.width <= out_info.width);

    Rect left_lap = get_image_overlap (idx, 1);
    Rect right_lap = get_image_overlap (idx_next, 0);

    int left_img_mid = XCAM_ALIGN_DOWN (in0_info.width / 2, XCAM_CL_BLENDER_ALIGNMENT_X);
    int right_img_mid = XCAM_ALIGN_DOWN (in1_info.width / 2, XCAM_CL_BLENDER_ALIGNMENT_X);

    int32_t prev_pos;
    prev_pos = left_lap.pos_x;
    left_lap.pos_x = XCAM_ALIGN_AROUND (left_lap.pos_x, XCAM_CL_BLENDER_ALIGNMENT_X);
    left_lap.width = XCAM_ALIGN_UP (left_lap.width, XCAM_CL_BLENDER_ALIGNMENT_X);
    right_lap.pos_x += left_lap.pos_x - prev_pos;
    right_lap.pos_x = XCAM_ALIGN_AROUND (right_lap.pos_x, XCAM_CL_BLENDER_ALIGNMENT_X);
    right_lap.width = left_lap.width;

    Rect area;
    area.pos_y = left_lap.pos_y;
    area.height = left_lap.height;
    area.pos_x = left_img_mid;
    area.width = left_lap.pos_x + left_lap.width - left_img_mid;
    _blender[idx]->set_input_valid_area (area, 0);

    area.pos_y = right_lap.pos_y;
    area.height = right_lap.height;
    area.pos_x = right_lap.pos_x;
    area.width = right_img_mid - right_lap.pos_x;
    _blender[idx]->set_input_valid_area (area, 1);

    Rect out_merge_window;
    out_merge_window.width = left_lap.width;
    out_merge_window.pos_x = cur_start_pos + (left_lap.pos_x - left_img_mid);
    out_merge_window.pos_y = 0;
    out_merge_window.height = out_info.height;
    _blender[idx]->set_merge_window (out_merge_window);

    _blender[idx]->set_input_merge_area (left_lap, 0);
    _blender[idx]->set_input_merge_area (right_lap, 1);

    cur_start_pos += left_lap.pos_x - left_img_mid + right_img_mid - right_lap.pos_x;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImage360Stitch::prepare_local_scale_blender_parameters (
    SmartPtr<VideoBuffer> &input0, SmartPtr<VideoBuffer> &input1, SmartPtr<VideoBuffer> &output, int idx, int idx_next)
{
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();

    XCAM_ASSERT (in0_info.height == in1_info.height);
    XCAM_ASSERT (in0_info.width <= out_info.width && in1_info.width <= out_info.width);

    Rect left_lap = get_image_overlap (idx, 1);
    Rect right_lap = get_image_overlap (idx_next, 0);

    int left_img_mid = XCAM_ALIGN_DOWN (in0_info.width / 2, XCAM_CL_BLENDER_ALIGNMENT_X);
    int right_img_mid = XCAM_ALIGN_DOWN (in1_info.width / 2, XCAM_CL_BLENDER_ALIGNMENT_X);
    int cur_start_pos = XCAM_ALIGN_DOWN (out_info.width / _fisheye_num * idx, XCAM_CL_BLENDER_ALIGNMENT_X);
    int merge_std_width = XCAM_ALIGN_DOWN (out_info.width / _fisheye_num, XCAM_CL_BLENDER_ALIGNMENT_X);

    int32_t prev_pos;
    prev_pos = left_lap.pos_x;
    left_lap.pos_x = XCAM_ALIGN_AROUND (left_lap.pos_x, XCAM_CL_BLENDER_ALIGNMENT_X);
    left_lap.width = XCAM_ALIGN_UP (left_lap.width, XCAM_CL_BLENDER_ALIGNMENT_X);
    right_lap.pos_x += left_lap.pos_x - prev_pos;
    right_lap.pos_x = XCAM_ALIGN_AROUND (right_lap.pos_x, XCAM_CL_BLENDER_ALIGNMENT_X);
    right_lap.width = left_lap.width;

    Rect area;
    area.pos_y = left_lap.pos_y;
    area.height = left_lap.height;
    area.pos_x = left_img_mid;
    area.width = left_lap.pos_x + left_lap.width - left_img_mid;
    _blender[idx]->set_input_valid_area (area, 0);

    area.pos_y = right_lap.pos_y;
    area.height = right_lap.height;
    area.pos_x = right_lap.pos_x;
    area.width = right_img_mid - right_lap.pos_x;
    _blender[idx]->set_input_valid_area (area, 1);

    Rect out_merge_window;
    int delta_width = merge_std_width - (right_img_mid - right_lap.pos_x) - (left_lap.pos_x - left_img_mid);
    out_merge_window.width = left_lap.width + delta_width;
    out_merge_window.pos_x = cur_start_pos + (left_lap.pos_x - left_img_mid);
    out_merge_window.pos_y = 0;
    out_merge_window.height = out_info.height;
    _blender[idx]->set_merge_window (out_merge_window);

    _blender[idx]->set_input_merge_area (left_lap, 0);
    _blender[idx]->set_input_merge_area (right_lap, 1);

    return XCAM_RETURN_NO_ERROR;
}

bool
CLImage360Stitch::create_buffer_pool (SmartPtr<BufferPool> &buf_pool, uint32_t width, uint32_t height)
{
    VideoBufferInfo buf_info;
    width = XCAM_ALIGN_UP (width, 16);
    buf_info.init (V4L2_PIX_FMT_NV12, width, height,
                   XCAM_ALIGN_UP (width, 16), XCAM_ALIGN_UP (height, 16));

    buf_pool = new CLVideoBufferPool ();
    XCAM_ASSERT (buf_pool.ptr ());
    buf_pool->set_video_info (buf_info);
    if (!buf_pool->reserve (6)) {
        XCAM_LOG_ERROR ("CLImage360Stitch init buffer pool failed");
        return false;
    }

    return true;
}

XCamReturn
CLImage360Stitch::reset_buffer_info (SmartPtr<VideoBuffer> &input)
{
    VideoBufferInfo reset_info;
    const VideoBufferInfo &buf_info = input->get_video_info ();

    uint32_t reset_width = 0;
    for (int i = 0; i < _fisheye_num; i++) {
        Rect img_left = get_image_overlap (i, 0);
        Rect img_right = get_image_overlap (i, 1);

        reset_width += img_right.pos_x - img_left.pos_x;
    }

    reset_width = XCAM_ALIGN_UP (reset_width, XCAM_CL_BLENDER_ALIGNMENT_X);
    reset_info.init (buf_info.format, reset_width, buf_info.height,
                     buf_info.aligned_width, buf_info.aligned_height);

    input->set_video_info (reset_info);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImage360Stitch::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (!_is_stitch_inited)
        set_stitch_info (get_default_stitch_info (_res_mode));

    ret = ensure_fisheye_parameters (input, output);
    STITCH_CHECK (ret, "ensure fisheye parameters failed");

    update_image_overlap ();
    if (_scale_mode == CLBlenderScaleLocal) {
        int idx_next = 1;
        for (int i = 0; i < _fisheye_num; i++) {
            idx_next = (i == (_fisheye_num - 1)) ? 0 : (i + 1);

            ret = prepare_local_scale_blender_parameters (
                      _fisheye[i].buf, _fisheye[idx_next].buf, output, i, idx_next);
            STITCH_CHECK (ret, "prepare local scale blender parameters failed");

            _fisheye[i].buf->attach_buffer (_fisheye[idx_next].buf);
            ret = ensure_handler_parameters (_blender[i], _fisheye[i].buf, output);
            STITCH_CHECK (ret, "blender: execute ensure_parameters failed");
            _fisheye[i].buf->detach_buffer (_fisheye[idx_next].buf);
        }
    } else { //global scale
        const VideoBufferInfo &buf_info = output->get_video_info ();
        if (!_scale_buf_pool.ptr ())
            create_buffer_pool (_scale_buf_pool, buf_info.width + XCAM_BLENDER_GLOBAL_SCALE_EXT_WIDTH, buf_info.height);
        SmartPtr<VideoBuffer> scale_input = _scale_buf_pool->get_buffer (_scale_buf_pool);
        XCAM_ASSERT (scale_input.ptr ());

        int idx_next = 1;
        int cur_start_pos = 0;
        for (int i = 0; i < _fisheye_num; i++) {
            idx_next = (i == (_fisheye_num - 1)) ? 0 : (i + 1);

            ret = prepare_global_scale_blender_parameters (
                      _fisheye[i].buf, _fisheye[idx_next].buf, scale_input, i, idx_next, cur_start_pos);
            STITCH_CHECK (ret, "prepare global scale blender parameters failed");

            _fisheye[i].buf->attach_buffer (_fisheye[idx_next].buf);
            ret = ensure_handler_parameters (_blender[i], _fisheye[i].buf, scale_input);
            STITCH_CHECK (ret, "blender: execute ensure_parameters failed");
            _fisheye[i].buf->detach_buffer (_fisheye[idx_next].buf);
        }

        reset_buffer_info (scale_input);
        _scale_global_input = scale_input;
        _scale_global_output = output;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImage360Stitch::execute_done (SmartPtr<VideoBuffer> &output)
{
#if HAVE_OPENCV
    for (int i = 0; i < _fisheye_num; i++) {
        if (!_feature_match[i]->is_ocl_path ()) {
            get_context ()->finish ();
            break;
        }
    }
#endif

    _scale_global_input.release ();
    _scale_global_output.release ();

    return CLMultiImageHandler::execute_done (output);
}

static void
convert_to_stitch_rect (Rect xcam_rect, Rect &stitch_rect)
{
    stitch_rect.pos_x = xcam_rect.pos_x;
    stitch_rect.pos_y = xcam_rect.pos_y + xcam_rect.height / 3;
    stitch_rect.width = xcam_rect.width;
    stitch_rect.height = xcam_rect.height / 3;
}

static void
convert_to_xcam_rect (Rect stitch_rect, Rect &xcam_rect)
{
    xcam_rect.pos_x = stitch_rect.pos_x;
    xcam_rect.width = stitch_rect.width;
}


XCamReturn
CLImage360Stitch::sub_handler_execute_done (SmartPtr<CLImageHandler> &handler)
{
#if HAVE_OPENCV
    XCAM_ASSERT (handler.ptr ());

    if (handler.ptr () == _fisheye[_fisheye_num - 1].handler.ptr ()) {
        int idx_next = 1;
        Rect crop_left, crop_right;

        for (int i = 0; i < _fisheye_num; i++) {
            idx_next = (i == (_fisheye_num - 1)) ? 0 : (i + 1);

            convert_to_stitch_rect (_img_merge_info[i].right, crop_left);
            convert_to_stitch_rect (_img_merge_info[idx_next].left, crop_right);

            _feature_match[i]->optical_flow_feature_match (
                _fisheye[i].buf, _fisheye[idx_next].buf, crop_left, crop_right, _fisheye[i].width);

            convert_to_xcam_rect (crop_left, _img_merge_info[i].right);
            convert_to_xcam_rect (crop_right, _img_merge_info[idx_next].left);
        }
    }
#else
    XCAM_UNUSED (handler);
#endif

    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<CLImageKernel>
create_blender_global_scale_kernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLImage360Stitch> &stitch,
    bool is_uv)
{
    char transform_option[1024];
    snprintf (transform_option, sizeof(transform_option), "-DPYRAMID_UV=%d", is_uv ? 1 : 0);

    static const XCamKernelInfo &kernel_info = {
        "kernel_pyramid_scale",
#include "kernel_gauss_lap_pyramid.clx"
        , 0
    };

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLBlenderGlobalScaleKernel (context, stitch, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernel_info, transform_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load blender global scaling kernel(%s) failed", is_uv ? "UV" : "Y");

    return kernel;
}

SmartPtr<CLImageHandler>
create_image_360_stitch (
    const SmartPtr<CLContext> &context, bool need_seam,
    CLBlenderScaleMode scale_mode, bool fisheye_map, bool need_lsc, SurroundMode surround_mode,
    StitchResMode res_mode, int fisheye_num, bool all_in_one_img)
{
    const int layer = 2;
    const bool need_uv = true;
    SmartPtr<CLFisheyeHandler> fisheye;
    SmartPtr<CLBlender> blender;
    SmartPtr<CLImage360Stitch> stitch = new CLImage360Stitch (
        context, scale_mode, surround_mode, res_mode, fisheye_num, all_in_one_img);
    XCAM_ASSERT (stitch.ptr ());

    for (int index = 0; index < fisheye_num; ++index) {
        fisheye = create_fisheye_handler (context, surround_mode, fisheye_map, need_lsc).dynamic_cast_ptr<CLFisheyeHandler> ();
        XCAM_FAIL_RETURN (ERROR, fisheye.ptr (), NULL, "image_360_stitch create fisheye handler failed");
        fisheye->disable_buf_pool (true);
        stitch->set_fisheye_handler (fisheye, index);
    }

    for (int index = 0; index < fisheye_num; ++index) {
        blender = create_pyramid_blender (context, layer, need_uv, need_seam, scale_mode).dynamic_cast_ptr<CLBlender> ();
        XCAM_FAIL_RETURN (ERROR, blender.ptr (), NULL, "image_360_stitch create blender failed");
        blender->disable_buf_pool (true);
        stitch->set_blender (blender, index);
    }

    if (scale_mode == CLBlenderScaleGlobal) {
        int max_plane = need_uv ? 2 : 1;
        bool uv_status[2] = {false, true};
        for (int plane = 0; plane < max_plane; ++plane) {
            SmartPtr<CLImageKernel> kernel = create_blender_global_scale_kernel (context, stitch, uv_status[plane]);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create blender global scaling kernel failed");
            stitch->add_kernel (kernel);
        }
    }

    return stitch;
}

}

