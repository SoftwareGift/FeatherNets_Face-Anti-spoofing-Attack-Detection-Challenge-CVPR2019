/*
 * stitcher.cpp - stitcher base
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "stitcher.h"
#include "xcam_utils.h"

// angle to position, output range [-180, 180]
#define OUT_WINDOWS_START 0.0f

#define constraint_margin (2 * _alignment_x)

#define XCAM_GL_RESTART_FIXED_INDEX 0xFFFF

namespace XCam {

static inline bool
merge_neighbor_area (
    const Stitcher::CopyArea &current,
    const Stitcher::CopyArea &next,
    Stitcher::CopyArea &merged)
{
    if (current.in_idx == next.in_idx &&
            current.in_area.pos_x + current.in_area.width == next.in_area.pos_x &&
            current.out_area.pos_x + current.out_area.width == next.out_area.pos_x)
    {
        merged = current;
        merged.in_area.pos_x = current.in_area.pos_x;
        merged.in_area.width = current.in_area.width + next.in_area.width;
        merged.out_area.pos_x = current.out_area.pos_x;
        merged.out_area.width = current.out_area.width + next.out_area.width;
        return true;
    }
    return false;
}

static inline bool
split_area_by_out (
    const Stitcher::CopyArea &area, const uint32_t round_width,
    Stitcher::CopyArea &split_a,  Stitcher::CopyArea &split_b)
{
    XCAM_ASSERT (area.out_area.pos_x >= 0 && area.out_area.pos_x < (int32_t)round_width);
    XCAM_ASSERT (area.out_area.width > 0 && area.out_area.width < (int32_t)round_width);
    if (area.out_area.pos_x + area.out_area.width > (int32_t)round_width) {
        split_a = area;
        split_a.out_area.width = round_width - area.out_area.pos_x;
        split_a.in_area.width = split_a.out_area.width;

        split_b = area;
        split_b.in_area.pos_x = area.in_area.pos_x + split_a.in_area.width;
        split_b.in_area.width = area.in_area.width - split_a.in_area.width;
        split_b.out_area.pos_x = 0;
        split_b.out_area.width = split_b.in_area.width;
        XCAM_ASSERT (split_b.out_area.width == area.out_area.pos_x + area.out_area.width - (int32_t)round_width);
        return true;

    }
    XCAM_ASSERT (area.out_area.width == area.in_area.width);
    return false;
}

Stitcher::Stitcher (uint32_t align_x, uint32_t align_y)
    : _is_crop_set (false)
    , _alignment_x (align_x)
    , _alignment_y (align_y)
    , _output_width (0)
    , _output_height (0)
    , _out_start_angle (OUT_WINDOWS_START)
    , _camera_num (0)
    , _is_round_view_set (false)
    , _is_overlap_set (false)
    , _is_center_marked (false)
{
    XCAM_ASSERT (align_x >= 1);
    XCAM_ASSERT (align_y >= 1);
}

Stitcher::~Stitcher ()
{
}

bool
Stitcher::set_bowl_config (const BowlDataConfig &config)
{
    _bowl_config = config;
    return true;
}

bool
Stitcher::set_camera_num (uint32_t num)
{
    XCAM_FAIL_RETURN (
        ERROR, num <= XCAM_STITCH_MAX_CAMERAS, false,
        "stitcher: set camera count failed, num(%d) is larger than max value(%d)",
        num, XCAM_STITCH_MAX_CAMERAS);
    _camera_num = num;
    return true;
}

bool
Stitcher::set_camera_info (uint32_t index, const CameraInfo &info)
{
    XCAM_FAIL_RETURN (
        ERROR, index < _camera_num, false,
        "stitcher: set camera info failed, index(%d) exceed max camera num(%d)",
        index, _camera_num);
    _camera_info[index] = info;
    return true;
}

bool
Stitcher::set_crop_info (uint32_t index, const ImageCropInfo &info)
{
    XCAM_FAIL_RETURN (
        ERROR, index < _camera_num, false,
        "stitcher: set camera info failed, index(%d) exceed max camera num(%d)",
        index, _camera_num);
    _crop_info[index] = info;
    _is_crop_set = true;
    return true;
}

bool
Stitcher::get_crop_info (uint32_t index, ImageCropInfo &info) const
{
    XCAM_FAIL_RETURN (
        ERROR, index < _camera_num, false,
        "stitcher: get crop info failed, index(%d) exceed camera num(%d)",
        index, _camera_num);
    info = _crop_info[index];
    return true;
}

#if 0
bool
Stitcher::set_overlap_info (uint32_t index, const ImageOverlapInfo &info)
{
    XCAM_FAIL_RETURN (
        ERROR, index < _camera_num, false,
        "stitcher: set overlap info failed, index(%d) exceed max camera num(%d)",
        index, _camera_num);
    _overlap_info[index] = info;
    _is_overlap_set = true;
    return true;
}

bool
Stitcher::get_overlap_info (uint32_t index, ImageOverlapInfo &info) const
{
    XCAM_FAIL_RETURN (
        ERROR, index < _camera_num, false,
        "stitcher: get overlap info failed, index(%d) exceed camera num(%d)",
        index, _camera_num);
    info = _overlap_info[index];
    return true;
}
#endif

bool
Stitcher::get_camera_info (uint32_t index, CameraInfo &info) const
{
    XCAM_FAIL_RETURN (
        ERROR, index < XCAM_STITCH_MAX_CAMERAS, false,
        "stitcher: get camera info failed, index(%d) exceed max camera value(%d)",
        index, XCAM_STITCH_MAX_CAMERAS);
    info = _camera_info[index];
    return true;
}

XCamReturn
Stitcher::estimate_round_slices ()
{
    if (_is_round_view_set)
        return XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        ERROR, _camera_num && _camera_num < XCAM_STITCH_MAX_CAMERAS, XCAM_RETURN_ERROR_PARAM,
        "stitcher: camera num was not set, or camera num(%d) exceed max camera value(%d)",
        _camera_num, XCAM_STITCH_MAX_CAMERAS);

    for (uint32_t i = 0; i < _camera_num; ++i) {
        CameraInfo &cam_info = _camera_info[i];
        RoundViewSlice &view_slice = _round_view_slices[i];

        view_slice.width = cam_info.angle_range / 360.0f * (float)_output_width;
        view_slice.width = XCAM_ALIGN_UP (view_slice.width, _alignment_x);
        view_slice.height = _output_height;
        view_slice.hori_angle_range = view_slice.width * 360.0f / (float)_output_width;

        uint32_t aligned_start = format_angle (cam_info.round_angle_start) / 360.0f * (float)_output_width;
        aligned_start = XCAM_ALIGN_AROUND (aligned_start, _alignment_x);
        if (_output_width <= constraint_margin + aligned_start || aligned_start <= constraint_margin)
            aligned_start = 0;
        view_slice.hori_angle_start = format_angle((float)aligned_start / (float)_output_width * 360.0f);
        if (XCAM_DOUBLE_EQUAL_AROUND (view_slice.hori_angle_start, 0.0001f))
            view_slice.hori_angle_start = 0.0f;

        cam_info.round_angle_start = view_slice.hori_angle_start;
        cam_info.angle_range = view_slice.hori_angle_range;
    }

    _is_round_view_set = true;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stitcher::estimate_coarse_crops ()
{
    if (_is_crop_set)
        return XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        ERROR, _camera_num > 0 && _is_round_view_set, XCAM_RETURN_ERROR_ORDER,
        "stitcher mark_centers failed, need set camera info and round_slices first");

    for (uint32_t i = 0; i < _camera_num; ++i) {
        _crop_info[i].left = 0;
        _crop_info[i].right = 0;
        _crop_info[i].top = 0;
        _crop_info[i].bottom = 0;
    }
    _is_crop_set = true;
    return XCAM_RETURN_NO_ERROR;
}

// after crop done
XCamReturn
Stitcher::mark_centers ()
{
    if (_is_center_marked)
        return XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        ERROR, _camera_num > 0 && _is_round_view_set, XCAM_RETURN_ERROR_ORDER,
        "stitcher mark_centers failed, need set camera info and round_view slices first");

    for (uint32_t i = 0; i < _camera_num; ++i) {
        const RoundViewSlice &slice = _round_view_slices[i];

        //calcuate final output postion
        float center_angle = i * 360.0f / _camera_num;
        uint32_t out_pos = format_angle (center_angle - _out_start_angle) / 360.0f * _output_width;
        XCAM_ASSERT (out_pos < _output_width);
        if (_output_width <= constraint_margin + out_pos || out_pos <= constraint_margin)
            out_pos = 0;

        // get slice center angle
        center_angle = XCAM_ALIGN_AROUND (out_pos, _alignment_x) / (float)_output_width * 360.0f - _out_start_angle;
        center_angle = format_angle (center_angle);

        float center_in_slice = center_angle - slice.hori_angle_start;
        center_in_slice = format_angle (center_in_slice);
        XCAM_FAIL_RETURN (
            ERROR, center_in_slice < slice.hori_angle_range,
            XCAM_RETURN_ERROR_PARAM,
            "stitcher mark center failed, slice:%d  calculated center-angle:%.2f is out of slice angle(start:%.2f, range:%.2f)",
            center_angle, slice.hori_angle_start, slice.hori_angle_range);

        uint32_t slice_pos = (uint32_t)(center_in_slice / slice.hori_angle_range * slice.width);
        slice_pos = XCAM_ALIGN_AROUND (slice_pos, _alignment_x);
        XCAM_ASSERT (slice_pos > _crop_info[i].left && slice_pos < slice.width - _crop_info[i].right);

        _center_marks[i].slice_center_x = slice_pos;
        _center_marks[i].out_center_x = out_pos;
    }
    _is_center_marked = true;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stitcher::estimate_overlap ()
{
    if (_is_overlap_set)
        return XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        ERROR, _is_round_view_set && _is_crop_set && _is_center_marked, XCAM_RETURN_ERROR_ORDER,
        "stitcher estimate_coarse_seam failed, need set round_view slices, crop info and mark centers first");

    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        uint32_t next_idx = (idx + 1) % _camera_num;
        const RoundViewSlice &left = _round_view_slices[idx];
        const RoundViewSlice &right = _round_view_slices[next_idx];
        const CenterMark &left_center = _center_marks[idx];
        const CenterMark &right_center = _center_marks[next_idx];
        const ImageCropInfo &left_img_crop = _crop_info[idx];
        const ImageCropInfo &right_img_crop = _crop_info[next_idx];

#if 0
        XCAM_FAIL_RETURN (
            ERROR,
            (format_angle (right.hori_angle_start - left.hori_angle_start) < left.hori_angle_range)
            XCAM_RETURN_ERROR_UNKNOWN,
            "stitcher estimate_coarse_seam failed and there is no seam between slice %d and slice %d", idx, next_idx);

        float seam_angle_start = right.hori_angle_start;
        float seam_angle_range =
            format_angle (left.hori_angle_start + left.hori_angle_range - right.hori_angle_start);

        XCAM_FAIL_RETURN (
            ERROR, seam_angle_range < right.hori_angle_range, XCAM_RETURN_ERROR_UNKNOWN,
            "stitcher estimate_coarse_seam failed and left slice(%d)over covered right slice(%d)", idx, next_idx);

        XCAM_ASSERT (!XCAM_DOUBLE_EQUAL_AROUND (left.hori_angle_range, 0.0f));
        XCAM_ASSERT (!XCAM_DOUBLE_EQUAL_AROUND (right.hori_angle_range, 0.0f));
#endif
        uint32_t out_right_center_x = right_center.out_center_x;
        if (out_right_center_x == 0)
            out_right_center_x = _output_width;

        Rect valid_left_img, valid_right_img;
        valid_left_img.pos_x = left_center.slice_center_x;
        valid_left_img.width = left.width - left_img_crop.right - valid_left_img.pos_x;
        valid_left_img.pos_y = left_img_crop.top;
        valid_left_img.height = left.height - left_img_crop.top - left_img_crop.bottom;

        valid_right_img.width = right_center.slice_center_x - right_img_crop.left;
        valid_right_img.pos_x = right_center.slice_center_x - valid_right_img.width;
        valid_right_img.pos_y = right_img_crop.top;
        valid_right_img.height = right.height - right_img_crop.top - right_img_crop.bottom;

        uint32_t merge_width = out_right_center_x - left_center.out_center_x;
        XCAM_FAIL_RETURN (
            ERROR,
            valid_left_img.width + valid_right_img.width > (int32_t)merge_width,
            XCAM_RETURN_ERROR_UNKNOWN,
            "stitcher estimate_overlap failed and there is no overlap area between slice %d and slice %d", idx, next_idx);

        uint32_t overlap_width = valid_left_img.width + valid_right_img.width - merge_width;

        Rect left_img_overlap, right_img_overlap;
        left_img_overlap.pos_x = valid_left_img.pos_x + valid_left_img.width - overlap_width;
        left_img_overlap.width = overlap_width;
        left_img_overlap.pos_y = valid_left_img.pos_y;
        left_img_overlap.height = valid_left_img.height;
        XCAM_ASSERT (left_img_overlap.pos_x >= (int32_t)left_center.slice_center_x &&  left_img_overlap.pos_x < (int32_t)left.width);

        right_img_overlap.pos_x = valid_right_img.pos_x;
        right_img_overlap.width = overlap_width;
        right_img_overlap.pos_y = valid_right_img.pos_y;
        right_img_overlap.height = valid_right_img.height;
        XCAM_ASSERT (right_img_overlap.pos_x >= (int32_t)right_img_crop.left && right_img_overlap.pos_x < (int32_t)right_center.slice_center_x);

        Rect out_overlap;
        out_overlap.pos_x = left_center.out_center_x + valid_left_img.width - overlap_width;
        out_overlap.width = overlap_width;
        // out_overlap.pos_y/height not useful by now
        out_overlap.pos_y = valid_left_img.pos_y;
        out_overlap.height = valid_left_img.height;

#if 0
        left_img_seam.pos_x =
            left.width * format_angle (seam_angle_start - left.hori_angle_start) / left.hori_angle_range;
        left_img_seam.pos_y = _crop_info[idx].top;
        left_img_seam.width = left.width * seam_angle_range / left.hori_angle_range;
        left_img_seam.height = left.height - _crop_info[idx].top - _crop_info[idx].bottom;

        //consider crop
        XCAM_ASSERT (left_img_seam.pos_x <  left.width - _crop_info[idx].right);
        if (left_img_seam.pos_x + left_img_seam.width > left.width - _crop_info[idx].right)
            left_img_seam.width = left.width - _crop_info[idx].right;

        right_img_seam.pos_x = 0;
        right_img_seam.pos_y = _crop_info[next_idx].top;
        right_img_seam.width = right.width * (seam_angle_range / right.hori_angle_range);
        right_img_seam.height = right.height - _crop_info[next_idx].top - _crop_info[next_idx].bottom;

        //consider crop
        XCAM_ASSERT (right_img_seam.pos_x + right_img_seam.width >  _crop_info[next_idx].left);
        if (_crop_info[next_idx].left) {
            right_img_seam.pos_x = _crop_info[next_idx].left;
            right_img_seam.width -= _crop_info[next_idx].left;
            left_img_seam.pos_x += _crop_info[next_idx].left;
            left_img_seam.width -= _crop_info[next_idx].left;
        }

        XCAM_ASSERT (abs (left_img_seam.width - right_img_seam.width) < 16);
        left_img_seam.pos_x = XCAM_ALIGN_DOWN (left_img_seam.pos_x, _alignment_x);
        right_img_seam.pos_x = XCAM_ALIGN_DOWN (right_img_seam.pos_x, _alignment_x);

        //find max seam width
        uint32_t seam_width, seam_height;
        seam_width = XCAM_MAX (left_img_seam.width, right_img_seam.width);
        if (left_img_seam.pos_x + seam_width > left.width)
            seam_width = left.width - left_img_seam.pos_x;
        if (right_img_seam.pos_x + seam_width > right.width)
            seam_width = right.width - right_img_seam.pos_x;

        XCAM_FAIL_RETURN (
            ERROR, seam_width >= XCAM_STITCH_MIN_SEAM_WIDTH, XCAM_RETURN_ERROR_UNKNOWN,
            "stitcher estimate_coarse_seam failed, the seam(w:%d) is very narrow between(slice %d and %d)",
            seam_width, idx, next_idx);
        left_img_seam.width = right_img_seam.width = XCAM_ALIGN_DOWN (seam_width, _alignment_x);

        // min height
        uint32_t top = XCAM_MAX (left_img_seam.pos_y, right_img_seam.pos_y);
        uint32_t bottom0 = left_img_seam.pos_y + left_img_seam.height;
        uint32_t bottom1 = right_img_seam.pos_y + right_img_seam.height;
        uint32_t bottom = XCAM_MIN (bottom0, bottom1);
        top = XCAM_ALIGN_UP (top, _alignment_y);
        left_img_seam.pos_y = right_img_seam.pos_y = top;
        left_img_seam.height = right_img_seam.height = XCAM_ALIGN_DOWN (bottom - top, _alignment_y);
#endif
        // set overlap info
        _overlap_info[idx].left = left_img_overlap;
        _overlap_info[idx].right = right_img_overlap;
        _overlap_info[idx].out_area = out_overlap;
    }

    _is_overlap_set = true;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stitcher::update_copy_areas ()
{
    XCAM_FAIL_RETURN (
        ERROR, _camera_num > 1 && _is_round_view_set && _is_crop_set && _is_overlap_set, XCAM_RETURN_ERROR_ORDER,
        "stitcher update_copy_areas failed, check orders, need"
        "camera_info, round_view slices, crop_info and overlap_info set first.");

    CopyAreaArray tmp_areas;
    uint32_t i = 0;
    uint32_t next_i = 0;
    for (i = 0; i < _camera_num; ++i) {
        next_i = (i + 1 ) % _camera_num;
        const CenterMark &mark_left = _center_marks[i];
        const CenterMark &mark_right = _center_marks[next_i];
        const ImageOverlapInfo  &overlap = _overlap_info[i];

        CopyArea split_a, split_b;

        CopyArea left;
        left.in_idx = i;
        left.in_area.pos_x = mark_left.slice_center_x;
        left.in_area.width = overlap.left.pos_x - left.in_area.pos_x;
        XCAM_ASSERT (left.in_area.width > 0);
        left.in_area.pos_y = _crop_info[i].top;
        left.in_area.height = _round_view_slices[i].height - _crop_info[i].top - _crop_info[i].bottom;
        XCAM_ASSERT (left.in_area.height > 0);

        left.out_area.pos_x = mark_left.out_center_x;
        left.out_area.width = left.in_area.width;
        left.out_area.pos_y = 0;
        left.out_area.height = left.in_area.height;

        if (split_area_by_out (left, _output_width, split_a, split_b)) {
            tmp_areas.push_back (split_a);
            tmp_areas.push_back (split_b);
        } else {
            tmp_areas.push_back (left);
        }

        CopyArea right;
        right.in_idx = next_i;
        right.in_area.pos_x = _overlap_info[i].right.pos_x + _overlap_info[i].right.width;
        right.in_area.width =  (int32_t)mark_right.slice_center_x - right.in_area.pos_x;
        XCAM_ASSERT (right.in_area.width > 0);
        right.in_area.pos_y = _crop_info[next_i].top;
        right.in_area.height = _round_view_slices[next_i].height - _crop_info[next_i].top - _crop_info[next_i].bottom;
        XCAM_ASSERT (right.in_area.height > 0);

        uint32_t out_right_center_x = mark_right.out_center_x;
        if (out_right_center_x == 0)
            out_right_center_x = _output_width;
        right.out_area.width = right.in_area.width;
        right.out_area.pos_x = out_right_center_x - right.out_area.width;
        right.out_area.pos_y = 0;
        right.out_area.height = right.in_area.height;

        if (split_area_by_out (right, _output_width, split_a, split_b)) {
            tmp_areas.push_back (split_a);
            tmp_areas.push_back (split_b);
        } else {
            tmp_areas.push_back (right);
        }
    }
    XCAM_ASSERT (tmp_areas.size () > _camera_num && _camera_num >= 2);

    CopyArea merged;
    int32_t start = 0;
    int32_t end = tmp_areas.size () - 1;
    if (tmp_areas.size () > 2) {
        const CopyArea &first = tmp_areas[0];
        const CopyArea &last = tmp_areas[end];
        // merge first and last
        if (merge_neighbor_area (last, first, merged)) {
            _copy_areas.push_back (merged);
            ++start;
            --end;
        }
    }

    // merge areas
    for (i = (uint32_t)start; (int32_t)i <= end; ) {
        const CopyArea &current = tmp_areas[i];
        if (i == (uint32_t)end) {
            _copy_areas.push_back (current);
            break;
        }

        const CopyArea &next = tmp_areas[i + 1];
        if (merge_neighbor_area (current, next, merged)) {
            _copy_areas.push_back (merged);
            i += 2;
        } else {
            _copy_areas.push_back (current);
            i += 1;
        }
    }

    XCAM_ASSERT (_copy_areas.size() >= _camera_num);

    return XCAM_RETURN_NO_ERROR;
}

BowlModel::BowlModel (const BowlDataConfig &config, const uint32_t image_width, const uint32_t image_height)
    : _config (config)
    , _bowl_img_width (image_width)
    , _bowl_img_height (image_height)
{
    //max area => x/a = y/b
    XCAM_ASSERT (fabs(_config.center_z) < _config.c);
    float mid = sqrt ((1.0f - _config.center_z * _config.center_z / (_config.c * _config.c)) / 2.0f);
    _max_topview_length_mm = mid * _config.a * 2.0f;
    _max_topview_width_mm = mid * _config.b * 2.0f;
}

bool
BowlModel::get_max_topview_area_mm (float &length_mm, float &width_mm)
{
    if (_max_topview_width_mm <= 0.0f || _max_topview_length_mm <= 0.0f)
        return false;
    length_mm = _max_topview_length_mm;
    width_mm = _max_topview_width_mm;
    return true;
}

bool
BowlModel::get_topview_rect_map (
    PointMap &texture_points,
    uint32_t res_width, uint32_t res_height,
    float length_mm, float width_mm)
{
    if (XCAM_DOUBLE_EQUAL_AROUND (length_mm, 0.0f) ||
            XCAM_DOUBLE_EQUAL_AROUND (width_mm, 0.0f)) {
        get_max_topview_area_mm (length_mm, width_mm);
    }

    XCAM_FAIL_RETURN (
        ERROR,
        length_mm * length_mm / (_config.a * _config.a) / 4.0f + width_mm * width_mm / (_config.b * _config.b) / 4.0f +
        _config.center_z * _config.center_z / (_config.c * _config.c) <= 1.0f + 0.001f,
        false,
        "bowl model topview input area(L:%.2fmm, W:%.2fmm) is larger than max area", length_mm, width_mm);

    float center_pos_x = res_width / 2.0f;
    float center_pos_y = res_height / 2.0f;
    float mm_per_pixel_x = length_mm / res_width;
    float mm_per_pixel_y = width_mm / res_height;

    texture_points.resize (res_width * res_height);

    for(uint32_t row = 0; row < res_height; row++) {
        for(uint32_t col = 0; col < res_width; col++) {
            PointFloat3 world_pos (
                (col - center_pos_x) * mm_per_pixel_x,
                (center_pos_y - row) * mm_per_pixel_y,
                0.0f);

            PointFloat2 texture_pos = bowl_view_coords_to_image (
                                          _config, world_pos, _bowl_img_width, _bowl_img_height);

            texture_points [res_width * row + col] = texture_pos;
        }
    }
    return true;
}

bool
BowlModel::get_stitch_image_vertex_model (
    VertexMap &vertices, PointMap &texture_points, IndexVector &indeices,
    uint32_t res_width, uint32_t res_height, float vertex_height)
{
    vertices.reserve (2 * (res_width + 1) * (res_height + 1));
    texture_points.reserve (2 * (res_width + 1) * (res_height + 1));
    indeices.reserve (2 * (res_width + 1) * (res_height + 1) + (res_height + 1));

    float step_x = (float)_bowl_img_width / res_width;
    float step_y = vertex_height / res_height;
    float offset_y = (float)_bowl_img_height - vertex_height;

    int32_t indicator = 0;

    for (uint32_t row = 0; row < res_height - 1; row++) {
        PointFloat2 texture_pos0;
        texture_pos0.y = row * step_y + offset_y;

        PointFloat2 texture_pos1;
        texture_pos1.y = (row + 1) * step_y + offset_y;

        for (uint32_t col = 0; col <= res_width; col++) {

            texture_pos0.x = col * step_x;
            texture_pos1.x = col * step_x;

            PointFloat3 world_pos0 =
                bowl_view_image_to_world (
                    _config, _bowl_img_width, _bowl_img_height, texture_pos0);

            vertices.push_back (PointFloat3(world_pos0.x / _config.a, world_pos0.y / _config.b, world_pos0.z / _config.c));
            indeices.push_back (indicator++);
            texture_points.push_back (PointFloat2(texture_pos0.x / _bowl_img_width, texture_pos0.y / _bowl_img_height));

            PointFloat3 world_pos1 =
                bowl_view_image_to_world (
                    _config, _bowl_img_width, _bowl_img_height, texture_pos1);

            vertices.push_back (PointFloat3(world_pos1.x / _config.a, world_pos1.y / _config.b, world_pos1.z / _config.c));
            indeices.push_back (indicator++);
            texture_points.push_back (PointFloat2(texture_pos1.x / _bowl_img_width, texture_pos1.y / _bowl_img_height));
        }
    }
    return true;
}


bool
BowlModel::get_bowlview_vertex_model (
    VertexMap &vertices, PointMap &texture_points, IndexVector &indeices,
    uint32_t res_width, uint32_t res_height)
{
    return get_stitch_image_vertex_model (vertices, texture_points, indeices, res_width, res_height, (float)_bowl_img_height);
}

bool
BowlModel::get_topview_vertex_model (
    VertexMap &vertices, PointMap &texture_points, IndexVector &indeices,
    uint32_t res_width, uint32_t res_height)
{
    float wall_image_height = _config.wall_height / (float)(_config.wall_height + _config.ground_length) * (float)_bowl_img_height;
    float ground_image_height = (float)_bowl_img_height - wall_image_height;

    return get_stitch_image_vertex_model (vertices, texture_points, indeices, res_width, res_height, ground_image_height);
}


}
