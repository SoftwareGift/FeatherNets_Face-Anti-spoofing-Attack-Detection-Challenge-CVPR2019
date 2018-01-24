/*
 * feature_match.cpp - optical flow feature match
 *
 *  Copyright (c) 2016-2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
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

#include "feature_match.h"

#define XCAM_FM_DEBUG 0

namespace XCam {

FeatureMatch::FeatureMatch ()
    : _x_offset (0.0f)
    , _y_offset (0.0f)
    , _mean_offset (0.0f)
    , _mean_offset_y (0.0f)
    , _valid_count (0)
    , _fm_idx (-1)
    , _frame_num (0)
{
}

void
FeatureMatch::set_config (CVFMConfig config)
{
    _config = config;
}

CVFMConfig
FeatureMatch::get_config ()
{
    return _config;
}

void
FeatureMatch::set_fm_index (int idx)
{
    _fm_idx = idx;
}

void
FeatureMatch::reset_offsets ()
{
    _x_offset = 0.0f;
    _y_offset = 0.0f;
    _mean_offset = 0.0f;
    _mean_offset_y = 0.0f;
}

bool
FeatureMatch::get_mean_offset (std::vector<float> &offsets, float sum, int &count, float &mean_offset)
{
    if (count < _config.min_corners)
        return false;

    mean_offset = sum / count;

#if XCAM_FM_DEBUG
    XCAM_LOG_INFO (
        "FeatureMatch(idx:%d): X-axis mean offset:%.2f, pre_mean_offset:%.2f (%d times, count:%d)",
        _fm_idx, mean_offset, 0.0f, 0, count);
#endif

    bool ret = true;
    float delta = 20.0f;//mean_offset;
    float pre_mean_offset = mean_offset;
    for (int try_times = 1; try_times < 4; ++try_times) {
        int recur_count = 0;
        sum = 0.0f;

        for (size_t i = 0; i < offsets.size (); ++i) {
            if (fabs (offsets[i] - mean_offset) >= _config.recur_offset_error)
                continue;
            sum += offsets[i];
            ++recur_count;
        }

        if (recur_count < _config.min_corners) {
            ret = false;
            break;
        }

        mean_offset = sum / recur_count;
#if XCAM_FM_DEBUG
        XCAM_LOG_INFO (
            "FeatureMatch(idx:%d): X-axis mean_offset:%.2f, pre_mean_offset:%.2f (%d times, count:%d)",
            _fm_idx, mean_offset, pre_mean_offset, try_times, recur_count);
#endif

        if (mean_offset == pre_mean_offset && recur_count == count)
            return true;

        if (fabs (mean_offset - pre_mean_offset) > fabs (delta) * 1.2f) {
            ret = false;
            break;
        }

        delta = mean_offset - pre_mean_offset;
        pre_mean_offset = mean_offset;
        count = recur_count;
    }

    return ret;
}

void
FeatureMatch::adjust_stitch_area (int dst_width, float &x_offset, Rect &stitch0, Rect &stitch1)
{
    if (fabs (x_offset) < 5.0f)
        return;

    int last_overlap_width = stitch1.pos_x + stitch1.width + (dst_width - (stitch0.pos_x + stitch0.width));
    // int final_overlap_width = stitch1.pos_x + stitch1.width + (dst_width - (stitch0.pos_x - x_offset + stitch0.width));
    if ((stitch0.pos_x - x_offset + stitch0.width) > dst_width)
        x_offset = dst_width - (stitch0.pos_x + stitch0.width);
    int final_overlap_width = last_overlap_width + x_offset;
    final_overlap_width = XCAM_ALIGN_AROUND (final_overlap_width, 8);
    XCAM_ASSERT (final_overlap_width >= _config.sitch_min_width);
    int center = final_overlap_width / 2;
    XCAM_ASSERT (center >= _config.sitch_min_width / 2);

    stitch1.pos_x = XCAM_ALIGN_AROUND (center - _config.sitch_min_width / 2, 8);
    stitch1.width = _config.sitch_min_width;
    stitch0.pos_x = dst_width - final_overlap_width + stitch1.pos_x;
    stitch0.width = _config.sitch_min_width;

    float delta_offset = final_overlap_width - last_overlap_width;
    x_offset -= delta_offset;
}

}
