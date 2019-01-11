/*
 * feature_match.h - optical flow feature match
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

#ifndef XCAM_FEATURE_MATCH_H
#define XCAM_FEATURE_MATCH_H

#include <xcam_std.h>
#include <video_buffer.h>
#include <interface/data_types.h>

namespace XCam {

struct FMConfig {
    int sitch_min_width;
    int min_corners;           // number of minimum efficient corners
    float offset_factor;       // last_offset * offset_factor + cur_offset * (1.0f - offset_factor)
    float delta_mean_offset;   // cur_mean_offset - last_mean_offset
    float recur_offset_error;  // cur_offset - mean_offset
    float max_adjusted_offset; // maximum offset of each adjustment
    float max_valid_offset_y;  // valid maximum offset in vertical direction
    float max_track_error;     // maximum track error

    FMConfig ()
        : sitch_min_width (56)
        , min_corners (8)
        , offset_factor (0.8f)
        , delta_mean_offset (5.0f)
        , recur_offset_error (8.0f)
        , max_adjusted_offset (12.0f)
        , max_valid_offset_y (8.0f)
        , max_track_error (24.0f)
    {}
};

class FeatureMatch
{
public:
    explicit FeatureMatch ();
    virtual ~FeatureMatch () {};

    virtual void feature_match (
        const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf) = 0;

    void set_fm_index (int idx);
    void set_config (const FMConfig &config);

    void set_crop_rect (const Rect &left_rect, const Rect &right_rect);
    void get_crop_rect (Rect &left_rect, Rect &right_rect);

    void reset_offsets ();
    float get_current_left_offset_x ();
    float get_current_left_offset_y ();

    virtual void set_dst_width (int width);
    virtual void enable_adjust_crop_area ();

protected:
    bool get_mean_offset (const std::vector<float> &offsets, float sum, int &count, float &mean_offset);

private:
    XCAM_DEAD_COPY (FeatureMatch);

protected:
    float                _x_offset;
    float                _y_offset;
    float                _mean_offset;
    float                _mean_offset_y;
    int                  _valid_count;
    FMConfig             _config;

    Rect                 _left_rect;
    Rect                 _right_rect;

    // debug parameters
    int                  _fm_idx;
    uint32_t             _frame_num;
};

}

#endif // XCAM_FEATURE_MATCH_H
