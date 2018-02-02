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

struct CVFMConfig {
    int sitch_min_width;
    int min_corners;           // number of minimum efficient corners
    float offset_factor;       // last_offset * offset_factor + cur_offset * (1.0f - offset_factor)
    float delta_mean_offset;   // cur_mean_offset - last_mean_offset
    float recur_offset_error;  // cur_offset - mean_offset
    float max_adjusted_offset; // maximum offset of each adjustment
    float max_valid_offset_y;  // valid maximum offset in vertical direction
    float max_track_error;     // maximum track error

    CVFMConfig ()
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

    void set_config (CVFMConfig config);
    CVFMConfig get_config ();

    void set_fm_index (int idx);

    void reset_offsets ();

    virtual void optical_flow_feature_match (
        const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf,
        Rect &left_crop_rect, Rect &right_crop_rect, int dst_width = 0) = 0;

    float get_current_left_offset_x () const {
        return _x_offset;
    }

    float get_current_left_offset_y () const {
        return _y_offset;
    }

    virtual void set_ocl (bool use_ocl) = 0;
    virtual bool is_ocl_path () = 0;

protected:
    bool get_mean_offset (std::vector<float> &offsets, float sum, int &count, float &mean_offset);

private:
    XCAM_DEAD_COPY (FeatureMatch);

protected:
    float                _x_offset;
    float                _y_offset;
    float                _mean_offset;
    float                _mean_offset_y;
    int                  _valid_count;
    CVFMConfig           _config;

    // debug parameters
    int                  _fm_idx;
    uint                 _frame_num;
};

}

#endif // XCAM_FEATURE_MATCH_H
