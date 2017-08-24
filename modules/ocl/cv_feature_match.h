/*
 * cv_feature_match.h - optical flow feature match
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

#ifndef XCAM_CV_FEATURE_MATCH_H
#define XCAM_CV_FEATURE_MATCH_H

#include <base/xcam_common.h>
#include <base/xcam_buffer.h>
#include <dma_video_buffer.h>
#include <smartptr.h>
#include "xcam_obj_debug.h"
#include "image_file_handle.h"
#include "cv_base_class.h"

#include <ocl/cl_context.h>
#include <ocl/cl_device.h>
#include <ocl/cl_memory.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#define XCAM_CV_FM_MATCH_NUM  2

namespace XCam {

struct CVFMConfig {
    int sitch_min_width;
    int min_corners;           // number of minimum efficient corners
    float offset_factor;       // last_offset * offset_factor + cur_offset * (1.0f - offset_factor)
    float delta_mean_offset;   // cur_mean_offset - last_mean_offset
    float max_adjusted_offset; // max offset of each adjustment

    CVFMConfig ()
        : sitch_min_width (56)
        , min_corners (8)
        , offset_factor (0.8f)
        , delta_mean_offset (5.0f)
        , max_adjusted_offset (12.0f)
    {}
};

class CVFeatureMatch : public CVBaseClass
{
public:
    explicit CVFeatureMatch ();

    void set_config (CVFMConfig config);
    CVFMConfig get_config ();

    void set_fm_index (int idx);

    void optical_flow_feature_match (
        SmartPtr<DrmBoBuffer> left_buf, SmartPtr<DrmBoBuffer> right_buf,
        cv::Rect &left_img_crop, cv::Rect &right_img_crop, int dst_width);

protected:
    bool get_crop_image (SmartPtr<DrmBoBuffer> buffer, cv::Rect img_crop, cv::UMat &img);

    void add_detected_data (cv::InputArray image, cv::Ptr<cv::Feature2D> detector, std::vector<cv::Point2f> &corners);
    void get_valid_offsets (cv::InputOutputArray out_image, cv::Size img0_size,
                            std::vector<cv::Point2f> corner0, std::vector<cv::Point2f> corner1,
                            std::vector<uchar> status, std::vector<float> error,
                            std::vector<float> &offsets, float &sum, int &count);
    bool get_mean_offset (std::vector<float> offsets, float sum, int &count, float &mean_offset);

    void calc_of_match (cv::InputArray image0, cv::InputArray image1,
                        std::vector<cv::Point2f> corner0, std::vector<cv::Point2f> corner1,
                        std::vector<uchar> &status, std::vector<float> &error,
                        int &last_count, float &last_mean_offset, float &out_x_offset);
    void adjust_stitch_area (int dst_width, float &x_offset, cv::Rect &stitch0, cv::Rect &stitch1);
    void detect_and_match (
        cv::InputArray img_left, cv::InputArray img_right, cv::Rect &crop_left, cv::Rect &crop_right,
        int &valid_count, float &mean_offset, float &x_offset, int dst_width);

private:
    XCAM_DEAD_COPY (CVFeatureMatch);

private:
    CVFMConfig           _config;

    float                _x_offset;
    float                _mean_offset;
    int                  _valid_count;

    // debug parameters
    int                  _fm_idx;
    uint                 _frame_num;
};

}

#endif // XCAM_CV_FEATURE_MATCH_H
