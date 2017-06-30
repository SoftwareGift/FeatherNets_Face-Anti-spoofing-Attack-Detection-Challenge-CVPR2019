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

class CVFeatureMatch
{
public:
    explicit CVFeatureMatch (const SmartPtr<CLContext> &context);

    void set_ocl (bool use_ocl) {
        _use_ocl = use_ocl;
    }
    bool is_ocl_path () {
        return _use_ocl;
    }

    void set_config (CVFMConfig config);
    CVFMConfig get_config ();

    void optical_flow_feature_match (
        int output_width, SmartPtr<DrmBoBuffer> buf0, SmartPtr<DrmBoBuffer> buf1,
        cv::Rect &img0_crop_left, cv::Rect &img0_crop_right, cv::Rect &img1_crop_left, cv::Rect &img1_crop_right);

protected:
    void init_opencv_ocl ();

    bool convert_to_mat (SmartPtr<CLContext> context, SmartPtr<DrmBoBuffer> buffer, cv::Mat &image);
    bool get_crop_image (SmartPtr<DrmBoBuffer> buffer,
                         cv::Rect img_crop_left, cv::Rect img_crop_right, cv::UMat &img_left, cv::UMat &img_right);

    void add_detected_data (cv::InputArray image, cv::Ptr<cv::Feature2D> detector, std::vector<cv::Point2f> &corners);
    void get_valid_offsets (cv::InputOutputArray out_image, cv::Size img0_size,
                            std::vector<cv::Point2f> corner0, std::vector<cv::Point2f> corner1,
                            std::vector<uchar> status, std::vector<float> error, std::vector<float> &offsets, float &sum, int &count);
    bool get_mean_offset (std::vector<float> offsets, float sum, int &count, float &mean_offset);

    void calc_of_match (cv::InputArray image0, cv::InputArray image1,
                        std::vector<cv::Point2f> corner0, std::vector<cv::Point2f> corner1,
                        std::vector<uchar> &status, std::vector<float> &error,
                        int &last_count, float &last_mean_offset, float &out_x_offset, int frame_num, int idx);
    void adjust_stitch_area (int dst_width, float &x_offset, cv::Rect &stitch0, cv::Rect &stitch1);
    void detect_and_match (
        cv::InputArray img_left, cv::InputArray img_right, cv::Rect &crop_left, cv::Rect &crop_right,
        int &valid_count, float &mean_offset, float &x_offset, int dst_width);

private:
    XCAM_DEAD_COPY (CVFeatureMatch);

private:
    SmartPtr<CLContext>  _context;
    CVFMConfig           _config;

    float                _x_offset[XCAM_CV_FM_MATCH_NUM];
    float                _mean_offset[XCAM_CV_FM_MATCH_NUM];
    int                  _valid_count[XCAM_CV_FM_MATCH_NUM];

    bool                 _use_ocl;
    bool                 _is_ocl_inited;
};

}

#endif // XCAM_CV_FEATURE_MATCH_H
