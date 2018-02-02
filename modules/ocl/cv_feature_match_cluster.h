/*
 * cv_feature_match_cluster.h - optical flow feature match selected by clustering
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
 * Author: Wu Junkai <junkai.Wu@intel.com>
 */

#ifndef XCAM_CV_FEATURE_MATCH_CLUSTER_H
#define XCAM_CV_FEATURE_MATCH_CLUSTER_H

#include <ocl/cv_feature_match.h>

namespace XCam {

class CVFeatureMatchCluster
    : public CVFeatureMatch
{
public:
    explicit CVFeatureMatchCluster ();

    void optical_flow_feature_match (
        const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf,
        Rect &left_img_crop, Rect &right_img_crop, int dst_width = 0);

protected:
    bool calc_mean_offset (std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
                           std::vector<uchar> &status, std::vector<float> &error,
                           float &mean_offset_x, float &mean_offset_y,
                           cv::InputOutputArray debug_img, cv::Size &img0_size, cv::Size &img1_size);

    void calc_of_match_cluster (cv::InputArray image0, cv::InputArray image1,
                                std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
                                std::vector<uchar> &status, std::vector<float> &error,
                                float &last_mean_offset_x, float &last_mean_offset_y,
                                float &out_x_offset, float &out_y_offset);

    void detect_and_match_cluster (cv::InputArray img_left, cv::InputArray img_right, Rect &crop_left, Rect &crop_right,
                                   float &mean_offset_x, float &mean_offset_y,
                                   float &x_offset, float &y_offset);


private:
    XCAM_DEAD_COPY (CVFeatureMatchCluster);

};

}

#endif // XCAM_CV_FEATURE_MATCH_CLUSTER_H
