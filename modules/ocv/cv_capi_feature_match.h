/*
 * cv_capi_feature_match.h - optical flow feature match
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef CV_CAPI_FEATURE_MATCH_H
#define CV_CAPI_FEATURE_MATCH_H

#include <video_buffer.h>
#include <interface/feature_match.h>
#include "cv_utils.h"

#ifdef ANDROID
#include <cv.h>
#else
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking_c.h>
#endif

namespace XCam {

class CVCapiFeatureMatch
    : public FeatureMatch
{
public:
    explicit CVCapiFeatureMatch ();

    virtual void feature_match (
        const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf);

private:
    bool get_crop_image (
        const SmartPtr<VideoBuffer> &buffer, const Rect &crop_rect, std::vector<char> &crop_image, CvMat &img);

    void detect_and_match (CvArr* img_left, CvArr* img_right);
    void add_detected_data (CvArr* image, std::vector<CvPoint2D32f> &corners);

    void calc_of_match (
        CvArr* image0, CvArr* image1, std::vector<CvPoint2D32f> &corner0, std::vector<CvPoint2D32f> &corner1,
        std::vector<char> &status, std::vector<float> &error);

    void get_valid_offsets (
        std::vector<CvPoint2D32f> &corner0, std::vector<CvPoint2D32f> &corner1,
        std::vector<char> &status, std::vector<float> &error,
        std::vector<float> &offsets, float &sum, int &count,
        CvArr* out_image, CvSize &img0_size);

private:
    XCAM_DEAD_COPY (CVCapiFeatureMatch);

    std::vector<char>    _left_crop_image;
    std::vector<char>    _right_crop_image;
};

}

#endif // CV_CAPI_FEATURE_MATCH_H
