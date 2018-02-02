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

#include <xcam_std.h>
#include <video_buffer.h>
#include <interface/feature_match.h>

#ifdef ANDROID
#include <cv.h>
#else
#include <opencv2/opencv.hpp>
#endif

namespace XCam {

class CVCapiFeatureMatch
    : public FeatureMatch
{
public:
    explicit CVCapiFeatureMatch ();

    void optical_flow_feature_match (
        const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf,
        Rect &left_img_crop, Rect &right_img_crop, int dst_width = 0);

    void set_ocl (bool use_ocl) {
        XCAM_UNUSED (use_ocl);
    }
    bool is_ocl_path () {
        return false;
    }

protected:
    bool get_crop_image (const SmartPtr<VideoBuffer> &buffer, const Rect &crop_rect,
                         std::vector<char> &crop_image, CvMat &img);

    void add_detected_data (CvArr* image, std::vector<CvPoint2D32f> &corners);
    void get_valid_offsets (std::vector<CvPoint2D32f> &corner0, std::vector<CvPoint2D32f> &corner1,
                            std::vector<char> &status, std::vector<float> &error,
                            std::vector<float> &offsets, float &sum, int &count,
                            CvArr* out_image, CvSize &img0_size);

    void calc_of_match (CvArr* image0, CvArr* image1,
                        std::vector<CvPoint2D32f> &corner0, std::vector<CvPoint2D32f> &corner1,
                        std::vector<char> &status, std::vector<float> &error,
                        int &last_count, float &last_mean_offset, float &out_x_offset);

    void detect_and_match (CvArr* img_left, CvArr* img_right,
                           int &valid_count, float &mean_offset, float &x_offset);

private:
    XCAM_DEAD_COPY (CVCapiFeatureMatch);

    std::vector<char> _left_crop_image;
    std::vector<char> _right_crop_image;
};

}

#endif // CV_CAPI_FEATURE_MATCH_H
