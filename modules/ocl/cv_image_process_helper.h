/*
 * cv_image_process_helper.h - OpenCV image processing helpers functions
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
 * Author: Andrey Parfenov <a1994ndrey@gmail.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_CV_IMAGE_PROCESS_HELPER_H
#define XCAM_CV_IMAGE_PROCESS_HELPER_H

#include <xcam_std.h>
#include <video_buffer.h>
#include <ocl/cv_base_class.h>

namespace XCam {


class CVImageProcessHelper : public CVBaseClass
{

public:
    explicit CVImageProcessHelper ();

    void compute_dft (const cv::Mat &image, cv::Mat &result);
    void compute_idft (cv::Mat *input, cv::Mat &result);
    void apply_constraints (cv::Mat &image, float threshold_min_value = 0.0f, float threshold_max_value = 255.0f, float min_value = 0.0f, float max_value = 255.0f);
    float get_snr (const cv::Mat &noisy, const cv::Mat &noiseless);
    cv::Mat erosion (const cv::Mat &image, int erosion_size, int erosion_type);
    void normalize_weights (cv::Mat &weights);

    XCAM_DEAD_COPY (CVImageProcessHelper);
};

}

#endif // XCAM_CV_IMAGE_PROCESS_HELPER_H
