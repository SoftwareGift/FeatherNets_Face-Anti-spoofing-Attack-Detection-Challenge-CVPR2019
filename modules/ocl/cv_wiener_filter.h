/*
 * cv_wiener_filter.h - wierner filter for non-blind deblurring
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

#ifndef XCAM_CV_WIENER_FILTER_H
#define XCAM_CV_WIENER_FILTER_H

#include <xcam_std.h>
#include <video_buffer.h>
#include <ocl/cv_base_class.h>
#include <ocl/cv_image_process_helper.h>

namespace XCam {

class CVWienerFilter : public CVBaseClass
{

public:
    explicit CVWienerFilter ();

    void wiener_filter (const cv::Mat &blurred_image, const cv::Mat &known, cv::Mat &unknown, float noise_power);

private:

    XCAM_DEAD_COPY (CVWienerFilter);

    SmartPtr<CVImageProcessHelper>  _helpers;
};

}

#endif // XCAM_CV_WIENER_FILTER_H
