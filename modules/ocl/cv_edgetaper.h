/*
 * cv_edgetaper.h - used in deblurring to remove ringing artifacts
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

#ifndef XCAM_CV_EDGETAPER_H
#define XCAM_CV_EDGETAPER_H

#include <xcam_std.h>
#include <ocl/cv_base_class.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

namespace XCam {


class CVEdgetaper : public CVBaseClass
{

public:
    explicit CVEdgetaper ();
    void edgetaper (const cv::Mat &image, const cv::Mat &psf, cv::Mat &output);

private:
    void create_weights (const cv::Mat &image, const cv::Mat &psf, cv::Mat &coefficients);

    XCAM_DEAD_COPY (CVEdgetaper);
};

}

#endif // XCAM_CV_EDGETAPER_H
