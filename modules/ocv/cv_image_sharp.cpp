/*
 * cv_image_sharp.cpp - image sharp
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

#include "cv_image_sharp.h"

namespace XCam {


CVImageSharp::CVImageSharp ()
    : CVBaseClass ()
{

}

cv::Mat
CVImageSharp::sharp_image_gray (const cv::Mat &image, float sigmar)
{
    cv::Mat temp_image;
    image.convertTo (temp_image, CV_32FC1);
    cv::Mat bilateral_image;
    cv::bilateralFilter (temp_image, bilateral_image, 5, sigmar, 2);

    cv::Mat sharp_filter = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    cv::Mat filtered_image;
    cv::filter2D (bilateral_image, filtered_image, -1, sharp_filter);
    cv::normalize (filtered_image, filtered_image, 0, 255.0f, cv::NORM_MINMAX);
    cv::Mat sharpened = temp_image + filtered_image;
    cv::normalize (sharpened, sharpened, 0, 255.0f, cv::NORM_MINMAX);
    return sharpened.clone ();
}

float
CVImageSharp::measure_sharp (const cv::Mat &image)
{
    cv::Mat dst;
    cv::Laplacian (image, dst, -1, 3, 1, 0, cv::BORDER_CONSTANT);
    dst.convertTo (dst, CV_8UC1);
    float sum = cv::sum (dst)[0];
    sum /= (image.rows * image.cols);
    return sum;
}

}
